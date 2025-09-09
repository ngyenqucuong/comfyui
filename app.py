import torch
from nodes import NODE_CLASS_MAPPINGS,load_custom_node
import warnings
import os
import gc
import random
import numpy as np
import io
from PIL import Image
from totoro_extras import nodes_custom_sampler
from totoro_extras import nodes_post_processing
from totoro_extras import nodes_flux
from totoro_extras import nodes_mask
from totoro import model_management
from fastapi import FastAPI, File, UploadFile, Form, HTTPException 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from concurrent.futures import ThreadPoolExecutor
import uvicorn

from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime
import uuid
import asyncio
import json
import logging


if not load_custom_node("custom_nodes/TotoroUI-GGUF"):
  raise Exception("Failed to load GGUF custom node")

if not load_custom_node("custom_nodes/TotoroUI-PuLID-Flux"):
  raise Exception("Failed to load PuLID Flux Enhanced custom node")


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

DualCLIPLoaderGGUF = NODE_CLASS_MAPPINGS["DualCLIPLoaderGGUF"]()
UnetLoaderGGUF = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
PulidFluxModelLoader = NODE_CLASS_MAPPINGS["PulidFluxModelLoader"]()
PulidFluxInsightFaceLoader = NODE_CLASS_MAPPINGS["PulidFluxInsightFaceLoader"]()
PulidFluxEvaClipLoader = NODE_CLASS_MAPPINGS["PulidFluxEvaClipLoader"]()
executor = ThreadPoolExecutor(max_workers=1)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CLIPTextEncodeFlux = nodes_flux.NODE_CLASS_MAPPINGS["CLIPTextEncodeFlux"]()
RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
MaskToImage = nodes_mask.NODE_CLASS_MAPPINGS["MaskToImage"]()
ImageToMask = nodes_mask.NODE_CLASS_MAPPINGS["ImageToMask"]()
LoraLoader = NODE_CLASS_MAPPINGS["LoraLoader"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
VAEEncode = NODE_CLASS_MAPPINGS["VAEEncode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
ImageScaleToTotalPixels = nodes_post_processing.NODE_CLASS_MAPPINGS["ImageScaleToTotalPixels"]()
ApplyPulidFlux = NODE_CLASS_MAPPINGS["ApplyPulidFlux"]()

def initialize_pipelines():
  """Initialize the diffusion pipelines with InstantID and SDXL-Lightning - GPU optimized"""
  global unet_f, clip_f,pulid,face_analysis,eva_clip,vae,unet,clip
  with torch.inference_mode():
    eva_clip = PulidFluxEvaClipLoader.load_eva_clip()[0]
    print("Loading VAE...")
    vae = VAELoader.load_vae("ae.sft")[0]
    print(f"Loading Flux1-dev-Q4_K_S...")
    unet = UnetLoaderGGUF.load_unet(f"flux1-dev-Q4_K_S.gguf")[0]
    print("Loading Clips...")
    clip = DualCLIPLoaderGGUF.load_clip("t5-v1_1-xxl-encoder-Q6_K.gguf", "clip_l.safetensors", "flux")[0]
    print("Loading PuLID...")
    pulid = PulidFluxModelLoader.load_model(f"pulid_flux_v0.9.1.safetensors")[0]
    face_analysis = PulidFluxInsightFaceLoader.load_insightface("CPU")[0]

    unet_f, clip_f = unet, clip


def img_tensor_to_np(img_tensor):
  img_tensor = img_tensor.clone() * 255.0
  return img_tensor.squeeze().numpy().astype(np.uint8)

def img_np_to_tensor(img_np_list):
  return torch.from_numpy(img_np_list.astype(np.float32) / 255.0).unsqueeze(0)

def cuda_gc():
  try:
    model_management.soft_empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
  except:
    pass

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2

@torch.inference_mode()
def generate(prompt, fixed_seed, guidance, steps, sampler_name, scheduler, weight, start_at, end_at, face_img=None, pose_image=None, mask_img=None, denoise=1.0):
  global unet, clip, unet_f, clip_f

  unet_f, clip_f = unet, clip

  print("Prompt Received")

  image = LoadImage.load_image(face_img)[0]
  latent_image = ImageScaleToTotalPixels.upscale(image, "lanczos", 1.0)[0]
  latent_image = VAEEncode.encode(vae, latent_image)[0]

  cond = CLIPTextEncodeFlux.encode(clip_f, prompt, prompt, guidance)[0]

  pulid_image = LoadImage.load_image(pose_image)[0]

  mask_np = np.array(mask_img).astype(np.uint8)
  mask_image = img_np_to_tensor(mask_np)

  mask = ImageToMask.image_to_mask(mask_image, "red")[0]

  unet_f = ApplyPulidFlux.apply_pulid_flux(unet_f, pulid, eva_clip, face_analysis, pulid_image, weight, start_at, end_at, attn_mask=mask)[0]

  print("PuLID Applied")

  guider = BasicGuider.get_guider(unet_f, cond)[0]
  sampler = KSamplerSelect.get_sampler(sampler_name)[0]
  sigmas = BasicScheduler.get_sigmas(unet_f, scheduler, steps, denoise)[0]

  if fixed_seed == 0:
    seed = random.randint(0, 18446744073709551615)
  else:
    seed = fixed_seed

  logger.info("Seed: %d", seed)

  noise = RandomNoise.get_noise(seed)[0]
  sample, sample_denoised = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
  model_management.soft_empty_cache()
  decoded = VAEDecode.decode(vae, sample)[0].detach()
  result = Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0])
  del unet_f
  del clip_f
  cuda_gc()
  return result




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipelines on startup"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, initialize_pipelines)
    yield


app = FastAPI(title="SDXL Face Swap API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="."), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)







class Img2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
    seed: Optional[int] = None
    strength: float = 0.8
    ip_adapter_scale: float = 0.8  # Lower for InstantID
    controlnet_conditioning_scale: float = 0.8
    guidance_scale: float = 0.0  # Zero for LCM
    num_inference_steps: int = 8

class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    result_url: Optional[str] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    created_at: datetime
    completed_at: Optional[datetime] = None

jobs = {}
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

async def gen_img2img(job_id: str, face_image : Image.Image, pose_image: Image.Image, mask_image: Image.Image, request: Img2ImgRequest):
  sampler_name = "euler" # @param ["euler","heun","heunpp2","heunpp2","dpm_2","lms","dpmpp_2m","ipndm","deis","ddim","uni_pc","uni_pc_bh2"]
  scheduler = "simple" # @param ["normal","sgm_uniform","simple","ddim_uniform"]
  denoise = 0.75 # @param {"type":"slider","min":0,"max":1,"step":0.01}
  pulid_weight = 1 # @param {"type":"slider","min":-1,"max":5,"step":0.05}
  pulid_start_at = 0.015 # @param {"type":"slider","min":0,"max":1,"step":0.001}
  pulid_end_at = 1 # @param {"type":"slider","min":0,"max":1,"step":0.001}
  # prompt = ""
  result = generate(request.prompt, request.seed, request.guidance_scale, request.num_inference_steps, sampler_name, scheduler, pulid_weight, pulid_start_at, pulid_end_at, face_image, pose_image, mask_image, denoise)
  filename = f"{job_id}_base.png"
  filepath = os.path.join(results_dir, filename)
  result.save(filepath)
      
  metadata = {
      "job_id": job_id,
      "type": "head_swap",
      "prompt": request.prompt,
      "parameters": request.dict(),
      "filename": filename,
      "device_used": 'cuda',
  }
      
  metadata_path = os.path.join(results_dir, f"{job_id}_metadata.json")
  with open(metadata_path, 'w') as f:
      json.dump(metadata, f, indent=2, default=str)
  
  jobs[job_id]["status"] = "completed"
  jobs[job_id]["progress"] = 1.0
  jobs[job_id]["result_url"] = f"/results/{filename}"
  jobs[job_id]["metadata"] = metadata
  jobs[job_id]["completed_at"] = datetime.now()
  
  logger.info(f"Img2img completed successfully on cuda")



@app.get("/", response_class=HTMLResponse)
async def serve_web_interface():
    """Serve the web interface"""
    try:
        with open("img2img_interface.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Web interface not found</h1>")

@app.get("/web", response_class=HTMLResponse)
async def serve_web_interface_alt():
    """Alternative route for web interface"""
    return await serve_web_interface()

@app.get("/health")
async def health_check():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
            "gpu_memory_allocated": torch.cuda.memory_allocated(),
            "gpu_memory_cached": torch.cuda.memory_reserved()
        }
   
    
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "gpu_info": gpu_info
    }


@app.post("/img2img")
async def img2img(
    base_image: UploadFile = File(...),
    pose_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    prompt: str = Form(""),
    negative_prompt: str = Form("(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"),
    strength: float = Form(0.85),
    ip_adapter_scale: float = Form(0.8),  # Lower for InstantID
    controlnet_conditioning_scale: float = Form(0.8),
    num_inference_steps: int = Form(8),
    guidance_scale: float = Form(0),  # Zero for LCM
    seed: Optional[int] = Form(None),
    
):
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "created_at": datetime.now(),
        "type": "head_swap"
    }
    
    try:
        # Load images
        base_img = Image.open(io.BytesIO(await base_image.read())).convert('RGB')
        pose_img = Image.open(io.BytesIO(await pose_image.read())).convert('RGB')
        mask_img = Image.open(io.BytesIO(await mask_image.read())).convert('RGBA')
        request = Img2ImgRequest(
            num_inference_steps=num_inference_steps,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            strength=strength,
            ip_adapter_scale=ip_adapter_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            mask_img=mask_img
        )
        # Start background task
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, lambda: asyncio.run(
            gen_img2img(job_id, base_img, pose_img,mask_img, request)
        ))
        
        return {"job_id": job_id, "status": "pending"}
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error_message"] = str(e)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "result_url": job.get("result_url"),
        "seed": job.get("metadata", {}).get("seed"),
        "error_message": job.get("error_message"),
        "created_at": job["created_at"].isoformat(),
        "completed_at": job.get("completed_at").isoformat() if job.get("completed_at") else None
    }

@app.get("/results/{filename}")
async def get_result(filename: str):
    """Get result image"""
    filepath = os.path.join(results_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(filepath)


@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    try:
        job_list = []
        for job_id, job_data in jobs.items():
            job_list.append({
                "job_id": job_id,
                "status": job_data.get("status", "unknown"),
                "created_at": job_data.get("created_at", datetime.now()).isoformat(),
                "completed_at": job_data.get("completed_at").isoformat() if job_data.get("completed_at") else None,
                "result_url": job_data.get("result_url"),
                "error_message": job_data.get("error_message")
            })
        
        job_list.sort(key=lambda x: x["created_at"], reverse=True)
        return job_list
    except Exception as e:
        logger.error(f"Error getting jobs: {e}")
        return []

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete files
    job = jobs[job_id]
    if "metadata" in job and "filename" in job["metadata"]:
        filename = job["metadata"]["filename"]
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Delete metadata file
        metadata_path = os.path.join(results_dir, f"{job_id}_metadata.json")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
    
    # Remove from jobs
    del jobs[job_id]
    
    return {"message": "Job deleted successfully"}

if __name__ == "__main__":
    
    # Set environment variables for better CUDA error reporting
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    uvicorn.run(app, host="0.0.0.0", port=8888)