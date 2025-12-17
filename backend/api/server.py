from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

from core.queue_manager import queue_manager
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Story-Image Generator API",
    description="Bidirectional Story-Image Generation with RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class ImageToStoryRequest(BaseModel):
    image: str

class StoryToImageRequest(BaseModel):
    story: str

# Response Models
class JobResponse(BaseModel):
    job_id: str
    status: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: dict | None = None
    error: str | None = None

class HealthResponse(BaseModel):
    status: str
    queues: dict

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check and queue status"""
    return HealthResponse(
        status="healthy",
        queues={
            "story_queue": queue_manager.get_queue_length("story"),
            "image_queue": queue_manager.get_queue_length("image")
        }
    )

@app.post("/generate-story", response_model=JobResponse)
async def generate_story(request: ImageToStoryRequest):
    """
    Generate story from image (async)
    Returns job_id to poll for results
    """
    try:
        job_id = queue_manager.create_job(
            job_type="story",
            data={"image": request.image}
        )
        
        return JobResponse(
            job_id=job_id,
            status="pending"
        )
        
    except Exception as e:
        logger.error(f"Error creating story job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-image", response_model=JobResponse)
async def generate_image(request: StoryToImageRequest):
    """
    Generate image from story (async)
    Returns job_id to poll for results
    """
    try:
        job_id = queue_manager.create_job(
            job_type="image",
            data={"story": request.story}
        )
        
        return JobResponse(
            job_id=job_id,
            status="pending"
        )
        
    except Exception as e:
        logger.error(f"Error creating image job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll job status and retrieve results"""
    try:
        job_data = queue_manager.get_job(job_id)
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobStatusResponse(
            job_id=job_data["job_id"],
            status=job_data["status"],
            result=job_data.get("result"),
            error=job_data.get("error")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level="info"
    )