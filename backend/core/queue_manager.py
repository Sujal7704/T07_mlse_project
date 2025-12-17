import redis
import json
import uuid
from enum import Enum
from typing import Optional, Dict, Any
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class QueueManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )
        self.story_queue = "story_generation_queue"
        self.image_queue = "image_generation_queue"
    
    def create_job(self, job_type: str, data: Dict[str, Any]) -> str:
        """Create a new job and add to queue"""
        job_id = str(uuid.uuid4())
        
        job_data = {
            "job_id": job_id,
            "type": job_type,
            "status": JobStatus.PENDING,
            "data": data,
            "result": None,
            "error": None
        }
        
        # Store job metadata
        self.redis_client.setex(
            f"job:{job_id}",
            settings.JOB_TIMEOUT,
            json.dumps(job_data)
        )
        
        # Add to appropriate queue
        queue_name = self.story_queue if job_type == "story" else self.image_queue
        self.redis_client.lpush(queue_name, job_id)
        
        logger.info(f"Created job {job_id} of type {job_type}")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job status and data"""
        job_json = self.redis_client.get(f"job:{job_id}")
        if job_json:
            return json.loads(job_json)
        return None
    
    def update_job_status(self, job_id: str, status: JobStatus, 
                         result: Any = None, error: str = None):
        """Update job status"""
        job_data = self.get_job(job_id)
        if not job_data:
            return
        
        job_data["status"] = status
        if result is not None:
            job_data["result"] = result
        if error is not None:
            job_data["error"] = error
        
        self.redis_client.setex(
            f"job:{job_id}",
            settings.JOB_TIMEOUT,
            json.dumps(job_data)
        )
    
    def get_next_job(self, job_type: str) -> Optional[str]:
        """Get next job from queue (blocking)"""
        queue_name = self.story_queue if job_type == "story" else self.image_queue
        result = self.redis_client.brpop(queue_name, timeout=1)
        
        if result:
            return result[1]
        return None
    
    def get_queue_length(self, job_type: str) -> int:
        """Get current queue length"""
        queue_name = self.story_queue if job_type == "story" else self.image_queue
        return self.redis_client.llen(queue_name)

queue_manager = QueueManager()