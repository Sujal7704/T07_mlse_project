import logging
import time
import threading
from typing import Dict, Any
from PIL import Image

from core.model_manager import model_manager
from core.rag_engine import rag_engine
from core.queue_manager import queue_manager, JobStatus
from core.observability import observer
from utils.image_utils import decode_base64_image, encode_image_to_base64
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------- small helpers ----------
def _safe_excerpt(text: str, length: int = 200) -> str:
    if not isinstance(text, str):
        return ""
    return text[:length] + ("..." if len(text) > length else "")


def _maybe_heartbeat(job_id: str):
    """
    Optionally update the job heartbeat so external orchestrators know the worker is alive.
    If your queue_manager doesn't expose this method, this is a no-op (safe).
    """
    try:
        if hasattr(queue_manager, "update_job_heartbeat"):
            queue_manager.update_job_heartbeat(job_id)
    except Exception as e:
        logger.debug(f"Heartbeat update failed for job {job_id}: {e}")


# ---------- Generation worker ----------
class GenerationWorker:
    def __init__(self, worker_type: str):
        self.worker_type = worker_type  # 'story' or 'image'
        self._shutdown_event = threading.Event()
        logger.info(f"Initialized {worker_type} worker")

    # Decorate the high-level processing functions with observer trace
    @observer.trace_generation("story_generation")
    def process_story_job(self, job_id: str, job_data: dict) -> Dict[str, Any]:
        """
        Process: image -> retrieve -> generate story -> index story
        Returns a dict (including any observer summary injected by the observer).
        """
        start_ts = time.time()
        story_result = {"job_id": job_id}

        try:
            queue_manager.update_job_status(job_id, JobStatus.RUNNING)
            _maybe_heartbeat(job_id)

            # --- Decode image ---
            with observer.step("decode_image"):
                image = decode_base64_image(job_data["image"])
                observer.log_artifact("input_image_meta", {"mode": image.mode, "size": image.size}, content_type="image_meta")

            _maybe_heartbeat(job_id)

            # --- Image embedding ---
            with observer.step("image_embedding"):
                image_embedding = model_manager.get_image_embedding(image)
                # embedding shape assumed (1, dim) or (dim,) - try to get dimension robustly
                dim = image_embedding.shape[-1] if hasattr(image_embedding, "shape") else None
                observer.log_embedding("image", dimension=dim)
                story_result["embedding_dim"] = dim

            _maybe_heartbeat(job_id)

            # --- RAG retrieval ---
            with observer.step("rag_retrieval"):
                similar_stories = rag_engine.retrieve_stories(image_embedding, k=settings.TOP_K_RETRIEVAL)
                observer.log_rag_metrics("story", settings.TOP_K_RETRIEVAL, len(similar_stories))

                # log a small safe sample of retrieved contexts
                sample_snippets = [(_safe_excerpt(s), None) if isinstance(s, str) else ("<non-text>", None) for s in similar_stories[:3]]
                observer.log_artifact("rag_retrieval_sample", [s for s, _ in sample_snippets], content_type="json", extra={"k": settings.TOP_K_RETRIEVAL})

            _maybe_heartbeat(job_id)

            # --- Build context and generate ---
            with observer.step("build_context_and_generate"):
                context = "\n\n".join([f"Example {i+1}:\n{s}" for i, s in enumerate(similar_stories)])
                # enforce some size limits on context
                if len(context) > settings.MAX_CONTEXT_CHARS:
                    context = context[: settings.MAX_CONTEXT_CHARS]
                story = model_manager.generate_story(image, context)
                observer.log_artifact("generated_story_excerpt", _safe_excerpt(story, 500), content_type="text")

            _maybe_heartbeat(job_id)

            # --- Index generated story into RAG store ---
            with observer.step("index_story"):
                story_embedding = model_manager.get_text_embedding(story)
                rag_engine.add_story(story_embedding, story, {"job_id": job_id})
                observer.log_rag_metrics("story_indexed", k=1, num_results=1)  # simple metric to show we indexed

            # --- Finalize job ---
            queue_manager.update_job_status(job_id, JobStatus.COMPLETED, result={"story": story})
            logger.info(f"Completed story job {job_id} in {time.time() - start_ts:.2f}s")
            return {"story": story}

        except Exception as e:
            # structured failure telemetry and job update
            logger.error(f"Story job {job_id} failed: {e}", exc_info=True)
            observer.log_failure(stage="process_story_job", error=e, context={"job_id": job_id})
            try:
                queue_manager.update_job_status(job_id, JobStatus.FAILED, error=str(e))
            except Exception as q_e:
                logger.debug(f"Failed to update job status to FAILED for {job_id}: {q_e}")
            raise

    @observer.trace_generation("image_generation")
    def process_image_job(self, job_id: str, job_data: dict) -> Dict[str, Any]:
        """
        Process: story -> retrieve -> extract style -> generate image -> index image
        """
        start_ts = time.time()
        try:
            queue_manager.update_job_status(job_id, JobStatus.RUNNING)
            _maybe_heartbeat(job_id)

            story: str = job_data["story"]
            short_story = _safe_excerpt(story, getattr(settings, "MAX_STORY_EXCERPT", 500))

            # --- Text embedding ---
            with observer.step("text_embedding"):
                text_embedding = model_manager.get_text_embedding(story)
                dim = text_embedding.shape[-1] if hasattr(text_embedding, "shape") else None
                observer.log_embedding("text", dimension=dim)
            
            _maybe_heartbeat(job_id)

            # --- RAG retrieval for image metadata ---
            with observer.step("rag_retrieve_image_metadata"):
                similar_metadata = rag_engine.retrieve_image_metadata(text_embedding, k=settings.TOP_K_RETRIEVAL)
                observer.log_rag_metrics("image", settings.TOP_K_RETRIEVAL, len(similar_metadata))
                # capture style hints safely
                style_hints = ", ".join([meta.get("style", "") for meta in similar_metadata if meta.get("style")])
                observer.log_artifact("style_hints", _safe_excerpt(style_hints, 200), content_type="text")

            _maybe_heartbeat(job_id)

            # --- Compose prompt and generate image ---
            with observer.step("compose_prompt_and_generate_image"):
                # create a concise prompt for image generation (trim long stories)
                prompt_for_image = short_story if len(short_story) < 500 else short_story[:500]
                image = model_manager.generate_image(prompt_for_image, style_hints)
                # inspect/limit image size if needed
                if hasattr(image, "size") and (image.size[0] * image.size[1] > settings.MAX_PIXEL_COUNT):
                    # optionally resize to guard against massively large images
                    resized = image.copy()
                    resized.thumbnail(settings.MAX_IMAGE_DIM)
                    image = resized
                    observer.log_artifact("image_resized", {"original_size": image.size, "max_dim": settings.MAX_IMAGE_DIM}, content_type="image_meta")

                # store image metadata, not raw bytes
                observer.log_artifact("generated_image_meta", {"size": image.size, "mode": image.mode}, content_type="image_meta")

            _maybe_heartbeat(job_id)

            # --- Persist image and index embedding ---
            with observer.step("encode_and_index_image"):
                image_b64 = encode_image_to_base64(image)  
                image_embedding = model_manager.get_image_embedding(image)
                rag_engine.add_image(image_embedding, {
                    "job_id": job_id,
                    "style": style_hints,
                    "story_excerpt": short_story
                })

                # Don't add prefix again - it's already there!
                queue_manager.update_job_status(
                    job_id,
                    JobStatus.COMPLETED,
                    result={"image_url": image_b64}  # Use image_b64 directly
                )
                return {"image_url": image_b64}

        except Exception as e:
            logger.error(f"Image job {job_id} failed: {e}", exc_info=True)
            observer.log_failure(stage="process_image_job", error=e, context={"job_id": job_id})
            try:
                queue_manager.update_job_status(job_id, JobStatus.FAILED, error=str(e))
            except Exception as q_e:
                logger.debug(f"Failed to update job status to FAILED for {job_id}: {q_e}")
            raise

    def _shutdown(self):
        self._shutdown_event.set()

    def run(self):
        """
        Main worker loop.
        Loads models and processes jobs until shutdown requested.
        """
        logger.info(f"Starting {self.worker_type} worker...")
        # Load models up front (fail early)
        model_manager.load_embedding_model()
        rag_engine.load_or_create_indices()

        if self.worker_type == "story":
            model_manager.load_vlm()
        else:
            model_manager.load_t2i()

        logger.info("Worker ready, entering main loop")

        backoff = 0.5
        try:
            while not self._shutdown_event.is_set():
                try:
                    job_id = queue_manager.get_next_job(self.worker_type)
                    if not job_id:
                        # no work, back off slightly
                        time.sleep(backoff)
                        backoff = min(backoff * 1.5, 5.0)
                        continue
                    backoff = 0.5

                    job_data = queue_manager.get_job(job_id)
                    if not job_data:
                        logger.debug(f"Job {job_id} disappeared or invalid; skipping")
                        continue

                    logger.info(f"Processing {self.worker_type} job {job_id}")

                    # depending on worker_type call the appropriate processing routine
                    if self.worker_type == "story":
                        self.process_story_job(job_id, job_data["data"])
                    else:
                        self.process_image_job(job_id, job_data["data"])

                except KeyboardInterrupt:
                    logger.info("Worker received KeyboardInterrupt, shutting down gracefully")
                    self._shutdown()
                except Exception as e:
                    # top-level worker exception handling; keep worker alive
                    logger.exception(f"Worker loop error: {e}")
                    time.sleep(1)
        finally:
            logger.info("Worker exiting, attempting graceful shutdown of models if supported")
            try:
                if hasattr(model_manager, "unload_all"):
                    model_manager.unload_all()
            except Exception as e:
                logger.debug(f"Error during model unload: {e}")


# ---------- CLI ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generation_worker.py [story|image]")
        sys.exit(1)

    worker_type = sys.argv[1]
    if worker_type not in ["story", "image"]:
        print("Worker type must be 'story' or 'image'")
        sys.exit(1)

    worker = GenerationWorker(worker_type)
    try:
        worker.run()
    except Exception:
        logger.exception("Uncaught worker exception, exiting")
        raise
