import time
import logging
import hashlib
import json
from functools import wraps
from typing import Optional, Dict, Any, Iterable
from collections import deque
import threading

import torch

from config.settings import settings

logger = logging.getLogger(__name__)

# Conditional Langfuse import
if settings.ENABLE_LANGFUSE:
    try:
        from langfuse import Langfuse, get_client
        langfuse_client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
            sample_rate=getattr(settings, "LANGFUSE_SAMPLE_RATE", 1.0) 
        )
    except Exception as e:
        logger.warning(f"Langfuse init failed: {e}")
        langfuse_client = None
else:
    langfuse_client = None


def _safe_hash(data: Any) -> str:
    """Return a compact SHA1 hex for a serializable artifact (not the content)."""
    try:
        raw = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        raw = str(data).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


class StepTrace:
    def __init__(self, name: str, parent_trace_id: Optional[str] = None):
        self.name = name
        self.parent_trace_id = parent_trace_id
        self._obs_ctx = None
        self.start = None

    def __enter__(self):
        self.start = time.time()
        logger.debug(f"[STEP START] {self.name}")
        if langfuse_client:
            try:
                # start a child span; if you need to force trace_id, you can pass trace_context
                trace_context = None
                if self.parent_trace_id:
                    # create a minimal trace_context so the observation attaches to the parent trace
                    trace_context = {"trace_id": self.parent_trace_id, "parent_span_id": "0000000000000000"}
                self._obs_ctx = langfuse_client.start_as_current_observation(as_type="span", name=self.name, trace_context=trace_context)
                self._obs_ctx.__enter__()
            except Exception as e:
                logger.debug(f"Langfuse step start failed: {e}")
        return self

    def __exit__(self, exc_type, exc, tb):
        duration = time.time() - (self.start or time.time())
        logger.info(f"[STEP END] {self.name}: {duration:.3f}s")
        if langfuse_client:
            try:
                # you can add metadata/event for duration
                try:
                    trace_id = langfuse_client.get_current_trace_id()
                except Exception:
                    trace_id = None
                langfuse_client.event(name=f"step_{self.name}_end", metadata={"duration": duration}, trace_id=trace_id)
            except Exception as e:
                logger.debug(f"Langfuse step end failed: {e}")
            finally:
                try:
                    if self._obs_ctx:
                        self._obs_ctx.__exit__(exc_type, exc, tb)
                except Exception:
                    pass

class Observer:
    def __init__(self, gpu_history_len: int = 100):
        # ring buffer for GPU allocation history (thread-safe)
        self._gpu_history = deque(maxlen=gpu_history_len)
        self._gpu_lock = threading.Lock()

    # ---- GPU helpers ----
    @staticmethod
    def get_gpu_memory() -> Dict[str, float]:
        """Get current GPU memory usage (per-process numbers)"""
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9
        }

    def update_gpu_history(self):
        """Append current allocated memory to the history buffer."""
        mem = self.get_gpu_memory()
        val = mem.get("allocated_gb")
        if val is None:
            return
        with self._gpu_lock:
            self._gpu_history.append((time.time(), val))

    def gpu_trend(self, window_seconds: float = 300.0) -> Dict[str, Any]:
        """Return simple GPU trending stats over the recent window (seconds)."""
        cutoff = time.time() - window_seconds
        with self._gpu_lock:
            items = [v for (t, v) in self._gpu_history if t >= cutoff]
        if not items:
            return {"count": 0}
        import statistics
        return {
            "count": len(items),
            "mean_allocated_gb": statistics.mean(items),
            "max_allocated_gb": max(items),
            "min_allocated_gb": min(items),
            "std_allocated_gb": statistics.pstdev(items) if len(items) > 1 else 0.0,
        }

    # ---- Tracing and decorators ----
    def trace_generation(self, generation_type: str):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                initial_memory = self.get_gpu_memory()
                self.update_gpu_history()

                trace_id = None
                # Start a Langfuse trace/observation context if client exists
                if langfuse_client:
                    try:
                        # use the context manager to start an observation (span/trace)
                        # as_type can be "span" or "trace" depending on SDK - using span is compatible
                        obs_ctx = langfuse_client.start_as_current_observation(as_type="span", name=f"{generation_type}_generation")
                    except Exception as e:
                        logger.debug(f"Langfuse start observation failed: {e}")
                        obs_ctx = None
                else:
                    obs_ctx = None

                summary = {
                    "generation_type": generation_type,
                    "start_time": start_time,
                    "initial_memory": initial_memory,
                    "steps": {},
                }

                try:
                    # enter the observation context if present
                    if obs_ctx:
                        obs_cm = obs_ctx.__enter__()
                        try:
                            # optionally get trace id from client helper (safe guard)
                            try:
                                trace_id = langfuse_client.get_current_trace_id()
                            except Exception:
                                trace_id = None
                        except Exception:
                            pass

                    result = func(*args, **kwargs)

                    duration = time.time() - start_time
                    final_memory = self.get_gpu_memory()
                    self.update_gpu_history()
                    trend = self.gpu_trend()

                    summary.update({
                        "duration_seconds": duration,
                        "final_memory": final_memory,
                        "gpu_trend": trend
                    })

                    logger.info(
                        f"{generation_type} generation: {duration:.2f}s, "
                        f"GPU allocated: {final_memory.get('allocated_gb', 0):.2f}GB"
                    )

                    # send concise metrics/events to langfuse
                    if langfuse_client and trace_id:
                        try:
                            # attach a numeric score / metric
                            langfuse_client.score(trace_id=trace_id, name="generation_time", value=duration)
                            # attach full summary as an event
                            langfuse_client.event(name="generation_summary", metadata=summary, trace_id=trace_id)
                        except Exception as e:
                            logger.debug(f"Langfuse scoring/event failed: {e}")

                    # flush buffered data to ensure delivery (important for short lived jobs)
                    if langfuse_client:
                        try:
                            langfuse_client.flush()
                        except Exception as e:
                            logger.debug(f"Langfuse flush failed: {e}")

                    # attach trace id into result for correlation
                    if isinstance(result, dict):
                        result.setdefault("_observer", {})["trace_id"] = trace_id
                        result["_observer"]["summary"] = summary
                    return result

                except Exception as e:
                    duration = time.time() - start_time
                    final_memory = self.get_gpu_memory()
                    summary.update({
                        "duration_seconds": duration,
                        "final_memory": final_memory,
                        "error": str(e)
                    })
                    logger.exception(f"{generation_type} failed after {duration:.2f}s: {e}")

                    if langfuse_client and trace_id:
                        try:
                            langfuse_client.score(trace_id=trace_id, name="error", value=str(e))
                            langfuse_client.flush()
                        except Exception:
                            logger.debug("Langfuse error score/flush failed")

                    raise
                finally:
                    # ensure we exit the observation context
                    if obs_ctx:
                        try:
                            obs_ctx.__exit__(None, None, None)
                        except Exception:
                            pass

            return wrapper
        return decorator

    # ---- Step-level context manager (exposed) ----
    def step(self, name: str, parent_trace_id: Optional[str] = None) -> StepTrace:
        """Return a StepTrace context manager usable in `with` blocks."""
        return StepTrace(name=name, parent_trace_id=parent_trace_id)

    # ---- Structured logging helpers ----
    def log_artifact(self, name: str, content: Any, content_type: str = "text", extra: Optional[Dict[str, Any]] = None):
        """
        Log an artifact. We avoid logging raw binaries; instead we hash them and record size/summary.
        content_type: "text" | "image_meta" | "json" | etc.
        """
        try:
            size = None
            if hasattr(content, "__len__") and not isinstance(content, (dict,)):
                try:
                    size = len(content)
                except Exception:
                    size = None

            artifact_hash = _safe_hash({"name": name, "type": content_type, "content_sample": content if isinstance(content, str) and len(content) <= 512 else str(content)[:256]})
            metadata = {"name": name, "type": content_type, "hash": artifact_hash, "size": size}
            if extra:
                metadata.update(extra)

            logger.debug(f"[ARTIFACT] {name} meta={metadata}")
            if langfuse_client:
                try:
                    langfuse_client.event(name=f"artifact_{name}", metadata=metadata)
                except Exception as e:
                    logger.debug(f"Langfuse artifact logging failed: {e}")
        except Exception:
            logger.exception("artifact logging failed")

    def log_embedding(self, embedding_type: str, dimension: int, latency_s: Optional[float] = None, model_name: Optional[str] = None, coherence_score: Optional[float] = None):
        """Log embedding extraction with optional coherence and latency metadata."""
        payload = {
            "embedding_type": embedding_type,
            "dimension": dimension,
            "latency_s": latency_s,
            "model": model_name,
            "coherence": coherence_score
        }
        logger.debug(f"[EMBEDDING] {payload}")
        if langfuse_client:
            try:
                langfuse_client.generation(name=f"{embedding_type}_embedding", metadata=payload)
            except Exception as e:
                logger.debug(f"Langfuse embedding logging failed: {e}")

    def log_rag_metrics(self, content_type: str, k: int, num_results: int, scores: Optional[Iterable[float]] = None, used: Optional[int] = None):
        """Log RAG retrieval diagnostics. 'scores' - iterable of similarity scores if available."""
        import statistics
        payload = {
            "content_type": content_type,
            "requested_k": k,
            "returned": num_results,
            "used": used if used is not None else min(num_results, k),
        }
        if scores:
            scores = list(scores)
            payload.update({
                "avg_score": float(statistics.mean(scores)) if scores else None,
                "min_score": float(min(scores)) if scores else None,
                "max_score": float(max(scores)) if scores else None,
            })
        logger.info(f"[RAG] {payload}")
        if langfuse_client:
            try:
                langfuse_client.generation(name=f"rag_{content_type}", metadata=payload)
            except Exception as e:
                logger.debug(f"Langfuse RAG logging failed: {e}")

    def log_failure(self, stage: str, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Structured failure telemetry so you can filter by stage and error types in observability tooling."""
        payload = {
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        logger.error(f"[FAILURE] {payload}")
        if langfuse_client:
            try:
                langfuse_client.generation(name="failure_event", metadata=payload)
            except Exception as e:
                logger.debug(f"Langfuse failure logging failed: {e}")

    def summarize_trace(self, story_id: Optional[str], generation_type: str, start_time: float, end_time: float, stages: Dict[str, float], initial_memory: Dict[str, float], final_memory: Dict[str, float]) -> Dict[str, Any]:
        """Create a compact summary for end-to-end traces (serializable)."""
        total = end_time - start_time
        summary = {
            "story_id": story_id,
            "generation_type": generation_type,
            "duration_seconds": total,
            "stages": stages,
            "initial_memory": initial_memory,
            "final_memory": final_memory,
            "gpu_trend_5m": self.gpu_trend(window_seconds=300)
        }
        logger.info(f"[TRACE SUMMARY] {summary}")
        if langfuse_client:
            try:
                langfuse_client.event(name="trace_summary", metadata=summary)
            except Exception as e:
                logger.debug(f"Langfuse trace summary failed: {e}")
        return summary


# Singleton instance for convenience
observer = Observer()
