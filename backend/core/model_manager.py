import torch
import time
from typing import Optional, Union
from transformers import (
    AutoModel, 
    AutoModelForVision2Seq,
    AutoModelForImageClassification,
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BitsAndBytesConfig
)

from diffusers import DiffusionPipeline, StableDiffusionPipeline
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import logging
from typing import Optional
import gc
from config.settings import settings

logger = logging.getLogger(__name__)


class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self.device = settings.DEVICE
        self.dtype = torch.float16 if settings.USE_FP16 else torch.float32
        logger.info(f"Device: {self.device}, Dtype: {self.dtype}")
        
        # Model placeholders
        self.embedding_model: Optional[PreTrainedModel] = None
        self.embedding_processor: Optional[Union[AutoProcessor, PreTrainedTokenizerBase]] = None

        # Vision-Language Model (LLaVA → also supports Qwen-VL, Phi-Vision, etc.)
        self.vlm_model: Optional[AutoModelForVision2Seq] = None
        self.vlm_processor: Optional[AutoProcessor] = None

        # Text-to-Image Model (SD3 → supports SDXL, Flux, HunyuanDiT…)
        self.t2i_pipeline: Optional[DiffusionPipeline] = None
        self._initialized = True
    
    def _clear_cache(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def load_embedding_model(self):
        """Load embedding model (GPU preferred, automatic fallback if incompatible)."""
        if self.embedding_model is not None:
            logger.info("Embedding model already loaded; skipping.")
            return

        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")

        # Detect GPU capability
        gpu_ok = torch.cuda.is_available()
        fp16_ok = False
        target_device = self.device

        if gpu_ok:
            cc = torch.cuda.get_device_capability()
            logger.info(f"GPU Compute Capability: {cc}")
            # FP16 supported on CC >= 7.0 (Turing+)
            fp16_ok = cc[0] >= 7
        else:
            target_device = "cpu"

        try:
            self.embedding_processor = AutoProcessor.from_pretrained(
                settings.EMBEDDING_MODEL,
                cache_dir=settings.MODELS_DIR,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Embedding processor load failed: {e}")
            raise

        # Dynamic dtype selection
        dtype = torch.float16 if fp16_ok else torch.float32

        try:
            self.embedding_model = AutoModel.from_pretrained(
                settings.EMBEDDING_MODEL,
                cache_dir=settings.MODELS_DIR,
                torch_dtype=dtype,
                trust_remote_code=True
            ).to(target_device)

            self.embedding_model.eval()
            logger.info(f"Embedding model loaded on {target_device} (dtype={dtype}).")

        except Exception as e:
            logger.error(f"GPU load failed ({e}), falling back to CPU.")
            # Final guaranteed fallback
            self.embedding_model = AutoModel.from_pretrained(
                settings.EMBEDDING_MODEL,
                cache_dir=settings.MODELS_DIR,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to("cpu")
            self.embedding_model.eval()
            logger.info("Embedding model successfully loaded on CPU.")


    
    def load_vlm(self):
        """Load a Vision-Language model (LLaVA 1.6) on GPU without quantization."""
        if self.vlm_model is not None:
            logger.info("VLM already loaded; skipping.")
            return

        logger.info(f"Loading VLM: {settings.VISION_LANGUAGE_MODEL}")

        try:
            from transformers import LlavaForConditionalGeneration, LlavaProcessor
        except ImportError:
            raise ImportError(
                "Please install latest transformers version: pip install transformers>=4.40.0"
            )

        model_name = "llava-hf/llava-1.5-7b-hf"

        # Load processor
        try:
            self.vlm_processor = LlavaProcessor.from_pretrained(
                model_name,
                cache_dir=settings.MODELS_DIR
            )
        except Exception as e:
            logger.error(f"Processor load failed: {e}")
            raise

        # Load model
        try:
            self.vlm_model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                cache_dir=settings.MODELS_DIR,
                device_map="auto"
            )
        except Exception as e:
            logger.error(f"VLM load failed: {e}")
            raise

        self.vlm_model.eval()
        logger.info("LLaVA VLM successfully loaded on GPU")


    
    def load_t2i(self):
        """Load Stable Diffusion 1.5 for text-to-image with quantization"""
        if self.t2i_pipeline is not None:
            return

        logger.info("Loading Stable Diffusion 1.5 pipeline...")

        # Load the pipeline in half precision
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            cache_dir=settings.MODELS_DIR   
        ).to("cuda")                  # move to GPU


        # Assign pipeline
        self.t2i_pipeline = pipeline
        logger.info("Stable Diffusion 1.5 pipeline loaded on GPU.")

    
    def unload_vlm(self):
        """Free VLM memory"""
        if self.vlm_model is not None:
            del self.vlm_model
            del self.vlm_processor
            self.vlm_model = None
            self.vlm_processor = None
            self._clear_cache()
    
    def unload_t2i(self):
        """Free T2I memory"""
        if self.t2i_pipeline is not None:
            del self.t2i_pipeline
            self.t2i_pipeline = None
            self._clear_cache()
    
    @torch.no_grad()
    def get_image_embedding(self, image: Image.Image) -> torch.Tensor:
        """Extract image embedding using CLIP"""
        inputs = self.embedding_processor(
            images=image, 
            return_tensors="pt"
        ).to(self.device, self.dtype)
        
        outputs = self.embedding_model.get_image_features(**inputs)
        return outputs.cpu().float()
    
    @torch.no_grad()
    def get_text_embedding(self, text: str) -> torch.Tensor:
        """Extract text embedding using CLIP"""
        inputs = self.embedding_processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)
        
        outputs = self.embedding_model.get_text_features(**inputs)
        return outputs.cpu().float()
    

    # @torch.no_grad()
    # def generate_story(self, image: Image.Image, context: str) -> str:
    #     # Try conversation-style format for better results
    #     prompt = (
    #         "USER: <image>\n"
    #         "You are a creative storyteller. Carefully look at the image.\n"
    #         f"Context: {context}\n\n"
    #         "Write a detailed short story (2-3 paragraphs) describing the scene with rich emotion.\n"
    #         "ASSISTANT:"
    #     )

    #     inputs = self.vlm_processor(
    #         images=image,
    #         text=prompt,
    #         return_tensors="pt"
    #     ).to(self.device)

    #     output_ids = self.vlm_model.generate(
    #         **inputs,
    #         max_new_tokens=500,
    #         do_sample=True,
    #         temperature=0.8,
    #         top_p=0.95
    #     )

    #     # Decode only new tokens
    #     input_length = inputs['input_ids'].shape[1]
    #     generated_ids = output_ids[0][input_length:]
        
    #     story = self.vlm_processor.decode(
    #         generated_ids,
    #         skip_special_tokens=True
    #     ).strip()

    @torch.no_grad()
    def generate_story(self, image: Image.Image, context: str) -> str:
        prompt = (
            "USER: <image>\n"
            "You are a masterful storyteller in the style of Studio Ghibli and Hayao Miyazaki. "
            "Look deeply at this serene, whimsical scene and feel its quiet magic.\n"
            f"Context: {context}\n\n"
            "Write a heartfelt, poetic short story (1–2 paragraphs) as if narrating a Ghibli film. "
            "Use gentle, flowing prose, rich in emotion, nature, wonder, and subtle melancholy. "
            "Describe soft light, wind in the grass, glowing spirits, childhood innocence, and the beauty of fleeting moments.\n"
            "ASSISTANT:"
        )

        inputs = self.vlm_processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        output_ids = self.vlm_model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1
        )

        input_length = inputs['input_ids'].shape[1]
        generated_ids = output_ids[0][input_length:]
        story = self.vlm_processor.decode(generated_ids, skip_special_tokens=True).strip()

        # return story if story else "A quiet wind passes through the ancient forest..."



        return story if story else "No story generated."

    
    # @torch.no_grad()
    # def generate_image(self, prompt: str, style_hints: str = "") -> Image.Image:
    #     """Generate image from story with style guidance"""
    #     full_prompt = f"{prompt}"
    #     if style_hints:
    #         full_prompt += f", {style_hints}"
    #     full_prompt += ", masterpiece, highly detailed, 8k, cinematic"
        
    #     negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy"
        
    #     image = self.t2i_pipeline(
    #         prompt=full_prompt,
    #         negative_prompt=negative_prompt,
    #         num_inference_steps=settings.NUM_INFERENCE_STEPS,
    #         guidance_scale=settings.GUIDANCE_SCALE,
    #         height=settings.IMAGE_SIZE,
    #         width=settings.IMAGE_SIZE
    #     ).images[0]
        
    #     return image


    @torch.no_grad()
    def generate_image(self, prompt: str, style_hints: str = "") -> Image.Image:
        """Generate PERFECT Ghibli-style image with beautiful, consistent faces"""
        full_prompt = (
            f"{prompt}, "
            "studio ghibli style, hayao miyazaki masterpiece, "
            "soft warm lighting, gentle colors, detailed background, whimsical atmosphere, "
            "beautiful delicate face, large expressive eyes, smooth skin, perfect proportions, "
            "elegant hair flow, emotionally resonant expression, cinematic composition, "
            "highly detailed, 8k, flawless anatomy, ethereal glow, magical realism"
        )
        
        if style_hints:
            full_prompt += f", {style_hints}"

        negative_prompt = (
            "distorted face, deformed, bad anatomy, extra limbs, missing arms, fused fingers, "
            "too many fingers, long neck, username, watermark, text, ugly, tiling, "
            "poorly drawn hands, poorly drawn feet, blurry, low quality, "
            "out of frame, mutation, mutated, cross-eyed, disfigured, "
            "jpeg artifacts, lowres, cropped"
        )

        image = self.t2i_pipeline(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=settings.NUM_INFERENCE_STEPS,
            guidance_scale=settings.GUIDANCE_SCALE,
            height=settings.IMAGE_SIZE,
            width=settings.IMAGE_SIZE,
            generator=torch.Generator(device=self.device).manual_seed(int(time.time()))  # optional variety
        ).images[0]

        return image

model_manager = ModelManager()