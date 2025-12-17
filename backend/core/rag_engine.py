import faiss
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict
import torch

from config.settings import settings

logger = logging.getLogger(__name__)

# rag_engine.py
import json
import faiss
import torch
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from PIL import Image
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(
        self,
        dataset_csv: str = "/home/ashish/Desktop/202418007/RAVSG/backend/data/merged_dataset.csv",
        index_dir: Optional[str] = None
    ):
        self.dataset_csv = Path(dataset_csv)
        self.index_dir = settings.DATABASE_DIR / "rag_indices" if index_dir is None else Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.caption_index_path = self.index_dir / "caption_index.faiss"      # text embeddings
        self.image_index_path   = self.index_dir / "image_index.faiss"        # image embeddings
        self.caption_db_path     = self.index_dir / "captions.json"
        self.image_db_path       = self.index_dir / "images.json"

        # CLIP
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(settings.EMBEDDING_MODEL)
        self.model = CLIPModel.from_pretrained(settings.EMBEDDING_MODEL).to(self.device)
        self.model.eval()

        self.dimension = settings.EMBEDDING_DIM

        # Indices & storage
        self.caption_index: faiss.Index = None   # text embeddings (used for text queries)
        self.image_index:   faiss.Index = None   # image embeddings (used for both text & image queries)
        self.caption_data: List[Dict] = []       # metadata with cached text embeddings
        self.image_data:   List[Dict] = []       # metadata with cached image embeddings

    def _create_index(self) -> faiss.IndexFlatL2:
        index = faiss.IndexFlatL2(self.dimension)
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        return index

    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(emb, p=2, dim=-1)

    def embed_images(self, image_paths: List[str]) -> torch.Tensor:
        images = []
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load image {p}: {e}")
                images.append(Image.new("RGB", (336, 336), (0, 0, 0)))
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        return torch.nn.functional.normalize(emb, p=2, dim=-1)
    
    def add_image(self, image_embedding: torch.Tensor, metadata: Dict):
        """Add a newly generated image + its embedding"""
        emb_np = image_embedding.cpu().numpy().astype('float32').reshape(1, -1)

        if self.image_index is None:
            self.image_index = self._create_index()

        self.image_index.add(emb_np)
        self.image_data.append({
            **metadata,
            "embedding": image_embedding.tolist()  # cache for future reloads
        })
        self._save_images()
        logger.info(f"Added new generated image to RAG (total: {len(self.image_data)})")

    def add_story(self, text_embedding: torch.Tensor, caption: str, metadata: Dict = None):
        """Optional: also add caption embedding if you want text→text search"""
        emb_np = text_embedding.cpu().numpy().astype('float32').reshape(1, -1)
        if self.caption_index is None:
            self.caption_index = self._create_index()

        self.caption_index.add(emb_np)
        self.caption_data.append({
            "caption": caption,
            "metadata": metadata or {},
            "embedding": text_embedding.tolist()
        })
        self._save_captions()

    # ==============================================================
    # Public API – matches your old naming exactly
    # ==============================================================

    def retrieve_stories(self, query_embedding: torch.Tensor, k: int = None) -> List[str]:
        """
        Used as: similar_stories = rag_engine.retrieve_stories(image_embedding, k=settings.TOP_K_RETRIEVAL)
        → Returns list of captions (stories)
        """
        if k is None:
            k = settings.TOP_K_RETRIEVAL

        if not self.caption_index or self.caption_index.ntotal == 0:
            return []

        query_np = query_embedding.cpu().numpy().astype('float32')
        _, I = self.caption_index.search(query_np, k)
        return [
            self.caption_data[i]["caption"]
            for i in I[0] if 0 <= i < len(self.caption_data)
        ]

    def retrieve_image_metadata(self, query_embedding: torch.Tensor, k: int = None) -> List[Dict]:
        """
        Used as: similar_metadata = rag_engine.retrieve_image_metadata(text_embedding, k=settings.TOP_K_RETRIEVAL)
        → Returns list of image metadata dicts (path + caption + etc.)
        """
        if k is None:
            k = settings.TOP_K_RETRIEVAL

        if not self.image_index or self.image_index.ntotal == 0:
            return []

        query_np = query_embedding.cpu().numpy().astype('float32')
        _, I = self.image_index.search(query_np, k)
        return [
            self.image_data[i]
            for i in I[0] if 0 <= i < len(self.image_data)
        ]

    # ==============================================================
    # Building & Loading
    # ==============================================================

    def build_from_csv(self, batch_size: int = 64):
        if not self.dataset_csv.exists():
            raise FileNotFoundError(f"CSV not found: {self.dataset_csv}")

        logger.info("Loading dataset for RAG build...")
        df = pd.read_csv(self.dataset_csv)
        if not {"image", "caption"}.issubset(df.columns):
            raise ValueError("CSV must contain 'image' and 'caption' columns")

        image_paths = [str(Path(p).resolve()) for p in df["image"].astype(str)]
        captions = df["caption"].fillna("").tolist()

        # Reset
        self.caption_index = self._create_index()
        self.image_index   = self._create_index()
        self.caption_data.clear()
        self.image_data.clear()

        logger.info(f"Embedding {len(captions)} pairs (batch_size={batch_size})...")
        for i in tqdm(range(0, len(captions), batch_size), desc="Building RAG"):
            batch_paths = image_paths[i:i+batch_size]
            batch_caps  = captions[i:i+batch_size]

            text_embs = self.embed_texts(batch_caps)
            img_embs  = self.embed_images(batch_paths)

            text_np = text_embs.cpu().numpy().astype('float32')
            img_np  = img_embs.cpu().numpy().astype('float32')

            self.caption_index.add(text_np)
            self.image_index.add(img_np)

            for j in range(len(batch_caps)):
                idx = i + j
                self.caption_data.append({
                    "index": idx,
                    "caption": batch_caps[j],
                    "image_path": batch_paths[j],
                    "embedding": text_embs[j].cpu().tolist()
                })
                self.image_data.append({
                    "index": idx,
                    "caption": batch_caps[j],
                    "image_path": batch_paths[j],
                    "embedding": img_embs[j].cpu().tolist()
                })

        self.save()
        logger.info(f"RAG build complete! Saved to {self.index_dir}")

    def load(self):
        """Fast load – call this on every app start"""
        if self.caption_index_path.exists() and self.caption_db_path.exists():
            self.caption_index = faiss.read_index(str(self.caption_index_path))
            if faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.caption_index = faiss.index_cpu_to_gpu(res, 0, self.caption_index)
            with open(self.caption_db_path) as f:
                self.caption_data = json.load(f)

        if self.image_index_path.exists() and self.image_db_path.exists():
            self.image_index = faiss.read_index(str(self.image_index_path))
            if faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.image_index = faiss.index_cpu_to_gpu(res, 0, self.image_index)
            with open(self.image_db_path) as f:
                self.image_data = json.load(f)

        logger.info(f"RAG loaded: {len(self.image_data)} items")

    def load_or_create_indices(self):
        """This is the method your generation_worker.py calls"""
        self.load()

    def _save_images(self):
            with open(self.image_db_path, "w") as f:
                json.dump(self.image_data, f, indent=2)
            if self.image_index:
                cpu_idx = faiss.index_gpu_to_cpu(self.image_index) if faiss.get_num_gpus() > 0 else self.image_index
                faiss.write_index(cpu_idx, str(self.image_index_path))

    def _save_captions(self):
        with open(self.caption_db_path, "w") as f:
            json.dump(self.caption_data, f, indent=2)
        if self.caption_index:
            cpu_idx = faiss.index_gpu_to_cpu(self.caption_index) if faiss.get_num_gpus() > 0 else self.caption_index
            faiss.write_index(cpu_idx, str(self.caption_index_path))

    def save(self):
        self._save_images()
        self._save_captions()


    def reset(self):
        """Delete everything and start fresh"""
        for p in [self.caption_index_path, self.image_index_path,
                  self.caption_db_path, self.image_db_path]:
            if p.exists():
                p.unlink()
        logger.info("All RAG data reset.")

rag_engine = RAGEngine()
