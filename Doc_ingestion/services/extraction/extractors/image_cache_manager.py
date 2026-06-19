"""
Image Cache Manager
Caches extracted image text using perceptual hashing to avoid redundant OCR work.
"""

import logging
import io
import hashlib
from typing import Dict, Optional

from PIL import Image
import imagehash

logger = logging.getLogger(__name__)


class ImageCacheManager:
    """
    Manage image text caching using perceptual hashing.
    Features: perceptual hashing, in-memory cache, MD5 fallback, statistics.
    """

    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size
        self.cache: Dict[str, str] = {}
        self.hits = 0
        self.misses = 0
        logger.debug(f"Image cache initialized (hash_size={hash_size})")

    def compute_hash(self, image_bytes: bytes) -> str:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return str(imagehash.phash(img, hash_size=self.hash_size))
        except Exception as e:
            logger.debug(f"Perceptual hash failed, using MD5: {e}")
            return f"md5_{hashlib.md5(image_bytes).hexdigest()}"

    def get(self, image_bytes: bytes) -> Optional[str]:
        img_hash = self.compute_hash(image_bytes)
        if img_hash in self.cache:
            self.hits += 1
            return self.cache[img_hash]
        self.misses += 1
        return None

    def put(self, image_bytes: bytes, description: str) -> str:
        img_hash = self.compute_hash(image_bytes)
        self.cache[img_hash] = description
        return img_hash

    def has(self, image_bytes: bytes) -> bool:
        return self.compute_hash(image_bytes) in self.cache

    def clear(self):
        size = len(self.cache)
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info(f"Cache cleared ({size} entries removed)")

    def get_stats(self) -> Dict:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "cache_size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate": round(hit_rate, 3),
            "hit_rate_percent": round(hit_rate * 100, 1),
        }

    def __len__(self):
        return len(self.cache)

    def __contains__(self, image_bytes: bytes):
        return self.has(image_bytes)


_cache_manager: Optional[ImageCacheManager] = None


def get_image_cache() -> ImageCacheManager:
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = ImageCacheManager()
    return _cache_manager


def reset_image_cache():
    global _cache_manager
    if _cache_manager is not None:
        _cache_manager.clear()
