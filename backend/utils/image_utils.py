import base64
from io import BytesIO
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise ValueError("Invalid base64 image data")

def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"

# def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
#     """Encode PIL Image to base64 string (without data URI prefix)"""
#     buffered = BytesIO()
#     image.save(buffered, format=format)
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     return img_str

def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize image maintaining aspect ratio"""
    width, height = image.size
    
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    return image