from .llm_info import COSTS
import mimetypes
import base64
from io import BytesIO
#import google.generativeai as genai
import google.genai as genai

def calculate_cost(model_name: str, prompt_token: int, completion_token: int) -> float:
    return COSTS.get(model_name, dict()).get("prompt", 0) * prompt_token \
        + COSTS.get(model_name, dict()).get("completion", 0) * completion_token

def encode_image(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError("Cannot determine the MIME type of the file")
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
    return f"data:{mime_type};base64,{base64_image}"

def decode_image(data_url):
    header, base64_image = data_url.split(",", 1)
    mime_type = header.split(";")[0].split(":")[1]
    
    return mime_type, base64_image

def compress_image(image, max_size):
    quality = 95 
    while True:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        if len(base64_data) <= max_size or quality <= 10:
            return base64_data
        quality -= 5


def upload_to_gemini(path, mime_type=None):
    file = genai.upload_file(path, mime_type=mime_type)
    return file