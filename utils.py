"""
utils.py
────────
Utility Helper Functions

Contains:
  - encode_image_to_base64  : PIL Image → base64 string for API
  - validate_image          : file-size / format checks
  - resize_for_display      : safe resize for Streamlit display
"""

import base64
import io
from PIL import Image


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MAX_FILE_SIZE_MB  = 10
ALLOWED_FORMATS   = {"JPEG", "JPG", "PNG", "WEBP"}
API_JPEG_QUALITY  = 90    # JPEG quality when encoding for API (smaller payload)


def encode_image_to_base64(image: Image.Image) -> str:
    """
    Convert a PIL Image to a base64-encoded JPEG string.

    The Claude Vision API requires images as base64 strings.
    We re-encode as JPEG to reduce payload size.

    Args:
        image : PIL.Image (already preprocessed)

    Returns:
        Base64-encoded string (no prefix)
    """
    # Convert to RGB (JPEG doesn't support alpha channel)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save to in-memory buffer as JPEG
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=API_JPEG_QUALITY, optimize=True)
    buffer.seek(0)

    # Encode to base64
    img_bytes  = buffer.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return img_base64


def validate_image(uploaded_file) -> tuple[bool, str]:
    """
    Validate the uploaded image file.

    Checks:
      1. File size ≤ 10 MB
      2. Format is JPEG / PNG / WEBP
      3. File is a valid, openable image

    Args:
        uploaded_file : Streamlit UploadedFile object

    Returns:
        (is_valid: bool, message: str)
    """
    # ── Check file size ─────────────────────────────────────────
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({file_size_mb:.1f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB."

    # ── Check format ────────────────────────────────────────────
    try:
        img    = Image.open(uploaded_file)
        fmt    = img.format.upper() if img.format else ""
        if fmt not in ALLOWED_FORMATS:
            return False, f"Unsupported format '{fmt}'. Use JPG, PNG, or WEBP."
    except Exception as e:
        return False, f"Could not open image: {str(e)}"
    finally:
        # Reset file pointer for later use
        uploaded_file.seek(0)

    return True, "OK"


def resize_for_display(image: Image.Image, max_width: int = 600) -> Image.Image:
    """
    Resize an image proportionally for Streamlit display.
    Does NOT modify the original; returns a new image.
    """
    if image.width <= max_width:
        return image

    ratio  = max_width / image.width
    height = int(image.height * ratio)
    return image.resize((max_width, height), Image.LANCZOS)
