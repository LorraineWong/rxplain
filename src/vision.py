# vision.py - Image input pipeline for Rxplain, powered by Gemma 4 local inference

import re
from PIL import Image
import torch


def preprocess_image(pil_image: Image.Image) -> Image.Image:
    """Resize to max 1024px on longest side, convert to RGB."""
    img = pil_image.convert("RGB")
    max_side = 1024
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def extract_drug_name_gemma4(pil_image: Image.Image, model, processor) -> str:
    """Use local Gemma 4 vision to extract drug name from medication box photo."""
    img = preprocess_image(pil_image)

    prompt = (
        "What is the drug name or medicine name shown on this box? "
        "Reply with ONLY the drug name (e.g. Panadol, Warfarin, Metformin). "
        "Do not include dosage, brand taglines, or manufacturer. "
        "If you cannot read a drug name, reply UNKNOWN."
    )

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt}
    ]}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=32, do_sample=False)

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[-1]
    decoded = processor.decode(output[0][input_len:], skip_special_tokens=True).strip()

    # Strip dosage info trailing the name
    decoded = re.sub(r'\s+\d+\s*(mg|mcg|ml|%|iu).*$', '', decoded, flags=re.IGNORECASE).strip()

    if not decoded or decoded.upper() == "UNKNOWN" or len(decoded) > 50:
        return ""
    return decoded


def image_to_drug_name(pil_image: Image.Image, model=None, processor=None, tokenizer=None) -> tuple[str, str]:
    """
    Main entry point for image input.
    Uses local Gemma 4 vision model.
    Returns: (drug_name, method_used)
    """
    if model is None or processor is None:
        print("[vision] Model not loaded, cannot process image.")
        return "", "failed"

    print("[vision] Using Gemma 4 local vision to identify drug...")
    try:
        drug_name = extract_drug_name_gemma4(pil_image, model, processor)
        if drug_name:
            print(f"[vision] Gemma 4 identified: {drug_name}")
            return drug_name, "gemma4"
        print("[vision] Gemma 4 returned empty result.")
        return "", "failed"
    except Exception as e:
        print(f"[vision] Gemma 4 vision failed: {e}")
        return "", "failed"
