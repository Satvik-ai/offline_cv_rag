"""Image question answering with a local Ollama vision model or local vLLM endpoint."""
from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


SYSTEM_PROMPT = """You are a careful visual analysis assistant.
Report only details that are directly visible in the image.
Do not invent blur, tilt, distortion, fading, lighting problems, or other
image-quality issues. If a property is uncertain, say that it is uncertain.
Do not repeat yourself. Use concise sentences and stay within the requested
answer length. Do not infer details that cannot be established from the pixels."""


def _image_data_url(image_path: str | Path) -> str:
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_ollama_content(response) -> str:
    message = getattr(response, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None and isinstance(response, dict):
        content = response.get("message", {}).get("content")
    return str(content or "").strip()


def _limit_response(text: str) -> str:
    text = (text or "").strip()
    max_chars = int(os.getenv("VISION_MAX_RESPONSE_CHARS", "1800"))
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
    if not text:
        raise RuntimeError("The local vision model returned an empty response.")
    return text


def ask_vision_model(
    image_path: str | Path,
    question: str,
    *,
    backend: str | None = None,
) -> str:
    """Ask a local VLM about one image. No external hosted API is used."""
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    selected_backend = (backend or os.getenv("VISION_BACKEND", "ollama")).strip().lower()
    user_prompt = (
        "Inspect the supplied image and answer this question:\n"
        f"{question}\n\n"
        "Keep the answer factual and concise. Mention uncertainty rather than guessing."
    )

    if selected_backend == "ollama":
        import ollama

        client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        response = client.chat(
            model=os.getenv("OLLAMA_VISION_MODEL", "llava-phi3"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt, "images": [str(path)]},
            ],
            options={
                "temperature": float(os.getenv("VISION_TEMPERATURE", "0")),
                "num_predict": int(os.getenv("VISION_NUM_PREDICT", "180")),
                "repeat_penalty": float(os.getenv("VISION_REPEAT_PENALTY", "1.2")),
            },
        )
        return _limit_response(_extract_ollama_content(response))

    if selected_backend in {"openai_compatible", "vllm"}:
        from openai import OpenAI

        client = OpenAI(
            base_url=os.getenv("VISION_OPENAI_BASE_URL", "http://localhost:8001/v1"),
            api_key=os.getenv("VISION_OPENAI_API_KEY", "EMPTY"),
        )
        response = client.chat.completions.create(
            model=os.getenv("VISION_OPENAI_MODEL", "local-vision"),
            temperature=float(os.getenv("VISION_TEMPERATURE", "0")),
            max_tokens=int(os.getenv("VISION_NUM_PREDICT", "180")),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": _image_data_url(path)}},
                    ],
                },
            ],
        )
        return _limit_response(response.choices[0].message.content or "")

    raise ValueError("VISION_BACKEND must be 'ollama' or 'openai_compatible'.")
