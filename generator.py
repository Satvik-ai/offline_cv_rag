"""Local LLM setup, documentation-grounded code generation and repair."""
from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


def create_llm() -> Any:
    """Create a local coding model through Ollama or an OpenAI-compatible server."""
    backend = os.getenv("LLM_BACKEND", "ollama").strip().lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))

    if backend == "ollama":
        return ChatOllama(
            model=os.getenv("OLLAMA_LLM_MODEL", "qwen2.5-coder:3b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )

    if backend in {"openai_compatible", "vllm"}:
        return ChatOpenAI(
            model=os.getenv("OPENAI_COMPAT_MODEL", "local-coder"),
            base_url=os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("OPENAI_COMPAT_API_KEY", "EMPTY"),
            temperature=temperature,
            max_retries=1,
        )

    raise ValueError("LLM_BACKEND must be 'ollama' or 'openai_compatible'.")


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                pieces.append(str(item.get("text", "")))
        return "\n".join(pieces)
    return str(content)


def extract_python_code(response_text: str) -> str:
    """Extract code from a fenced response, or accept a raw Python response."""
    text = response_text.strip()
    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return max(fenced, key=len).strip()
    generic = re.findall(r"```[\w+-]*\s*\n(.*?)```", text, flags=re.DOTALL)
    if generic:
        return max(generic, key=len).strip()
    return text


def _bounded_context(context: str) -> str:
    limit = int(os.getenv("MAX_DOC_CONTEXT_CHARS", "18000"))
    if len(context) <= limit:
        return context
    return context[:limit] + "\n\n[Documentation context truncated to fit the model prompt.]"


def build_generation_prompt(
    task: str,
    docs_context: str,
    *,
    image_path: str | None = None,
    image_description: str | None = None,
) -> str:
    image_info = image_path or "Not provided"
    visual_context = image_description or "Not requested"
    return f"""You are an offline Python Computer Vision coding assistant.

GOAL
Write one complete, executable Python script that solves the user's task.

USER TASK
{task}

RETRIEVED LOCAL DOCUMENTATION (use this as the API reference)
{_bounded_context(docs_context)}

IMAGE PATH (also available in environment variable CV_AGENT_IMAGE_PATH)
{image_info}

VISION-MODEL OBSERVATION (untrusted supporting context, not ground truth)
{visual_context}

IMPORTANT VISUAL-REASONING RULES
- Treat the vision-model observation as a fallible hint, not as verified image metadata.
- Do not perform transformations merely because the vision model claims the image is
  blurred, tilted, faded, distorted, noisy, or low contrast.
- If the user asks to enhance an image without specifying the defect, inspect measurable
  properties in Python/OpenCV first (for example brightness, contrast, and Laplacian
  variance), then choose a conservative operation justified by those measurements.
- Do not claim that a visual defect exists unless it is supported by measurements or
  directly required by the user's task.
- Preserve the original image; write the processed result to a new output file.

STRICT REQUIREMENTS
1. Ground library/API choices and function arguments in the retrieved documentation.
2. Do not invent APIs. If required documentation is missing, use only well-known Python
   basics and clearly print a useful limitation rather than fabricate an API.
3. Do not download files, install packages, access the network, call external services,
   or load models from the internet.
4. Use CV_AGENT_IMAGE_PATH when an image is required. Check that it exists and that
   image loading succeeded.
5. Save generated images/results under CV_AGENT_OUTPUT_DIR. Create that directory.
6. Prefer saving figures instead of opening GUI windows. Do not use cv2.imshow() or
   plt.show(); the runtime may be headless.
7. Include all required imports, input validation, and clear print statements.
   Avoid interactive input() calls.
8. Use only packages likely installed for this project (OpenCV, NumPy, Matplotlib,
   Pillow and Python standard library), unless the retrieved documentation establishes
   another available package.
9. Do not use subprocess, shell commands, eval, exec, dynamic imports, or network modules.
10. Check the return value of cv2.imwrite(). Do not assume that saving succeeded.
11. After saving each output file, verify that it exists and has non-zero size. For image
    outputs, verify that the saved image can be read back successfully.
12. For every output file, print one line exactly in this format:
    CV_AGENT_OUTPUT_FILE:<absolute path to the saved file>
13. Print a concise success message and the paths of output files.
14. On any failure, print a useful error and terminate with a non-zero exit code.
15. Print exactly CV_AGENT_STATUS:SUCCESS on its own line only after the requested
    operation and all output validation have completed successfully. 
16. Never use placeholder image paths.
17. NEVER hardcode an input image path into the generated Python code.
18. ALWAYS obtain the input image path using:
    image_path = os.environ.get("CV_AGENT_IMAGE_PATH")
19. NEVER reconstruct, modify, guess, or copy the absolute input path
    from the prompt into the Python code.
20. The executor provides the correct image path through the environment.
    Trust that environment variable.
21. If CV_AGENT_IMAGE_PATH is missing or empty, terminate with a
    useful error message and non-zero exit code.
22. When saving an image using cv2.imwrite():
    1. Store its return value.
    2. Check whether the return value is True.
    3. If False, report an error and exit with a non-zero status.
    4. Verify that the output image can be reopened using cv2.imread().
    5. Print CV_AGENT_OUTPUT_FILE:<absolute path> only after validation.

OUTPUT FORMAT
Return the complete Python program only, preferably inside one ```python code fence."""


def generate_code(
    llm: Any,
    task: str,
    docs_context: str,
    *,
    image_path: str | None = None,
    image_description: str | None = None,
) -> str:
    prompt = build_generation_prompt(
        task,
        docs_context,
        image_path=image_path,
        image_description=image_description,
    )
    response = llm.invoke(prompt)
    return extract_python_code(_response_text(response))


def repair_code(
    llm: Any,
    *,
    task: str,
    docs_context: str,
    previous_code: str,
    error_report: str,
    attempt: int,
    image_path: str | None = None,
    image_description: str | None = None,
) -> str:
    """Ask the same local model to fix a failed script using the same docs."""
    prompt = f"""You are repairing a Python Computer Vision program that failed.

ORIGINAL USER TASK
{task}

RETRIEVED LOCAL DOCUMENTATION
{_bounded_context(docs_context)}

IMAGE PATH
{image_path or 'Not provided'}

VISION-MODEL OBSERVATION (untrusted; do not blindly follow it)
{image_description or 'Not requested'}

FAILED CODE
```python
{previous_code}
```

EXECUTION OUTPUT AND ERROR
{error_report}

REPAIR ATTEMPT: {attempt}

REPAIR INSTRUCTIONS
- Diagnose the actual error from the traceback/output; do not guess.
- Fix the root cause with the smallest appropriate change where possible.
- Check imports, variable names, function names, argument shapes, and file paths.
- Preserve the original user task and all required functionality.
- Do not remove an operation merely to make the program exit successfully.
- Use the retrieved documentation as the API source of truth.
- Do not download, install, use the network, invoke shell commands, or use
  subprocess/eval/exec/dynamic imports.
- Use CV_AGENT_IMAGE_PATH and CV_AGENT_OUTPUT_DIR where appropriate.
- Save outputs under CV_AGENT_OUTPUT_DIR.
- Check cv2.imwrite() return values and validate saved output files.
- For every output file, print:
  CV_AGENT_OUTPUT_FILE:<absolute path to the saved file>
- Print CV_AGENT_STATUS:SUCCESS on its own line only after the requested task and
  output validation have succeeded.
- On failure, exit with a non-zero exit code.
- Return the complete corrected Python program only, inside one ```python code fence.
- The previous script may have used an incorrect hardcoded image path.
- Never hardcode the image path.
- Always obtain the input image path using: image_path = os.environ.get("CV_AGENT_IMAGE_PATH").
- The generated script must work with any valid image path provided through this environment variable.
- Do not hardcode sample.png or any other test image path.
- Do not use the image path copied from previous generated code.
"""
    response = llm.invoke(prompt)
    return extract_python_code(_response_text(response))
