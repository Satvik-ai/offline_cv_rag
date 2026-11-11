"""Run generated scripts, validate their outputs, and support repair retries.

NOTE: subprocess isolation and the AST checks below are not a security sandbox.
Use an OS-level container/restricted account for untrusted generated code.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import re
import tempfile


@dataclass
class ExecutionResult:
    success: bool
    code: str
    stdout: str
    stderr: str
    attempts: int
    output_dir: Path
    script_path: Path | None = None


def create_script_filename(task: str, workspace: Path) -> Path:
    """
    Generate a unique Python filename from the user's task.
    Prevents overwriting previously generated scripts.
    """

    # Take the first five words
    first_five_words = task.split()[:5]

    # Join the words
    filename = "_".join(first_five_words).lower()

    # Replace special characters and spaces with underscores
    filename = re.sub(r"[^a-z0-9]+", "_", filename)

    # Remove leading/trailing underscores
    filename = filename.strip("_")

    # Limit filename length
    filename = filename[:60].rstrip("_")

    # Fallback filename
    if not filename:
        filename = "cv_task"

    script_path = workspace / f"{filename}.py"

    # Avoid overwriting existing scripts
    counter = 1

    while script_path.exists():
        script_path = workspace / f"{filename}_{counter}.py"
        counter += 1

    return script_path


def _static_safety_issues(code: str) -> list[str]:
    """Basic guardrails against obviously unsafe generated code; not a sandbox."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    blocked_imports = {
        "subprocess", "socket", "requests", "urllib", "http", "ftplib",
        "smtplib", "paramiko", "ctypes", "multiprocessing",
    }
    blocked_calls = {"eval", "exec", "compile", "__import__", "breakpoint"}
    blocked_attributes = {"system", "popen", "Popen", "rmtree", "remove", "unlink"}
    issues: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in blocked_imports:
                    issues.append(f"Blocked import: {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in blocked_imports:
                issues.append(f"Blocked import: {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in blocked_calls:
                issues.append(f"Blocked call: {node.func.id}()")
            if isinstance(node.func, ast.Attribute) and node.func.attr in blocked_attributes:
                issues.append(f"Blocked call: ...{node.func.attr}()")
    return sorted(set(issues))


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."


def _declared_output_paths(stdout: str, output_dir: Path) -> tuple[list[Path], list[str]]:
    """Read CV_AGENT_OUTPUT_FILE lines and resolve paths safely under output_dir."""
    paths: list[Path] = []
    errors: list[str] = []
    prefix = "CV_AGENT_OUTPUT_FILE:"

    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith(prefix):
            continue
        raw_path = stripped[len(prefix):].strip().strip('"')
        if not raw_path:
            errors.append("An empty CV_AGENT_OUTPUT_FILE path was reported.")
            continue

        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = output_dir / candidate
        candidate = candidate.resolve()

        try:
            candidate.relative_to(output_dir.resolve())
        except ValueError:
            errors.append(f"Reported output is outside the output directory: {candidate}")
            continue

        paths.append(candidate)

    return paths, errors


def _validate_outputs(
    stdout: str,
    output_dir: Path,
    before_run: dict[Path, tuple[int, int]],
) -> list[str]:
    paths, errors = _declared_output_paths(stdout, output_dir)
    if not paths:
        errors.append(
            "No output file was reported. The script must print "
            "CV_AGENT_OUTPUT_FILE:<absolute path> for each saved result."
        )
        return errors

    for path in paths:
        if not path.is_file():
            errors.append(f"Reported output file does not exist: {path}")
            continue

        try:
            stat = path.stat()
        except OSError as exc:
            errors.append(f"Cannot inspect output file {path}: {exc}")
            continue

        if stat.st_size <= 0:
            errors.append(f"Output file is empty: {path}")
            continue

        previous = before_run.get(path)
        if previous is not None and stat.st_mtime_ns <= previous[0] and stat.st_size == previous[1]:
            errors.append(f"Output file was not changed during this execution: {path}")
            continue

        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
            try:
                import cv2
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None or image.size == 0:
                    errors.append(f"Saved image cannot be decoded: {path}")
            except ImportError:
                # Existence/non-zero-size checks still apply if OpenCV is unavailable.
                pass
            except Exception as exc:
                errors.append(f"Could not validate image {path}: {exc}")

    return errors


def run_generated_code(
    code: str,
    *,
    task: str,
    workspace_dir: str | Path = "./workspace",
    output_dir: str | Path = "./outputs",
    image_path: str | Path | None = None,
    timeout_seconds: int = 90,
) -> ExecutionResult:

    workspace = Path(workspace_dir).expanduser().resolve()
    outputs = Path(output_dir).expanduser().resolve()

    workspace.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)

    temp_script_path = None
    final_script_path = None

    # --------------------------------------------------
    # 1. Static safety checks
    # --------------------------------------------------

    if os.getenv("ENABLE_CODE_SAFETY_CHECK", "true").strip().lower() in {
        "1", "true", "yes"
    }:
        issues = _static_safety_issues(code)

        if issues:
            return ExecutionResult(
                success=False,
                code=code,
                stdout="",
                stderr=(
                    "Code rejected by the basic safety checks:\n- "
                    + "\n- ".join(issues)
                ),
                attempts=1,
                script_path=None,
                output_dir=outputs,
            )

    # --------------------------------------------------
    # 2. Prepare execution environment
    # --------------------------------------------------

    env = os.environ.copy()

    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"

    env["CV_AGENT_OUTPUT_DIR"] = str(outputs)

    if image_path:
        resolved_image = Path(image_path).expanduser().resolve()
        env["CV_AGENT_IMAGE_PATH"] = str(resolved_image)
    else:
        env.pop("CV_AGENT_IMAGE_PATH", None)

    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    # --------------------------------------------------
    # 3. Record existing output files
    # --------------------------------------------------

    before_run: dict[Path, tuple[int, int]] = {}

    for existing in outputs.rglob("*"):
        if existing.is_file():
            try:
                stat = existing.stat()

                before_run[existing.resolve()] = (
                    stat.st_mtime_ns,
                    stat.st_size,
                )

            except OSError:
                pass

    # --------------------------------------------------
    # 4. Create temporary Python script
    # --------------------------------------------------

    try:
        fd, temp_name = tempfile.mkstemp(
            prefix="_cv_agent_temp_",
            suffix=".py",
            dir=str(workspace),
        )

        os.close(fd)

        temp_script_path = Path(temp_name)

        temp_script_path.write_text(
            code,
            encoding="utf-8",
        )

        print("[EXECUTOR] Running generated script...")

        # --------------------------------------------------
        # 5. Execute generated Python code
        # --------------------------------------------------

        process = subprocess.run(
            [
                sys.executable,
                "-u",
                str(temp_script_path),
            ],
            cwd=str(workspace),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )

        stdout = process.stdout or ""
        stderr = process.stderr or ""

        # --------------------------------------------------
        # 6. Validate generated outputs
        # --------------------------------------------------

        validation_errors = []

        if process.returncode == 0:
            validation_errors = _validate_outputs(
                stdout,
                outputs,
                before_run,
            )

        task_success = (
            process.returncode == 0
            and not validation_errors
        )

        # --------------------------------------------------
        # 7. Optional success marker
        # --------------------------------------------------

        success_marker = "CV_AGENT_STATUS:SUCCESS"

        has_success_marker = success_marker in {
            line.strip()
            for line in stdout.splitlines()
        }

        if process.returncode == 0 and not has_success_marker:
            print(
                "[EXECUTOR] Info: Success marker missing. "
                "Actual output validation determines success."
            )

        if validation_errors:
            stderr += (
                "\nOutput validation failed:\n- "
                + "\n- ".join(validation_errors)
            )

        # --------------------------------------------------
        # 8. Save script ONLY after successful execution
        # --------------------------------------------------

        if task_success:

            try:
                final_script_path = create_script_filename(
                    task,
                    workspace,
                )

                final_script_path.write_text(
                    code,
                    encoding="utf-8",
                )

                print(
                    f"[EXECUTOR] Successfully saved script: "
                    f"{final_script_path.name}"
                )

            except OSError as exc:
                # Execution succeeded, but permanent saving failed.
                stderr += (
                    f"\n[WARNING] Execution succeeded, but "
                    f"the script could not be saved: {exc}"
                )

                final_script_path = None

        else:
            print(
                "[EXECUTOR] Execution failed. "
                "Temporary script will be discarded."
            )

        return ExecutionResult(
            success=task_success,
            code=code,
            stdout=_clip(stdout, 12000),
            stderr=_clip(stderr, 16000),
            attempts=1,
            script_path=final_script_path,
            output_dir=outputs,
        )

    # --------------------------------------------------
    # 9. Timeout handling
    # --------------------------------------------------

    except subprocess.TimeoutExpired as exc:

        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )

        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )

        return ExecutionResult(
            success=False,
            code=code,
            stdout=_clip(stdout, 12000),
            stderr=_clip(
                stderr
                + f"\nExecution timed out after "
                f"{timeout_seconds} seconds.",
                16000,
            ),
            attempts=1,
            script_path=None,
            output_dir=outputs,
        )

    # --------------------------------------------------
    # 10. Subprocess startup / file errors
    # --------------------------------------------------

    except OSError as exc:

        return ExecutionResult(
            success=False,
            code=code,
            stdout="",
            stderr=f"Execution error: {exc}",
            attempts=1,
            script_path=None,
            output_dir=outputs,
        )

    # --------------------------------------------------
    # 11. Always remove temporary script
    # --------------------------------------------------

    finally:

        if temp_script_path is not None:
            try:
                temp_script_path.unlink(missing_ok=True)
            except OSError as exc:
                print(
                    f"[EXECUTOR] Warning: Could not remove "
                    f"temporary script: {exc}"
                )

def execute_with_repair(
    initial_code: str,
    *,
    task: str,
    repair_callback: Callable[[str, str, int], str],
    max_retries: int = 3,
    workspace_dir: str | Path = "./workspace",
    output_dir: str | Path = "./outputs",
    image_path: str | Path | None = None,
    timeout_seconds: int = 90,
) -> ExecutionResult:
    """Run once, then ask repair_callback for up to max_retries corrections."""
    code = initial_code
    last_result: ExecutionResult | None = None

    for attempt in range(max_retries + 1):
        result = run_generated_code(
            code,
            task=task,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            image_path=image_path,
            timeout_seconds=timeout_seconds,
        )
        result.attempts = attempt + 1
        if result.success:
            return result

        last_result = result
        if attempt >= max_retries:
            break

        error_report = (
            f"STDOUT:\n{result.stdout or '(empty)'}\n\n"
            f"STDERR / ERROR:\n{result.stderr or '(empty)'}"
        )
        print(f"[REPAIR] Execution failed (attempt {attempt + 1}). Asking the LLM to correct it...")
        code = repair_callback(code, error_report, attempt + 1)
        if not code.strip():
            result.stderr += "\nThe model returned an empty correction."
            last_result = result
            break

    assert last_result is not None
    return last_result
