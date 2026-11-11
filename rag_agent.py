"""Offline documentation-grounded CV coding agent with optional image understanding."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

from executor import execute_with_repair
from generator import create_llm, generate_code, repair_code
from ingest import build_vectorstore, load_vectorstore
from vision import ask_vision_model

load_dotenv()


def _retrieve_context(vectorstore, query: str, top_k: int):
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    sections: list[str] = []
    for index, (doc, distance) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page")
        location = f", page {page}" if page is not None else ""
        sections.append(
            f"[Retrieved source {index}: {source}{location}; vector distance={distance:.4f}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(sections), results


def run_agent(
    task: str,
    *,
    docs_dir: str | Path = "./docs",
    persist_dir: str | Path = "./chroma_db",
    image_path: str | Path | None = None,
    use_vision: bool = False,
    vision_only: bool = False,
    top_k: int = 5,
    max_retries: int = 3,
    timeout_seconds: int = 90,
    rebuild_index: bool = False,
    auto_ingest: bool = True,
) -> int:
    if not task.strip():
        raise ValueError("Task/query cannot be empty.")
    if (use_vision or vision_only) and not image_path:
        raise ValueError("Provide --image PATH when using --vision or --vision-only.")

    resolved_image = None
    if image_path:
        resolved_image = Path(image_path).expanduser().resolve()
        if not resolved_image.is_file():
            raise FileNotFoundError(f"Image not found: {resolved_image}")

    image_description = None
    if use_vision or vision_only:
        print("[VISION] Sending image to the local vision model...")
        image_description = ask_vision_model(resolved_image, task)
        print("\n[VISION RESPONSE]\n" + image_description + "\n")
        if vision_only:
            return 0

    if rebuild_index:
        print("[RAG] Rebuilding the documentation index...")
        vectorstore = build_vectorstore(docs_dir, persist_dir, reset=True)
    else:
        try:
            vectorstore = load_vectorstore(persist_dir)
        except (FileNotFoundError, RuntimeError) as exc:
            if not auto_ingest:
                raise
            print(f"[RAG] Existing index unavailable ({exc}). Building it from local docs...")
            vectorstore = build_vectorstore(docs_dir, persist_dir, reset=True)

    # Keep retrieval focused on the requested operation. VLM prose can be noisy
    # and should not dominate retrieval; it is passed separately to the generator.
    retrieval_query = task
    print("[RAG] Retrieving relevant documentation...")
    docs_context, retrieved = _retrieve_context(vectorstore, retrieval_query, top_k)
    if not docs_context:
        raise RuntimeError("No documentation chunks were retrieved. Check that docs were indexed.")

    print("\n[RETRIEVED SOURCES]")
    for index, (doc, distance) in enumerate(retrieved, start=1):
        page = doc.metadata.get("page")
        suffix = f" (page {page})" if page is not None else ""
        print(f"  {index}. {doc.metadata.get('source', 'unknown')}{suffix} | distance={distance:.4f}")

    print("\n[LLM] Generating Python code...")
    llm = create_llm()
    resolved_image_text = str(resolved_image) if resolved_image else None
    code = generate_code(
        llm,
        task,
        docs_context,
        image_path=resolved_image_text,
        image_description=image_description,
    )
    if not code.strip():
        raise RuntimeError("The coding model returned an empty response.")

    def repair_callback(previous_code: str, error_report: str, attempt: int) -> str:
        return repair_code(
            llm,
            task=task,
            docs_context=docs_context,
            previous_code=previous_code,
            error_report=error_report,
            attempt=attempt,
            image_path=resolved_image_text,
            image_description=image_description,
        )

    result = execute_with_repair(
        code,
        task=task,
        repair_callback=repair_callback,
        max_retries=max_retries,
        workspace_dir=os.getenv("WORKSPACE_DIR", "./workspace"),
        output_dir=os.getenv("OUTPUT_DIR", "./outputs"),
        image_path=resolved_image,
        timeout_seconds=timeout_seconds,
    )

    print("\n" + "=" * 68)
    print("FINAL RESULT")
    print("=" * 68)
    print(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Execution attempts: {result.attempts}")
    print(f"Generated script: {result.script_path}")
    print(f"Output directory: {result.output_dir}")
    if result.stdout:
        print("\nSTDOUT:\n" + result.stdout)
    if result.stderr:
        print("\nSTDERR:\n" + result.stderr)
    print("\nFINAL PYTHON CODE:\n")
    print(result.code)
    return 0 if result.success else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline RAG agent: retrieve local CV docs, generate code, execute and self-correct."
    )
    parser.add_argument("task", nargs="?", help="Natural-language Computer Vision task")
    parser.add_argument("--interactive", action="store_true", help="Prompt for a task in the terminal")
    parser.add_argument("--docs-dir", default=os.getenv("DOCS_DIR", "./docs"))
    parser.add_argument("--persist-dir", default=os.getenv("CHROMA_DIR", "./chroma_db"))
    parser.add_argument("--image", help="Optional input image path")
    parser.add_argument("--vision", action="store_true", help="Ask the local VLM for image context first")
    parser.add_argument("--vision-only", action="store_true", help="Answer with the VLM; do not retrieve docs or generate code")
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", "5")))
    parser.add_argument("--max-retries", type=int, default=int(os.getenv("MAX_RETRIES", "3")))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("EXECUTION_TIMEOUT", "90")))
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild ChromaDB from the current docs")
    parser.add_argument("--no-auto-ingest", action="store_true", help="Fail instead of building a missing index")
    args = parser.parse_args()

    task = args.task
    if args.interactive or not task:
        task = input("Enter your Computer Vision task: ").strip()

    if args.top_k < 1 or args.max_retries < 0 or args.timeout < 1:
        parser.error("--top-k and --timeout must be positive; --max-retries cannot be negative.")

    if args.image:
        # User explicitly supplied an image
        execution_image_path = Path(args.image).expanduser().resolve()

        if not execution_image_path.is_file():
            print(f"[ERROR] Image not found: {execution_image_path}")
            return 1

        print(f"[INPUT] Using user-provided image: {execution_image_path}")
    else:
        # No image supplied: use sample.png for testing
        execution_image_path = (
            Path(__file__).resolve().parent / "sample.png"
        )

        if not execution_image_path.is_file():
            print(
                f"[ERROR] No input image supplied and "
                f"sample.png was not found: {execution_image_path}"
            )
            return 1

        print(
            "[TEST] No input image supplied. "
            "Using sample.png to validate generated code."
        )

    try:
        exit_code = run_agent(
            task,
            docs_dir=args.docs_dir,
            persist_dir=args.persist_dir,
            image_path=execution_image_path,
            use_vision=args.vision or args.vision_only,
            vision_only=args.vision_only,
            top_k=args.top_k,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout,
            rebuild_index=args.rebuild_index,
            auto_ingest=not args.no_auto_ingest,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        if os.getenv("CV_AGENT_DEBUG", "false").strip().lower() in {"1", "true", "yes"}:
            traceback.print_exc()
        exit_code = 2

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
