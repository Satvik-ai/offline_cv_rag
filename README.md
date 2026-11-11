# Offline CV RAG Agent

An offline, documentation-grounded Computer Vision coding agent. It:

1. Reads local API documentation (PDF, Markdown, text, reStructuredText, HTML, Python, Jupyter notebooks, JSON, YAML, TOML and CSV).
2. Splits it into chunks, embeds the chunks with a locally running Ollama embedding model, and stores them in persistent ChromaDB.
3. Retrieves relevant documentation for a natural-language CV task.
4. Uses a local coding LLM (Ollama by default; optional OpenAI-compatible local server such as vLLM) to generate Python code.
5. Executes the generated script in a subprocess, captures errors and asks the LLM to repair failures for a bounded number of retries.
6. Optionally sends an input image to a local vision model (Ollama/LLaVA or an OpenAI-compatible local VLM endpoint) and adds its textual observation to the code-generation context.

## 1. Project layout

```text
offline_cv_agent/
├── docs/           # Put the provided API docs here (PDF/MD/TXT/HTML/etc.)
├── chroma_db/      # Persistent ChromaDB data (created during ingestion)
├── models/         # Optional: keep local GGUF/model files here
├── workspace/      # Generated Python script is written here
├── outputs/        # Generated images/results are written here
├── rag_agent.py    # Main CLI/orchestrator
├── ingest.py       # Multi-format document loading, chunking and indexing
├── generator.py    # Local LLM selection, prompting, code extraction and repair
├── executor.py     # Subprocess execution, timeout, error capture and retry loop
├── vision.py       # Local vision-language model calls
├── requirements.txt
├── .env.example
└── README.md
```

## 2. Environment prerequisites

### Create and activate a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the Python packages:

```bash
python -m pip install -r requirements.txt
```

If the exam environment already provides compatible packages, do not unnecessarily reinstall them.

## 3. Prepare local models (before going offline)

Ollama is a separate application/service; the Python package in `requirements.txt` is only its client.

Start/check Ollama:

```bash
ollama --version
ollama list
```

Example model roles:

```bash
ollama pull qwen2.5-coder:3b
ollama pull nomic-embed-text
ollama pull llava-phi3
```

- `qwen2.5-coder:3b`: coding model (about 3B parameters).
- `nomic-embed-text`: text embedding model for ChromaDB; it is not the code LLM.
- `llava-phi3`: compact vision-language model.

Check that the models are actually present:

```bash
ollama list
```

Do not rely on a model tag merely because it appears in `.env`; the model must be downloaded/imported locally. On the machine, the Ollama service should be running and reachable at `http://localhost:11434`.

### Import a provided GGUF coding model into Ollama

Ollama can import a supported GGUF file. Copy the supplied `.gguf` into `models/`, then create a file named `Modelfile` in the project root. Example contents (replace the path with the real absolute path):

```text
FROM /absolute/path/to/offline_cv_agent/models/qwen-coder.gguf
PARAMETER temperature 0
```

Create and test the local Ollama model:

```bash
ollama create qwen-coder-local -f Modelfile
ollama run qwen-coder-local
```

Then set this in `.env`:

```text
OLLAMA_LLM_MODEL=qwen-coder-local
LLM_BACKEND=ollama
```

For a split GGUF model, keep all shard filenames intact and follow the current Ollama import documentation. GGUF architecture/chat-template compatibility still matters; not every arbitrary GGUF file is guaranteed to run.

### Optional: use a local vLLM server for code generation

Use this only if vLLM and a compatible local model are already installed. vLLM is an inference server, not a model. Its OpenAI-compatible endpoint can be used by this project.

Example (run in a separate terminal, using a local model directory):

```bash
vllm serve /absolute/path/to/local-coder-model \
  --served-model-name local-coder \
  --host 127.0.0.1 \
  --port 8000
```

Set `.env`:

```text
LLM_BACKEND=openai_compatible
OPENAI_COMPAT_BASE_URL=http://localhost:8000/v1
OPENAI_COMPAT_API_KEY=EMPTY
OPENAI_COMPAT_MODEL=local-coder
```

The model directory and all tokenizer/config files must already be local. vLLM model/quantization support is version- and architecture-dependent; do not assume every GGUF file can be served by vLLM. For GGUF, Ollama import is the simpler route in this project.

### Optional: use a local vLLM-compatible vision endpoint

Only use this when the served VLM supports image input. Start it on a separate port, for example `8001`, with the model and processor files already available locally. Then set:

```text
VISION_BACKEND=openai_compatible
VISION_OPENAI_BASE_URL=http://localhost:8001/v1
VISION_OPENAI_API_KEY=EMPTY
VISION_OPENAI_MODEL=local-vision
```

Otherwise keep `VISION_BACKEND=ollama` and set `OLLAMA_VISION_MODEL` to the local LLaVA-compatible model tag.

## 4. Configure the project

Copy `.env.example` to `.env`:

Linux/macOS:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set the exact model names and local endpoint URLs available on the exam machine. Important settings:

| Variable | Purpose |
|---|---|
| `LLM_BACKEND` | `ollama` or `openai_compatible` |
| `OLLAMA_LLM_MODEL` | Local code-generation model tag |
| `OLLAMA_EMBED_MODEL` | Local embedding model tag |
| `VISION_BACKEND` | `ollama` or `openai_compatible` |
| `OLLAMA_VISION_MODEL` | Local vision model tag |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Documentation chunking parameters |
| `TOP_K` | Number of retrieved chunks |
| `MAX_RETRIES` | Number of correction attempts after the first run |
| `EXECUTION_TIMEOUT` | Maximum seconds per generated-code run |

## 5. Add documentation in any supported format

Copy the documentation recursively into `docs/`. You may mix formats:

```text
docs/
├── opencv/
│   ├── image_processing.pdf
│   ├── filtering.md
│   └── contours.html
├── numpy/
│   └── user_guide.pdf
└── matplotlib/
    └── pyplot.md
```

Supported extensions: `.pdf`, `.md`, `.markdown`, `.txt`, `.rst`, `.html`, `.htm`, `.py`, `.ipynb`, `.json`, `.yaml`, `.yml`, `.toml`, `.csv`.

PDFs are extracted page by page. The page number is retained in metadata and shown in retrieved source references. Markdown and other text formats are read as text. HTML tags/scripts are removed before indexing.

**Scanned PDFs:** If a PDF consists of page images rather than selectable text, PyMuPDF may extract no text. The ingestion program prints a warning. OCR must be performed beforehand (for example with a locally installed OCR/Tesseract workflow); OCR is not silently assumed to be available. Always inspect the ingestion output and make sure the extracted document count/chunk count is non-zero.

## 6. Build the vector database

Start the Ollama service, then run:

```bash
python ingest.py --reset
```

Or specify paths:

```bash
python ingest.py --docs-dir ./docs --persist-dir ./chroma_db --reset
```

`--reset` deletes and rebuilds the project collection. Use it when documents are replaced/removed or when changing the embedding model. Without `--reset`, the same chunk IDs are upserted, but old chunks from deleted/renamed files can remain; therefore a reset is recommended after changing the documentation set.

The database is stored locally in `chroma_db/`. Do not delete this folder if you want to reuse the index. Back it up if the exam environment is temporary.

## 7. Run the agent

### Text-only CV coding task

```bash
python rag_agent.py "Read an image, convert it to grayscale, and save the result"
```

### Image-processing task with a specific input image

```bash
python rag_agent.py "Apply Canny edge detection to the image and save the edge map" --image ./sample.jpg
```

The image path is passed to the generated script using `CV_AGENT_IMAGE_PATH`. Generated output files should be saved under `outputs/` using `CV_AGENT_OUTPUT_DIR`.

### Ask the vision model about an image, then generate code

```bash
python rag_agent.py "Identify the visible objects and write code to draw bounding boxes if a suitable local detector is available" --image ./sample.jpg --vision
```

The VLM's answer is treated as a description/context hint, not as ground truth. The code-generation model still needs to use the retrieved CV documentation and an actual detector/API for tasks requiring bounding boxes or precise counts.

### Vision question only (no code generation)

```bash
python rag_agent.py "Describe the main objects in this image" --image ./sample.jpg --vision-only
```

### Interactive mode

```bash
python rag_agent.py --interactive
```

### Rebuild the index and run a task

```bash
python rag_agent.py "Apply Gaussian blur" --rebuild-index --image ./sample.jpg
```

## 8. What happens internally

```text
User task (+ optional image)
        |
        +--> Optional local VLM --> image description
        |
        v
ChromaDB similarity search <--- local embedding model
        |
        v
Retrieved documentation chunks + source metadata
        |
        v
Local coding LLM (Qwen Coder / Mistral / local vLLM)
        |
        v
Extract Python code --> workspace/generated_cv_task.py
        |
        v
Python subprocess (timeout, captured stdout/stderr)
        |
        +--> success --> print code, stdout and output location
        |
        +--> failure --> send traceback + same documentation to LLM
                              |
                              +--> corrected script (bounded retry count)
```

## 9. Official references

- LangChain Python: https://docs.langchain.com/oss/python/
- Chroma: https://docs.trychroma.com/
- Ollama API: https://docs.ollama.com/
- Ollama GGUF import: https://github.com/ollama/ollama/blob/main/docs/import.mdx
- vLLM OpenAI-compatible serving: https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
- PyMuPDF text extraction: https://pymupdf.readthedocs.io/en/latest/recipes-text.html
- Python subprocess: https://docs.python.org/3/library/subprocess.html
