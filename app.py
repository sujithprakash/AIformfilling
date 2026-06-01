import os
import json
import glob
import time

import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

from PyPDF2 import PdfReader
import docx

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

load_dotenv()

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 8
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-mini"


# ---------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------

def extract_text_from_pdf_path(path: str) -> str:
    text = ""
    try:
        with open(path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
    except Exception as e:
        print(f"[PDF ERROR] {path}: {e}")
    return text


def extract_text_from_docx_path(path: str) -> str:
    try:
        with open(path, "rb") as f:
            document = docx.Document(f)
            return "\n".join(p.text for p in document.paragraphs)
    except Exception as e:
        print(f"[DOCX ERROR] {path}: {e}")
        return ""


def extract_text_from_txt_path(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"[TXT ERROR] {path}: {e}")
        return ""


def collect_text_from_folder(folder_path: str) -> dict:
    """
    Read all PDF, DOCX and TXT files in the given folder (non-recursive).
    Returns combined_text and per-file stats.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    patterns = [
        os.path.join(folder_path, "*.pdf"),
        os.path.join(folder_path, "*.docx"),
        os.path.join(folder_path, "*.txt"),
    ]

    all_paths = []
    for pattern in patterns:
        all_paths.extend(glob.glob(pattern))

    print(f"[FOLDER] Scanning {folder_path}")
    print(f"[FOLDER] Files found: {all_paths}")

    files_info = []
    combined_text = ""

    if not all_paths:
        return {"combined_text": "", "files": files_info}

    for path in all_paths:
        ext = os.path.splitext(path)[1].lower()
        file_text = ""

        if ext == ".pdf":
            file_text = extract_text_from_pdf_path(path)
        elif ext == ".docx":
            file_text = extract_text_from_docx_path(path)
        elif ext == ".txt":
            file_text = extract_text_from_txt_path(path)

        had_text = bool(file_text.strip())
        files_info.append({
            "path": path,
            "filename": os.path.basename(path),
            "extension": ext,
            "chars": len(file_text),
            "had_text": had_text,
        })

        if had_text:
            combined_text += f"\n\n===== FILE: {os.path.basename(path)} =====\n"
            combined_text += file_text
        else:
            print(f"[WARN] No text extracted from {path} (maybe image-only scan)")

    print(f"[FOLDER] Total characters collected: {len(combined_text)}")
    return {"combined_text": combined_text, "files": files_info}


# ---------------------------------------------------------
# RAG helpers: chunking, embedding, retrieval
# ---------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window chunker with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts using the OpenAI embeddings API."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


def cosine_similarity(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    corpus_norms = corpus_vecs / (np.linalg.norm(corpus_vecs, axis=1, keepdims=True) + 1e-10)
    return corpus_norms @ query_norm


def retrieve_top_chunks(query: str, chunks: list[str], chunk_vecs: np.ndarray, k: int = TOP_K) -> list[str]:
    """Embed query, retrieve top-k most similar chunks by cosine similarity."""
    query_vec = embed_texts([query])[0]
    scores = cosine_similarity(query_vec, chunk_vecs)
    top_indices = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_indices]


# ---------------------------------------------------------
# RAG extraction pipeline
# ---------------------------------------------------------

EXTRACTION_QUERY = (
    "personal details name email phone date of birth nationality address city "
    "programme choice level of study mode intake qualification institution "
    "field of study graduation year grade work experience employer position "
    "industry motivation background statement"
)

EXTRACTION_FIELDS = [
    "name", "email", "phone", "date_of_birth", "nationality",
    "address_line1", "address_line2", "city", "state_region", "country",
    "preferred_contact",
    "level_of_study", "programme_interest", "study_mode", "intake",
    "highest_qualification", "highest_institution", "field_of_study",
    "graduation_year", "grade", "other_qualifications",
    "total_experience", "current_employer", "current_position",
    "industry", "experience_summary",
    "background_statement", "ai_notes",
]

SYSTEM_PROMPT = (
    "You extract structured data from CVs and academic certificates "
    "to help fill university application forms. You must ALWAYS "
    "return valid JSON with the exact keys requested. "
    "Use ONLY information present in the supplied text. "
    "Return an empty string for any field not clearly supported — never invent details."
)


def analyse_folder(folder_path: str) -> dict:
    """
    Full RAG pipeline:
    1. Extract text from all documents in folder
    2. Chunk the combined corpus
    3. Embed all chunks
    4. Retrieve top-k chunks most relevant to the extraction query
    5. Pass retrieved context to LLM for structured JSON extraction
    """
    collected = collect_text_from_folder(folder_path)
    raw_text = collected["combined_text"]
    files_info = collected["files"]

    if not raw_text.strip():
        return {
            "error": "No readable text found in folder (PDF/DOCX/TXT).",
            "files": files_info,
        }

    # Step 1: Chunk
    chunks = chunk_text(raw_text)
    print(f"[RAG] {len(chunks)} chunks created from corpus ({len(raw_text)} chars)")

    # Step 2: Embed all chunks
    chunk_vecs = embed_texts(chunks)
    print(f"[RAG] Embedded {len(chunks)} chunks using {EMBED_MODEL}")

    # Step 3: Retrieve top-k relevant chunks
    top_chunks = retrieve_top_chunks(EXTRACTION_QUERY, chunks, chunk_vecs, k=TOP_K)
    context = "\n\n---\n\n".join(top_chunks)
    print(f"[RAG] Retrieved top-{TOP_K} chunks, context length: {len(context)} chars")

    # Step 4: LLM extraction over retrieved context
    keys_str = ", ".join(EXTRACTION_FIELDS)
    user_prompt = f"""
You are assisting admissions staff at a university.

Using ONLY the retrieved document excerpts below, extract the following fields
and return them as a single JSON object with exactly these keys:
{keys_str}

If a field is not clearly present in the excerpts, return an empty string for that key.
Do NOT invent or infer details not explicitly stated.

RETRIEVED EXCERPTS:
\"\"\"{context}\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=600,
            temperature=0.2,
        )

        content = response.choices[0].message.content
        print("[AI RAW JSON]", content)
        data = json.loads(content)
        data["files"] = files_info
        data["retrieved_chunks"] = top_chunks  # pass evidence to frontend
        return data

    except Exception as e:
        print(f"[AI ERROR] {folder_path}: {e}")
        return {
            "error": f"Error while contacting AI service: {e}",
            "files": files_info,
        }


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/folder-assist", methods=["POST"])
def folder_assist():
    """
    Receive a folder path, run the RAG pipeline over all documents,
    and return structured data + file list + retrieved chunks + execution time.
    """
    data = request.get_json(silent=True) or {}
    folder_path_input = data.get("folder_path", "").strip()

    if not folder_path_input:
        return jsonify({"error": "No folder path provided."}), 200

    folder_path = folder_path_input

    if not os.path.isdir(folder_path):
        print(f"[ERROR] Folder does not exist: {folder_path}")
        return jsonify({"error": f"Folder does not exist: {folder_path}"}), 200

    start = time.perf_counter()
    result = analyse_folder(folder_path)
    elapsed = time.perf_counter() - start

    result["elapsed_seconds"] = elapsed
    print(f"[METRICS] analyse_folder({folder_path}) took {elapsed:.2f} seconds")

    return jsonify(result), 200


@app.route("/ai-assist", methods=["POST"])
def ai_assist():
    data = request.get_json(silent=True) or {}
    raw_text = data.get("raw_text", "")

    if not raw_text.strip():
        return jsonify({"error": "No text provided."}), 400

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You help applicants to rewrite motivation/background statements "
                        "clearly, professionally and concisely for university admissions."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Rewrite and improve the following motivation/background text, "
                        "keeping all important information:\n\n" + raw_text
                    ),
                },
            ],
            max_completion_tokens=300,
            temperature=0.4,
        )
        improved = response.choices[0].message.content
        return jsonify({"improved_text": improved})
    except Exception as e:
        print(f"[AI ERROR] /ai-assist: {e}")
        return jsonify({"error": "Error while contacting the AI service."}), 500


if __name__ == "__main__":
    app.run(debug=True)
