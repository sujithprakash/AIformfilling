import io
import os
import json
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
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload limit

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 8
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-mini"


# ---------------------------------------------------------
# Text extraction helpers (work on in-memory bytes)
# ---------------------------------------------------------

def extract_text_from_pdf_bytes(data: bytes) -> str:
    text = ""
    try:
        reader = PdfReader(io.BytesIO(data))
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
    except Exception as e:
        print(f"[PDF ERROR] {e}")
    return text


def extract_text_from_docx_bytes(data: bytes) -> str:
    try:
        document = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs)
    except Exception as e:
        print(f"[DOCX ERROR] {e}")
        return ""


def extract_text_from_txt_bytes(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


# ---------------------------------------------------------
# RAG helpers: chunking, embedding, retrieval
# ---------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def embed_texts(texts: list) -> np.ndarray:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def cosine_similarity(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    corpus_norms = corpus_vecs / (np.linalg.norm(corpus_vecs, axis=1, keepdims=True) + 1e-10)
    return corpus_norms @ query_norm


def retrieve_top_chunks(query: str, chunks: list, chunk_vecs: np.ndarray, k: int = TOP_K) -> list:
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


def analyse_uploaded_files(uploaded_files) -> dict:
    """
    Full RAG pipeline over uploaded file objects (werkzeug FileStorage).
    1. Extract text from each file in memory
    2. Chunk the combined corpus
    3. Embed all chunks
    4. Retrieve top-k chunks most relevant to the extraction query
    5. LLM structured extraction over retrieved context
    """
    files_info = []
    combined_text = ""

    for f in uploaded_files:
        filename = f.filename
        ext = os.path.splitext(filename)[1].lower()
        data = f.read()
        file_text = ""

        if ext == ".pdf":
            file_text = extract_text_from_pdf_bytes(data)
        elif ext == ".docx":
            file_text = extract_text_from_docx_bytes(data)
        elif ext == ".txt":
            file_text = extract_text_from_txt_bytes(data)
        else:
            print(f"[SKIP] Unsupported file type: {filename}")

        had_text = bool(file_text.strip())
        files_info.append({
            "filename": filename,
            "extension": ext,
            "chars": len(file_text),
            "had_text": had_text,
        })

        if had_text:
            combined_text += f"\n\n===== FILE: {filename} =====\n" + file_text
        else:
            print(f"[WARN] No text extracted from {filename}")

    if not combined_text.strip():
        return {
            "error": "No readable text found in uploaded files. Image-only PDFs require OCR.",
            "files": files_info,
        }

    # Chunk → embed → retrieve
    chunks = chunk_text(combined_text)
    print(f"[RAG] {len(chunks)} chunks from {len(combined_text)} chars")

    chunk_vecs = embed_texts(chunks)
    top_chunks = retrieve_top_chunks(EXTRACTION_QUERY, chunks, chunk_vecs, k=TOP_K)
    context = "\n\n---\n\n".join(top_chunks)
    print(f"[RAG] Retrieved top-{TOP_K} chunks ({len(context)} chars)")

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
        data = json.loads(response.choices[0].message.content)
        data["files"] = files_info
        data["retrieved_chunks"] = top_chunks
        return data

    except Exception as e:
        print(f"[AI ERROR] {e}")
        return {"error": f"AI service error: {e}", "files": files_info}


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload-assist", methods=["POST"])
def upload_assist():
    uploaded = request.files.getlist("files")
    if not uploaded or all(f.filename == "" for f in uploaded):
        return jsonify({"error": "No files uploaded."}), 200

    start = time.perf_counter()
    result = analyse_uploaded_files(uploaded)
    result["elapsed_seconds"] = time.perf_counter() - start
    print(f"[METRICS] upload-assist took {result['elapsed_seconds']:.2f}s")
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
        return jsonify({"improved_text": response.choices[0].message.content})
    except Exception as e:
        print(f"[AI ERROR] /ai-assist: {e}")
        return jsonify({"error": "Error while contacting the AI service."}), 500


if __name__ == "__main__":
    app.run(debug=True)
