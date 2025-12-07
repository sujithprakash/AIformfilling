import os
import json
import glob
import time

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

# Optional base folder for students (not enforced here)
BASE_STUDENT_FOLDER = r""  # leave empty to allow full paths from UI


# ---------------------------------------------------------
# Helper functions
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
    Read all PDF, DOCX and TXT files in the given folder (non-recursive)
    and return:
      - combined_text: concatenated text from all files that had text
      - files: list of file stats (name, extension, chars, had_text)
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


def analyse_folder(folder_path: str) -> dict:
    """
    Use the combined text from CV + certificates in a folder to extract
    structured data for the application form.
    """
    collected = collect_text_from_folder(folder_path)
    raw_text = collected["combined_text"]
    files_info = collected["files"]

    if not raw_text.strip():
        return {
            "error": "No readable text found in folder (PDF/DOCX/TXT).",
            "files": files_info,
        }

    # Truncate for token safety
    text_short = raw_text[:20000]

    user_prompt = f"""
You are assisting admissions staff at a university.

You are given text extracted from a student's CV and certificates,
combined from multiple documents. Using ONLY this information,
extract the following details where possible and return them as JSON.

Personal / contact:
- name
- email
- phone
- date_of_birth
- nationality
- address_line1
- address_line2
- city
- state_region
- country
- preferred_contact

Programme choice:
- level_of_study
- programme_interest
- study_mode
- intake

Previous qualifications (highest completed qualification):
- highest_qualification
- highest_institution
- field_of_study
- graduation_year
- grade

Other academic info:
- other_qualifications

Work experience:
- total_experience
- current_employer
- current_position
- industry
- experience_summary

Motivation / background:
- background_statement

AI helper:
- ai_notes

If a field is not clearly available in the documents, return an empty string.
Do NOT invent details.

Return a single JSON object with exactly these keys:
name, email, phone, date_of_birth, nationality,
address_line1, address_line2, city, state_region, country,
preferred_contact,
level_of_study, programme_interest, study_mode, intake,
highest_qualification, highest_institution, field_of_study,
graduation_year, grade, other_qualifications,
total_experience, current_employer, current_position,
industry, experience_summary,
background_statement, ai_notes.

TEXT FROM FOLDER (truncated):
\"\"\"{text_short}\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract structured data from CVs and academic certificates "
                        "to help fill university application forms. You must ALWAYS "
                        "return valid JSON with the exact keys requested and avoid "
                        "inventing information that is not supported by the text."
                    ),
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=500,
            temperature=0.2,
        )

        content = response.choices[0].message.content
        print("[AI RAW JSON]", content)
        data = json.loads(content)
        data["files"] = files_info          # attach file stats to result
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
    Receive a folder path, scan all PDFs/DOCXs/TXTs inside,
    and return structured data + file list + execution time.
    """
    data = request.get_json(silent=True) or {}
    folder_path_input = data.get("folder_path", "").strip()

    if not folder_path_input:
        return jsonify({"error": "No folder path provided."}), 200

    # For now, accept the full path directly from UI
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
            model="gpt-4.1-mini",
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
            max_completion_tokens=250,
            temperature=0.4,
        )
        improved = response.choices[0].message.content
        return jsonify({"improved_text": improved})
    except Exception as e:
        print(f"[AI ERROR] /ai-assist: {e}")
        return jsonify({"error": "Error while contacting the AI service."}), 500


if __name__ == "__main__":
    app.run(debug=True)
