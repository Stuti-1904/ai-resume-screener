import os
from flask import Flask, render_template, request, send_file
from PyPDF2 import PdfReader
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import io

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load AI model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper to extract text from PDF
def extract_text_from_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_description = request.form.get("job_description")
        if not job_description:
            return "Please provide a job description"

        # Read resumes
        resume_texts = []
        resume_names = []
        files = request.files.getlist("resumes")
        for file in files:
            if file.filename == "":
                continue
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            if file.filename.endswith(".pdf"):
                text = extract_text_from_pdf(filepath)
            else:
                text = file.read().decode("utf-8")
            resume_texts.append(text)
            resume_names.append(file.filename)

        if not resume_texts:
            return "No resumes uploaded"

        # AI embeddings
        job_vec = model.encode(job_description, convert_to_tensor=True)
        resume_vecs = model.encode(resume_texts, convert_to_tensor=True)

        # Cosine similarity
        scores = util.cos_sim(job_vec, resume_vecs)[0].cpu().tolist()

        # Extract top 5 keywords/skills for display (simple example)
        keywords = ["Python", "Flask", "Machine Learning", "SQL", "Data", "API", "Cloud", "Git"]
        top_skills_list = []
        for text in resume_texts:
            skills_found = [kw for kw in keywords if kw.lower() in text.lower()]
            top_skills_list.append(", ".join(skills_found[:5]))

        # Prepare results DataFrame
        results = pd.DataFrame({
            "Resume": resume_names,
            "Score": [round(s * 100, 2) for s in scores],
            "Top Skills": top_skills_list
        }).sort_values(by="Score", ascending=False)

        # Save results to session or temp CSV
        results.to_csv("uploads/results.csv", index=False)

        return render_template("results.html", tables=results.to_dict(orient="records"))

    return render_template("index.html")

# Download CSV route
@app.route("/download")
def download():
    path = "uploads/results.csv"
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
