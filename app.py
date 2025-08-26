import os
import io
import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import streamlit as st

# Load AI model once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Helper to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# --- Streamlit App ---
st.title("📄 AI Resume Screener")
st.write("Upload resumes and compare them to a job description.")

# Job description input
job_description = st.text_area("✍️ Paste Job Description")

# Resume upload
uploaded_files = st.file_uploader(
    "📂 Upload Resumes (PDF or TXT)", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

if st.button("Analyze") and job_description and uploaded_files:
    resume_texts = []
    resume_names = []

    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        else:  # txt file
            text = file.read().decode("utf-8")

        resume_texts.append(text)
        resume_names.append(file.name)

    if resume_texts:
        # Embeddings
        job_vec = model.encode(job_description, convert_to_tensor=True)
        resume_vecs = model.encode(resume_texts, convert_to_tensor=True)

        # Cosine similarity
        scores = util.cos_sim(job_vec, resume_vecs)[0].cpu().tolist()

        # Simple keyword check
        keywords = ["Python", "Flask", "Machine Learning", "SQL", "Data", "API", "Cloud", "Git"]
        top_skills_list = []
        for text in resume_texts:
            skills_found = [kw for kw in keywords if kw.lower() in text.lower()]
            top_skills_list.append(", ".join(skills_found[:5]))

        # Results DataFrame
        results = pd.DataFrame({
            "Resume": resume_names,
            "Score (%)": [round(s * 100, 2) for s in scores],
            "Top Skills": top_skills_list
        }).sort_values(by="Score (%)", ascending=False)

        st.success("✅ Analysis Complete!")
        st.dataframe(results, use_container_width=True)

        # Download CSV
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv,
            file_name="results.csv",
            mime="text/csv",
        )
    else:
        st.warning("No valid resumes uploaded.")
