import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

def read_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

st.title("AI Resume Screening System")

uploaded_files = st.file_uploader("Upload Resume PDFs", accept_multiple_files=True, type=["pdf"])
job_desc = st.text_area("Paste Job Description")

if st.button("Screen Resumes") and uploaded_files and job_desc:
    resumes = [read_pdf(file) for file in uploaded_files]
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(resumes + [job_desc])
    
    resume_vectors = vectors[:-1]
    job_vector = vectors[-1]
    
    scores = cosine_similarity(resume_vectors, job_vector)
    
    st.subheader("Resume Match Scores:")
    for i, score in enumerate(scores):
        st.write(f"{uploaded_files[i].name}: {round(score[0]*100, 2)}%")
