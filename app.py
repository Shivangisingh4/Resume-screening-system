from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import os

def read_pdf(file_path):
    text = ""
    pdf = PyPDF2.PdfReader(file_path)
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Step 1: Place your PDF resumes in a folder
resume_folder = "resumes"  # create this folder and put PDFs inside
resume_files = os.listdir(resume_folder)

resumes = []
for file in resume_files:
    if file.endswith(".pdf"):
        resumes.append(read_pdf(os.path.join(resume_folder, file)))

# Step 2: Input job description
print("Paste Job Description:")
job_desc = input()

# Step 3: TF-IDF & similarity
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(resumes + [job_desc])

resume_vectors = vectors[:-1]
job_vector = vectors[-1]

scores = cosine_similarity(resume_vectors, job_vector)

# Step 4: Show match %
for i, score in enumerate(scores):
    print(f"{resume_files[i]} Match: {round(score[0]*100, 2)} %")
