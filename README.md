# AI Resume Screening System

## 📌 Project Description
This project is an AI-based Resume Screening System that automatically ranks resumes based on how well they match a given job description using Natural Language Processing (NLP).

## 🚀 Features
- Upload multiple resume PDFs
- Paste job description
- Automatic resume ranking
- Match percentage using TF-IDF & Cosine Similarity
- Simple and interactive UI using Streamlit

## 🛠 Tech Stack
- Python
- Scikit-learn
- NLP (TF-IDF)
- Streamlit
- PyPDF2

## ▶️ How to Run

1. Create and activate environment
```bash
conda create -n resume_env python=3.10
conda activate resume_env
'''2.Install dependencies
'''bash
pip install -r requirements.txt
'''3.Run application
'''bash
streamlit run app_ui.py

