import streamlit as st
import PyPDF2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Extract text from uploaded PDF
def extract_text(pdf_file):
    text = ""

    reader = PyPDF2.PdfReader(pdf_file)

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text


# Calculate similarity
def calculate_similarity(resume_text, job_description):

    documents = [resume_text, job_description]

    tfidf = TfidfVectorizer()

    tfidf_matrix = tfidf.fit_transform(documents)

    score = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )

    return score[0][0] * 100


# UI
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄"
)

st.title("📄 AI Resume Screening System")

st.write(
    "Upload a Resume and compare it with a Job Description"
)

uploaded_resume = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "Enter Job Description"
)

if uploaded_resume and job_description:

    resume_text = extract_text(uploaded_resume)

    score = calculate_similarity(
        resume_text,
        job_description
    )

    st.subheader("Match Score")

    st.metric(
        label="Resume Match",
        value=f"{score:.2f}%"
    )

    if score >= 60:
        st.success("✅ Candidate Recommended")
    else:
        st.error("❌ Candidate Not Recommended")