import streamlit as st
import PyPDF2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================
# Extract Text From PDF
# ==========================

def extract_text(pdf_file):

    text = ""

    try:
        reader = PyPDF2.PdfReader(pdf_file)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

    except Exception:
        return ""

    return text


# ==========================
# Calculate Similarity
# ==========================

def calculate_similarity(resume_text, job_description):

    documents = [resume_text, job_description]

    tfidf = TfidfVectorizer()

    tfidf_matrix = tfidf.fit_transform(documents)

    score = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )

    return score[0][0] * 100


# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="centered"
)

# ==========================
# Sidebar
# ==========================

st.sidebar.title("📊 Project Information")

st.sidebar.markdown("""
### AI Techniques
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Cosine Similarity

### Technologies
- Python
- Streamlit
- Scikit-Learn
- PyPDF2

### Purpose
Compare resumes with job descriptions and generate a matching score.
""")

# ==========================
# Main UI
# ==========================

st.title("📄 AI Resume Screening System")

st.caption(
    "Upload a resume and compare it with a job description using NLP."
)

st.divider()

uploaded_resume = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "Enter Job Description",
    height=200,
    placeholder="Paste the job description here..."
)

# ==========================
# Analyze Button
# ==========================

if st.button("🔍 Analyze Resume"):

    if not uploaded_resume:
        st.warning("Please upload a resume.")
        st.stop()

    if not job_description.strip():
        st.warning("Please enter a job description.")
        st.stop()

    resume_text = extract_text(uploaded_resume)

    if len(resume_text.strip()) == 0:
        st.error(
            "Unable to extract text from the PDF. Please upload a text-based PDF."
        )
        st.stop()

    score = calculate_similarity(
        resume_text,
        job_description
    )

    st.divider()

    st.subheader("📈 Match Analysis")

    st.metric(
        label="Resume Match Score",
        value=f"{score:.2f}%"
    )

    st.progress(int(score))

    if score >= 75:
        st.success("✅ Highly Recommended")

    elif score >= 60:
        st.info("👍 Recommended")

    else:
        st.error("❌ Not Recommended")

    st.divider()

    st.subheader("Result Summary")

    st.write(
        f"The resume matches approximately **{score:.2f}%** "
        f"of the provided job description."
    )

    st.info(
        "This score is calculated using TF-IDF Vectorization and Cosine Similarity."
    )