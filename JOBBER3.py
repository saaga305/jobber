import streamlit as st
import pandas as pd
import joblib
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

from io import StringIO, BytesIO
import PyPDF2
import docx2txt

st.set_page_config(page_title="SmartHireAI", layout="centered", page_icon="🤖")

# ✅ Inject minimal CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f3f6fc;
        }
        .title-container {
            background-color: #e8f0fe;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
            box-shadow: 0 0 8px rgba(0, 0, 0, 0.05);
        }
        .title-text {
            font-size: 32px;
            font-weight: bold;
            color: #1a73e8;
            margin-bottom: 0;
        }
        .subtext {
            font-size: 15px;
            color: #666;
            margin-top: 0;
        }
        .footer {
            text-align: center;
            font-size: 13px;
            margin-top: 3rem;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)

# Title Block
st.markdown("""
<div class="title-container">
    <p class="title-text">🤖 SmartHireAI: Intelligent Job Matching</p>
    <p class="subtext">Match your resume with the job you're aiming for. Understand your skill gaps. Learn and grow.</p>
</div>
""", unsafe_allow_html=True)
# Skill categories and associated keywords
SKILL_DICT = {
    'Java Developer': ['java', 'spring', 'hibernate', 'junit'],
    'Python Developer': ['python', 'django', 'flask', 'numpy', 'pandas'],
    'Testing': ['selenium', 'testng', 'junit', 'manual testing', 'automation'],
    'DevOps Engineer': ['jenkins', 'docker', 'kubernetes', 'ci/cd', 'ansible'],
    'Web Designing': ['html', 'css', 'javascript', 'photoshop'],
    'HR': ['recruitment', 'employee engagement', 'onboarding', 'hrms'],
    'Hadoop': ['hadoop', 'pig', 'hive', 'spark', 'mapreduce'],
    'Sales': ['negotiation', 'crm', 'lead generation', 'client management'],
    'Data Science': ['machine learning', 'pandas', 'scikit-learn', 'matplotlib', 'modeling'],
    'Mechanical Engineer': ['autocad', 'solidworks', 'thermodynamics', 'manufacturing'],
    'ETL Developer': ['informatica', 'ssrs', 'ssas', 'data warehouse'],
    'Blockchain': ['blockchain', 'ethereum', 'solidity', 'web3'],
    'Operations Manager': ['logistics', 'kpi', 'supply chain', 'inventory'],
    'Arts': ['drawing', 'painting', 'animation', 'illustration'],
    'Database': ['sql', 'oracle', 'mongodb', 'database design'],
    'Health and fitness': ['yoga', 'nutrition', 'workout', 'exercise'],
    'PMO': ['project management', 'pmo tools', 'milestones', 'ms project'],
    'Electrical Engineering': ['circuit', 'voltage', 'power', 'transformer'],
    'Business Analyst': ['requirement gathering', 'uml', 'brd', 'data analysis'],
    'DotNet Developer': ['c#', '.net', 'asp.net', 'visual studio'],
    'Automation Testing': ['selenium', 'qtp', 'loadrunner', 'automation'],
    'Network Security Engineer': ['firewall', 'vpn', 'intrusion detection', 'encryption'],
    'Civil Engineer': ['autocad', 'staad', 'construction', 'estimation'],
    'SAP Developer': ['sap', 'abap', 'fiori', 'hana'],
    'Advocate': ['litigation', 'legal research', 'drafting', 'case law']
}
# ✅ Define LEARNING_RESOURCES
LEARNING_RESOURCES = {
    'Python Developer': [
        ("DataCamp", "https://www.datacamp.com"),
        ("Coursera", "https://www.coursera.org"),
        ("freeCodeCamp", "https://www.freecodecamp.org"),
        ("YouTube", "https://www.youtube.com/results?search_query=python+developer")
    ],
    'Civil Engineer': [
        ("Coursera", "https://www.coursera.org/search?query=civil%20engineering"),
        ("Skill-Lync", "https://skill-lync.com/civil-engineering-courses"),
        ("YouTube", "https://www.youtube.com/results?search_query=civil+engineering")
    ],
    'Data Science': [
        ("DataCamp", "https://www.datacamp.com"),
        ("Coursera", "https://www.coursera.org"),
        ("freeCodeCamp", "https://www.freecodecamp.org"),
        ("YouTube", "https://www.youtube.com/results?search_query=data+science")
    ]
    # Add more roles and links if needed
}

# ✅ Define detect_job_category to avoid NameError

def detect_job_category(job_text):
    job_text = job_text.lower()
    for category in SKILL_DICT:
        if category.lower() in job_text:
            return category
    return None

# (Rest of the app logic continues)
@st.cache_resource
def load_models():
    clf_model = joblib.load("model.pkl")
    tfidf_vec = joblib.load("resume_vectorizer.pkl")
    sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return clf_model, tfidf_vec, sbert

classifier_model, tfidf_vectorizer, sbert_model = load_models()

# Helper functions for text extraction

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    return " ".join([page.extract_text() or '' for page in pdf_reader.pages])

def extract_text_from_docx(file):
    return docx2txt.process(file)

def extract_text_from_uploaded_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ]:
        return extract_text_from_docx(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    else:
        st.warning("Unsupported file type. Please upload PDF, DOCX, or TXT.")
        return ""

# (Remaining code continues unchanged)

# Input Section
st.markdown("""
### 📄 Upload Section
Upload your resume and job description, or paste the text directly.
*Supported formats:* PDF, DOCX, TXT
""")

col1, col2 = st.columns(2)
with col1:
    uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
    resume_input = st.text_area("Or paste your resume here")

with col2:
    uploaded_job = st.file_uploader("Upload Job Description", type=["pdf", "txt"])
    job_input = st.text_area("Or paste job description here")

st.markdown("""
---
📌 **Example Job Descriptions:**
- Looking for a *Civil Engineer* with **AutoCAD**, **construction**, and **project supervision** skills.
- Hiring a *Python Developer* experienced in **Flask**, **Pandas**, and **API development**.
- Seeking a *Data Scientist* with knowledge of **machine learning**, **data visualization**, and **statistics**.
---
""")

if st.button("🔍 Match Now"):
    with st.spinner("Processing and matching your resume..."):
        resume_text = extract_text_from_uploaded_file(uploaded_resume) if uploaded_resume else resume_input
        job_text = extract_text_from_uploaded_file(uploaded_job) if uploaded_job else job_input

        if not resume_text.strip() or not job_text.strip():
            st.warning("Please provide both resume and job description.")
        else:
            resume_clean = re.sub(r'\r|\n|\t', ' ', resume_text)
            resume_clean = unicodedata.normalize('NFKD', resume_clean).encode('ascii', 'ignore').decode('ascii')
            resume_clean = re.sub(r'[^a-zA-Z ]', '', resume_clean).lower()

            job_clean = re.sub(r'\r|\n|\t', ' ', job_text)
            job_clean = unicodedata.normalize('NFKD', job_clean).encode('ascii', 'ignore').decode('ascii')
            job_clean = re.sub(r'[^a-zA-Z ]', '', job_clean).lower()

            resume_vec = tfidf_vectorizer.transform([resume_clean])
            predicted_cat = classifier_model.predict(resume_vec)[0]

            resume_embed = sbert_model.encode([resume_clean], convert_to_tensor=True)
            job_embed = sbert_model.encode([job_clean], convert_to_tensor=True)
            similarity = util.cos_sim(resume_embed, job_embed)[0][0].item() * 100

            override_category = st.selectbox("Optional: Select job category manually", [""] + list(SKILL_DICT.keys()))
            job_category = override_category if override_category else detect_job_category(job_text)

            resume_skills = [skill for skill in SKILL_DICT.get(predicted_cat, []) if skill in resume_clean]
            job_skills = [skill for skill in SKILL_DICT.get(job_category, []) if skill in job_clean]
            lacking_skills = list(set(job_skills) - set(resume_skills))

            st.markdown("## 🎯 Match Results")
            st.progress(int(similarity))
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**✅ Matching Score:** `{similarity:.2f}%`")
                st.markdown(f"**🧠 Predicted Resume Category:** `{predicted_cat}`")
                st.markdown(f"**✔️ Resume Skills:** `{', '.join(resume_skills)}`")
            with col2:
                st.markdown(f"**🏷️ Detected Job Category:** `{job_category or 'Not Found'}`")
                st.markdown(f"**📌 Required Skills:** `{', '.join(job_skills)}`")
                st.markdown(f"**❌ To Learn:** `{', '.join(lacking_skills) if lacking_skills else 'None 🎉'}`")

            summary = f"""
Match Score: {similarity:.2f}%
Predicted Category: {predicted_cat}
Detected Category: {job_category or 'N/A'}

Skills in Resume: {', '.join(resume_skills)}
Required Skills: {', '.join(job_skills)}
Skills To Learn: {', '.join(lacking_skills) if lacking_skills else 'None'}
"""
            st.download_button("📥 Save Summary", summary, file_name="smart_match_summary.txt")

            if lacking_skills:
                st.info(f"📚 You can learn these skills to grow in `{job_category}`:")
                if job_category in LEARNING_RESOURCES:
                    for name, url in LEARNING_RESOURCES[job_category]:
                        st.markdown(f"- [{name}]({url})")
                else:
                    st.markdown("- [YouTube](https://www.youtube.com)")

# Footer
st.markdown("""
<div class="footer">
SmartHireAI &copy; 2025 | Streamlit + scikit-learn + Sentence-BERT | Built with ❤️
</div>
""", unsafe_allow_html=True)
