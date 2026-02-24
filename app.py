import streamlit as st
import pickle
import re
import pdfplumber
from docx import Document

# 1. PAGE SETUP & OBJECTIVE
st.set_page_config(page_title="HRM Document Classifier", layout="wide")

st.sidebar.markdown("""
### 🎯 Business Objective
The document classification solution aims to **significantly reduce manual human effort** in HRM. 

It targets a **higher level of accuracy and automation** with minimal human intervention.

**Scope:**
* Automated Resume Screening
* Skill Categorization
* (Future) Financial Document Sorting
""")

st.sidebar.info("ℹ️ **Current Version:** Optimized for Tech Resumes (React, SQL, Workday, PeopleSoft).")

# 2. LOAD MODEL (USING SPECIFIED PATH)
@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            clf, tfidf, le = pickle.load(file)
        return clf, tfidf, le

    except FileNotFoundError:
        st.error(f"⚠️ Error: Model file not found at:\n{model_path}")
        return None, None, None

    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, None, None


# 🔽 PASTE YOUR FULL PATH HERE
MODEL_PATH = r"C:\Users\LENOVO\OneDrive\Desktop\Resume classification\resume_classifier.pkl"

clf, tfidf, le = load_model(MODEL_PATH)


# 3. HELPER FUNCTIONS
def clean_resume_text(text):
    text = text.lower()
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'www\S+\s*', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9#\+\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text(uploaded_file):
    text = ""
    try:
        if uploaded_file.name.endswith('.pdf'):
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t: text += t + "\n"
        elif uploaded_file.name.endswith('.docx'):
            doc = Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return text

def is_valid_resume(text):
    resume_keywords = ['experience', 'education', 'skills', 'summary', 'projects', 'professional', 'career']
    tech_keywords = ['react', 'javascript', 'sql', 'database', 'workday', 'peoplesoft', 'oracle', 'html', 'css', 'java']
    
    text_lower = text.lower()
    has_resume_keyword = any(keyword in text_lower for keyword in resume_keywords)
    has_tech_keyword = any(keyword in text_lower for keyword in tech_keywords)
    
    if has_resume_keyword and has_tech_keyword:
        return True
    return False

# 4. MAIN DASHBOARD

st.markdown("<h1 style='text-align: center; color: #4F8BF9;'>Resume Classification</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center;'>Automating Candidate Screening</h3>", unsafe_allow_html=True)
st.markdown("---")  # Add a line separator for a cleaner look

# Define Max File Size (5MB)
MAX_FILE_SIZE = 5 * 1024 * 1024 

# Initialize raw_text to None so we can access it outside the columns later
raw_text = None

# Layout: Two columns (Upload on left, Results on right)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a file (PDF, DOCX)", type=["pdf", "docx"])

with col2:
    st.subheader("Analysis Result")
    
    if uploaded_file is not None:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"⚠️ **File Too Large:** Please upload a file smaller than 5MB.")
            st.caption(f"Your file size: {uploaded_file.size / (1024*1024):.2f} MB")
        
        else:
            with st.spinner('Processing document...'):
                raw_text = extract_text(uploaded_file)
                
                if raw_text:
                    # GUARDRAIL CHECK
                    if not is_valid_resume(raw_text):
                        st.warning("⚠️ **Flagged:** Document does not appear to be a relevant technical resume.")
                        st.caption("Missing standard sections (Experience, Education) or technical keywords.")
                    else:
                        # PREDICTION
                        cleaned_text = clean_resume_text(raw_text)
                        vectorized_text = tfidf.transform([cleaned_text])
                        
                        # Confidence Check
                        probs = clf.predict_proba(vectorized_text)
                        max_prob = probs.max()
                        
                        if max_prob < 0.5:
                             st.warning(f"⚠️ **Uncertain:** Confidence is low ({max_prob:.2%}). Verify manually.")
                        else:
                            predicted_id = clf.predict(vectorized_text)[0]
                            category_name = le.inverse_transform([predicted_id])[0]
                            
                            # SUCCESS DISPLAY
                            st.success(f"✅ Classified as: **{category_name}**")
                            st.progress(max_prob)
                            st.caption(f"Confidence Score: {max_prob:.2%}")
                else:
                    st.error("Could not read document text.")
    else:
        st.info("Waiting for upload...")

# 5. EXTRACTED CONTENT VIEW (Full Width)

# This is now outside the columns, so it spans the whole page width
if raw_text:
    st.markdown("---")
    st.subheader("📄 Extracted Document Content")
    with st.expander("Click to view full resume text"):
        st.text_area("Raw Text", raw_text, height=300)

# Footer
st.markdown("---")
st.markdown("<h6 style='text-align: center; color: #4F8BF9;'>HRM Automation Tool v1.0 | Built with Python & Streamlit by Dongre Prashanth  and Doli Karthik </h6>", unsafe_allow_html=True)
