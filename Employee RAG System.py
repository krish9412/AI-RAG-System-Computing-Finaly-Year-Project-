import streamlit as st
import os
import tempfile
import json
import io
import requests
import pdfplumber
import uuid
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Page Configuration
st.set_page_config(page_title="📚 Employee Training System Using RAG ", layout="wide")

# 🔐 SECURE API KEY EXTRACTION
# Attempt to load from Streamlit Secrets (Cloud Deployment) or local environment variables first
if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
elif os.environ.get("OPENAI_API_KEY"):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
else:
    # Fallback to allow manual input ONLY if secrets are not set up yet
    openai_api_key = ""

# Initializing session state variables
if 'course_content' not in st.session_state:
    st.session_state.course_content = None
if 'course_generated' not in st.session_state:
    st.session_state.course_generated = False
if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False
if 'completed_questions' not in st.session_state:
    st.session_state.completed_questions = set()
if 'total_questions' not in st.session_state:
    st.session_state.total_questions = 0
if 'extracted_texts' not in st.session_state:
    st.session_state.extracted_texts = []
if 'employer_queries' not in st.session_state:
    st.session_state.employer_queries = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'uploaded_file_names' not in st.session_state:
    st.session_state.uploaded_file_names = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Sidebars Appearance
st.sidebar.title("🎓 Professional Learning System")

# Clear Sessions Button & Session Management
if st.sidebar.button("🔄 Reset Application"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.extracted_texts = []
    st.session_state.uploaded_files = []
    st.session_state.uploaded_file_names = []
    st.session_state.vector_store = None
    st.rerun()

# Only display the input field if the secret key hasn't been injected in the backend
if not openai_api_key:
    openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API key", type="password")
    if not openai_api_key:
        st.info("💡 Backend API key missing. Please enter your OpenAI API key to use the application.")

# 📄 Multi-File Uploader for PDFs
uploaded_files = st.sidebar.file_uploader("📝 Upload Training PDFs", type=['pdf'], accept_multiple_files=True)

# Function to extract text from PDF
def extract_pdf_text(pdf_file):
    try:
        pdf_file.seek(0)
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return ""

# Process uploaded files and add to session state
if uploaded_files and openai_api_key:
    current_filenames = [file.name for file in uploaded_files]
    if current_filenames != st.session_state.uploaded_file_names:
        st.session_state.extracted_texts = []
        st.session_state.uploaded_files = []
        st.session_state.uploaded_file_names = current_filenames
        documents = []
        
        with st.spinner("Processing PDF files..."):
            for pdf_file in uploaded_files:
                extracted_text = extract_pdf_text(pdf_file)
                if extracted_text:
                    st.session_state.extracted_texts.append({
                        "filename": pdf_file.name,
                        "text": extracted_text
                    })
                    st.session_state.uploaded_files.append(pdf_file)
                    documents.append(Document(page_content=extracted_text, metadata={"filename": pdf_file.name}))
                    
        if st.session_state.extracted_texts:
            st.sidebar.success(f"✅ {len(st.session_state.extracted_texts)} PDF files processed successfully!")
            
            try:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_docs = text_splitter.split_documents(documents)
                
                embeddings = OpenAIEmbeddings(api_key=openai_api_key)
                st.session_state.vector_store = FAISS.from_documents(
                    documents=split_docs,
                    embedding=embeddings
                )
            except Exception as e:
                st.error(f"Error initializing vector store: {e}")
elif not uploaded_files:
    st.info("📥 Please upload PDF files to begin.")

# 🎯 Model and Role selection
model_options = ["gpt-4o-mini", "gpt-4o"]
selected_model = st.sidebar.selectbox("Select OpenAI Model", model_options, index=0)

role_options = ["Manager", "Executive", "Developer", "Designer", "Marketer", "Human Resources", "Other", "Fresher"]
role = st.sidebar.selectbox("Select Your Role", role_options)

learning_focus_options = ["Leadership", "Technical Skills", "Communication", "Project Management", "Innovation", "Team Building", "Finance"]
learning_focus = st.sidebar.multiselect("Select Learning Focus", learning_focus_options)

# Display uploaded files in sidebar
if st.session_state.uploaded_file_names:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Uploaded Files")
    for i, filename in enumerate(st.session_state.uploaded_file_names):
        st.sidebar.text(f"{i+1}. {filename}")

# Enhanced RAG function using LangChain with FAISS
def generate_rag_answer(question):
    try:
        if not openai_api_key:
            return "API key is required to generate answers."
        
        if not st.session_state.vector_store:
            return "Vector store not initialized. Please process documents first."
            
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.get_relevant_documents(question)
        
        if not retrieved_docs:
            return "I couldn't find relevant information in the uploaded documents to answer this question."
        
        context = ""
        for i, doc in enumerate(retrieved_docs):
            context += f"\nDocument {i+1}: {doc.metadata.get('filename', 'Unknown')}\nContent: {doc.page_content}\n"
        
        prompt = f"""
        You are an AI assistant for a professional learning platform. Answer the following question 
        based STRICTLY on the provided document content. You must only use information found in the uploaded documents.
        
        Question: {question}
        
        Document Content: {context}
        
        IMPORTANT INSTRUCTIONS:
        1. Use ONLY the information from the uploaded documents to answer the question
        2. If the question cannot be answered based on the provided documents, say "I can answer questions based on the uploaded documents only"
        3. Do not use any external knowledge or information not present in the documents
        4. Reference specific document names when providing information
        5. If the documents contain conflicting information, mention this in your answer
        6. Be concise and accurate
        """
        
        llm = ChatOpenAI(api_key=openai_api_key, model=selected_model, temperature=0.2)
        response = llm.invoke(prompt)
        answer = response.content
        
        if retrieved_docs:
            answer += "\n\n**References:**\n"
            unique_files = set()
            for doc in retrieved_docs:
                filename = doc.metadata.get("filename", "Unknown")
                unique_files.add(filename)
            for filename in unique_files:
                answer += f"- {filename}\n"
        
        return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Employee Queries Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Employee Queries")

new_query = st.sidebar.text_area("Add a new question:", height=100)
if st.sidebar.button("Submit Question"):
    if new_query:
        answer = ""
        if st.session_state.vector_store:
            with st.spinner("Generating answer from uploaded documents..."):
                try:
                    answer = generate_rag_answer(new_query)
                except Exception as e:
                    answer = f"Error generating answer: {str(e)}"
        else:
            answer = "Please upload and process documents first to enable question answering."
        
        st.session_state.employer_queries.append({
            "question": new_query,
            "answer": answer,
            "answered": bool(answer)
        })
        st.sidebar.success("Question submitted and answered!")
        st.rerun()

# Functions to check answer and update progress
def check_answer(question_id, user_answer, correct_answer):
    if user_answer.strip().lower() == correct_answer.strip().lower():
        st.success("🎉 Correct! Well done!")
        st.session_state.completed_questions.add(question_id)
        return True
    else:
        st.error(f"Not quite. The correct answer is: {correct_answer}")
        return False

# Generate Progress Report
def generate_progress_report():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 50, "Professional Learning Platform")
    c.setFont("Helvetica", 12)
    c.drawCentredString(width / 2, height - 70, "Training Progress Report")
    c.line(50, height - 80, width - 50, height - 80)

    c.setFont("Helvetica", 12)
    y_position = height - 110

    c.drawString(50, y_position, f"User Role: {role}")
    y_position -= 20
    c.drawString(50, y_position, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    y_position -= 20

    if hasattr(st.session_state, 'course_content') and st.session_state.course_content:
        course_title = st.session_state.course_content.get('course_title', 'N/A')
        c.drawString(50, y_position, f"Course: {course_title}")
        y_position -= 20
    if learning_focus:
        c.drawString(50, y_position, f"Learning Focus: {', '.join(learning_focus)}")
        y_position -= 20

    y_position -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Progress Overview:")
    c.setFont("Helvetica", 12)
    y_position -= 20
    completed = len(st.session_state.completed_questions)
    total = st.session_state.total_questions
    progress_percentage = (completed / total * 100) if total > 0 else 0
    c.drawString(50, y_position, f"Questions Completed: {completed}/{total}")
    y_position -= 20
    c.drawString(50, y_position, f"Progress Percentage: {progress_percentage:.1f}%")
    y_position -= 20

    y_position -= 10
    if progress_percentage >= 75:
        summary = "Excellent progress! You're almost done!"
    elif progress_percentage >= 50:
        summary = "Great job! You're more than halfway there!"
    elif progress_percentage > 0:
        summary = "Good start! Keep going to complete more modules!"
    else:
        summary = "Let's get started! Answer some quiz questions to track your progress."
    c.drawString(50, y_position, f"Summary: {summary}")
    y_position -= 20

    if st.session_state.completed_questions:
        y_position -= 10
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, "Completed Questions:")
        c.setFont("Helvetica", 12)
        y_position -= 20
        for qid in sorted(st.session_state.completed_questions):
            try:
                module_num = qid.split('_')[1]
                question_num = qid.split('_')[3]
                c.drawString(50, y_position, f"Module {module_num}, Question {question_num}")
                y_position -= 15
                if y_position < 50:
                    c.showPage()
                    y_position = height - 50
            except IndexError:
                continue

    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(width / 2, 30, f"Generated by Professional Learning Platform on {datetime.now().strftime('%Y-%m-%d')}")
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def generate_course():
    st.session_state.is_generating = True
    st.session_state.course_generated = False
    st.rerun()

def perform_course_generation():
    try:
        combined_docs = ""
        for i, doc in enumerate(st.session_state.extracted_texts):
            doc_summary = f"\n--- DOCUMENT {i+1}: {doc['filename']} ---\n"
            doc_summary += doc['text'][:3000]
            combined_docs += doc_summary + "\n\n"
        
        professional_context = f"Role: {role}, Focus: {', '.join(learning_focus)}"
        
        summary_query = "Create a comprehensive summary of these documents highlighting key concepts, theories, and practical applications across all materials."
        document_summary = generate_rag_answer(summary_query)
        
        prompt = f"""
        Design a comprehensive professional learning course based on the multiple documents provided.
        Context: {professional_context}
        Document Summary: {document_summary}
        
        Document Contents: {combined_docs[:5000]}
        
        Create an engaging, thorough and well-structured course by analyzing all provided documents.
        
        Return the response strictly in the following JSON format:
        {{
            "course_title": "Your Course Title",
            "course_description": "Detailed description of the course",
            "modules": [
                {{
                    "title": "Module 1 Title",
                    "learning_objectives": ["Objective 1", "Objective 2"],
                    "content": "Module content text with detailed explanations, examples, and practical applications",
                    "quiz": {{
                        "questions": [
                            {{
                                "question": "Question text?",
                                "options": ["Option A", "Option B", "Option C", "Option D"],
                                "correct_answer": "Option A"
                            }}
                        ]
                    }}
                }}
            ]
        }}
        """
        
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        response_content = response.choices[0].message.content
        try:
            st.session_state.course_content = json.loads(response_content)
            st.session_state.course_generated = True
            total_questions = 0
            for module in st.session_state.course_content.get("modules", []):
                quiz = module.get("quiz", {})
                total_questions += len(quiz.get("questions", []))
            st.session_state.total_questions = total_questions
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON response: {e}")
            st.text(response_content)
        
    except Exception as e:
        st.error(f"Error: {e}")
    
    st.session_state.is_generating = False

# Main contents area with tabs
tab1, tab2, tab3 = st.tabs(["📚 Course Content", "❓ Employer Queries", "📑 Document Sources"])

if st.session_state.is_generating:
    with st.spinner("Generating your personalized course from multiple documents..."):
        st.session_state.completed_questions = set()
        perform_course_generation()
        st.success("✅ Your Comprehensive Course is Ready!")
        st.rerun()

with tab1:
    if hasattr(st.session_state, 'course_generated') and st.session_state.course_generated and
