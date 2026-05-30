import streamlit as st
import os
import io
import json
import pdfplumber
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

st.set_page_config(page_title="📚 Employee Training System Using RAG", layout="wide")

def get_groq_api_key():
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    return os.environ.get("GROQ_API_KEY", "")

groq_api_key = get_groq_api_key()

if not groq_api_key:
    st.error("Missing GROQ_API_KEY. Add it to Streamlit Secrets or your environment variables.")
    st.stop()

if "course_content" not in st.session_state:
    st.session_state.course_content = None
if "course_generated" not in st.session_state:
    st.session_state.course_generated = False
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "completed_questions" not in st.session_state:
    st.session_state.completed_questions = set()
if "total_questions" not in st.session_state:
    st.session_state.total_questions = 0
if "extracted_texts" not in st.session_state:
    st.session_state.extracted_texts = []
if "employer_queries" not in st.session_state:
    st.session_state.employer_queries = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

st.sidebar.title("🎓 Professional Learning System")

if st.sidebar.button("🔄 Reset Application"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.extracted_texts = []
    st.session_state.uploaded_files = []
    st.session_state.uploaded_file_names = []
    st.session_state.vector_store = None
    st.rerun()

uploaded_files = st.sidebar.file_uploader("📝 Upload Training PDFs", type=["pdf"], accept_multiple_files=True)

def extract_pdf_text(pdf_file):
    try:
        pdf_file.seek(0)
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return ""

if uploaded_files:
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
                    st.session_state.extracted_texts.append(
                        {"filename": pdf_file.name, "text": extracted_text}
                    )
                    st.session_state.uploaded_files.append(pdf_file)
                    documents.append(
                        Document(page_content=extracted_text, metadata={"filename": pdf_file.name})
                    )

        if st.session_state.extracted_texts:
            st.sidebar.success(f"✅ {len(st.session_state.extracted_texts)} PDF files processed successfully!")

            try:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_docs = text_splitter.split_documents(documents)

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vector_store = FAISS.from_documents(
                    documents=split_docs,
                    embedding=embeddings,
                )
            except Exception as e:
                st.error(f"Error initializing vector store: {e}")
else:
    st.info("📥 Please upload PDFs to begin.")

model_options = ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"]
selected_model = st.sidebar.selectbox("Select Free LLM Engine", model_options, index=0)

role_options = ["Manager", "Executive", "Developer", "Designer", "Marketer", "Human Resources", "Other", "Fresher"]
role = st.sidebar.selectbox("Select Your Role", role_options)

learning_focus_options = ["Leadership", "Technical Skills", "Communication", "Project Management", "Innovation", "Team Building", "Finance"]
learning_focus = st.sidebar.multiselect("Select Learning Focus", learning_focus_options)

if st.session_state.uploaded_file_names:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Uploaded Files")
    for i, filename in enumerate(st.session_state.uploaded_file_names):
        st.sidebar.text(f"{i+1}. {filename}")

def generate_rag_answer(question):
    try:
        if not st.session_state.vector_store:
            return "Vector store not initialized. Please process documents first."

        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
        retrieved_docs = retriever.get_relevant_documents(question)

        if not retrieved_docs:
            return "I couldn't find relevant information in the uploaded documents to answer this question."

        context = ""
        for i, doc in enumerate(retrieved_docs):
            context += f"\nDocument {i+1}: {doc.metadata.get('filename', 'Unknown')}\nContent: {doc.page_content}\n"

        prompt = f"""
You are an AI assistant for a professional learning platform. Answer the following question
based STRICTLY on the provided document content.

Question: {question}

Document Content: {context}

IMPORTANT INSTRUCTIONS:
1. Use ONLY the information from the uploaded documents.
2. If the question cannot be answered based on the documents, say:
   "I can answer questions based on the uploaded documents only"
3. Do not use any external knowledge.
4. Reference specific document names.
5. Be concise and accurate.
"""

        llm = ChatGroq(model_name=selected_model, temperature=0.2, groq_api_key=groq_api_key)
        response = llm.invoke(prompt)
        answer = response.content

        if retrieved_docs:
            answer += "\n\n**References:**\n"
            unique_files = set()
            for doc in retrieved_docs:
                unique_files.add(doc.metadata.get("filename", "Unknown"))
            for filename in unique_files:
                answer += f"- {filename}\n"

        return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def check_answer(question_id, user_answer, correct_answer):
    if user_answer.strip().lower() == correct_answer.strip().lower():
        st.success("🎉 Correct! Well done!")
        st.session_state.completed_questions.add(question_id)
        return True
    else:
        st.error(f"Not quite. The correct answer is: {correct_answer}")
        return False

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

    if hasattr(st.session_state, "course_content") and st.session_state.course_content:
        course_title = st.session_state.course_content.get("course_title", "N/A")
        c.drawString(50, y_position, f"Course: {course_title}")
        y_position -= 20

    completed = len(st.session_state.completed_questions)
    total = st.session_state.total_questions
    progress_percentage = (completed / total * 100) if total > 0 else 0
    c.drawString(50, y_position, f"Questions Completed: {completed}/{total}")
    y_position -= 20
    c.drawString(50, y_position, f"Progress Percentage: {progress_percentage:.1f}%")
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
            combined_docs += f"\n--- DOCUMENT {i+1}: {doc['filename']} ---\n"
            combined_docs += doc["text"][:2000] + "\n\n"

        professional_context = f"Role: {role}, Focus: {', '.join(learning_focus)}"

        prompt = f"""
Design a comprehensive professional learning course based on the multiple documents provided.

Context: {professional_context}
Document Contents: {combined_docs[:4000]}

Return ONLY a valid JSON object with this schema:
{{
  "course_title": "Your Synthesized Title",
  "course_description": "Detailed summary describing how these combined documents relate.",
  "modules": [
    {{
      "title": "Module 1 Title",
      "learning_objectives": ["Objective A", "Objective B"],
      "content": "Comprehensive textbook style reading module text...",
      "quiz": {{
        "questions": [
          {{
            "question": "Question text?",
            "options": ["Choice A", "Choice B", "Choice C", "Choice D"],
            "correct_answer": "Choice A"
          }}
        ]
      }}
    }}
  ]
}}
"""

        llm = ChatGroq(model_name=selected_model, temperature=0.4, groq_api_key=groq_api_key)
        response = llm.invoke(prompt)
        response_content = response.content.strip()

        if response_content.startswith("```json"):
            response_content = response_content[7:]
        if response_content.endswith("```"):
            response_content = response_content[:-3]
        response_content = response_content.strip()

        try:
            st.session_state.course_content = json.loads(response_content)
            st.session_state.course_generated = True
            total_questions = 0
            for module in st.session_state.course_content.get("modules", []):
                quiz = module.get("quiz", {})
                total_questions += len(quiz.get("questions", []))
            st.session_state.total_questions = total_questions
        except json.JSONDecodeError:
            st.error("The model returned invalid JSON. Please try generating the course again.")

    except Exception as e:
        st.error(f"Error: {e}")
    st.session_state.is_generating = False

tab1, tab2, tab3 = st.tabs(["📚 Course Content", "❓ Employer Queries", "📑 Document Sources"])

if st.session_state.is_generating:
    with st.spinner("Compiling structural coursework syllabus..."):
        st.session_state.completed_questions = set()
        perform_course_generation()
        st.rerun()

with tab1:
    if st.session_state.course_generated and st.session_state.course_content:
        course = st.session_state.course_content
        st.title(f"🌟 {course.get('course_title', 'Professional Course')}")
        st.write(course.get("course_description", ""))

        completed = len(st.session_state.completed_questions)
        total = st.session_state.total_questions
        progress_percentage = (completed / total * 100) if total > 0 else 0
        st.progress(progress_percentage / 100)

        modules = course.get("modules", [])
        for i, module in enumerate(modules, 1):
            with st.expander(f"📚 Module {i}: {module.get('title')}"):
                st.write(module.get("content"))

                questions = module.get("quiz", {}).get("questions", [])
                for q_idx, q in enumerate(questions, 1):
                    question_id = f"module_{i}_question_{q_idx}"
                    st.markdown(f"**Question {q_idx}:** {q.get('question')}")
                    options = q.get("options", [])
                    if options:
                        user_answer = st.radio("Select answer:", options, key=f"q_{i}_{q_idx}")
                        if question_id in st.session_state.completed_questions:
                            st.success("✓ Solved")
                        else:
                            if st.button("Check Answer", key=f"btn_{i}_{q_idx}"):
                                check_answer(question_id, user_answer, q.get("correct_answer"))
    else:
        st.title("Welcome to the Professional Training Workspace")
        st.markdown("Upload multiple training PDFs to synthesize an interactive curriculum program.")
        if st.session_state.extracted_texts and not st.session_state.is_generating:
            if st.button("🚀 Generate My Course", use_container_width=True):
                generate_course()

with tab2:
    st.title("💬 Employer Queries")
    if not st.session_state.employer_queries:
        st.info("No queries filed yet.")
    else:
        for i, query in enumerate(st.session_state.employer_queries):
            with st.expander(f"Question {i+1}: {query['question']}"):
                st.write(query["answer"])

with tab3:
    st.title("📑 Document Sources")
    if not st.session_state.extracted_texts:
        st.info("No documents uploaded yet.")
    else:
        for i, doc in enumerate(st.session_state.extracted_texts):
            with st.expander(f"Document {i+1}: {doc['filename']}"):
                st.text_area("Preview:", value=doc["text"][:1000], height=200, disabled=True)
