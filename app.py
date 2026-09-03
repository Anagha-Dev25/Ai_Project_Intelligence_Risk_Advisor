import hashlib
import os
import tempfile

import streamlit as st

from app.pipeline import process_document
from app.rag.langchain_rag import LangChainRAG


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ProjectIQ: An AI-Driven Enterprise Project Intelligence & Risk Management Platform",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }

    .metric-label {
        font-size: 14px;
        opacity: 0.7;
    }

    .result-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 15px;
    }

    .result-title {
        font-size: 19px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .source-text {
        font-size: 13px;
        opacity: 0.65;
        margin-top: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "processed_file_hash" not in st.session_state:
    st.session_state.processed_file_hash = None

if "current_source" not in st.session_state:
    st.session_state.current_source = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "search_count" not in st.session_state:
    st.session_state.search_count = 0


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 ProjectIQ: An AI-Driven Enterprise Project Intelligence & Risk Management Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Transform project documents into actionable intelligence using '
    'semantic search and AI-powered analysis.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Project Workspace")

    st.write("Upload a project document to begin analysis.")

    uploaded_file = st.file_uploader(
        "📄 Upload Document",
        type=["pdf", "docx", "txt", "csv"]
    )

    st.divider()

    


# --------------------------------------------------
# DOCUMENT PROCESSING
# --------------------------------------------------

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    file_hash = hashlib.md5(file_bytes).hexdigest()

    source_name = uploaded_file.name

    if st.session_state.processed_file_hash != file_hash:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(source_name)[1]
        ) as temp_file:

            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        try:

            with st.spinner("🔄 Processing project document..."):

                chunk_count = process_document(
                    temp_file_path,
                    source_name=source_name
                )

            st.session_state.processed_file_hash = file_hash
            st.session_state.current_source = source_name
            st.session_state.chunk_count = chunk_count

            st.success(
                f"✅ {source_name} processed successfully."
            )

        except Exception as e:

            st.error(
                f"❌ Document processing failed: {e}"
            )

        finally:

            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    else:

        st.info(
            f"✅ {source_name} is already loaded into the project workspace."
        )


# --------------------------------------------------
# PROJECT METRICS
# --------------------------------------------------

if st.session_state.current_source:

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    📄 {st.session_state.chunk_count}
                </div>
                <div class="metric-label">
                    Document Chunks
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">
                    🧠 384
                </div>
                <div class="metric-label">
                    Embedding Dimensions
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    🔍 {st.session_state.search_count}
                </div>
                <div class="metric-label">
                    Queries Analyzed
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


st.divider()


# --------------------------------------------------
# PROJECT INTELLIGENCE SEARCH
# --------------------------------------------------

st.subheader("🔎 Project Intelligence")

query = st.text_input(
    "Ask a question about your project",
    placeholder="Example: What are the major risks mentioned in this project?"
)


if st.button("🚀 Analyze Project", use_container_width=True):

    if not st.session_state.current_source:

        st.warning(
            "⚠️ Please upload a project document first."
        )

    elif not query.strip():

        st.warning(
            "⚠️ Please enter a project question."
        )

    else:

        with st.spinner("🧠 Analyzing project knowledge..."):

            rag = LangChainRAG()

            results = rag.search(
                query,
                top_k=3,
                source=st.session_state.current_source
            )

        st.session_state.search_count += 1

        st.subheader("💡 Relevant Project Intelligence")

    if results:

        for i, result in enumerate(results, start=1):

            st.markdown(f"### 🔹 Result {i}")
  
            st.write(result.page_content)

            st.caption(
                   f"📁 Source: {st.session_state.current_source}"
             )

        st.divider() 
st.caption(
    "🔐 Enterprise Project Intelligence • "
    "Powered by LangChain, OpenAI, and Streamlit"
)