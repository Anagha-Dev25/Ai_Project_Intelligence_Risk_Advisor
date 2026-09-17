import hashlib
import os
import tempfile

import streamlit as st

from app.pipeline import process_document
from app.rag.langchain_rag import LangChainRAG

from app.agents.scope_agent import ScopeAgent
from app.agents.risk_agent import RiskAgent
from app.agents.blocker_agent import BlockerAgent


# ============================================================
# CACHED COMPONENTS
# ============================================================

@st.cache_resource
def get_agents():
    return ScopeAgent(), RiskAgent(), BlockerAgent()


@st.cache_resource
def get_rag():
    return LangChainRAG()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ProjectIQ",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "processed_file_hash" not in st.session_state:
    st.session_state.processed_file_hash = None

if "current_source" not in st.session_state:
    st.session_state.current_source = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "search_count" not in st.session_state:
    st.session_state.search_count = 0

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

if "search_results" not in st.session_state:
    st.session_state.search_results = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Project Workspace")

    st.write(
        "Upload a project document to begin "
        "AI-powered project analysis."
    )

    # --------------------------------------------------------
    # DOCUMENT UPLOAD
    # --------------------------------------------------------

    uploaded_files = st.file_uploader(
        "📄 Upload Project Document",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True
    )

    st.divider()

    # --------------------------------------------------------
    # SEMANTIC RETRIEVAL
    # --------------------------------------------------------

    st.subheader("🔎 Ask Questions")

    st.caption(
        "Ask questions about your project "
        "using RAG-powered semantic search."
    )

    document_available = (
        st.session_state.current_source is not None
        or uploaded_files is not None
    )

    sidebar_query = st.text_input(
        "Ask a question",
        placeholder="e.g. What are the project dependencies?",
        disabled=not document_available
    )

    search_button = st.button(
        "🔍 Search",
        use_container_width=True,
        disabled=not document_available
    )

    st.divider()

    # --------------------------------------------------------
    # ACTIVE PROJECT
    # --------------------------------------------------------

    if st.session_state.current_source:

        st.subheader("📌 Active Project")

        st.caption(
            st.session_state.current_source
        )

        st.success("🟢 Document Loaded")


# ============================================================
# MAIN PROJECTIQ HEADER
# ============================================================

st.title("🤖 ProjectIQ")

st.caption(
    "AI-Driven Enterprise Project Intelligence & "
    "Risk Management Platform"
)

st.success(
    "🟢 Local AI • Qwen 2.5 • RAG Powered"
)


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

if uploaded_files:

    combined_hash = hashlib.md5()

    for uploaded_file in uploaded_files:
        combined_hash.update(
            uploaded_file.name.encode()
        )
        combined_hash.update(
            uploaded_file.getvalue()
        )

    file_hash = combined_hash.hexdigest()

    if st.session_state.processed_file_hash != file_hash:

        total_chunks = 0
        processed_sources = []

        try:

            st.session_state.analysis_results = None
            st.session_state.search_results = None

            with st.spinner(
                "🔄 Processing project documents..."
            ):

                for uploaded_file in uploaded_files:

                    file_bytes = uploaded_file.getvalue()
                    source_name = uploaded_file.name

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=os.path.splitext(source_name)[1]
                    ) as temp_file:

                        temp_file.write(file_bytes)
                        temp_file_path = temp_file.name

                    try:

                        chunk_count = process_document(
                            temp_file_path,
                            source_name=source_name
                        )

                        total_chunks += chunk_count
                        processed_sources.append(
                            source_name
                        )

                    finally:

                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)

            st.session_state.processed_file_hash = file_hash

            st.session_state.current_source = (
                processed_sources
            )

            st.session_state.chunk_count = total_chunks

            st.success(
                f"✅ {len(processed_sources)} project documents "
                f"processed successfully."
            )

            for source in processed_sources:
                st.caption(f"📄 {source}")

        except Exception as e:

            st.error(
                f"❌ Document processing failed: {e}"
            )

    else:

        st.info(
            "✅ These project documents are already loaded "
            "into the project workspace."
        )


# ============================================================
# SEMANTIC RETRIEVAL EXECUTION
# ============================================================

if search_button:

    if not sidebar_query.strip():

        st.warning(
            "⚠️ Please enter a project question "
            "in the sidebar."
        )

    elif not st.session_state.current_source:

        st.warning(
            "⚠️ Please wait for the document "
            "to finish processing."
        )

    else:

        with st.spinner(
            "🧠 Searching project knowledge..."
        ):

            rag = get_rag()

            search_results = rag.search(
                sidebar_query,
                top_k=3,
                source=None
            )

        st.session_state.search_count += 1

        st.session_state.search_results = {
            "query": sidebar_query,
            "results": search_results
        }


# ============================================================
# SIDEBAR SEARCH RESULTS
# ============================================================

if st.session_state.search_results:

    with st.sidebar:

        st.divider()

        st.subheader("🔎 Search Results")

        search_data = st.session_state.search_results

        st.caption(
            f"Question: {search_data['query']}"
        )

        results = search_data["results"]

        if results:

            for i, document in enumerate(
                results,
                start=1
            ):

                with st.container(border=True):

                    st.markdown(
                        f"**Result {i}**"
                    )

                    st.write(
                        document.page_content
                    )

                    source = document.metadata.get(
                        "source",
                        "Unknown source"
                    )

                    st.caption(
                        f"📄 Source: {source}"
                    )

        else:

            st.info(
                "No relevant information found."
            )


# ============================================================
# PROJECT OVERVIEW
# ============================================================

if st.session_state.current_source:

    st.divider()

    st.header("📊 Project Overview")

    st.caption(
        "Document intelligence and AI analysis status"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📄 Document Chunks",
            st.session_state.chunk_count
        )

    with col2:

        st.metric(
            "🧠 Embedding Dimensions",
            384
        )

    with col3:

        st.metric(
            "🔍 Queries Analyzed",
            st.session_state.search_count
        )

    with col4:

        if st.session_state.analysis_results:

            st.metric(
                "🟢 AI Status",
                "Analyzed"
            )

        else:

            st.metric(
                "🟢 AI Status",
                "Ready"
            )


# ============================================================
# AI AGENTS OVERVIEW
# ============================================================

if st.session_state.current_source:

    st.divider()

    st.header("🧠 Project Intelligence Agents")

    st.caption(
        "Three specialized AI agents analyze different "
        "dimensions of the project using retrieved "
        "document evidence."
    )

    agent1, agent2, agent3 = st.columns(3)

    with agent1:

        with st.container(border=True):

            st.subheader(
                "🧠 Scope & Deliverables"
            )

            st.write(
                "Extracts project goals, deliverables, "
                "milestones, timelines and responsibilities."
            )

    with agent2:

        with st.container(border=True):

            st.subheader(
                "⚠️ Risk & Delivery Forecast"
            )

            st.write(
                "Detects evidence-based risks, dependencies "
                "and potential delivery challenges."
            )

    with agent3:

        with st.container(border=True):

            st.subheader(
                "🚧 Blockers & Actions"
            )

            st.write(
                "Identifies blockers, pending decisions, "
                "action items and responsible persons."
            )


# ============================================================
# AI PROJECT ANALYSIS
# ============================================================

if st.session_state.current_source:

    st.divider()

    st.header("🚀 AI Project Analysis")

    st.caption(
        "Run the specialized AI agents to generate "
        "evidence-grounded project intelligence."
    )

    if st.button(
        "🚀 Run Project Analysis",
        use_container_width=True
    ):

        source = st.session_state.current_source

        scope_agent, risk_agent, blocker_agent = get_agents()

        # ----------------------------------------------------
        # SCOPE AGENT
        # ----------------------------------------------------

        with st.spinner(
            "🧠 Extracting project scope and deliverables..."
        ):

            scope_results = scope_agent.extract_scope(
                source
            )

        # ----------------------------------------------------
        # RISK AGENT
        # ----------------------------------------------------

        with st.spinner(
            "⚠️ Detecting risks and forecasting delivery..."
        ):

            risk_results = risk_agent.analyze_risks(
                source
            )

        # ----------------------------------------------------
        # BLOCKER AGENT
        # ----------------------------------------------------

        with st.spinner(
            "🚧 Identifying blockers and action items..."
        ):

            blocker_results = blocker_agent.identify_blockers(
                source
            )

        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        st.session_state.analysis_results = {
            "scope": scope_results,
            "risk": risk_results,
            "blocker": blocker_results
        }

        st.success(
            "✅ Project analysis completed successfully!"
        )


# ============================================================
# AI ANALYSIS RESULTS
# ============================================================

if st.session_state.analysis_results:

    results = st.session_state.analysis_results

    st.divider()

    st.header("📊 AI Analysis Results")

    st.caption(
        "Evidence-grounded intelligence generated from "
        "the uploaded project document."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "🧠 Project Scope",
            "⚠️ Risks & Delivery",
            "🚧 Blockers & Actions"
        ]
    )

    # ========================================================
    # SCOPE
    # ========================================================

    with tab1:

        with st.container(border=True):

            st.subheader(
                "🎯 Scope & Deliverables"
            )

            st.markdown(
                results["scope"]["analysis"]
            )

    # ========================================================
    # RISK
    # ========================================================

    with tab2:

        with st.container(border=True):

            st.subheader(
                "📈 Risk & Delivery Forecast"
            )

            st.markdown(
                results["risk"]["analysis"]
            )

        

    # ========================================================
    # BLOCKERS
    # ========================================================

    with tab3:

        with st.container(border=True):

            st.subheader(
                "📋 Blocker & Action Analysis"
            )

            st.markdown(
                results["blocker"]["analysis"]
            )

        


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🔐 Enterprise Project Intelligence • "
    "Powered by LangChain • Ollama • Qwen 2.5 • "
    "ChromaDB • Streamlit"
)