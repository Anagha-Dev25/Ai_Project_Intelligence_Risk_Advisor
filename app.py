import hashlib
import os
import tempfile
from typing import List, Dict, Any

import streamlit as st
from langchain_ollama import ChatOllama

from app.config import (
    DEFAULT_LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_KEEP_ALIVE,
    SUPPORTED_EXTENSIONS,
    check_ollama_status,
)
from app.pipeline import process_document
from app.rag.langchain_rag import LangChainRAG
from app.agents.scope_agent import ScopeAgent
from app.agents.risk_agent import RiskAgent
from app.agents.blocker_agent import BlockerAgent
from app.health.health_scorer import ProjectHealthScorer
from app.agents.documentation_agent import DocumentationAgent
from app.export import (
    export_risk_register_csv,
    export_action_items_csv,
    export_user_stories_csv,
    generate_executive_report_markdown,
)


# ============================================================
# CACHED COMPONENTS
# ============================================================

@st.cache_resource
def get_agents():
    return ScopeAgent(), RiskAgent(), BlockerAgent()


@st.cache_resource
def get_rag():
    return LangChainRAG()


@st.cache_resource
def get_health_scorer():
    return ProjectHealthScorer()


@st.cache_resource
def get_documentation_agent():
    return DocumentationAgent()


@st.cache_resource
def get_assistant_llm():
    return ChatOllama(
        model=DEFAULT_LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        num_predict=800,
        keep_alive=LLM_KEEP_ALIVE
    )


# ============================================================
# PAGE CONFIGURATION & ENTERPRISE STYLING
# ============================================================

st.set_page_config(
    page_title="ProjectIQ • Enterprise Project Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS
st.markdown(
    """
    <style>
    /* Clean modern enterprise styling */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-healthy { background-color: #dcfce7; color: #166534; }
    .badge-stable { background-color: #fef9c3; color: #854d0e; }
    .badge-risk { background-color: #ffedd5; color: #9a3412; }
    .badge-critical { background-color: #fee2e2; color: #991b1b; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

defaults = {
    "processed_file_hash": None,
    "current_source": None,
    "chunk_count": 0,
    "search_count": 0,
    "analysis_results": None,
    "health_results": None,
    "documentation_results": None,
    "chat_history": [],
    "document_stats": {}
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER — GET ACTIVE SOURCES
# ============================================================

def get_active_sources() -> List[str]:
    source = st.session_state.current_source
    if source is None:
        return []
    if isinstance(source, list):
        return source
    return [source]


# ============================================================
# HELPER — CONVERSATIONAL ASSISTANT
# ============================================================

def answer_project_question(question: str) -> Dict[str, Any]:
    rag = get_rag()
    llm = get_assistant_llm()
    sources = get_active_sources()

    if not sources:
        return {
            "answer": "Please upload a project document first.",
            "evidence": []
        }

    # --------------------------------------------------------
    # Determine whether this is a comprehensive list question
    # --------------------------------------------------------
    list_question_keywords = [
        "all", "list", "main deliverables", "deliverables",
        "objectives", "milestones", "responsibilities",
        "roles", "components", "features", "tasks"
    ]

    is_list_question = any(keyword in question.lower() for keyword in list_question_keywords)

    # --------------------------------------------------------
    # Retrieve evidence
    # --------------------------------------------------------
    if is_list_question:
        retrieval_query = f"""
Find the complete project section relevant to this question.

USER QUESTION:
{question}

Retrieve evidence containing the full list requested by the user.
The requested list may continue across multiple document chunks. Include evidence from all chunks.
"""
        retrieval_k = 8
    else:
        retrieval_query = question
        retrieval_k = 5

    retrieved_documents = rag.search(
        retrieval_query,
        top_k=retrieval_k,
        source=sources
    )

    if not retrieved_documents:
        return {
            "answer": "I could not find relevant evidence in the uploaded project documents.",
            "evidence": []
        }

    # --------------------------------------------------------
    # Remove duplicate chunks
    # --------------------------------------------------------
    unique_documents = []
    seen = set()

    for document in retrieved_documents:
        content = document.page_content.strip()
        if content and content not in seen:
            seen.add(content)
            unique_documents.append(document)

    # --------------------------------------------------------
    # Build evidence text
    # --------------------------------------------------------
    evidence_text = "\n\n".join(
        f"EVIDENCE {index + 1}:\n"
        f"Source: {document.metadata.get('source', 'Unknown')}\n"
        f"{document.page_content}"
        for index, document in enumerate(unique_documents)
    )

    # --------------------------------------------------------
    # Health context
    # --------------------------------------------------------
    health_context = ""
    health_keywords = [
        "health", "healthy", "status", "on track", "track",
        "project status", "project doing", "overall"
    ]

    if any(keyword in question.lower() for keyword in health_keywords):
        try:
            health_scorer = get_health_scorer()
            health = health_scorer.calculate_health(
                sources[0] if len(sources) == 1 else sources
            )

            health_context = f"""
SYSTEM-CALCULATED PROJECT HEALTH:
Overall Score: {health['overall_score']}/100
Status: {health['status']}
Scope Clarity: {health['dimensions']['scope_clarity']}/100
Timeline Health: {health['dimensions']['timeline_health']}/100
Risk Status: {health['dimensions']['risk_status']}/100
Blocker Status: {health['dimensions']['blocker_status']}/100
Identified Risks: {health['identified_risks']}
Identified Blockers: {health['identified_blockers']}

This health score is calculated by the ProjectIQ system from retrieved project evidence.
"""
        except Exception:
            health_context = ""

    # --------------------------------------------------------
    # Conversation context
    # --------------------------------------------------------
    recent_history = st.session_state.chat_history[-6:]
    conversation_context = "".join(
        f"{msg['role'].upper()}: {msg['content']}\n" for msg in recent_history
    )

    # --------------------------------------------------------
    # Grounded assistant prompt
    # --------------------------------------------------------
    prompt = f"""
You are ProjectIQ, an enterprise project intelligence assistant.
Your job is to answer project questions using ONLY:
1. Evidence retrieved from uploaded project documents.
2. The system-calculated project health information, when provided.

============================================================
STRICT FACT-GROUNDING RULES
============================================================
1. NEVER invent project facts.
2. NEVER invent people, dates, risks, blockers, dependencies, approvals, budgets, resources, responsibilities, or activities.
3. Every factual statement must be directly supported by the retrieved evidence.
4. If the evidence does not contain the requested fact, state that the information is not available.
5. Never use outside knowledge.

============================================================
PERSON-TO-RESPONSIBILITY RULE
============================================================
When the question asks WHO is responsible for something:
- Find the exact person-to-responsibility mapping in the evidence.
- Preserve the mapping EXACTLY as written.
- NEVER transfer a responsibility from one person to another person.
- If the requested mapping is present in the evidence, state it directly.

============================================================
COMPLETE LIST RULE
============================================================
When the user asks for a list, deliverables, objectives, milestones, responsibilities, roles, components, or features:
- Provide the COMPLETE list supported by the evidence.
- Combine information from multiple evidence chunks when relevant.
- Do NOT omit later items simply because they appear in another chunk.

============================================================
PROJECT HEALTH & STATUS
============================================================
For "Are we on track?" or project health questions:
- Use the system-calculated health context when available.
- Clearly distinguish documented project facts from calculated health scores.

============================================================
CONVERSATION HISTORY
============================================================
{conversation_context}

============================================================
SYSTEM HEALTH CONTEXT
============================================================
{health_context}

============================================================
RETRIEVED PROJECT EVIDENCE
============================================================
{evidence_text}

============================================================
USER QUESTION
============================================================
{question}

============================================================
FINAL INSTRUCTION
============================================================
Answer directly, concisely, and completely using ONLY the evidence above.
"""

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        answer = f"Error generating response from local LLM ({DEFAULT_LLM_MODEL}): {e}. Please ensure Ollama is running."

    evidence = [
        {
            "source": doc.metadata.get("source", "Unknown source"),
            "content": doc.page_content
        }
        for doc in unique_documents
    ]

    return {
        "answer": answer,
        "evidence": evidence
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Project Workspace")
    st.caption("Upload project documents to activate ProjectIQ intelligence.")

    # Ingestion uploader
    uploader_types = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]
    uploaded_files = st.file_uploader(
        "📄 Upload Project Documents",
        type=uploader_types,
        accept_multiple_files=True
    )

    st.divider()

    # Active Project Status
    if st.session_state.current_source:
        st.markdown("#### 📌 Active Knowledge Base")
        for source in get_active_sources():
            chunk_info = st.session_state.document_stats.get(source, {})
            c_cnt = chunk_info.get("chunks", "—")
            st.caption(f"📄 **{source}** ({c_cnt} chunks)")

        st.success("🟢 Knowledge Base Loaded")

        # Clear project button
        if st.button("🔄 Reset / Clear Workspace", use_container_width=True):
            rag = get_rag()
            rag.clear_all()
            st.session_state.processed_file_hash = None
            st.session_state.processed_files_map = {}
            st.session_state.current_source = None
            st.session_state.chunk_count = 0
            st.session_state.analysis_results = None
            st.session_state.health_results = None
            st.session_state.documentation_results = None
            st.session_state.chat_history = []
            st.session_state.document_stats = {}
            st.rerun()
    else:
        st.info("Upload project documents (PDF, DOCX, TXT, CSV, MD, JSON) to begin.")

    st.divider()

    # Ollama Diagnostics in Sidebar
    st.markdown("#### 🧠 AI Engine Diagnostic")
    is_ok, msg, details = check_ollama_status()
    if is_ok:
        st.caption(f"🟢 **LLM Status:** {msg}")
    else:
        st.caption(f"⚠️ **LLM Status:** {msg}")
    st.caption(f"Model: `{DEFAULT_LLM_MODEL}` • Embeddings: `all-MiniLM-L6-v2`")


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🤖 ProjectIQ")
st.caption(
    "AI-Driven Enterprise Project Intelligence & Risk Management Platform"
)

col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.info("💡 **Local Enterprise AI Stack**: Qwen 2.5 • ChromaDB • LangChain RAG • Multi-Agent Engine")


# ============================================================
# INCREMENTAL DOCUMENT PROCESSING PIPELINE
# ============================================================

if "processed_files_map" not in st.session_state:
    st.session_state.processed_files_map = {}

if uploaded_files:
    current_files_map = {}
    for uploaded_file in uploaded_files:
        hasher = hashlib.md5()
        hasher.update(uploaded_file.getvalue())
        current_files_map[uploaded_file.name] = (hasher.hexdigest(), uploaded_file)

    # Detect new or updated files
    new_or_updated = [
        name for name, (f_hash, _) in current_files_map.items()
        if name not in st.session_state.processed_files_map or st.session_state.processed_files_map[name] != f_hash
    ]

    if new_or_updated:
        is_first_load = len(st.session_state.processed_files_map) == 0
        newly_added_chunks = 0
        added_names = []

        try:
            with st.spinner(f"🔄 {'Vectorizing initial project documents' if is_first_load else 'Incrementally updating knowledge base with new documents'}..."):
                for name in new_or_updated:
                    f_hash, uploaded_file = current_files_map[name]
                    file_bytes = uploaded_file.getvalue()

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=os.path.splitext(name)[1]
                    ) as temp_file:
                        temp_file.write(file_bytes)
                        temp_file_path = temp_file.name

                    try:
                        chunk_count = process_document(
                            temp_file_path,
                            source_name=name
                        )
                        newly_added_chunks += chunk_count
                        added_names.append(name)
                        st.session_state.document_stats[name] = {
                            "chunks": chunk_count,
                            "bytes": len(file_bytes)
                        }
                        st.session_state.processed_files_map[name] = f_hash
                    finally:
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)

            # Update active sources
            active_list = list(st.session_state.processed_files_map.keys())
            st.session_state.current_source = active_list
            st.session_state.chunk_count = sum(
                info.get("chunks", 0) for info in st.session_state.document_stats.values()
            )

            # Invalidate cached analysis to prompt re-evaluation
            st.session_state.analysis_results = None
            st.session_state.health_results = None
            st.session_state.documentation_results = None

            if is_first_load:
                st.session_state.chat_history = []
                st.success(
                    f"✅ Successfully processed {len(added_names)} document(s) into {newly_added_chunks} vectorized chunks."
                )
            else:
                # Milestone 4.2: Incremental update notification while preserving chat history
                added_str = ", ".join(added_names)
                st.toast(f"📥 Knowledge base updated with {added_str} (+{newly_added_chunks} chunks)!", icon="🚀")
                st.info(
                    f"📥 **Incremental Update:** Added `{added_str}` (+{newly_added_chunks} chunks). "
                    "The knowledge base has been automatically updated. You can re-run analysis or ask questions about this new information."
                )
        except Exception as e:
            st.error(f"Document processing failed: {e}")


# ============================================================
# PROJECT OVERVIEW METRICS
# ============================================================

if st.session_state.current_source:
    st.divider()
    st.subheader("📊 Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📄 Total Chunks", st.session_state.chunk_count)

    with col2:
        st.metric("🧠 Embedding Dims", 384)

    with col3:
        st.metric("💬 Inquiries Handled", len(st.session_state.chat_history) // 2)

    with col4:
        if st.session_state.health_results:
            status_val = st.session_state.health_results["status"]
            score_val = st.session_state.health_results["overall_score"]
            st.metric("🟢 Project Health", f"{status_val} ({score_val}/100)")
        else:
            st.metric("🤖 AI Engine", "Ready")


# ============================================================
# PROJECT HEALTH
# ============================================================

if st.session_state.current_source:

    st.divider()

    st.header("❤️ Project Health")

    if st.button(
        "📈 Calculate Project Health",
        use_container_width=True
    ):

        with st.spinner(
            "🧠 Evaluating project health..."
        ):

            scorer = get_health_scorer()
            sources = get_active_sources()

            health_results = scorer.calculate_health(
                sources[0] if len(sources) == 1 else sources
            )

            st.session_state.health_results = health_results

    if st.session_state.health_results:

        health = st.session_state.health_results

        # ----------------------------------------------------
        # Main Health Metrics
        # ----------------------------------------------------

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Overall",
                f"{health['overall_score']}/100"
            )

        with col2:
            st.metric(
                "Status",
                health["status"]
            )

        with col3:
            st.metric(
                "Scope",
                f"{health['dimensions']['scope_clarity']}/100"
            )

        with col4:
            st.metric(
                "Timeline",
                f"{health['dimensions']['timeline_health']}/100"
            )

        with col5:
            st.metric(
                "Risks",
                len(health["identified_risks"])
            )

        # ----------------------------------------------------
        # Detailed Health Dimensions
        # ----------------------------------------------------

        st.markdown("### 📊 Health Dimensions")

        dim1, dim2, dim3, dim4 = st.columns(4)

        with dim1:
            st.metric(
                "🎯 Scope Clarity",
                f"{health['dimensions']['scope_clarity']}/100"
            )

        with dim2:
            st.metric(
                "📅 Timeline Health",
                f"{health['dimensions']['timeline_health']}/100"
            )

        with dim3:
            st.metric(
                "⚠️ Risk Status",
                f"{health['dimensions']['risk_status']}/100"
            )

        with dim4:
            st.metric(
                "🚧 Blocker Status",
                f"{health['dimensions']['blocker_status']}/100"
            )

        # ----------------------------------------------------
        # Active Risks & Blockers
        # ----------------------------------------------------

        risk_col, blocker_col = st.columns(2)

        with risk_col:

            st.markdown("### ⚠️ Active Risks")

            risks = health.get("identified_risks", [])

            if risks:

                for i, risk in enumerate(risks, start=1):
                    st.warning(
                        f"**Risk {i}:** {risk}"
                    )

            else:

                st.success(
                    "No evidence-based risks identified."
                )

        with blocker_col:

            st.markdown("### 🚧 Active Blockers")

            blockers = health.get(
                "identified_blockers",
                []
            )

            if blockers:

                for i, blocker in enumerate(
                    blockers,
                    start=1
                ):
                    st.error(
                        f"**Blocker {i}:** {blocker}"
                    )

            else:

                st.success(
                    "No evidence-based blockers identified."
                )

        st.caption(
            "Health is calculated from evidence retrieved "
            "from the uploaded project documents."
        )


# ============================================================
# CONVERSATIONAL PROJECT INTELLIGENCE ASSISTANT
# ============================================================

if st.session_state.current_source:
    st.divider()
    st.subheader("💬 Project Intelligence Assistant")
    st.caption("Ask questions about your project. ProjectIQ answers strictly using verified project evidence.")

    # Quick prompt suggestions
    st.markdown("**Suggested Inquiries:**")
    quick_cols = st.columns(4)
    suggested_q = None

    if quick_cols[0].button("🎯 Key Deliverables", use_container_width=True):
        suggested_q = "What are the main deliverables and objectives of this project?"
    if quick_cols[1].button("👥 Roles & Owners", use_container_width=True):
        suggested_q = "Who is responsible for what in the project?"
    if quick_cols[2].button("⚠️ Risks & Blockers", use_container_width=True):
        suggested_q = "What are the current blockers and risks?"
    if quick_cols[3].button("📅 Timeline & Dates", use_container_width=True):
        suggested_q = "What are the key project milestones and timeline deadlines?"

    # Render previous chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("evidence"):
                with st.expander("📚 Evidence used"):
                    for evidence in message["evidence"]:
                        st.markdown(f"**📄 {evidence['source']}**")
                        st.caption(evidence["content"])

    # Chat input
    chat_prompt = st.chat_input("Ask ProjectIQ about your project...")
    active_question = suggested_q or chat_prompt

    if active_question:
        st.session_state.chat_history.append({"role": "user", "content": active_question})

        with st.chat_message("user"):
            st.markdown(active_question)

        with st.chat_message("assistant"):
            with st.spinner("🧠 Analyzing project evidence..."):
                result = answer_project_question(active_question)

            st.markdown(result["answer"])

            if result["evidence"]:
                with st.expander("📚 Evidence used"):
                    for evidence in result["evidence"]:
                        st.markdown(f"**📄 {evidence['source']}**")
                        st.caption(evidence["content"])

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "evidence": result["evidence"]
        })


# ============================================================
# AI PROJECT ANALYSIS
# ============================================================

if st.session_state.current_source:
    st.divider()
    st.subheader("🚀 AI Multi-Agent Project Analysis")
    st.caption("Run specialized agents (Scope, Risk, Blocker) for deep, structured project intelligence.")

    if st.button("🚀 Run Comprehensive Project Analysis", use_container_width=True):
        source = st.session_state.current_source
        scope_agent, risk_agent, blocker_agent = get_agents()

        with st.spinner("🎯 Scope Agent: Extracting project scope & deliverables..."):
            scope_results = scope_agent.extract_scope(source)

        with st.spinner("⚠️ Risk Agent: Detecting delivery risks & forecasts..."):
            risk_results = risk_agent.analyze_risks(source)

        with st.spinner("🚧 Blocker Agent: Identifying active blockers & action items..."):
            blocker_results = blocker_agent.identify_blockers(source)

        st.session_state.analysis_results = {
            "scope": scope_results,
            "risk": risk_results,
            "blocker": blocker_results
        }
        st.success("✅ Multi-agent project analysis completed successfully.")

    if st.session_state.analysis_results:
        results = st.session_state.analysis_results

        tab1, tab2, tab3 = st.tabs([
            "🎯 Project Scope",
            "📈 Risks & Delivery",
            "📋 Blockers & Actions"
        ])

        with tab1:
            st.markdown(results["scope"]["analysis"])

        with tab2:
            st.markdown(results["risk"]["analysis"])

        with tab3:
            st.markdown(results["blocker"]["analysis"])


# ============================================================
# DOCUMENTATION GENERATION & EXPORT
# ============================================================

if st.session_state.current_source:
    st.divider()
    st.subheader("📑 Documentation Intelligence & Agile Export")
    st.caption("Generate structured user stories, risk registers, and traceable action items with one-click export.")

    if st.button("📑 Generate Agile Documentation", use_container_width=True):
        with st.spinner("📝 Generating structured agile documentation..."):
            documentation_agent = get_documentation_agent()
            documentation_results = documentation_agent.generate_documentation(
                st.session_state.current_source
            )
            st.session_state.documentation_results = documentation_results

    if st.session_state.documentation_results:
        doc_res = st.session_state.documentation_results

        tab_us, tab_rr, tab_act, tab_rep = st.tabs([
            "👤 User Stories",
            "⚠️ Risk Register",
            "📋 Action Items",
            "📊 Executive Report Export"
        ])

        with tab_us:
            st.markdown(doc_res.get("user_stories", "No user stories generated."))
            us_csv = export_user_stories_csv(doc_res)
            st.download_button(
                "⬇️ Download User Stories (CSV)",
                data=us_csv,
                file_name="user_stories.csv",
                mime="text/csv"
            )

        with tab_rr:
            st.markdown(doc_res.get("risk_register", "No risk register generated."))
            rr_csv = export_risk_register_csv(doc_res)
            st.download_button(
                "⬇️ Download Risk Register (CSV)",
                data=rr_csv,
                file_name="risk_register.csv",
                mime="text/csv"
            )

        with tab_act:
            st.markdown(doc_res.get("action_items", "No action items generated."))
            act_csv = export_action_items_csv(doc_res)
            st.download_button(
                "⬇️ Download Action Items (CSV)",
                data=act_csv,
                file_name="action_items.csv",
                mime="text/csv"
            )

        with tab_rep:
            st.markdown("#### Full Executive Project Intelligence Report")
            full_report = generate_executive_report_markdown(
                get_active_sources(),
                st.session_state.health_results,
                st.session_state.analysis_results,
                st.session_state.documentation_results
            )
            st.download_button(
                "⬇️ Download Full Executive Report (Markdown)",
                data=full_report,
                file_name="executive_project_report.md",
                mime="text/markdown",
                use_container_width=True
            )
            with st.expander("👁️ Preview Full Report"):
                st.markdown(full_report)


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "ProjectIQ • AI-Driven Enterprise Project Intelligence & Risk Management Platform • Fully Local & Private"
)
