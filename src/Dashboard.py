"""
dashboard.py — AI Resume Screening Dashboard
Place inside: src/dashboard.py
Run from project root: streamlit run src/dashboard.py
"""

import os
import sys
import json
import glob
import shutil
import tempfile
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from dataclasses import asdict

# =========================================================
# PATH SETUP
# =========================================================
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "outputs")
TEMP_DIR    = os.path.join(PROJECT_DIR, "temp_uploads")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR,   exist_ok=True)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="RecruitIQ — AI Resume Screener",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:      #0a0a0f;
    --surface: #111118;
    --surface2:#18181f;
    --border:  #2a2a35;
    --accent:  #e8b84b;
    --accent2: #4be8b8;
    --text:    #e8e8f0;
    --muted:   #6b6b80;
    --hire:    #4be8b8;
    --maybe:   #e8b84b;
    --reject:  #e84b6a;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%,
            rgba(232,184,75,0.07) 0%, transparent 60%),
        repeating-linear-gradient(0deg, transparent, transparent 40px,
            rgba(255,255,255,0.01) 40px, rgba(255,255,255,0.01) 41px),
        repeating-linear-gradient(90deg, transparent, transparent 40px,
            rgba(255,255,255,0.01) 40px, rgba(255,255,255,0.01) 41px);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
h1,h2,h3 { font-family: 'Syne', sans-serif; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
.metric-card:hover { border-color: var(--accent); }
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted);
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800;
    color: var(--text); line-height: 1;
}
.metric-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem; color: var(--muted); margin-top: 0.3rem;
}
.upload-zone {
    background: var(--surface);
    border: 1px dashed var(--border);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: var(--accent); }
.upload-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted);
    margin-bottom: 0.6rem;
}
.upload-success {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem; color: var(--hire);
    margin-top: 0.4rem;
}
.upload-info {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem; color: var(--muted);
    line-height: 1.8; margin-top: 0.3rem;
}
.score-bar-track {
    background: var(--border); border-radius: 2px;
    height: 3px; margin: 0.3rem 0 0.15rem; overflow: hidden;
}
.score-bar-fill {
    height: 100%; border-radius: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.skill-chip {
    display: inline-block;
    background: rgba(232,184,75,0.1);
    border: 1px solid rgba(232,184,75,0.25);
    color: var(--accent);
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem; padding: 0.18rem 0.5rem;
    border-radius: 3px; margin: 0.12rem;
}
.skill-chip.missing {
    background: rgba(232,75,106,0.08);
    border-color: rgba(232,75,106,0.2);
    color: var(--reject);
}
.decision-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.08em;
    padding: 0.22rem 0.65rem; border-radius: 3px;
}
.decision-hire   { background: rgba(75,232,184,0.12); color: var(--hire);   border: 1px solid rgba(75,232,184,0.25); }
.decision-maybe  { background: rgba(232,184,75,0.12); color: var(--maybe);  border: 1px solid rgba(232,184,75,0.25); }
.decision-reject { background: rgba(232,75,106,0.12); color: var(--reject); border: 1px solid rgba(232,75,106,0.25); }
.review-flag {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem; color: var(--maybe);
}
.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem; margin-bottom: 0.9rem;
}
.domain-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem; letter-spacing: 0.1em;
    text-transform: uppercase;
    background: rgba(232,184,75,0.08);
    border: 1px solid var(--accent);
    color: var(--accent);
    padding: 0.18rem 0.7rem; border-radius: 2px;
}
.stTextArea textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    border-radius: 6px !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.55rem 1.8rem !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #f0c75a !important;
    transform: translateY(-1px) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important; color: var(--muted) !important;
    background: transparent !important; border: none !important;
    padding: 0.6rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
hr { border-color: var(--border) !important; }
/* Style Streamlit file uploader */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--muted) !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SAFE PIPELINE IMPORT
# =========================================================
PIPELINE_AVAILABLE = False
pipeline_error     = None

try:
    from ranker        import process_single_resume, collect_pdf_paths
    from domain_config import detect_domain
    PIPELINE_AVAILABLE = True
except ImportError as e:
    pipeline_error = str(e)


# =========================================================
# HELPERS
# =========================================================
def decision_class(decision: str) -> str:
    d = decision.lower()
    if "strong" in d or ("hire" in d and "maybe" not in d and "manual" not in d):
        return "decision-hire"
    elif "maybe" in d or "consider" in d or "review" in d:
        return "decision-maybe"
    return "decision-reject"

def normalize_result(r: dict) -> dict:
    for key in ["final_score","semantic_score","skill_score",
                "total_exp_score","role_exp_score","parse_confidence"]:
        val = r.get(key, r.get(f"{key}_pct", 0))
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.0
        r[key] = val / 100.0 if val > 1.0 else val
    for key in ["matched_skills","missing_skills","semantic_skills"]:
        if not isinstance(r.get(key), list):
            r[key] = []
    for key in ["name","email","phone","domain","decision",
                "parse_method","linkedin","github","file_path"]:
        if not r.get(key):
            r[key] = "N/A"
    for key in ["total_exp_years","role_exp_months","required_exp_years","rank"]:
        try:
            r[key] = float(r.get(key, 0))
        except (TypeError, ValueError):
            r[key] = 0.0
    r["needs_review"] = bool(r.get("needs_review", False))
    return r

def get_output_files() -> list:
    files = glob.glob(os.path.join(OUTPUT_DIR, "results_*.json"))
    return sorted(files, reverse=True)

def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [normalize_result(r) for r in raw]

def to_df(results: list) -> pd.DataFrame:
    return pd.DataFrame([{
        "Rank":         int(r.get("rank", 0)),
        "Name":         r.get("name", ""),
        "Score":        r.get("final_score", 0),
        "Semantic":     r.get("semantic_score", 0),
        "Skills":       r.get("skill_score", 0),
        "Exp":          r.get("total_exp_score", 0),
        "Role Exp":     r.get("role_exp_score", 0),
        "Exp Years":    r.get("total_exp_years", 0),
        "Domain":       r.get("domain", ""),
        "Decision":     r.get("decision", ""),
        "Parse Method": r.get("parse_method", ""),
        "Needs Review": r.get("needs_review", False),
    } for r in results])

def save_uploaded_resumes(uploaded_files: list) -> str:
    """
    Saves Streamlit uploaded file objects to a temp folder on disk.
    Returns the folder path for the pipeline to process.
    """
    # Fresh temp folder each run — avoid mixing old and new resumes
    session_folder = os.path.join(
        TEMP_DIR,
        f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(session_folder, exist_ok=True)

    saved = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.lower().endswith(".pdf"):
            dest = os.path.join(session_folder, uploaded_file.name)
            with open(dest, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved.append(dest)

    return session_folder, saved

def extract_jd_from_upload(uploaded_file) -> str:
    """
    Reads JD content from uploaded .txt or .pdf file.
    For PDF JDs, uses pdfplumber to extract text.
    """
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        try:
            import pdfplumber, io
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Could not read JD PDF: {e}")
            return ""

    return ""


# =========================================================
# CHARTS
# =========================================================
BG   = "rgba(0,0,0,0)"
FONT = dict(family="DM Mono", color="#6b6b80", size=10)
GRID = "#2a2a35"

def radar(r: dict) -> go.Figure:
    cats = ["Semantic","Skills","Experience","Role Exp"]
    vals = [r.get(k,0)*100 for k in
            ["semantic_score","skill_score","total_exp_score","role_exp_score"]]
    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]],
        fill="toself", fillcolor="rgba(232,184,75,0.1)",
        line=dict(color="#e8b84b", width=2),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(17,17,24,0.8)",
            radialaxis=dict(visible=True, range=[0,100],
                tickfont=dict(size=8,color="#6b6b80",family="DM Mono"),
                gridcolor=GRID, linecolor=GRID),
            angularaxis=dict(
                tickfont=dict(size=9,color="#9b9baf",family="DM Mono"),
                gridcolor=GRID, linecolor=GRID),
        ),
        paper_bgcolor=BG, showlegend=False,
        margin=dict(l=30,r=30,t=20,b=20), height=250,
    )
    return fig

def breakdown_bar(r: dict) -> go.Figure:
    vals = [r.get(k,0)*100 for k in
            ["semantic_score","skill_score","total_exp_score","role_exp_score"]]
    fig = go.Figure(go.Bar(
        x=["Semantic","Skills","Total Exp","Role Exp"], y=vals,
        marker_color=["#e8b84b","#4be8b8","#b84be8","#4b8be8"],
        marker_line=dict(width=0),
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(family="DM Mono", size=9, color="#9b9baf"),
    ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor="rgba(17,17,24,0.5)", font=FONT,
        yaxis=dict(range=[0,115], gridcolor=GRID, color="#6b6b80", ticksuffix="%"),
        xaxis=dict(color="#6b6b80"),
        margin=dict(l=5,r=5,t=20,b=5), height=190, showlegend=False,
    )
    return fig

def distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=df["Score"]*100, nbinsx=15,
        marker=dict(color="#e8b84b", opacity=0.85,
                    line=dict(color="#0a0a0f", width=1)),
    ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor="rgba(17,17,24,0.5)", font=FONT,
        xaxis=dict(
            title=dict(text="MATCH SCORE (%)",
                       font=dict(size=9, family="DM Mono", color="#6b6b80")),
            gridcolor=GRID, color="#6b6b80",
        ),
        yaxis=dict(
            title=dict(text="COUNT",
                       font=dict(size=9, family="DM Mono", color="#6b6b80")),
            gridcolor=GRID, color="#6b6b80",
        ),
        bargap=0.08, margin=dict(l=10,r=10,t=10,b=10), height=220,
    )
    return fig

def donut(results: list) -> go.Figure:
    counts = {"Hire":0,"Maybe":0,"Reject":0}
    for r in results:
        d = r.get("decision","").lower()
        if "strong" in d or ("hire" in d and "maybe" not in d and "manual" not in d):
            counts["Hire"] += 1
        elif "maybe" in d or "consider" in d or "review" in d:
            counts["Maybe"] += 1
        else:
            counts["Reject"] += 1
    fig = go.Figure(go.Pie(
        labels=list(counts.keys()), values=list(counts.values()),
        hole=0.72,
        marker=dict(colors=["#4be8b8","#e8b84b","#e84b6a"],
                    line=dict(color="#0a0a0f",width=2)),
        textfont=dict(family="DM Mono",size=9), textinfo="label+percent",
    ))
    fig.update_layout(
        paper_bgcolor=BG, showlegend=False,
        margin=dict(l=10,r=10,t=10,b=10), height=220,
        annotations=[dict(
            text=f"<b>{len(results)}</b><br><span style='font-size:9px'>total</span>",
            font=dict(family="Syne",size=18,color="#e8e8f0"), showarrow=False,
        )],
    )
    return fig

def skills_chart(results: list) -> go.Figure | None:
    from collections import Counter
    counts = Counter(
        s for r in results for s in r.get("matched_skills", [])
    ).most_common(12)
    if not counts:
        return None
    skills, freqs = zip(*counts)
    fig = go.Figure(go.Bar(
        x=list(freqs), y=list(skills), orientation="h",
        marker=dict(color=list(freqs),
                    colorscale=[[0,"#2a2a35"],[1,"#e8b84b"]],
                    line=dict(width=0)),
        text=list(freqs), textposition="outside",
        textfont=dict(family="DM Mono",size=9,color="#6b6b80"),
    ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor="rgba(17,17,24,0.5)", font=FONT,
        xaxis=dict(gridcolor=GRID, color="#6b6b80"),
        yaxis=dict(color="#9b9baf",
                   tickfont=dict(family="DM Mono",size=9),
                   autorange="reversed"),
        margin=dict(l=10,r=40,t=10,b=10), height=320,
    )
    return fig


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.5rem;'>
        <div style='font-family:Syne,sans-serif;font-size:1.4rem;
                    font-weight:800;color:#e8b84b;'>⬡ RecruitIQ</div>
        <div style='font-family:DM Mono,monospace;font-size:0.62rem;
                    letter-spacing:0.12em;color:#6b6b80;margin-top:0.2rem;'>
            AI RESUME SCREENING SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">MODE</div>', unsafe_allow_html=True)
    mode = st.radio("Mode", ["Screen New Resumes","Load Saved Results"],
                    label_visibility="collapsed")
    st.markdown("---")

    # Filters (shown when results exist)
    st.markdown('<div class="section-header">FILTERS</div>', unsafe_allow_html=True)
    min_score       = st.slider("Min score", 0, 100, 0)
    decision_filter = st.multiselect(
        "Decision", ["Strong Hire","Hire","Maybe","Reject"], default=[]
    )
    show_review = st.checkbox("Needs review only", value=False)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-family:DM Mono,monospace;font-size:0.6rem;
                color:#3a3a50;line-height:2;'>
    PIPELINE<br>
    <span style='color:{"#4be8b8" if PIPELINE_AVAILABLE else "#e84b6a"};'>
        {"● CONNECTED" if PIPELINE_AVAILABLE else "● IMPORT ERROR"}
    </span><br>
    {"" if PIPELINE_AVAILABLE else f"<span style='color:#e84b6a;font-size:0.55rem;'>{pipeline_error}</span>"}
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div style='margin-bottom:2rem;'>
    <div style='font-family:Syne,sans-serif;font-size:2.2rem;
                font-weight:800;letter-spacing:-0.03em;
                color:#e8e8f0;line-height:1.1;'>
        Candidate Intelligence
        <span style='color:#e8b84b;'>Dashboard</span>
    </div>
    <div style='font-family:DM Mono,monospace;font-size:0.7rem;
                color:#6b6b80;letter-spacing:0.08em;margin-top:0.4rem;'>
        AI-POWERED · OVERLAP-CORRECTED EXPERIENCE · DOMAIN-AWARE SCORING
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MODE A — SCREEN NEW RESUMES
# =========================================================
results = []
domain  = ""

if mode == "Screen New Resumes":

    if not PIPELINE_AVAILABLE:
        st.error(f"Pipeline unavailable — {pipeline_error}\n\nEnsure dashboard.py is in src/ and all modules are installed.")
        st.stop()

    # ── Upload Section ────────────────────────────────────
    st.markdown('<div class="section-header">UPLOAD JOB DESCRIPTION</div>',
                unsafe_allow_html=True)

    col_jd, col_resume = st.columns([1, 1], gap="large")

    with col_jd:
        st.markdown("""
        <div class='upload-label'>Job Description File</div>
        <div class='upload-info'>Accepts .txt or .pdf &nbsp;·&nbsp; One file only</div>
        """, unsafe_allow_html=True)

        jd_file = st.file_uploader(
            "Upload JD",
            type=["txt", "pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key="jd_uploader"
        )

        # Also allow manual paste as fallback
        st.markdown("""
        <div style='font-family:DM Mono,monospace;font-size:0.62rem;
                    color:#6b6b80;margin:0.8rem 0 0.3rem;'>
            OR PASTE JD MANUALLY
        </div>
        """, unsafe_allow_html=True)
        jd_paste = st.text_area(
            "Paste JD manually",
            height=150,
            placeholder="We are looking for a Machine Learning Engineer...",
            label_visibility="collapsed",
            key="jd_paste"
        )

        # Resolve final JD text — uploaded file takes priority
        jd_text = ""
        if jd_file:
            jd_text = extract_jd_from_upload(jd_file)
            if jd_text:
                st.markdown(f"""
                <div class='upload-success'>
                    ✓ JD loaded from: {jd_file.name}
                    ({len(jd_text.split())} words)
                </div>
                """, unsafe_allow_html=True)
        elif jd_paste.strip():
            jd_text = jd_paste.strip()
            st.markdown(f"""
            <div class='upload-success'>
                ✓ JD from manual input ({len(jd_text.split())} words)
            </div>
            """, unsafe_allow_html=True)

        # Preview JD
        if jd_text:
            with st.expander("Preview JD text", expanded=False):
                st.markdown(f"""
                <div style='font-family:DM Mono,monospace;font-size:0.7rem;
                            color:#9b9baf;line-height:1.8;white-space:pre-wrap;'>
                {jd_text[:800]}{"..." if len(jd_text) > 800 else ""}
                </div>
                """, unsafe_allow_html=True)

    with col_resume:
        st.markdown("""
        <div class='upload-label'>Resume Files</div>
        <div class='upload-info'>
            Accepts .pdf &nbsp;·&nbsp; Upload multiple files<br>
            All uploaded resumes will be screened and ranked
        </div>
        """, unsafe_allow_html=True)

        resume_files = st.file_uploader(
            "Upload Resumes",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="resume_uploader"
        )

        if resume_files:
            pdf_count = len([f for f in resume_files if f.name.lower().endswith(".pdf")])
            st.markdown(f"""
            <div class='upload-success'>✓ {pdf_count} PDF(s) ready to screen</div>
            <div style='font-family:DM Mono,monospace;font-size:0.62rem;
                        color:#6b6b80;margin-top:0.5rem;max-height:120px;
                        overflow-y:auto;'>
                {"<br>".join(f.name for f in resume_files[:20])}
                {"<br>..." if len(resume_files) > 20 else ""}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Run Button ────────────────────────────────────────
    col_btn, col_hint = st.columns([1, 4])
    with col_btn:
        run_btn = st.button("⬡  Screen Resumes")
    with col_hint:
        if not jd_text:
            st.markdown("""
            <div style='font-family:DM Mono,monospace;font-size:0.68rem;
                        color:#e84b6a;padding-top:0.6rem;'>
                ⚠ Upload or paste a job description first
            </div>
            """, unsafe_allow_html=True)
        elif not resume_files:
            st.markdown("""
            <div style='font-family:DM Mono,monospace;font-size:0.68rem;
                        color:#e8b84b;padding-top:0.6rem;'>
                ⚠ Upload at least one resume PDF
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='font-family:DM Mono,monospace;font-size:0.68rem;
                        color:#4be8b8;padding-top:0.6rem;'>
                ✓ Ready — {len(resume_files)} resumes · JD loaded
            </div>
            """, unsafe_allow_html=True)

    # ── Pipeline Execution ────────────────────────────────
    if run_btn:
        if not jd_text.strip():
            st.error("Please upload or paste a job description.")
        elif not resume_files:
            st.error("Please upload at least one resume PDF.")
        else:
            status   = st.empty()
            prog_bar = st.progress(0)

            try:
                # Step 1: Detect domain
                domain_name, domain_config = detect_domain(jd_text)
                status.markdown(f"""
                <div style='font-family:DM Mono,monospace;font-size:0.75rem;color:#e8b84b;'>
                    ◈ Domain detected: <b>{domain_name.upper().replace("_"," ")}</b>
                    &nbsp;·&nbsp; Weights: {domain_config["scoring_weights"]}
                </div>
                """, unsafe_allow_html=True)

                # Step 2: Save uploaded PDFs to temp folder
                status.markdown("""
                <div style='font-family:DM Mono,monospace;font-size:0.72rem;color:#6b6b80;'>
                    Saving uploaded files...
                </div>
                """, unsafe_allow_html=True)

                session_folder, saved_paths = save_uploaded_resumes(resume_files)
                total = len(saved_paths)

                if total == 0:
                    st.error("No valid PDF files found in upload.")
                    st.stop()

                status.markdown(f"""
                <div style='font-family:DM Mono,monospace;font-size:0.72rem;color:#6b6b80;'>
                    ✓ {total} files saved · starting pipeline...
                </div>
                """, unsafe_allow_html=True)

                # Step 3: Process each resume with real progress
                processed = []
                for i, pdf_path in enumerate(saved_paths, 1):
                    fname = os.path.basename(pdf_path)
                    pct   = int(i / total * 100)

                    status.markdown(f"""
                    <div style='font-family:DM Mono,monospace;font-size:0.7rem;color:#6b6b80;'>
                        [{i}/{total}] {fname}
                    </div>
                    """, unsafe_allow_html=True)
                    prog_bar.progress(pct)

                    result = process_single_resume(
                        pdf_path      = pdf_path,
                        jd_text       = jd_text,
                        domain_name   = domain_name,
                        domain_config = domain_config,
                    )
                    if result:
                        processed.append(result)

                # Step 4: Sort + rank
                processed.sort(key=lambda x: x.final_score, reverse=True)
                for i, r in enumerate(processed, 1):
                    r.rank = i

                # Step 5: Convert to dicts
                results = [normalize_result(asdict(r)) for r in processed]
                domain  = domain_name

                # Step 6: Auto-save to outputs/
                ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(OUTPUT_DIR, f"results_{ts}.json")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, default=str)

                # Step 7: Clean up temp folder
                try:
                    shutil.rmtree(session_folder)
                except Exception:
                    pass

                prog_bar.progress(100)
                status.markdown(f"""
                <div style='font-family:DM Mono,monospace;font-size:0.75rem;color:#4be8b8;'>
                    ✓ Complete — {len(results)}/{total} candidates scored
                    · Auto-saved to outputs/{os.path.basename(save_path)}
                </div>
                """, unsafe_allow_html=True)

                # Cache in session
                st.session_state["results"] = results
                st.session_state["domain"]  = domain

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                import traceback
                st.code(traceback.format_exc(), language="python")

    # Restore from session if pipeline ran earlier this session
    if not results:
        results = st.session_state.get("results", [])
        domain  = st.session_state.get("domain", "")


# =========================================================
# MODE B — LOAD SAVED RESULTS
# =========================================================
else:
    st.markdown('<div class="section-header">SAVED RESULTS</div>',
                unsafe_allow_html=True)
    output_files = get_output_files()

    if not output_files:
        st.markdown("""
        <div style='font-family:DM Mono,monospace;font-size:0.75rem;
                    color:#e84b6a;padding:1rem 0;'>
            No saved results found in outputs/ folder.<br>
            Screen some resumes first using the other mode.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    selected_file = st.selectbox(
        "Select results file",
        output_files,
        format_func=lambda x: os.path.basename(x),
        label_visibility="collapsed"
    )

    if selected_file:
        try:
            results = load_json(selected_file)
            domain  = results[0].get("domain","") if results else ""
            st.markdown(f"""
            <div style='font-family:DM Mono,monospace;font-size:0.72rem;
                        color:#4be8b8;margin:0.5rem 0;'>
                ✓ Loaded {len(results)} candidates from {os.path.basename(selected_file)}
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to load: {e}")


# =========================================================
# EMPTY STATE
# =========================================================
if not results:
    st.markdown("""
    <div style='text-align:center;padding:5rem 2rem;'>
        <div style='font-family:Syne,sans-serif;font-size:4rem;
                    color:#1e1e2a;font-weight:800;'>⬡</div>
        <div style='font-family:DM Mono,monospace;font-size:0.78rem;
                    color:#3a3a50;margin-top:1rem;letter-spacing:0.1em;'>
            NO RESULTS YET<br>
            <span style='font-size:0.62rem;color:#2a2a30;'>
                Upload a JD + resumes above and click Screen Resumes
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# =========================================================
# FILTERS
# =========================================================
filtered = []
for r in results:
    if r.get("final_score",0) * 100 < min_score:
        continue
    if show_review and not r.get("needs_review"):
        continue
    if decision_filter:
        dl = r.get("decision","").lower()
        if not any(f.lower() in dl for f in decision_filter):
            continue
    filtered.append(r)

if not filtered:
    st.warning("No candidates match current filters. Adjust in sidebar.")
    st.stop()

df = to_df(filtered)


# =========================================================
# DOMAIN STRIP + METRICS
# =========================================================
if domain:
    st.markdown(f"""
    <div style='margin-bottom:1.5rem;display:flex;align-items:center;gap:1rem;'>
        <span class='domain-badge'>{domain.upper().replace("_"," ")}</span>
        <span style='font-family:DM Mono,monospace;font-size:0.65rem;color:#6b6b80;'>
            {len(filtered)} candidates · {datetime.now().strftime("%d %b %Y")}
        </span>
    </div>
    """, unsafe_allow_html=True)

hired  = sum(1 for r in filtered
             if "hire" in r.get("decision","").lower()
             and "maybe" not in r.get("decision","").lower()
             and "manual" not in r.get("decision","").lower())
maybe  = sum(1 for r in filtered
             if any(k in r.get("decision","").lower() for k in ["maybe","consider"]))
review = sum(1 for r in filtered if r.get("needs_review"))
avg_sc = sum(r.get("final_score",0) for r in filtered) / len(filtered)
top_sc = max(r.get("final_score",0) for r in filtered)

for col, lbl, val, sub in zip(
    st.columns(5),
    ["TOTAL","RECOMMEND","MAYBE","AVG SCORE","TOP SCORE"],
    [len(filtered), hired, maybe, f"{avg_sc*100:.0f}%", f"{top_sc*100:.0f}%"],
    ["candidates","hire","review","mean match","best match"],
):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{lbl}</div>
            <div class='metric-value'>{val}</div>
            <div class='metric-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='margin:1.5rem 0;'></div>", unsafe_allow_html=True)


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["  Candidates  ","  Analytics  ","  Export  "])


# ── TAB 1: CANDIDATES ─────────────────────────────────────
with tab1:
    for r in filtered:
        score    = r.get("final_score", 0)
        name     = r.get("name", "Unknown")
        decision = r.get("decision", "")
        d_class  = decision_class(decision)
        matched  = r.get("matched_skills", [])
        missing  = r.get("missing_skills", [])
        rank     = int(r.get("rank", 0))

        with st.expander(
            f"#{rank}  {name}  ·  {score*100:.1f}%  ·  {decision}",
            expanded=(rank == 1)
        ):
            col_l, col_m, col_r = st.columns([2, 1.5, 1.5])

            with col_l:
                st.markdown(f"""
                <div style='font-family:DM Mono,monospace;font-size:0.62rem;
                            color:#6b6b80;'>RANK #{rank}</div>
                <div style='font-family:Syne,sans-serif;font-size:1.15rem;
                            font-weight:700;color:#e8e8f0;margin:0.15rem 0;'>
                    {name}</div>
                <div style='font-family:DM Mono,monospace;font-size:0.65rem;
                            color:#6b6b80;margin:0.3rem 0 0.8rem;'>
                    {r.get("email","N/A")} &nbsp;·&nbsp; {r.get("phone","N/A")}
                </div>
                <span class='decision-badge {d_class}'>{decision}</span>
                {"<br><span class='review-flag' style='margin-top:0.4rem;display:inline-block;'>⚠ LOW PARSE CONFIDENCE</span>" if r.get("needs_review") else ""}
                <div style='margin-top:1.2rem;'>
                """, unsafe_allow_html=True)

                for bar_lbl, bar_key in [
                    ("SEMANTIC",   "semantic_score"),
                    ("SKILLS",     "skill_score"),
                    ("EXPERIENCE", "total_exp_score"),
                    ("ROLE EXP",   "role_exp_score"),
                ]:
                    pct = r.get(bar_key,0) * 100
                    st.markdown(f"""
                    <div style='margin-bottom:0.55rem;'>
                        <div style='display:flex;justify-content:space-between;
                                    font-family:DM Mono,monospace;font-size:0.6rem;
                                    color:#6b6b80;margin-bottom:0.15rem;'>
                            <span>{bar_lbl}</span><span>{pct:.1f}%</span>
                        </div>
                        <div class='score-bar-track'>
                            <div class='score-bar-fill' style='width:{pct:.1f}%;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                exp_y = r.get("total_exp_years",0)
                pm    = r.get("parse_method","N/A").upper()
                pc    = r.get("parse_confidence",1.0) * 100
                li    = r.get("linkedin","")
                gh    = r.get("github","")
                extras = ""
                if li and li != "N/A": extras += f"LINKEDIN · {li}<br>"
                if gh and gh != "N/A": extras += f"GITHUB · {gh}"

                st.markdown(f"""
                <div style='font-family:DM Mono,monospace;font-size:0.62rem;
                            color:#6b6b80;margin-top:0.9rem;line-height:2;'>
                    EXP · {exp_y:.1f} yrs &nbsp;|&nbsp; PARSE · {pm} ({pc:.0f}%)
                    {"<br>" + extras if extras else ""}
                </div>
                """, unsafe_allow_html=True)

            with col_m:
                st.markdown('<div class="section-header">SCORE PROFILE</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(radar(r), use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown('<div class="section-header">BREAKDOWN</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(breakdown_bar(r), use_container_width=True,
                                config={"displayModeBar": False})

            with col_r:
                st.markdown('<div class="section-header">MATCHED SKILLS</div>',
                            unsafe_allow_html=True)
                if matched:
                    st.markdown(
                        "".join(f"<span class='skill-chip'>{s}</span>" for s in matched[:15]),
                        unsafe_allow_html=True)
                else:
                    st.markdown("""<div style='font-family:DM Mono,monospace;
                        font-size:0.65rem;color:#3a3a50;'>no matches</div>""",
                        unsafe_allow_html=True)

                st.markdown(
                    '<div class="section-header" style="margin-top:1.2rem;">MISSING SKILLS</div>',
                    unsafe_allow_html=True)
                if missing:
                    st.markdown(
                        "".join(f"<span class='skill-chip missing'>{s}</span>" for s in missing[:12]),
                        unsafe_allow_html=True)
                else:
                    st.markdown("""<div style='font-family:DM Mono,monospace;
                        font-size:0.65rem;color:#4be8b8;'>all skills matched ✓</div>""",
                        unsafe_allow_html=True)

                sem = r.get("semantic_skills", [])
                if sem:
                    st.markdown(
                        '<div class="section-header" style="margin-top:1.2rem;">SEMANTIC MATCHES</div>',
                        unsafe_allow_html=True)
                    st.markdown(
                        "".join(f"<span class='skill-chip' style='opacity:0.6;'>{s}</span>" for s in sem[:8]),
                        unsafe_allow_html=True)


# ── TAB 2: ANALYTICS ──────────────────────────────────────
with tab2:
    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.markdown('<div class="section-header">SCORE DISTRIBUTION</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(distribution(df), use_container_width=True,
                        config={"displayModeBar": False})
    with c2:
        st.markdown('<div class="section-header">HIRING DECISIONS</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(donut(filtered), use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown("---")
    c3, c4 = st.columns([1.4, 1.6])

    with c3:
        st.markdown('<div class="section-header">TOP MATCHED SKILLS</div>',
                    unsafe_allow_html=True)
        fig_sk = skills_chart(filtered)
        if fig_sk:
            st.plotly_chart(fig_sk, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.markdown("""<div style='font-family:DM Mono,monospace;font-size:0.7rem;
                color:#3a3a50;padding:2rem 0;'>No skill data.</div>""",
                unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="section-header">CANDIDATE COMPARISON</div>',
                    unsafe_allow_html=True)
        disp = df[["Rank","Name","Score","Semantic","Skills","Exp","Exp Years","Decision"]].copy()
        for c in ["Score","Semantic","Skills","Exp"]:
            disp[c] = disp[c].apply(lambda x: f"{x*100:.1f}%")
        disp["Exp Years"] = disp["Exp Years"].apply(lambda x: f"{x:.1f}y")
        try:
            st.dataframe(disp, use_container_width=True, hide_index=True, height=320)
        except TypeError:
            st.dataframe(disp, use_container_width=True, height=320)

    st.markdown("---")
    st.markdown('<div class="section-header">PARSE QUALITY</div>',
                unsafe_allow_html=True)
    pdf_c  = sum(1 for r in filtered if r.get("parse_method") == "pdfplumber")
    ocr_c  = sum(1 for r in filtered if r.get("parse_method") == "ocr")
    avg_cf = sum(r.get("parse_confidence",1) for r in filtered) / len(filtered)
    low_cf = sum(1 for r in filtered if r.get("parse_confidence",1) < 0.5)

    for col, lbl, val, sub in zip(
        st.columns(4),
        ["PDF PARSED","OCR FALLBACK","AVG CONFIDENCE","LOW CONFIDENCE"],
        [pdf_c, ocr_c, f"{avg_cf*100:.0f}%", low_cf],
        ["pdfplumber","scanned","parse quality","need review"],
    ):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{lbl}</div>
                <div class='metric-value' style='font-size:1.5rem;'>{val}</div>
                <div class='metric-sub'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)


# ── TAB 3: EXPORT ─────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">DOWNLOAD RESULTS</div>',
                unsafe_allow_html=True)
    ec1, ec2 = st.columns(2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    export_rows = [{
        "Rank":             int(r.get("rank",0)),
        "Name":             r.get("name",""),
        "Email":            r.get("email",""),
        "Phone":            r.get("phone",""),
        "Final Score %":    round(r.get("final_score",0)*100, 1),
        "Semantic %":       round(r.get("semantic_score",0)*100, 1),
        "Skill %":          round(r.get("skill_score",0)*100, 1),
        "Exp Score %":      round(r.get("total_exp_score",0)*100, 1),
        "Role Exp Score %": round(r.get("role_exp_score",0)*100, 1),
        "Total Exp Years":  r.get("total_exp_years",0),
        "Domain":           r.get("domain",""),
        "Decision":         r.get("decision",""),
        "Matched Skills":   ", ".join(r.get("matched_skills",[])),
        "Missing Skills":   ", ".join(r.get("missing_skills",[])),
        "LinkedIn":         r.get("linkedin",""),
        "GitHub":           r.get("github",""),
        "Parse Method":     r.get("parse_method",""),
        "Parse Confidence": round(r.get("parse_confidence",1)*100, 1),
        "Needs Review":     r.get("needs_review",False),
        "File":             os.path.basename(r.get("file_path","")),
    } for r in filtered]

    with ec1:
        st.download_button(
            "⬡  Download CSV",
            data=pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8"),
            file_name=f"screening_{ts}.csv",
            mime="text/csv",
        )
        st.markdown("""<div style='font-family:DM Mono,monospace;font-size:0.62rem;
            color:#6b6b80;margin-top:0.4rem;'>All candidates · full score breakdown</div>""",
            unsafe_allow_html=True)

    with ec2:
        st.download_button(
            "⬡  Download JSON",
            data=json.dumps(filtered, indent=2, default=str).encode("utf-8"),
            file_name=f"screening_{ts}.json",
            mime="application/json",
        )
        st.markdown("""<div style='font-family:DM Mono,monospace;font-size:0.62rem;
            color:#6b6b80;margin-top:0.4rem;'>Complete structured data</div>""",
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">SHORTLIST — HIRE RECOMMENDED</div>',
                unsafe_allow_html=True)
    shortlist = [r for r in filtered
                 if "hire" in r.get("decision","").lower()
                 and "maybe" not in r.get("decision","").lower()]
    if shortlist:
        for r in shortlist:
            st.markdown(f"""
            <div style='font-family:DM Mono,monospace;font-size:0.75rem;
                        color:#4be8b8;padding:0.4rem 0;border-bottom:1px solid #1a1a22;'>
                #{int(r.get("rank",0))} &nbsp;
                {r.get("name","?")} &nbsp;·&nbsp;
                {r.get("final_score",0)*100:.1f}% &nbsp;·&nbsp;
                {r.get("email","N/A")} &nbsp;·&nbsp;
                {r.get("total_exp_years",0):.1f} yrs
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""<div style='font-family:DM Mono,monospace;font-size:0.7rem;
            color:#3a3a50;'>No hire-recommended candidates under current filters.</div>""",
            unsafe_allow_html=True)