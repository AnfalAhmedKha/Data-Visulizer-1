# enhanced_dashboard.py
"""
Enhanced DataSci Studio Pro - Advanced GUI
===========================================
Features:
- Dark/Light mode toggle
- Chunked/Full data upload with progress bar
- Ollama & OpenAI integration
- Advanced data preprocessing
- Export to multiple formats with location picker
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
import io
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Import existing modules
from data_loader import load_file
from cleaning import validate, auto_clean, encode_categoricals
from eda import summary_stats, top_correlations, fig_to_bytes
from model import prepare_features, detect_task, train, train_compare_all, save_model
from exporter import df_to_csv_bytes, df_to_excel_bytes, build_pdf_report

# AI Integration imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="DataSci Studio Pro - Advanced",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for animations and modern UI
st.markdown("""
<style>
/* Smooth transitions */
* {
    transition: all 0.3s ease;
}

/* Progress bar animation */
@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}

.stProgress > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
    background-size: 200% 100%;
    animation: shimmer 2s infinite;
}

/* Card hover effects */
.element-container:has(.stMetric) {
    transition: transform 0.2s, box-shadow 0.2s;
}
.element-container:has(.stMetric):hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}

/* Custom file uploader */
.uploadedFile {
    border: 2px dashed #667eea;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}

/* Toggle switch styling */
.toggle-switch {
    position: relative;
    display: inline-block;
    width: 60px;
    height: 34px;
}
.toggle-switch input {
    opacity: 0;
    width: 0;
    height: 0;
}
.slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #ccc;
    transition: 0.4s;
    border-radius: 34px;
}
.slider:before {
    position: absolute;
    content: "";
    height: 26px;
    width: 26px;
    left: 4px;
    bottom: 4px;
    background-color: white;
    transition: 0.4s;
    border-radius: 50%;
}
input:checked + .slider {
    background-color: #667eea;
}
input:checked + .slider:before {
    transform: translateX(26px);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    defaults = {
        "df_raw": None,
        "df_clean": None,
        "clean_report": None,
        "validation": None,
        "model_result": None,
        "compare_df": None,
        "figs": {},
        "meta": {},
        "upload_progress": 0,
        "dark_mode": True,
        "chunk_mode": False,
        "chunk_size": 10000,
        "uploaded_file_info": None,
        "ai_config": {},
        "processing_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# Theme management
def apply_theme():
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }
        .stMarkdown, .stText, .stMetric label {
            color: #f1f5f9 !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        }
        .stMarkdown, .stText, .stMetric label {
            color: #0f172a !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
        }
        </style>
        """, unsafe_allow_html=True)

# Progress bar for data loading
def show_upload_progress(progress_value, status_text):
    progress_bar = st.progress(progress_value)
    status = st.empty()
    status.text(status_text)
    return progress_bar, status

# Chunked data loading
def load_data_in_chunks(file, chunk_size: int = 10000) -> Tuple[pd.DataFrame, dict]:
    """Load large files in chunks with progress tracking"""
    ext = Path(file.name).suffix.lower()
    
    if ext == '.csv':
        chunks = []
        total_rows = 0
        for i, chunk in enumerate(pd.read_csv(file, chunksize=chunk_size)):
            chunks.append(chunk)
            total_rows += len(chunk)
            progress = (i + 1) * chunk_size / (chunk_size * 10) if i < 9 else 1.0
            st.session_state.upload_progress = min(progress, 1.0)
        
        df = pd.concat(chunks, ignore_index=True)
        meta = {
            "file": file.name,
            "ext": ext,
            "rows": df.shape[0],
            "cols": df.shape[1],
            "columns": df.columns.tolist(),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
            "loaded_at": datetime.now().isoformat(),
            "chunk_mode": True,
            "chunk_size": chunk_size,
            "num_chunks": len(chunks),
        }
        return df, meta
    else:
        # For non-CSV files, load normally
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        
        df, meta = load_file(tmp_path)
        os.unlink(tmp_path)
        meta["chunk_mode"] = False
        return df, meta

# AI Integration Class
class AIInsights:
    def __init__(self, provider: str = "ollama", api_key: str = None, model: str = None):
        self.provider = provider
        self.api_key = api_key
        self.model = model or ("llama2" if provider == "ollama" else "gpt-3.5-turbo")
        
    def generate_insights(self, df: pd.DataFrame, analysis_type: str = "overview") -> str:
        """Generate AI-powered insights about the dataset"""
        
        # Prepare dataset summary
        summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=np.number).columns),
            "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "column_names": df.columns.tolist()[:10],
            "sample_stats": df.describe().to_dict() if not df.select_dtypes(include=np.number).empty else {},
        }
        
        prompt = f"""
        As a data science expert, analyze this dataset and provide:
        1. Key insights about data quality
        2. Potential business opportunities
        3. Recommended preprocessing steps
        4. Suitable machine learning approaches
        
        Dataset Summary:
        - Rows: {summary['rows']:,}
        - Columns: {summary['columns']}
        - Numeric features: {summary['numeric_columns']}
        - Categorical features: {summary['categorical_columns']}
        - Missing values: {summary['missing_values']:,}
        - Duplicate rows: {summary['duplicate_rows']:,}
        - Sample columns: {', '.join(summary['column_names'])}
        
        Provide actionable, concise insights.
        """
        
        if self.provider == "openai" and self.api_key:
            return self._call_openai(prompt)
        elif self.provider == "ollama":
            return self._call_ollama(prompt)
        else:
            return self._fallback_insights(summary)
    
    def _call_openai(self, prompt: str) -> str:
        if not OPENAI_AVAILABLE:
            return "OpenAI package not installed. Run: pip install openai"
        
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API Error: {str(e)}"
    
    def _call_ollama(self, prompt: str) -> str:
        if not OLLAMA_AVAILABLE:
            return "Ollama not available. Install with: pip install requests"
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False}
            )
            if response.status_code == 200:
                return response.json().get("response", "No response from Ollama")
            else:
                return f"Ollama error: {response.status_code}"
        except Exception as e:
            return f"Ollama connection error: {str(e)}. Make sure Ollama is running."
    
    def _fallback_insights(self, summary: dict) -> str:
        """Fallback insights when AI is not available"""
        insights = []
        
        # Data quality insights
        if summary['missing_values'] > 0:
            insights.append(f"⚠️ Dataset has {summary['missing_values']:,} missing values. Consider imputation or removal.")
        else:
            insights.append("✅ No missing values detected.")
        
        if summary['duplicate_rows'] > 0:
            insights.append(f"🔄 Found {summary['duplicate_rows']:,} duplicate rows. Remove them to avoid bias.")
        
        # Feature insights
        if summary['numeric_columns'] > 0:
            insights.append(f"📊 {summary['numeric_columns']} numeric features available for analysis.")
        
        if summary['categorical_columns'] > 0:
            insights.append(f"🏷️ {summary['categorical_columns']} categorical features. Consider encoding for ML models.")
        
        # Recommendations
        if summary['rows'] < 1000:
            insights.append("💡 Small dataset detected. Traditional ML models may work well.")
        elif summary['rows'] > 100000:
            insights.append("🚀 Large dataset detected. Consider using chunked processing or sampling for initial analysis.")
        
        insights.append("🎯 Recommended next steps: EDA → Feature Engineering → Model Training → Evaluation")
        
        return "\n".join(insights)

# Save dialog with location picker
def save_dialog(data: bytes, default_filename: str, file_type: str):
    """Custom save dialog with location suggestion"""
    
    col1, col2 = st.columns(2)
    with col1:
        save_dir = st.text_input("Save Directory", value="./exports", key=f"dir_{default_filename}")
    with col2:
        filename = st.text_input("Filename", value=default_filename, key=f"name_{default_filename}")
    
    full_path = os.path.join(save_dir, filename)
    
    if st.button(f"💾 Save {file_type.upper()}", key=f"save_{default_filename}"):
        try:
            os.makedirs(save_dir, exist_ok=True)
            with open(full_path, "wb") as f:
                f.write(data)
            st.success(f"✅ Saved to: {full_path}")
            return True
        except Exception as e:
            st.error(f"Save failed: {e}")
            return False
    return False

# Sidebar
with st.sidebar:
    st.markdown("## 🧬 DataSci Studio Pro")
    st.markdown("### Advanced Edition")
    st.divider()
    
    # Theme Toggle
    st.markdown("### 🎨 Theme")
    theme_col1, theme_col2 = st.columns(2)
    with theme_col1:
        if st.button("🌙 Dark", use_container_width=True):
            st.session_state.dark_mode = True
            st.rerun()
    with theme_col2:
        if st.button("☀️ Light", use_container_width=True):
            st.session_state.dark_mode = False
            st.rerun()
    
    apply_theme()
    
    st.divider()
    
    # Data Upload Section
    st.markdown("### 📂 Data Source")
    
    upload_method = st.radio("Upload Method", ["File Upload", "Database", "API"], horizontal=True)
    
    if upload_method == "File Upload":
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls", "json", "parquet"],
            help="Supported formats: CSV, Excel, JSON, Parquet"
        )
        
        if uploaded_file:
            # Chunk mode toggle
            st.markdown("### ⚙️ Loading Options")
            chunk_mode = st.toggle("Load in chunks (large files)", value=st.session_state.chunk_mode)
            if chunk_mode:
                chunk_size = st.number_input("Chunk size (rows)", min_value=1000, max_value=100000, value=10000, step=1000)
                st.session_state.chunk_size = chunk_size
            st.session_state.chunk_mode = chunk_mode
            
            if st.button("🚀 Load Dataset", type="primary", use_container_width=True):
                with st.spinner("Loading dataset..."):
                    try:
                        if st.session_state.chunk_mode and uploaded_file.name.endswith('.csv'):
                            df, meta = load_data_in_chunks(uploaded_file, st.session_state.chunk_size)
                        else:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                                tmp.write(uploaded_file.read())
                                tmp_path = tmp.name
                            df, meta = load_file(tmp_path)
                            os.unlink(tmp_path)
                        
                        st.session_state.df_raw = df
                        st.session_state.meta = meta
                        st.session_state.df_clean = None
                        st.session_state.validation = None
                        st.session_state.model_result = None
                        st.session_state.processing_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "action": "data_loaded",
                            "file": uploaded_file.name,
                            "rows": len(df),
                            "cols": len(df.columns)
                        })
                        st.success(f"✅ Loaded {meta['rows']:,} rows × {meta['cols']} columns")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Load failed: {e}")
    
    elif upload_method == "Database":
        st.info("Connect to database")
        db_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL", "SQLite"])
        if db_type == "SQLite":
            db_path = st.text_input("Database Path", "./data.db")
        else:
            host = st.text_input("Host", "localhost")
            port = st.text_input("Port", "5432" if db_type == "PostgreSQL" else "3306")
            database = st.text_input("Database")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
        
        query = st.text_area("SQL Query", "SELECT * FROM your_table LIMIT 1000")
        
        if st.button("🔌 Connect & Load", type="primary"):
            try:
                from sqlalchemy import create_engine
                if db_type == "SQLite":
                    conn_string = f"sqlite:///{db_path}"
                elif db_type == "PostgreSQL":
                    conn_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
                else:
                    conn_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
                
                engine = create_engine(conn_string)
                df = pd.read_sql(query, engine)
                st.session_state.df_raw = df
                st.session_state.meta = {"rows": len(df), "cols": len(df.columns), "source": "database"}
                st.success(f"✅ Loaded {len(df):,} rows")
            except Exception as e:
                st.error(f"Database error: {e}")
    
    else:  # API
        st.info("Load data from REST API")
        api_url = st.text_input("API Endpoint", "https://api.example.com/data")
        api_method = st.selectbox("Method", ["GET", "POST"])
        headers = st.text_area("Headers (JSON)", value='{"Content-Type": "application/json"}')
        params = st.text_area("Parameters (JSON)", value="{}")
        
        if st.button("🌐 Fetch Data", type="primary"):
            try:
                import requests
                headers_dict = json.loads(headers) if headers else {}
                params_dict = json.loads(params) if params else {}
                
                if api_method == "GET":
                    response = requests.get(api_url, headers=headers_dict, params=params_dict)
                else:
                    response = requests.post(api_url, headers=headers_dict, json=params_dict)
                
                data = response.json()
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict) and "data" in data:
                    df = pd.DataFrame(data["data"])
                else:
                    df = pd.DataFrame([data])
                
                st.session_state.df_raw = df
                st.session_state.meta = {"rows": len(df), "cols": len(df.columns), "source": "api"}
                st.success(f"✅ Loaded {len(df):,} rows")
            except Exception as e:
                st.error(f"API error: {e}")
    
    st.divider()
    
    # AI Configuration
    st.markdown("### 🤖 AI Insights")
    ai_provider = st.selectbox("AI Provider", ["Ollama (Local)", "OpenAI", "None"])
    
    if ai_provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
        if api_key:
            st.session_state.ai_config = {"provider": "openai", "api_key": api_key, "model": model}
    elif ai_provider == "Ollama (Local)":
        ollama_model = st.selectbox(
    "Select Ollama Model",
    ["llama3.1", "llama3", "qwen2.5", "mistral", "gemma2", "phi3", "qwen2", "mixtral", "llama3.2", "llama2"],
    index=9  # llama2 is at index 9
)
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
        st.session_state.ai_config = {"provider": "ollama", "url": ollama_url, "model": ollama_model}
        st.caption("Make sure Ollama is running: `ollama serve`")

    elif ai_provider == "Anthropic":
        api_key = st.text_input("Anthropic API Key", type="password")
        model = st.selectbox("Model", ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"])
        if api_key:
            ai_config = {"provider": "anthropic", "api_key": api_key, "model": model}
    else:
        st.session_state.ai_config = {"provider": None}
    
    st.divider()
    
    # Processing History
    if st.session_state.processing_history:
        st.markdown("### 📜 History")
        for item in st.session_state.processing_history[-5:]:
            st.caption(f"🕐 {item['timestamp'][:19]}\n  → {item['action']}")

# Main content area
if st.session_state.df_raw is None:
    # Welcome screen
    st.markdown("# 🧬 DataSci Studio Pro")
    st.markdown("## Advanced Data Science Pipeline")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 📊 Features
        - Multi-format support
        - Chunked loading
        - Dark/Light mode
        - AI-powered insights
        """)
    with col2:
        st.markdown("""
        ### 🤖 AI Integration
        - Ollama (local)
        - OpenAI GPT
        - Automated insights
        - Smart recommendations
        """)
    with col3:
        st.markdown("""
        ### 💾 Export Options
        - CSV / Excel / JSON
        - PDF Reports
        - Model files
        - Custom location
        """)
    
    st.info("👈 **Get Started**: Upload a dataset from the sidebar to begin your analysis")
    
    # Example datasets
    st.markdown("### 📁 Try with sample data")
    if st.button("Load Sample Dataset (Retail Sales)"):
        try:
            df, meta = load_file("data/retail_sales.csv")
            st.session_state.df_raw = df
            st.session_state.meta = meta
            st.rerun()
        except FileNotFoundError:
            st.error("Sample dataset not found. Please upload your own file.")
    
    st.stop()

# Main dashboard with loaded data
df_view = st.session_state.df_raw
total_rows = len(df_view)
total_cols = len(df_view.columns)

# KPI Row
st.markdown("## 📊 Dashboard")
kpi_cols = st.columns(5)

with kpi_cols[0]:
    st.metric("Total Rows", f"{total_rows:,}", delta=None)
with kpi_cols[1]:
    st.metric("Features", total_cols)
with kpi_cols[2]:
    missing_pct = (df_view.isnull().sum().sum() / (total_rows * total_cols)) * 100
    st.metric("Missing Values", f"{df_view.isnull().sum().sum():,}", delta=f"{missing_pct:.1f}%")
with kpi_cols[3]:
    duplicates = df_view.duplicated().sum()
    st.metric("Duplicates", f"{duplicates:,}", delta=f"{(duplicates/total_rows)*100:.1f}%" if total_rows > 0 else "0%")
with kpi_cols[4]:
    memory_mb = df_view.memory_usage(deep=True).sum() / 1e6
    st.metric("Memory", f"{memory_mb:.1f} MB")

# AI Insights Button
if st.session_state.ai_config.get("provider"):
    if st.button("🧠 Generate AI Insights", type="primary"):
        with st.spinner("AI is analyzing your dataset..."):
            ai = AIInsights(
                provider=st.session_state.ai_config["provider"],
                api_key=st.session_state.ai_config.get("api_key"),
                model=st.session_state.ai_config.get("model")
            )
            insights = ai.generate_insights(df_view)
            st.markdown("### 🤖 AI-Powered Analysis")
            st.info(insights)

# Tabs for different functionalities
tabs = st.tabs(["📋 Data Explorer", "🧹 Data Cleaning", "📊 EDA & Viz", "🤖 Modeling", "💾 Export", "📈 Advanced Analytics"])

# Tab 1: Data Explorer
with tabs[0]:
    st.markdown("### 🔍 Data Preview")
    
    # Search and filter
    search_col = st.selectbox("Search Column", ["All Columns"] + list(df_view.columns))
    if search_col != "All Columns":
        search_term = st.text_input(f"Search in {search_col}")
        if search_term:
            mask = df_view[search_col].astype(str).str.contains(search_term, case=False, na=False)
            display_df = df_view[mask]
            st.caption(f"Found {len(display_df)} matching rows")
        else:
            display_df = df_view
    else:
        display_df = df_view
    
    # Column filters
    with st.expander("🔧 Column Filters"):
        filter_cols = st.multiselect("Select columns to display", df_view.columns, default=df_view.columns[:10])
        if filter_cols:
            display_df = display_df[filter_cols]
    
    # Show data
    rows_to_show = st.slider("Rows to display", 10, min(500, len(display_df)), 50)
    st.dataframe(display_df.head(rows_to_show), use_container_width=True, height=400)
    
    # Data info
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📋 Column Info")
        info_df = pd.DataFrame({
            "Dtype": df_view.dtypes.astype(str),
            "Nulls": df_view.isnull().sum(),
            "Null %": (df_view.isnull().sum() / len(df_view) * 100).round(2),
            "Unique": df_view.nunique(),
        })
        st.dataframe(info_df, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Quick Stats")
        numeric_df = df_view.select_dtypes(include=np.number)
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe(), use_container_width=True)

# Tab 2: Data Cleaning
with tabs[1]:
    st.markdown("### 🧹 Advanced Data Cleaning")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("#### Cleaning Options")
        null_strategy = st.selectbox("Null Strategy", ["auto", "median", "mean", "zero", "ffill", "drop"])
        cap_outliers = st.checkbox("Cap Outliers (IQR)", value=True)
        null_threshold = st.slider("Drop columns with >% nulls", 0, 100, 60) / 100
        standardize_names = st.checkbox("Standardize column names", value=True)
        remove_duplicates = st.checkbox("Remove duplicates", value=True)
        
        if st.button("✨ Run Auto-Clean", type="primary", use_container_width=True):
            with st.spinner("Cleaning dataset..."):
                try:
                    from cleaning import standardize_column_names, remove_duplicates as rm_dupes
                    df_clean = df_view.copy()
                    
                    if standardize_names:
                        df_clean = standardize_column_names(df_clean)
                    if remove_duplicates:
                        df_clean = rm_dupes(df_clean)
                    
                    df_clean, report = auto_clean(
                        df_clean,
                        null_strategy=null_strategy,
                        null_threshold=null_threshold,
                        cap_outliers=cap_outliers
                    )
                    
                    st.session_state.df_clean = df_clean
                    st.session_state.clean_report = report
                    
                    st.success("✅ Cleaning completed!")
                    st.json(report)
                except Exception as e:
                    st.error(f"Cleaning error: {e}")
    
    with col1:
        if st.session_state.df_clean is not None:
            st.markdown("#### Before vs After")
            before_col, after_col = st.columns(2)
            with before_col:
                st.caption("Original Data")
                st.dataframe(df_view.head(10), use_container_width=True)
            with after_col:
                st.caption("Cleaned Data")
                st.dataframe(st.session_state.df_clean.head(10), use_container_width=True)
            
            st.markdown("#### Cleaning Report")
            if st.session_state.clean_report:
                rep = st.session_state.clean_report
                metric_cols = st.columns(4)
                metric_cols[0].metric("Rows Before", rep['shape_before'][0])
                metric_cols[1].metric("Rows After", rep['shape_after'][0], delta=f"-{rep['rows_removed']}")
                metric_cols[2].metric("Cols Before", rep['shape_before'][1])
                metric_cols[3].metric("Cols After", rep['shape_after'][1], delta=f"-{rep['cols_removed']}")
                
                with st.expander("Steps Applied"):
                    for step in rep["steps"]:
                        st.write(f"✅ {step}")
        else:
            st.info("Click 'Run Auto-Clean' to start preprocessing")

# Tab 3: EDA & Visualization
with tabs[2]:
    st.markdown("### 📊 Exploratory Data Analysis")
    
    viz_type = st.selectbox("Visualization Type", [
        "Correlation Heatmap", "Distribution Analysis", "Box Plots", 
        "Pair Plot", "Time Series", "3D Scatter Plot"
    ])
    
    numeric_cols = df_view.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_view.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if viz_type == "Correlation Heatmap":
        if len(numeric_cols) >= 2:
            corr_matrix = df_view[numeric_cols].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmin=-1, zmax=1,
                text=corr_matrix.round(2).values,
                texttemplate='%{text}',
                textfont={"size": 10},
            ))
            fig.update_layout(
                title="Feature Correlation Matrix",
                width=800, height=700,
                template='plotly_dark' if st.session_state.dark_mode else 'plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for correlation analysis")
    
    elif viz_type == "Distribution Analysis":
        if numeric_cols:
            selected_col = st.selectbox("Select Feature", numeric_cols)
            fig = make_subplots(rows=2, cols=1, subplot_titles=(f"Histogram: {selected_col}", f"Box Plot: {selected_col}"))
            
            # Histogram
            fig.add_trace(go.Histogram(x=df_view[selected_col].dropna(), nbinsx=50, name="Histogram"), row=1, col=1)
            # Box plot
            fig.add_trace(go.Box(y=df_view[selected_col].dropna(), name="Box Plot"), row=2, col=1)
            
            fig.update_layout(height=600, title=f"Distribution Analysis: {selected_col}", 
                            template='plotly_dark' if st.session_state.dark_mode else 'plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            data = df_view[selected_col].dropna()
            col1.metric("Mean", f"{data.mean():.2f}")
            col2.metric("Median", f"{data.median():.2f}")
            col3.metric("Std Dev", f"{data.std():.2f}")
            col4.metric("Skewness", f"{data.skew():.2f}")
    
    elif viz_type == "Box Plots":
        if numeric_cols:
            selected_cols = st.multiselect("Select Features", numeric_cols, default=numeric_cols[:5])
            if selected_cols:
                fig = go.Figure()
                for col in selected_cols:
                    fig.add_trace(go.Box(y=df_view[col].dropna(), name=col))
                fig.update_layout(title="Box Plot Analysis", height=500,
                                template='plotly_dark' if st.session_state.dark_mode else 'plotly_white')
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Pair Plot" and len(numeric_cols) >= 2:
        st.warning("Pair plot generation may be slow for large datasets")
        if st.button("Generate Pair Plot"):
            import plotly.figure_factory as ff
            fig = ff.create_scatterplotmatrix(df_view[numeric_cols[:4]], diag='histogram', height=800, width=800)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Time Series" and categorical_cols:
        date_cols = [c for c in categorical_cols if 'date' in c.lower() or 'time' in c.lower()]
        if date_cols:
            date_col = st.selectbox("Date Column", date_cols)
            try:
                df_view['_date'] = pd.to_datetime(df_view[date_col])
                value_col = st.selectbox("Value Column", numeric_cols)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_view['_date'], y=df_view[value_col], mode='lines+markers'))
                fig.update_layout(title=f"Time Series: {value_col} over {date_col}", height=500,
                                template='plotly_dark' if st.session_state.dark_mode else 'plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.error("Could not parse date column")
    
    elif viz_type == "3D Scatter Plot" and len(numeric_cols) >= 3:
        col_x = st.selectbox("X Axis", numeric_cols, index=0)
        col_y = st.selectbox("Y Axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        col_z = st.selectbox("Z Axis", numeric_cols, index=2 if len(numeric_cols) > 2 else 0)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=df_view[col_x].sample(min(1000, len(df_view))),
            y=df_view[col_y].sample(min(1000, len(df_view))),
            z=df_view[col_z].sample(min(1000, len(df_view))),
            mode='markers',
            marker=dict(size=3, color=df_view[col_z].sample(min(1000, len(df_view))), colorscale='Viridis')
        )])
        fig.update_layout(title="3D Scatter Plot", height=600,
                         template='plotly_dark' if st.session_state.dark_mode else 'plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Modeling
with tabs[3]:
    st.markdown("### 🤖 Machine Learning Modeling")
    
    if st.session_state.df_clean is not None:
        df_model = st.session_state.df_clean
    else:
        df_model = df_view
    
    target_col = st.selectbox("🎯 Target Column", df_model.columns, help="Column to predict")
    
    col1, col2 = st.columns(2)
    with col1:
        task = st.selectbox("Task Type", ["auto", "classification", "regression"])
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
    with col2:
        model_type = st.selectbox("Model", ["Random Forest", "Gradient Boosting", "Linear", "SVM", "Compare All"])
        cv_folds = st.slider("CV Folds", 3, 10, 5)
    
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training model..."):
            try:
                df_enc, _ = encode_categoricals(df_model, exclude=[target_col])
                X, y = prepare_features(df_enc, target_col)
                
                if task == "auto":
                    actual_task = detect_task(y)
                else:
                    actual_task = task
                
                if model_type == "Compare All":
                    compare_df = train_compare_all(X, y, task=actual_task, test_size=test_size)
                    st.session_state.compare_df = compare_df
                    st.success("✅ Comparison complete!")
                    st.dataframe(compare_df, use_container_width=True)
                else:
                    result = train(X, y, task=actual_task, model_name=model_type, test_size=test_size, cv_folds=cv_folds)
                    st.session_state.model_result = result
                    st.success(f"✅ {model_type} trained!")
                    
                    # Display metrics
                    st.markdown("### 📊 Model Performance")
                    metrics = result["metrics"]
                    metric_cols = st.columns(len(metrics))
                    for i, (k, v) in enumerate(metrics.items()):
                        metric_cols[i].metric(k.upper(), f"{v:.4f}")
                    
                    st.metric("Cross-validation Score", f"{result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
                    
                    # Feature importance
                    if result["feature_importance"]:
                        st.markdown("### 🔑 Feature Importance")
                        importance_df = pd.DataFrame(list(result["feature_importance"].items()), columns=["Feature", "Importance"])
                        importance_df = importance_df.sort_values("Importance", ascending=True).tail(15)
                        
                        fig = go.Figure(go.Bar(x=importance_df["Importance"], y=importance_df["Feature"], orientation='h'))
                        fig.update_layout(title="Top 15 Features", height=500,
                                        template='plotly_dark' if st.session_state.dark_mode else 'plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Save model option
                    if st.button("💾 Save Model"):
                        model_path = f"models/{model_type.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                        os.makedirs("models", exist_ok=True)
                        save_model(result["model"], model_path)
                        st.success(f"Model saved to {model_path}")
            except Exception as e:
                st.error(f"Training error: {e}")

# Tab 5: Export
with tabs[4]:
    st.markdown("### 💾 Export Results")
    
    export_df = st.session_state.df_clean if st.session_state.df_clean is not None else df_view
    
    st.markdown("#### 📄 Data Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export as CSV"):
            csv_data = df_to_csv_bytes(export_df)
            save_dialog(csv_data, f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "CSV")
    
    with col2:
        if st.button("📑 Export as Excel"):
            excel_data = df_to_excel_bytes(export_df, include_stats=True)
            save_dialog(excel_data, f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", "Excel")
    
    with col3:
        if st.button("📋 Export as JSON"):
            json_data = export_df.to_json(orient="records", indent=2).encode()
            save_dialog(json_data, f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "JSON")
    
    st.divider()
    
    st.markdown("#### 📄 PDF Report")
    if st.button("📑 Generate PDF Report", type="primary"):
        with st.spinner("Generating PDF..."):
            figs_for_pdf = []
            # Generate some default visualizations
            numeric_cols = export_df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                export_df[numeric_cols[:3]].hist(ax=ax, alpha=0.7)
                figs_for_pdf.append(fig)
            
            pdf_bytes = build_pdf_report(
                df=export_df,
                clean_report=st.session_state.clean_report,
                model_results=st.session_state.model_result,
                figures=figs_for_pdf
            )
            if pdf_bytes:
                save_dialog(pdf_bytes, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", "PDF")
    
    if st.session_state.model_result:
        st.divider()
        st.markdown("#### 🤖 Model Export")
        if st.button("💾 Save Trained Model"):
            import joblib
            model_bytes = io.BytesIO()
            joblib.dump(st.session_state.model_result["model"], model_bytes)
            save_dialog(model_bytes.getvalue(), f"model_{st.session_state.model_result['model_name'].replace(' ', '_')}.joblib", "model")

# Tab 6: Advanced Analytics
with tabs[5]:
    st.markdown("### 📈 Advanced Analytics")
    
    analysis_type = st.selectbox("Analysis Type", [
        "Statistical Tests", "Anomaly Detection", "Clustering Analysis",
        "Time Series Forecasting", "Feature Engineering Suggestions"
    ])
    
    if analysis_type == "Statistical Tests":
        st.markdown("#### Hypothesis Testing")
        col1, col2 = st.columns(2)
        with col1:
            test_type = st.selectbox("Test Type", ["t-test", "ANOVA", "Chi-Square", "Correlation Test"])
        with col2:
            alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
        
        if test_type == "t-test":
            if len(numeric_cols) >= 2:
                group_col = st.selectbox("Group Column (binary)", [c for c in df_view.columns if df_view[c].nunique() == 2])
                value_col = st.selectbox("Value Column", numeric_cols)
                
                if st.button("Run t-test"):
                    from scipy import stats
                    groups = df_view[group_col].unique()
                    group1 = df_view[df_view[group_col] == groups[0]][value_col].dropna()
                    group2 = df_view[df_view[group_col] == groups[1]][value_col].dropna()
                    t_stat, p_value = stats.ttest_ind(group1, group2)
                    
                    st.write(f"**t-statistic:** {t_stat:.4f}")
                    st.write(f"**p-value:** {p_value:.4f}")
                    st.write(f"**Result:** {'Reject H0' if p_value < alpha else 'Fail to reject H0'} - {'Significant difference' if p_value < alpha else 'No significant difference'}")
            else:
                st.warning("Need at least 2 numeric columns for t-test")
    
    elif analysis_type == "Anomaly Detection":
        st.markdown("#### Outlier Detection (IQR Method)")
        if numeric_cols:
            selected_col = st.selectbox("Select Column", numeric_cols)
            data = df_view[selected_col].dropna()
            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            st.metric("Outliers Found", len(outliers))
            st.metric("Outlier Percentage", f"{(len(outliers)/len(data))*100:.2f}%")
            st.metric("Lower Bound", f"{lower_bound:.2f}")
            st.metric("Upper Bound", f"{upper_bound:.2f}")
            
            if len(outliers) > 0:
                st.dataframe(outliers.to_frame(), use_container_width=True)
    
    elif analysis_type == "Clustering Analysis":
        if len(numeric_cols) >= 2:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            n_clusters = st.slider("Number of Clusters (k)", 2, 10, 3)
            cluster_cols = st.multiselect("Features for Clustering", numeric_cols, default=numeric_cols[:2])
            
            if st.button("Run K-Means Clustering"):
                X_cluster = df_view[cluster_cols].dropna()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_cluster)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                st.session_state.cluster_labels = labels
                st.metric("Clusters Created", n_clusters)
                st.metric("Silhouette Score", f"{sklearn.metrics.silhouette_score(X_scaled, labels):.3f}")
                
                # Visualize
                fig = go.Figure(data=go.Scatter(
                    x=X_scaled[:, 0], y=X_scaled[:, 1],
                    mode='markers',
                    marker=dict(color=labels, colorscale='Viridis', size=10, showscale=True)
                ))
                fig.update_layout(title=f"K-Means Clustering (k={n_clusters})", height=500,
                                template='plotly_dark' if st.session_state.dark_mode else 'plotly_white')
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Time Series Forecasting" and 'date' in str(df_view.columns).lower():
        st.info("Time series forecasting requires a date column and numeric value column")
        date_col = st.selectbox("Date Column", [c for c in df_view.columns if 'date' in c.lower() or 'time' in c.lower()])
        value_col = st.selectbox("Value Column", numeric_cols)
        forecast_steps = st.slider("Forecast Steps", 1, 30, 10)
        
        if st.button("Run Forecast"):
            try:
                df_view['_date'] = pd.to_datetime(df_view[date_col])
                ts_data = df_view.set_index('_date')[value_col].sort_index()
                
                from sklearn.linear_model import LinearRegression
                import numpy as np
                
                # Simple trend forecast
                X = np.arange(len(ts_data)).reshape(-1, 1)
                y = ts_data.values
                model = LinearRegression()
                model.fit(X, y)
                
                future_X = np.arange(len(ts_data), len(ts_data) + forecast_steps).reshape(-1, 1)
                forecast = model.predict(future_X)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, mode='lines', name='Historical'))
                fig.add_trace(go.Scatter(
                    x=pd.date_range(start=ts_data.index[-1], periods=forecast_steps+1, freq='D')[1:],
                    y=forecast, mode='lines+markers', name='Forecast', line=dict(dash='dash')
                ))
                fig.update_layout(title="Time Series Forecast", height=500,
                                template='plotly_dark' if st.session_state.dark_mode else 'plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Forecast error: {e}")
    
    elif analysis_type == "Feature Engineering Suggestions":
        st.markdown("#### 🤖 Automated Feature Engineering Suggestions")
        
        suggestions = []
        
        # Date features
        date_cols = [c for c in df_view.columns if 'date' in c.lower() or 'time' in c.lower()]
        if date_cols:
            for col in date_cols:
                suggestions.append(f"📅 Extract from '{col}': year, month, day, dayofweek, quarter, is_weekend")
        
        # Text features
        text_cols = [c for c in df_view.columns if df_view[c].dtype == 'object' and df_view[c].nunique() > 10]
        if text_cols:
            for col in text_cols[:3]:
                suggestions.append(f"📝 Text features from '{col}': length, word_count, capital_ratio, special_char_count")
        
        # Interaction features
        if len(numeric_cols) > 1:
            suggestions.append(f"🔗 Interaction features: {numeric_cols[0]} * {numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]}")
            suggestions.append(f"📊 Ratio features: {numeric_cols[0]} / {numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]}")
        
        # Binning
        for col in numeric_cols[:3]:
            suggestions.append(f"📦 Binning for '{col}': create quartiles, deciles, or custom bins")
        
        # Aggregations
        cat_cols = [c for c in df_view.columns if df_view[c].dtype == 'object' and df_view[c].nunique() <= 20]
        if cat_cols and numeric_cols:
            for cat in cat_cols[:2]:
                for num in numeric_cols[:2]:
                    suggestions.append(f"📊 Group by '{cat}' aggregations: mean_{num}, median_{num}, std_{num}")
        
        for suggestion in suggestions:
            st.write(f"✨ {suggestion}")
        
        st.info("💡 Tip: Use these suggestions to create new features and improve model performance!")

# Footer
st.divider()
st.caption("🧬 DataSci Studio Pro Advanced | Built with Streamlit | AI-Powered Data Science Pipeline")