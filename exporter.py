"""
exporter.py
===========
All export functions — CSV, Excel (formatted), PDF report,
plot PNGs, and model files.  Returns bytes for Streamlit download buttons.
"""

import io
import logging
import os
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(df: pd.DataFrame,
                       sheet_name: str = "Data",
                       include_stats: bool = True) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        if include_stats:
            num = df.select_dtypes(include="number")
            if not num.empty:
                num.describe().round(3).to_excel(writer, sheet_name="Stats")
        # Auto-width columns
        ws = writer.sheets[sheet_name]
        for col_cells in ws.columns:
            max_len = max(len(str(c.value or "")) for c in col_cells) + 2
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len, 40)
    return buf.getvalue()


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="#0f172a")
    return buf.getvalue()


def model_to_bytes(model) -> bytes:
    import joblib
    buf = io.BytesIO()
    joblib.dump({"model": model, "saved_at": datetime.now().isoformat()}, buf)
    return buf.getvalue()


def build_pdf_report(df: pd.DataFrame,
                      clean_report: dict = None,
                      model_results: dict = None,
                      figures: List[plt.Figure] = None) -> bytes:
    """
    Full multi-section PDF report:
    1. Dataset Overview
    2. Cleaning Summary
    3. Descriptive Statistics
    4. Charts (up to 6 figures embedded)
    5. Model Results (if provided)
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, Image as RLImage,
                                         HRFlowable, PageBreak)
    except ImportError:
        logger.error("pip install reportlab")
        return b""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    S_title = ParagraphStyle("T", parent=styles["Title"], fontSize=20,
                              textColor=colors.HexColor("#667eea"))
    S_h1    = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13,
                              textColor=colors.HexColor("#334155"))
    S_body  = styles["Normal"]

    ACC  = colors.HexColor("#667eea")
    GRN  = colors.HexColor("#10b981")
    WARN = colors.HexColor("#f59e0b")
    LT   = colors.HexColor("#f8fafc")
    LT2  = colors.white

    def tbl(data, col_widths, header_color=ACC):
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0,0),(-1,0), header_color),
            ("TEXTCOLOR",   (0,0),(-1,0), colors.white),
            ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[LT, LT2]),
            ("GRID",        (0,0),(-1,-1), 0.4, colors.HexColor("#cbd5e1")),
            ("FONTSIZE",    (0,0),(-1,-1), 8),
            ("PADDING",     (0,0),(-1,-1), 5),
        ]))
        return t

    story = []
    story.append(Paragraph("DataSci Studio Pro — Analysis Report", S_title))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", S_body))
    story.append(HRFlowable(width="100%", thickness=1, color=ACC))
    story.append(Spacer(1, 0.4*cm))

    # ── 1. Dataset Overview ──────────────────────────────────────────────
    story.append(Paragraph("1.  Dataset Overview", S_h1))
    ov = [["Metric","Value"],
          ["Rows",           f"{df.shape[0]:,}"],
          ["Columns",        str(df.shape[1])],
          ["Numeric cols",   str(len(df.select_dtypes("number").columns))],
          ["Categorical cols",str(len(df.select_dtypes("object").columns))],
          ["Missing values", str(int(df.isnull().sum().sum()))],
          ["Duplicate rows", str(int(df.duplicated().sum()))],
          ["Memory",         f"{df.memory_usage(deep=True).sum()/1e6:.2f} MB"]]
    story += [tbl(ov, [7*cm, 10*cm]), Spacer(1, 0.4*cm)]

    # ── 2. Cleaning summary ──────────────────────────────────────────────
    if clean_report:
        story.append(Paragraph("2.  Data Cleaning Summary", S_h1))
        cdata = [["Step","Detail"]]
        for k, v in clean_report.items():
            if k != "steps":
                cdata.append([str(k), str(v)])
        story += [tbl(cdata, [8*cm, 10*cm], GRN), Spacer(1, 0.4*cm)]

    # ── 3. Descriptive stats ─────────────────────────────────────────────
    num = df.select_dtypes(include="number").iloc[:, :10]
    if not num.empty:
        story.append(Paragraph("3.  Descriptive Statistics", S_h1))
        desc = num.describe().round(3)
        sd = [[""] + list(desc.columns)]
        for idx in desc.index:
            sd.append([idx] + [str(v) for v in desc.loc[idx]])
        cw = [2.5*cm] + [max(1.5*cm, 15*cm/len(desc.columns))]*len(desc.columns)
        story += [tbl(sd, cw, GRN), Spacer(1, 0.4*cm)]

    # ── 4. Charts ────────────────────────────────────────────────────────
    if figures:
        story.append(PageBreak())
        story.append(Paragraph("4.  Charts", S_h1))
        for fig in figures[:6]:
            if fig is None:
                continue
            try:
                ibuf = io.BytesIO()
                fig.savefig(ibuf, format="png", dpi=130, bbox_inches="tight", facecolor="#0f172a")
                ibuf.seek(0)
                story += [RLImage(ibuf, width=15.5*cm, height=7.5*cm), Spacer(1, 0.3*cm)]
            except Exception as e:
                logger.warning(f"Chart embed failed: {e}")

    # ── 5. Model results ─────────────────────────────────────────────────
    if model_results:
        story.append(PageBreak())
        story.append(Paragraph("5.  Model Results", S_h1))
        mdata = [["Metric","Value"]]
        mdata.append(["Model",      model_results.get("model_name","?")])
        mdata.append(["Task",       model_results.get("task","?")])
        mdata.append(["CV Mean",    str(model_results.get("cv_mean","?"))])
        mdata.append(["CV Std",     str(model_results.get("cv_std","?"))])
        for k, v in model_results.get("metrics", {}).items():
            mdata.append([k.upper(), str(v)])
        story += [tbl(mdata, [7*cm, 10*cm], WARN), Spacer(1, 0.3*cm)]

        fi = model_results.get("feature_importance", {})
        if fi:
            fi_data = [["Feature","Importance"]] + [[k, str(v)] for k, v in list(fi.items())[:15]]
            story += [Paragraph("Feature Importance", S_h1),
                      tbl(fi_data, [10*cm, 7*cm], WARN)]

    doc.build(story)
    return buf.getvalue()
