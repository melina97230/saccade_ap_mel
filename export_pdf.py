from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from matplotlib import font_manager


COLOR_PRIMARY = (27, 94, 55)
COLOR_SECONDARY = (76, 175, 80)
COLOR_GREY = (90, 100, 95)


def safe_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def format_index(value) -> str:
    if value is None or pd.isna(value):
        return "Données insuffisantes"
    return f"{float(value):.1f} / 100"


def new_pdf() -> FPDF:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    regular_font = font_manager.findfont("DejaVu Sans")
    bold_font = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold")
    )
    pdf.add_font("DejaVu", "", regular_font)
    pdf.add_font("DejaVu", "B", bold_font)
    pdf.set_font("DejaVu", "", 11)
    return pdf


def pdf_bytes(pdf: FPDF) -> bytes:
    output = pdf.output()
    return bytes(output)


def pdf_title(pdf: FPDF, title: str) -> None:
    pdf.set_font("DejaVu", "B", 18)
    pdf.set_text_color(*COLOR_PRIMARY)
    pdf.multi_cell(
        0,
        10,
        safe_text(title),
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(4)


def pdf_subtitle(pdf: FPDF, subtitle: str) -> None:
    pdf.set_font("DejaVu", "B", 13)
    pdf.set_text_color(*COLOR_SECONDARY)
    pdf.multi_cell(
        0,
        8,
        safe_text(subtitle),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(1)


def pdf_text(pdf: FPDF, text: str) -> None:
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(
        0,
        6,
        safe_text(text),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )


def pdf_table(pdf: FPDF, table: dict) -> None:
    for key, value in table.items():
        pdf.set_font("DejaVu", "B", 9)
        pdf.set_text_color(*COLOR_PRIMARY)
        pdf.multi_cell(
            0,
            5,
            safe_text(key),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.set_font("DejaVu", "", 10)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(
            0,
            6,
            safe_text(value),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.ln(1)


def add_identity(pdf: FPDF, patient_data: pd.DataFrame) -> None:
    last = patient_data.iloc[-1]
    pdf_subtitle(pdf, "Patient pseudonymisé")
    pdf_text(pdf, f"Code : {safe_text(last.get('patient_id'))}")
    pdf_text(pdf, f"Âge : {safe_text(last.get('age'))} ans")
    pdf_text(pdf, f"Statut TDA/H déclaré : {safe_text(last.get('tdah'))}")
    pdf_text(pdf, f"Nombre de séances : {len(patient_data)}")
    pdf.ln(2)


def add_index(pdf: FPDF, neuro_index) -> None:
    pdf_subtitle(pdf, "Index comportemental neurovisuel")
    pdf.set_font("DejaVu", "B", 18)
    if neuro_index is None or pd.isna(neuro_index):
        pdf.set_text_color(*COLOR_GREY)
    elif neuro_index >= 70:
        pdf.set_text_color(0, 130, 50)
    elif neuro_index >= 40:
        pdf.set_text_color(190, 120, 0)
    else:
        pdf.set_text_color(170, 40, 40)
    pdf.multi_cell(
        0,
        10,
        format_index(neuro_index),
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def add_list(pdf: FPDF, title: str, items: list[str]) -> None:
    pdf_subtitle(pdf, title)
    for item in items:
        pdf_text(pdf, f"• {safe_text(item).lstrip('• ').strip()}")
    pdf.ln(2)


def chart_file(patient_data: pd.DataFrame, column: str, title: str) -> Path | None:
    if column not in patient_data.columns:
        return None

    series = pd.to_numeric(patient_data[column], errors="coerce").dropna()
    if series.empty:
        return None

    if "session_number" in patient_data.columns:
        x_values = pd.to_numeric(
            patient_data.loc[series.index, "session_number"],
            errors="coerce",
        )
    else:
        x_values = np.arange(1, len(series) + 1)

    temporary = NamedTemporaryFile(delete=False, suffix=".png")
    temporary.close()
    path = Path(temporary.name)

    figure, axis = plt.subplots(figsize=(7.2, 3.2))
    axis.plot(x_values, series.values, marker="o", color="#2E7D32", linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("Séance")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def add_chart(pdf: FPDF, patient_data: pd.DataFrame, column: str, title: str) -> None:
    path = chart_file(patient_data, column, title)
    if path is None:
        return
    try:
        pdf.image(str(path), x=15, w=180)
        pdf.ln(4)
    finally:
        path.unlink(missing_ok=True)


def add_disclaimer(pdf: FPDF) -> None:
    pdf.ln(3)
    pdf.set_font("DejaVu", "", 8)
    pdf.set_text_color(*COLOR_GREY)
    pdf.multi_cell(
        0,
        5,
        "Prototype de recherche. Les seuils, l’index et les estimations ne sont "
        "pas validés sur une population de référence et ne constituent ni un "
        "diagnostic ni une décision thérapeutique autonome.",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )


def export_simple_pdf(
    patient_data: pd.DataFrame,
    table: dict,
    interpretation: list[str],
    neuro_index,
) -> bytes:
    pdf = new_pdf()
    pdf_title(pdf, "Bilan neurovisuel – Résumé")
    add_identity(pdf, patient_data)
    add_index(pdf, neuro_index)
    pdf_subtitle(pdf, "Indicateurs")
    pdf_table(pdf, table)
    add_list(pdf, "Interprétation descriptive", interpretation)
    add_disclaimer(pdf)
    return pdf_bytes(pdf)


def export_full_pdf(
    patient_data: pd.DataFrame,
    table: dict,
    interpretation: list[str],
    recommendations: list[str],
    neuro_index,
) -> bytes:
    pdf = new_pdf()
    pdf_title(pdf, "Rapport de suivi neurovisuel")
    add_identity(pdf, patient_data)
    add_index(pdf, neuro_index)
    pdf_subtitle(pdf, "Indicateurs de la dernière séance")
    pdf_table(pdf, table)
    add_list(pdf, "Interprétation descriptive", interpretation)
    add_list(pdf, "Pistes à confronter au bilan", recommendations)

    last_comment = safe_text(patient_data.iloc[-1].get("commentaire", "")).strip()
    if last_comment:
        pdf_subtitle(pdf, "Commentaire de l’orthoptiste")
        pdf_text(pdf, last_comment)

    pdf.add_page()
    pdf_title(pdf, "Évolution")
    add_chart(
        pdf,
        patient_data,
        "neuro_index",
        "Évolution de l’index comportemental neurovisuel",
    )
    add_chart(pdf, patient_data, "median_rt", "Évolution du temps de réaction médian")
    add_chart(pdf, patient_data, "cvrt", "Évolution du coefficient de variation")
    add_disclaimer(pdf)
    return pdf_bytes(pdf)


def export_ortho_pro_pdf(
    patient_data: pd.DataFrame,
    table: dict,
    interpretation: list[str],
    recommendations: list[str],
    neuro_index,
) -> bytes:
    pdf = new_pdf()
    pdf_title(pdf, "Synthèse orthoptique neurovisuelle")
    add_identity(pdf, patient_data)
    add_index(pdf, neuro_index)
    add_list(pdf, "Éléments descriptifs", interpretation[:5])
    add_list(pdf, "Pistes de réflexion", recommendations[:5])

    if len(patient_data) >= 2:
        values = pd.to_numeric(patient_data["neuro_index"], errors="coerce").dropna()
        if len(values) >= 2:
            delta = float(values.iloc[-1] - values.iloc[0])
            pdf_subtitle(pdf, "Évolution observée")
            pdf_text(
                pdf,
                f"Variation entre la première et la dernière séance calculable : "
                f"{delta:+.1f} point(s).",
            )

    pdf.add_page()
    pdf_title(pdf, "Courbe longitudinale")
    add_chart(
        pdf,
        patient_data,
        "neuro_index",
        "Évolution de l’index comportemental neurovisuel",
    )
    add_disclaimer(pdf)
    return pdf_bytes(pdf)
