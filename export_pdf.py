import os
from fpdf import FPDF
import matplotlib.pyplot as plt
import pandas as pd

# ===================================================
# STYLE CLINIQUE – COULEURS APAISANTES
# ===================================================

COLOR_PRIMARY = (27, 58, 46)      # Vert foncé
COLOR_SECONDARY = (76, 175, 80)   # Vert clair
COLOR_LIGHT = (227, 247, 232)     # Vert pastel

# ===================================================
# OUTIL : CRÉATION D’UN TITRE
# ===================================================

def pdf_title(pdf, title):
    pdf.set_font("DejaVu", "B", 20)
    pdf.set_text_color(*COLOR_PRIMARY)
    pdf.cell(0, 15, title, ln=True, align="C")
    pdf.ln(5)

# ===================================================
# OUTIL : SOUS-TITRE
# ===================================================

def pdf_subtitle(pdf, subtitle):
    pdf.set_font("DejaVu", "B", 14)
    pdf.set_text_color(*COLOR_SECONDARY)
    pdf.cell(0, 10, subtitle, ln=True)
    pdf.ln(2)

# ===================================================
# OUTIL : TEXTE SÉCURISÉ
# ===================================================

def safe_text(x):
    if x is None:
        return ""
    return str(x)

# ===================================================
# OUTIL : TEXTE NORMAL
# ===================================================

def pdf_text(pdf, text):
    pdf.set_font("DejaVu", "", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 8, safe_text(text))
    pdf.ln(2)

# ===================================================
# OUTIL : TABLEAU CLINIQUE
# ===================================================

def pdf_table(pdf, table_dict):
    pdf.set_font("DejaVu", "", 12)
    for key, value in table_dict.items():
        pdf.set_text_color(*COLOR_PRIMARY)
        pdf.cell(80, 8, f"{key} :", border=0)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(40, 8, safe_text(value), ln=True)
    pdf.ln(5)

# ===================================================
# OUTIL : COURBES
# ===================================================

def save_curve(series, title, filename):
    plt.figure(figsize=(6, 3))
    plt.plot(series.index, series.values, marker="o", color="#4CAF50")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ===================================================
# EXPORT PDF SIMPLE
# ===================================================

def export_simple_pdf(df_patient, table, interpretation, neuro_index):

    pdf = FPDF()
    pdf.add_page()

    # Police Unicode
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.add_font("DejaVu", "B", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", "", 12)

    pdf_title(pdf, "Bilan Neurovisuel – Résumé")

    pdf_subtitle(pdf, "Index neurovisuel")
    pdf_text(pdf, f"Score global : {round(neuro_index, 1)} / 100")

    pdf_subtitle(pdf, "Tableau clinique")
    pdf_table(pdf, table)

    pdf_subtitle(pdf, "Interprétation")
    for line in interpretation:
        pdf_text(pdf, f"- {line}")

    filename = "bilan_simple.pdf"
    pdf.output(filename)

    return filename

# ===================================================
# EXPORT PDF COMPLET – RAPPORT CLINIQUE
# ===================================================

def export_full_pdf(df_patient, table, interpretation, recos, neuro_index):

    pdf = FPDF()
    pdf.add_page()

    # Police Unicode
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.add_font("DejaVu", "B", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", "", 12)

    # TITRE
    pdf_title(pdf, "Rapport Clinique Orthoptique – Analyse Neurovisuelle")

    # IDENTITÉ PATIENT
    pdf_subtitle(pdf, "Identité patient (anonymisée)")
    last = df_patient.iloc[-1]
    pdf_text(pdf, f"ID patient : {safe_text(last['patient_id'])}")
    pdf_text(pdf, f"Âge : {safe_text(last['age'])} ans")
    pdf_text(pdf, f"Statut TDAH : {safe_text(last['tdah'])}")
    pdf.ln(3)

    # INDEX
    pdf_subtitle(pdf, "Index neurovisuel")
    pdf_text(pdf, f"Score global : {round(neuro_index, 1)} / 100")
    pdf.ln(3)

    # TABLEAU CLINIQUE
    pdf_subtitle(pdf, "Tableau clinique")
    pdf_table(pdf, table)

    # INTERPRÉTATION
    pdf_subtitle(pdf, "Interprétation orthoptique")
    for line in interpretation:
        pdf_text(pdf, f"- {safe_text(line)}")

    # RECOMMANDATIONS
    pdf_subtitle(pdf, "Recommandations orthoptiques")
    for r in recos:
        pdf_text(pdf, f"- {safe_text(r)}")
  
    # COMMENTAIRE ORTHOPTISTE
    pdf_subtitle(pdf, "Commentaire orthoptiste")
    last_comment = df_patient.iloc[-1].get("commentaire", "")

    if last_comment and str(last_comment).strip() != "":
        pdf_text(pdf, safe_text(last_comment))
    else:
        pdf_text(pdf, "Aucun commentaire enregistré pour cette séance.")

    # COURBES
    pdf.add_page()
    pdf_title(pdf, "Courbes d’évolution")

    save_curve(
        df_patient.set_index("session_number")["neuro_index"],
        "Évolution de l’index neurovisuel",
        "curve_index.png"
    )
    pdf.image("curve_index.png", w=180)
    pdf.ln(10)

    biom_cols = ["mean_rt", "variabilite", "cvrt", "impulsivity", "errors_inhibition"]

    for col in biom_cols:
        save_curve(
            df_patient.set_index("session_number")[col],
            f"Évolution : {col}",
            f"curve_{col}.png"
        )
        pdf.image(f"curve_{col}.png", w=180)
        pdf.ln(10)

    filename = "rapport_clinique.pdf"
    pdf.output(filename)

    return filename

# ===================================================
# OUTIL : COULEUR INDEX
# ===================================================

def get_index_color(neuro_index):
    if neuro_index >= 70:
        return (0, 150, 0)      # vert
    elif neuro_index >= 40:
        return (200, 140, 0)    # orange
    else:
        return (180, 0, 0)      # rouge

# ===================================================
# EXPORT PDF ORTHO PRO
# ===================================================

def export_ortho_pro_pdf(df_patient, table, interpretation, recos, neuro_index):
    pdf = FPDF()
    pdf.add_page()

    # Police Unicode
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.add_font("DejaVu", "B", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", "", 12)

    # ============================
    # PAGE 1 : SYNTHÈSE CLINIQUE
    # ============================

    # Titre
    pdf.set_font("DejaVu", "B", 18)
    pdf.set_text_color(*COLOR_PRIMARY)
    pdf.cell(0, 12, "Bilan neurovisuel – Synthèse clinique", ln=True, align="C")
    pdf.ln(5)

    # Identité patient
    last = df_patient.iloc[-1]
    pdf.set_font("DejaVu", "", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"ID patient : {safe_text(last['patient_id'])}", ln=True)
    pdf.cell(0, 8, f"Âge : {safe_text(last['age'])} ans", ln=True)
    pdf.cell(0, 8, f"Statut TDAH : {safe_text(last['tdah'])}", ln=True)
    pdf.ln(5)

    # Index global (gros, centré, couleur)
    color_index = get_index_color(neuro_index)
    pdf.set_font("DejaVu", "B", 22)
    pdf.set_text_color(*color_index)
    pdf.cell(0, 14, f"Index neurovisuel global : {round(neuro_index, 1)} / 100", ln=True, align="C")
    pdf.ln(4)

    # Phrase de synthèse simple
    pdf.set_font("DejaVu", "", 12)
    pdf.set_text_color(0, 0, 0)
    if neuro_index >= 70:
        synthese = "Profil attentionnel global satisfaisant."
    elif neuro_index >= 40:
        synthese = "Profil attentionnel global intermédiaire, avec fragilités modérées."
    else:
        synthese = "Profil attentionnel global fragile, nécessitant une prise en charge soutenue."
    pdf.multi_cell(0, 8, safe_text(synthese))
    pdf.ln(4)

    # Interprétation clinique (version courte)
    pdf.set_font("DejaVu", "B", 14)
    pdf.set_text_color(*COLOR_SECONDARY)
    pdf.cell(0, 10, "Interprétation clinique", ln=True)
    pdf.ln(2)

    pdf.set_font("DejaVu", "", 12)
    pdf.set_text_color(0, 0, 0)
    for line in interpretation[:5]:
        pdf.multi_cell(0, 7, f"• {safe_text(line)}")
    pdf.ln(4)

    # ============================
    # PAGE 2 : RECOMMANDATIONS
    # ============================

    pdf.add_page()

    pdf.set_font("DejaVu", "B", 16)
    pdf.set_text_color(*COLOR_PRIMARY)
    pdf.cell(0, 10, "Recommandations orthoptiques", ln=True)
    pdf.ln(4)

    pdf.set_font("DejaVu", "", 12)
    pdf.set_text_color(0, 0, 0)
    if not recos:
        pdf.multi_cell(0, 7, "Aucune recommandation spécifique générée.")
    else:
        for r in recos[:8]:
            pdf.multi_cell(0, 7, f"• {safe_text(r)}")
            pdf.ln(1)

    # ============================
    # PAGE 3 : ÉVOLUTION
    # ============================

    if "session_number" in df_patient.columns and "neuro_index" in df_patient.columns:
        pdf.add_page()

        pdf.set_font("DejaVu", "B", 16)
        pdf.set_text_color(*COLOR_PRIMARY)
        pdf.cell(0, 10, "Évolution de l’index neurovisuel", ln=True)
        pdf.ln(4)

        save_curve(
            df_patient.set_index("session_number")["neuro_index"],
            "Évolution de l’index neurovisuel",
            "curve_index_ortho.png"
        )

        pdf.image("curve_index_ortho.png", w=180)
        pdf.ln(6)

        pdf.set_font("DejaVu", "", 12)
        pdf.set_text_color(0, 0, 0)
        if len(df_patient) >= 2:
            delta = df_patient.iloc[-1]["neuro_index"] - df_patient.iloc[0]["neuro_index"]
            if delta > 5:
                txt = f"Amélioration globale de l’index (+{round(delta,1)} points) depuis le début du suivi."
            elif delta < -5:
                txt = f"Baisse globale de l’index ({round(delta,1)} points) depuis le début du suivi."
            else:
                txt = "Index global relativement stable au cours des séances."
        else:
            txt = "Une seule séance disponible : évolution non interprétable."
        pdf.multi_cell(0, 7, safe_text(txt))

    filename = "bilan_orthoptiste_pro.pdf"
    pdf.output(filename)

    return filename
