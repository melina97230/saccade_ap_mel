import streamlit as st
import random
import time
import pandas as pd
import numpy as np
import os
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from export_pdf import (
    export_simple_pdf,
    export_full_pdf,
    export_ortho_pro_pdf
)

# ===================================================
# CONFIGURATION GÉNÉRALE
# ===================================================
st.set_page_config(
    page_title="IA Neurovisuelle TDAH – Version Premium",
    layout="wide"
)

os.makedirs("data", exist_ok=True)
FILE_PATH = "data/results.csv"

# ===================================================
# CHARGEMENT UNIQUE DU CSV + AJOUT COLONNE
# ===================================================
if os.path.exists(FILE_PATH):
    df = pd.read_csv(FILE_PATH)
else:
    df = pd.DataFrame(columns=[
        "orthoptiste_id", "patient_id", "session_number", "timestamp",
        "age", "tdah", "mode_reeduc",
        "score", "errors", "impulsivity", "errors_inhibition",
        "mean_rt", "variabilite", "trend", "cvrt",
        "difficulty", "neuro_index",
        "commentaire"
    ])

# Ajout automatique de la colonne orthoptiste_id si absente
if "orthoptiste_id" not in df.columns:
    df["orthoptiste_id"] = ""

# ===================================================
# CHOIX DU MODE
# ===================================================

# ===================================================
# IDENTIFIANT ORTHOPTISTE
# ===================================================
orthoptiste = st.sidebar.text_input("Identifiant orthoptiste (obligatoire)")

# 🔒 Menu différent pour toi et pour les autres orthoptistes
if orthoptiste == "melina_admin":
    modes = ["Patient", "Démo Patient", "Test Orthoptiste", "Orthoptiste", "Liste des patients"]
else:
    modes = ["Patient", "Démo Patient", "Test Orthoptiste", "Orthoptiste"]

mode = st.sidebar.selectbox("Choisir un mode", modes)

if not orthoptiste:
    st.sidebar.warning("Veuillez entrer votre identifiant orthoptiste.")
    st.stop()

# ===================================================
# THÈME VISUEL – STYLE APAISANT
# ===================================================
def green_box(text):
    st.markdown(
        f"""
        <div style="
            background-color:#E3F7E8;
            padding:15px;
            border-radius:10px;
            border-left:6px solid #4CAF50;
            font-size:18px;">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )

def section_title(title, emoji="🌿"):
    st.markdown(
        f"""
        <h2 style="color:#1B3A2E; margin-top:40px; margin-bottom:10px;">
            {emoji} {title}
        </h2>
        """,
        unsafe_allow_html=True
    )

# ===================================================
# STRUCTURE DU DATAFRAME
# ===================================================
COLUMNS = [
    "patient_id", "session_number", "timestamp",
    "age", "tdah", "mode_reeduc",
    "score", "errors", "impulsivity", "errors_inhibition",
    "mean_rt", "variabilite", "trend", "cvrt",
    "difficulty", "neuro_index",
    "commentaire"  
]

# ⚠️ IMPORTANT : on ne recharge plus le CSV ici
# df = pd.read_csv(FILE_PATH) if os.path.exists(FILE_PATH) else pd.DataFrame(columns=COLUMNS)

# ===================================================
# IDENTIFIANT PATIENT (RGPD)
# ===================================================
def hash_patient(pid, birth):
    if not pid or not birth:
        return None
    return f"{pid}_{birth.strftime('%Y%m%d')}"

# ===================================================
# ANALYSE DES TEMPS DE RÉACTION
# ===================================================
def analyse_attention(times):
    if len(times) < 3:
        return {"variabilite": 0.0, "trend": 0.0, "mean_rt": 0.0}

    t = np.array(times)
    return {
        "variabilite": float(np.std(t)),
        "trend": float(np.polyfit(range(len(t)), t, 1)[0]),
        "mean_rt": float(np.mean(t))
    }

# ===================================================
# GÉNÉRATION DES STIMULI
# ===================================================
def generate_stimuli_full():
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    digits = list("0123456789")
    shapes = ["▲", "■", "◆", "●", "⬟", "⬢"]
    symbols = ["@", "#", "&", "%", "§", "¤"]
    emojis = ["🔵", "🟣", "🟠", "🔺", "🔻", "⭐"]
    return letters + digits + shapes + symbols + emojis

# ===================================================
# GESTION DES ERREURS
# ===================================================
def process_click(clicked, start_time):
    rt = time.time() - start_time

    if clicked == "🟢":
        return rt, None, "🟢 **Bravo !**"

    if rt < 0.5:
        return rt, "inhibition", "⚠️ **Impulsivité + erreur d’inhibition !**"
    else:
        return rt, "attention", "⚠️ **Erreur attentionnelle !**"

def update_error_counters(type_erreur):
    if type_erreur == "attention":
        st.session_state.errors += 1
    elif type_erreur == "inhibition":
        st.session_state.errors += 1
        st.session_state.errors_inhibition += 1
        st.session_state.impulsivity += 1

# ===================================================
# BIOMARQUEURS
# ===================================================
def compute_biomarkers(state, errors, impulsivity, inhib_errors):
    mean_rt = max(state["mean_rt"], 0.01)

    cvrt = state["variabilite"] / mean_rt
    impuls_ratio = impulsivity / max(1, (errors + impulsivity + inhib_errors))
    inhib_ratio = inhib_errors / max(1, (errors + impulsivity + inhib_errors))
    attn_ratio = errors / max(1, (errors + impulsivity + inhib_errors))

    return {
        "cvrt": float(cvrt),
        "impuls_ratio": float(impuls_ratio),
        "inhib_ratio": float(inhib_ratio),
        "attn_ratio": float(attn_ratio)
    }

# ===================================================
# INDEX NEUROVISUEL
# ===================================================
def compute_neuro_index(state, errors, impulsivity, inhib_errors, biomarkers):

    attention = 1 / (1 + state["variabilite"] + abs(state["trend"]) + biomarkers["attn_ratio"])
    speed = 1 / (1 + max(state["mean_rt"], 0.01))
    impuls = 1 / (1 + biomarkers["impuls_ratio"])
    inhib = 1 / (1 + biomarkers["inhib_ratio"])
    stability = 1 / (1 + biomarkers["cvrt"])

    index = (
        attention * 0.28 +
        speed * 0.22 +
        impuls * 0.18 +
        inhib * 0.18 +
        stability * 0.14
    )

    return float(index * 100)

# ===================================================
# TABLEAU CLINIQUE
# ===================================================
def clinical_table(state, biomarkers, neuro_index):

    table = {
        "Variabilité (σ RT)": round(state["variabilite"], 3),
        "Tendance (drift)": round(state["trend"], 3),
        "Temps moyen (RT)": round(state["mean_rt"], 3),
        "CVRT": round(biomarkers["cvrt"], 3),
        "Erreurs attentionnelles (%)": round(biomarkers["attn_ratio"] * 100, 1),
        "Impulsivité (%)": round(biomarkers["impuls_ratio"] * 100, 1),
        "Inhibition (%)": round(biomarkers["inhib_ratio"] * 100, 1),
        "Index neurovisuel": round(neuro_index, 1)
    }

    interpretation = []

    if state["variabilite"] > 0.35:
        interpretation.append("• Variabilité élevée → instabilité attentionnelle.")
    elif state["variabilite"] < 0.15:
        interpretation.append("• Variabilité faible → bonne stabilité attentionnelle.")

    if biomarkers["impuls_ratio"] > 0.25:
        interpretation.append("• Impulsivité marquée → difficulté de contrôle inhibiteur.")
    elif biomarkers["impuls_ratio"] < 0.10:
        interpretation.append("• Impulsivité faible → bon contrôle inhibiteur.")

    if biomarkers["inhib_ratio"] > 0.20:
        interpretation.append("• Erreurs d’inhibition élevées → difficulté à inhiber les distracteurs.")
    elif biomarkers["inhib_ratio"] < 0.10:
        interpretation.append("• Inhibition correcte.")

    if biomarkers["attn_ratio"] > 0.30:
        interpretation.append("• Erreurs attentionnelles fréquentes → distractibilité importante.")
    elif biomarkers["attn_ratio"] < 0.10:
        interpretation.append("• Bonne attention sélective.")

    if state["mean_rt"] > 1.2:
        interpretation.append("• Temps de réaction lent → ralentissement cognitif.")
    elif state["mean_rt"] < 0.6:
        interpretation.append("• Temps de réaction rapide.")

    if neuro_index < 40:
        interpretation.append("• Index faible → profil attentionnel fragile.")
    elif neuro_index > 70:
        interpretation.append("• Index élevé → bon fonctionnement attentionnel global.")

    return table, interpretation

# ===================================================
# RECOMMANDATIONS ORTHOPTIQUES
# ===================================================
def clinical_recommendations(state, biomarkers, neuro_index):
    recos = []

    if biomarkers["attn_ratio"] > 0.30:
        recos.append("• Renforcer l’attention sélective : discrimination visuelle, Go/No-Go.")
    elif biomarkers["attn_ratio"] < 0.10:
        recos.append("• Attention sélective satisfaisante.")

    if biomarkers["impuls_ratio"] > 0.25:
        recos.append("• Travailler le contrôle inhibiteur : Stop-Signal, délai de réponse.")
    elif biomarkers["impuls_ratio"] < 0.10:
        recos.append("• Contrôle inhibiteur efficace.")

    if biomarkers["inhib_ratio"] > 0.20:
        recos.append("• Rééducation de l’inhibition : exercices avec distracteurs.")
    elif biomarkers["inhib_ratio"] < 0.10:
        recos.append("• Inhibition correcte.")

    if state["variabilite"] > 0.35:
        recos.append("• Stabilisation attentionnelle : exercices rythmiques.")
    elif state["variabilite"] < 0.15:
        recos.append("• Bonne stabilité attentionnelle.")

    if state["mean_rt"] > 1.2:
        recos.append("• Améliorer la vitesse de traitement : exercices chronométrés.")
    elif state["mean_rt"] < 0.6:
        recos.append("• Vitesse correcte.")

    if neuro_index < 40:
        recos.append("• Programme intensif recommandé (2 séances/semaine).")
    elif neuro_index > 70:
        recos.append("• Suivi léger suffisant (1 séance/2 semaines).")

    return recos
# ===================================================
# PROFIL CLINIQUE SYNTHÉTIQUE
# ===================================================
def clinical_profile(df_patient):

    last = df_patient.iloc[-1]

    profile = f"""
    <b>Index neurovisuel :</b> {round(last['neuro_index'], 1)}<br><br>

    <b>Stabilité attentionnelle :</b> {round(last['variabilite'], 3)}<br>
    <b>Tendance (drift) :</b> {round(last['trend'], 3)}<br>
    <b>Temps moyen (RT) :</b> {round(last['mean_rt'], 3)} s<br><br>

    <b>Erreurs attentionnelles :</b> {last['errors']}<br>
    <b>Impulsivité :</b> {last['impulsivity']}<br>
    <b>Erreurs d’inhibition :</b> {last['errors_inhibition']}<br><br>

    <b>Difficulté actuelle :</b> {last['difficulty']}
    """

    return profile

def save_session(patient_id, age, tdah, mode_reeduc, state, biomarkers, difficulty, neuro_index):
    global df

    session_number = len(df[df["patient_id"] == patient_id]) + 1

    session = {
        "orthoptiste_id": orthoptiste,
        "patient_id": patient_id,
        "session_number": session_number,
        "timestamp": datetime.now().isoformat(),
        "age": age,
        "tdah": tdah,
        "mode_reeduc": mode_reeduc,
        "score": st.session_state.score,
        "errors": st.session_state.errors,
        "impulsivity": st.session_state.impulsivity,
        "errors_inhibition": st.session_state.errors_inhibition,
        "mean_rt": state["mean_rt"],
        "variabilite": state["variabilite"],
        "trend": state["trend"],
        "cvrt": biomarkers["cvrt"],
        "difficulty": difficulty,
        "neuro_index": neuro_index,
        "commentaire": st.session_state.get("commentaire", "")
    }

    df = pd.concat([df, pd.DataFrame([session])], ignore_index=True)
    df.to_csv(FILE_PATH, index=False)
    
def compute_difficulty(state, current_difficulty):
    """
    Ajuste la difficulté en fonction :
    - variabilité (CVRT)
    - impulsivité
    - erreurs d’inhibition
    - tendance (trend)
    """

    difficulty = current_difficulty

    # Variabilité élevée → difficulté diminue
    if state["variabilite"] > 0.35:
        difficulty -= 1

    # Impulsivité élevée → difficulté diminue
    if state["impulsivity"] > 0.25:
        difficulty -= 1

    # Erreurs d’inhibition → difficulté diminue
    if state.get("errors_inhibition", 0) > 2:
        difficulty -= 1

    # Bonne stabilité → difficulté augmente
    if state["variabilite"] < 0.20 and state["trend"] < 0:
        difficulty += 1

    # Bornes
    difficulty = max(3, min(10, difficulty))

    return difficulty
# ===================================================
# ENTRAÎNEMENT DU MODÈLE IA
# ===================================================
def train_neuro_model(df):

    # On vérifie qu'il y a assez de données
    if len(df) < 5:
        raise ValueError("Pas assez de données pour entraîner le modèle IA.")

    # Variables explicatives
    features = [
        "mean_rt", "variabilite", "cvrt", "impulsivity",
        "errors_inhibition", "errors", "difficulty", "age"
    ]

    # On retire les lignes incomplètes
    df_train = df.dropna(subset=features + ["neuro_index"]).copy()

    X = df_train[features]
    y = df_train["neuro_index"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modèle IA
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Sauvegarde
    joblib.dump(model, "data/model_neuro.pkl")

    return model
# ===================================================
# SESSION STATE – INITIALISATION
# ===================================================

defaults = {
    "started": 0,
    "score": 0,
    "errors": 0,
    "impulsivity": 0,
    "errors_inhibition": 0,
    "reaction_times": [],
    "start_time": None,
    "end_time": None,
    "difficulty": 5,
    "target": None,
    "force_end": False
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ===================================================
# MODE PATIENT (VERSION LÉGÈRE)
# ===================================================

if mode == "Patient":

    section_title("Mode Patient", "🎮")

    # -----------------------------
    # IDENTITÉ PATIENT
    # -----------------------------
    pid = st.text_input("Identifiant patient")
    birth = st.date_input("Date de naissance")
    age = st.number_input("Âge", 5, 99, 10)
    tdah = st.selectbox("Statut TDAH", ["Non", "Suspicion", "Diagnostiqué"])

    patient_id = hash_patient(pid, birth)

    # -----------------------------
    # PATIENT EXISTANT OU NOUVEAU
    # -----------------------------
    if pid and birth:
        if patient_id in df[df["orthoptiste_id"] == orthoptiste]["patient_id"].values:
            green_box("🟢 Patient trouvé. Prêt pour une nouvelle séance.")
        else:
            green_box("🟡 Nouveau patient. Aucune séance enregistrée.")

    # -----------------------------
    # RÉCUPÉRATION DIFFICULTÉ
    # -----------------------------
    if patient_id in df["patient_id"].values:
        subset = df[(df["patient_id"] == patient_id) & (df["orthoptiste_id"] == orthoptiste)]
        if len(subset) > 0:   # ← AJOUT SÉCURITÉ
            last = subset.sort_values("session_number").iloc[-1]
            st.session_state.difficulty = int(last["difficulty"])
        else:
            st.session_state.difficulty = 5
    else:
        st.session_state.difficulty = 5

    # -----------------------------
    # CHOIX MODE RÉÉDUCATION
    # -----------------------------
    mode_reeduc = st.selectbox("Type de rééducation", ["Standard (sans IA)", "Adaptative IA"])

    # -----------------------------
    # DURÉE DE LA SÉANCE
    # -----------------------------
    duree = st.selectbox("Durée de la séance", ["30 secondes", "1 minute", "2 minutes", "3 minutes"])
    duree_map = {"30 secondes": 30, "1 minute": 60, "2 minutes": 120, "3 minutes": 180}
    duree_sec = duree_map[duree]

    # -----------------------------
    # DÉMARRAGE
    # -----------------------------
    if st.button("🚀 Démarrer la séance"):
        st.session_state.started = 1
        st.session_state.start_time = time.time()
        st.session_state.end_time = time.time() + duree_sec
        st.session_state.force_end = False

# ===================================================
# LANCEMENT DU JEU
# ===================================================

if st.session_state.started and mode == "Patient":

    # -----------------------------
    # TIMER – TEMPS RESTANT
    # -----------------------------
    temps_restant = int(st.session_state.end_time - time.time())

    if temps_restant <= 0:
        st.session_state.force_end = True
        st.warning("⏳ Temps écoulé !")
    else:
        minutes = temps_restant // 60
        secondes = temps_restant % 60
        st.markdown(f"### ⏳ Temps restant : **{minutes:02d}:{secondes:02d}**")

    # ===================================================
    # MODE SANS IA – DIFFICULTÉ FIXE
    # ===================================================
    if mode_reeduc == "Standard (sans IA)":

        st.markdown("### 🎯 Consigne")
        st.markdown("""
Clique UNIQUEMENT sur la boule verte 🟢 dès qu’elle apparaît.  
Ignore les autres symboles, lettres ou formes.  
""")

        difficulty = st.number_input("Difficulté (fixe)", 3, 10, 5)

        # -----------------------------
        # AFFICHAGE DES STIMULI
        # -----------------------------
        target_index = random.randint(0, difficulty - 1)
        cols = st.columns(difficulty)
        clicked = None

        for i in range(difficulty):
            with cols[i]:
                if i == target_index:
                    if st.button("🟢", key=f"stim_noIA_green_{i}"):
                        clicked = "🟢"
                else:
                    distractor = random.choice(generate_stimuli_full())
                    if st.button(distractor, key=f"stim_noIA_dist_{i}"):
                        clicked = distractor

        # -----------------------------
        # GESTION DU CLIC
        # -----------------------------
        if clicked is not None:

            rt, type_erreur, feedback = process_click(clicked, st.session_state.start_time)

            if type_erreur is None:
                st.session_state.reaction_times.append(rt)
                st.session_state.score += 1
            else:
                update_error_counters(type_erreur)

            st.session_state.feedback = feedback
            st.session_state.start_time = time.time()

        if "feedback" in st.session_state:
            st.markdown(f"### {st.session_state.feedback}")
        # -----------------------------
        # ANALYSE + INDEX
        # -----------------------------
        analysis = analyse_attention(st.session_state.reaction_times)
        state = {
            **analysis,
            "errors": st.session_state.errors,
            "impulsivity": st.session_state.impulsivity
}

        biomarkers = compute_biomarkers(
            state,
            state["errors"],
            state["impulsivity"],
            st.session_state.errors_inhibition
)

        neuro_index = compute_neuro_index(
            state,
            state["errors"],
            state["impulsivity"],
            st.session_state.errors_inhibition,
            biomarkers
)

        # -----------------------------
        # FIN AUTOMATIQUE
        # -----------------------------
        if st.session_state.get("force_end", False):
            st.info("⏳ Temps écoulé — séance terminée.")
            if st.button("Enregistrer la séance maintenant"):
                save_session(patient_id, age, tdah, mode_reeduc, state, biomarkers, difficulty, neuro_index)
                st.success("Séance enregistrée avec succès !")
                st.rerun()

        # -----------------------------
        # BOUTON : TERMINER LA SÉANCE
        # -----------------------------
        if st.button("Terminer la séance"):
            save_session(patient_id, age, tdah, mode_reeduc, state, biomarkers, difficulty, neuro_index)
            st.success("Séance enregistrée avec succès !")
            st.rerun()

    # ===================================================
    # MODE ADAPTATIF IA – DIFFICULTÉ DYNAMIQUE
    # ===================================================
    elif mode_reeduc == "Adaptative IA":

        st.markdown("### 🤖 Mode Adaptatif IA")
        st.markdown("""
L’IA ajuste automatiquement la difficulté en fonction :
- de la variabilité des temps de réaction  
- de l’impulsivité  
- des erreurs d’inhibition  
- de la stabilité attentionnelle  
""")

        # Analyse en temps réel
        analysis = analyse_attention(st.session_state.reaction_times)
        state = {
            **analysis,
            "errors": st.session_state.errors,
            "impulsivity": st.session_state.impulsivity
        }

        biomarkers = compute_biomarkers(
            state,
            state["errors"],
            state["impulsivity"],
            st.session_state.errors_inhibition
        )
        neuro_index = compute_neuro_index(
    state,
    state["errors"],
    state["impulsivity"],
    st.session_state.errors_inhibition,
    biomarkers
)

        # Calcul de la difficulté adaptative
        st.session_state.difficulty = compute_difficulty(state, st.session_state.difficulty)
        difficulty = int(st.session_state.difficulty)

        st.caption(f"Difficulté adaptative : **{difficulty}**")

        # -----------------------------
        # AFFICHAGE DES STIMULI
        # -----------------------------
        target_index = random.randint(0, difficulty - 1)
        cols = st.columns(difficulty)
        clicked = None

        for i in range(difficulty):
            with cols[i]:
                if i == target_index:
                    if st.button("🟢", key=f"stim_IA_green_{i}"):
                        clicked = "🟢"
                else:
                    distractor = random.choice(generate_stimuli_full())
                    if st.button(distractor, key=f"stim_IA_dist_{i}"):
                        clicked = distractor

        # -----------------------------
        # GESTION DU CLIC
        # -----------------------------
        if clicked is not None:

            rt, type_erreur, feedback = process_click(clicked, st.session_state.start_time)

            if type_erreur is None:
                st.session_state.reaction_times.append(rt)
                st.session_state.score += 1
            else:
                update_error_counters(type_erreur)

            st.session_state.feedback = feedback
            st.session_state.start_time = time.time()

        if "feedback" in st.session_state:
            st.markdown(f"### {st.session_state.feedback}")

        # -----------------------------
        # FIN AUTOMATIQUE
        # -----------------------------
        if st.session_state.get("force_end", False):
            if st.button("Enregistrer la séance maintenant"):
               save_session(patient_id, age, tdah, mode_reeduc, state, biomarkers, difficulty, neuro_index)
               st.success("Séance enregistrée avec succès !")
               st.rerun()
# ===================================================
# MODE DÉMO PATIENT
# ===================================================

elif mode == "Démo Patient":

    section_title("Démo Patient", "🎬")

    st.markdown("Cette démo génère automatiquement une séance complète pour illustrer le fonctionnement.")

    if st.button("Lancer une démo automatique"):

        demo_id = "DEMO_PATIENT"
        reaction_times = []
        errors = 0
        impulsivity = 0
        inhib_errors = 0
        difficulty = 5

        for _ in range(20):

            rt = random.uniform(0.4, 1.2)
            reaction_times.append(rt)

            if random.random() < 0.2:
                errors += 1
                if rt < 0.5:
                    impulsivity += 1
                inhib_errors += 1

            state = analyse_attention(reaction_times)
            state["errors"] = errors
            state["impulsivity"] = impulsivity
            biomarkers = compute_biomarkers(state, errors, impulsivity, inhib_errors)
            difficulty = compute_difficulty(state, difficulty)

        neuro_index = compute_neuro_index(state, errors, impulsivity, inhib_errors, biomarkers)

        session_number = len(df[df["patient_id"] == demo_id]) + 1

        session = {
            "patient_id": demo_id,
            "session_number": session_number,
            "timestamp": datetime.now().isoformat(),
            "age": 12,
            "tdah": "Diagnostiqué",
            "mode_reeduc": "Adaptative IA",
            "score": len(reaction_times),
            "errors": errors,
            "impulsivity": impulsivity,
            "errors_inhibition": inhib_errors,
            "mean_rt": state["mean_rt"],
            "variabilite": state["variabilite"],
            "trend": state["trend"],
            "cvrt": biomarkers["cvrt"],
            "difficulty": difficulty,
            "neuro_index": neuro_index
        }

        df = pd.concat([df, pd.DataFrame([session])], ignore_index=True)
        df.to_csv(FILE_PATH, index=False)

        st.success("Démo générée ! Consultez l’onglet Orthoptiste.")

# ===================================================
# MODE TEST ORTHOPTISTE
# ===================================================

elif mode == "Test Orthoptiste":

    section_title("Mode Test Orthoptiste", "🧪")

    st.markdown("Ce test permet d’évaluer rapidement l’attention et l’inhibition.")

    if st.button("Démarrer le test"):
        st.session_state.started = 1
        st.session_state.start_time = time.time()
        st.session_state.test_clicks = 0
        st.session_state.difficulty = 5
        st.session_state.target = None
        st.session_state.reaction_times = []
        st.session_state.errors = 0
        st.session_state.impulsivity = 0
        st.session_state.errors_inhibition = 0

    if st.session_state.started:

        analysis = analyse_attention(st.session_state.reaction_times)
        state = {
            **analysis,
            "errors": st.session_state.errors,
            "impulsivity": st.session_state.impulsivity
        }
        biomarkers = compute_biomarkers(
            state,
            st.session_state.errors,
            st.session_state.impulsivity,
            st.session_state.errors_inhibition
        )

        st.session_state.difficulty = compute_difficulty(state, st.session_state.difficulty)
        difficulty = int(st.session_state.difficulty)

        st.caption(f"Difficulté actuelle : **{difficulty}**")

        if st.session_state.target is None or st.session_state.target >= difficulty:
            st.session_state.target = random.randint(0, difficulty - 1)

        cols = st.columns(difficulty)
        stimuli = generate_stimuli_full()

        for i in range(difficulty):
            with cols[i]:

                if i == st.session_state.target:
                    if st.button("🟢", key=f"test_target_{i}"):

                        rt = time.time() - st.session_state.start_time
                        st.session_state.reaction_times.append(rt)
                        st.session_state.test_clicks += 1

                        st.session_state.target = None
                        st.session_state.start_time = time.time()

                        if st.session_state.test_clicks >= 10:
                            st.success("Test terminé ! Merci.")
                            st.session_state.started = 0
                        else:
                            st.rerun()

                else:
                    symbol = random.choice(stimuli)
                    if st.button(symbol, key=f"test_distractor_{i}"):

                        st.session_state.errors += 1
                        rt = time.time() - st.session_state.start_time
                        if rt < 0.5:
                            st.session_state.impulsivity += 1
                        st.session_state.errors_inhibition += 1
                        
# ===================================================
# MODE : LISTE DES PATIENTS
# ===================================================

elif mode == "Liste des patients":
    
    # 🔒 Sécurité interne : accès réservé
    if orthoptiste != "melina_admin":
        st.error("⛔ Accès réservé. Cette section est uniquement pour l’administratrice.")
        st.stop()


    section_title("📋 Liste des patients enregistrés", "📋")

    # Charger le CSV
    try:
        df = pd.read_csv(FILE_PATH)
    except:
        st.error("Aucune donnée patient trouvée.")
        st.stop()

    if df.empty:
        st.info("Aucun patient enregistré pour le moment.")
        st.stop()

    # Regrouper par patient
    patients = df.groupby("patient_id").agg(
        nb_seances=("session_number", "count"),
        derniere_seance=("timestamp", "max")
    ).reset_index()

    st.subheader("Patients enregistrés")
    st.dataframe(patients)

    # Sélection du patient
    choix = st.selectbox("Choisir un patient", patients["patient_id"].unique())

    if st.button("Ouvrir le dossier"):
        st.session_state["patient_id"] = choix
        st.success(f"Dossier du patient {choix} chargé.")
        st.info("Rendez-vous dans l’onglet **Orthoptiste** pour voir les séances.")

# ===================================================
# MODE ORTHOPTISTE – DASHBOARD PRO
# ===================================================

elif mode == "Orthoptiste":

    section_title("Dashboard Orthoptiste", "🧠")

    if len(df) == 0:
        st.warning("Aucune donnée disponible.")
        st.stop()

    # -----------------------------
    # ENTRAÎNEMENT IA
    # -----------------------------
    st.markdown("### 🤖 Entraîner / Mettre à jour le modèle IA")

    if st.button("Entraîner l’IA maintenant"):
        model = train_neuro_model(df)
        st.success("Modèle IA entraîné et sauvegardé !")

    st.markdown("---")

    # -----------------------------
    # SÉLECTION PATIENT
    # -----------------------------
    st.markdown("### 👤 Sélection du patient")
    # Filtrer les données de l’orthoptiste connecté
    df_ortho = df[df["orthoptiste_id"] == orthoptiste]

    if len(df_ortho) == 0:
        st.warning("Aucun patient associé à cet orthoptiste.")
        st.stop()

    patient = st.selectbox("Patient", df_ortho["patient_id"].unique())
    p = df_ortho[df_ortho["patient_id"] == patient].sort_values("session_number")

    last = p.iloc[-1]

    st.markdown("---")

    # ===================================================
    # PROFIL CLINIQUE SYNTHÉTIQUE
    # ===================================================
    section_title("Profil clinique synthétique", "🧬")
    green_box(clinical_profile(p))

    st.metric("Index neurovisuel", round(last["neuro_index"], 2))
    st.metric("Difficulté actuelle", int(last["difficulty"]))

    st.markdown("---")

    # ===================================================
    # HISTORIQUE + ANALYSES CLINIQUES
    # ===================================================
    st.markdown("### 🕒 Dernière séance")
    st.write(last)

    st.markdown("### 📊 Progression de l’index neurovisuel")
    st.line_chart(p.set_index("session_number")["neuro_index"])

    st.markdown("### 🧪 Biomarqueurs attentionnels")
    biom_cols = ["mean_rt", "variabilite", "cvrt", "impulsivity", "errors_inhibition"]
    st.line_chart(p.set_index("session_number")[biom_cols])

    st.markdown("---")

    # ===================================================
    # COMMENTAIRE ORTHOPTISTE
    # ===================================================
    st.markdown("### 📝 Commentaire orthoptiste")
    commentaire = st.text_area("Notes cliniques / observations", "")
    st.session_state.commentaire = commentaire

    st.markdown("---")

    # ===================================================
    # ANALYSE CLINIQUE DÉTAILLÉE
    # ===================================================
    st.markdown("### 🧾 Tableau clinique (dernière séance)")

    state = {
        "variabilite": last["variabilite"],
        "trend": last["trend"],
        "mean_rt": last["mean_rt"],
        "errors": last["errors"],
        "impulsivity": last["impulsivity"]
    }

    biomarkers = compute_biomarkers(
        state,
        last["errors"],
        last["impulsivity"],
        last["errors_inhibition"]
    )

    neuro_index = last["neuro_index"]

    table, interpretation = clinical_table(state, biomarkers, neuro_index)

    st.table(table)

    st.markdown("### 📝 Commentaire de la dernière séance")
    st.write(last["commentaire"])

    st.markdown("### 🩺 Interprétation orthoptique")
    for line in interpretation:
        st.markdown(line)

    st.markdown("### 🧭 Recommandations orthoptiques")
    recos = clinical_recommendations(state, biomarkers, neuro_index)
    for r in recos:
        st.markdown(r)

    st.markdown("---")

    # ===================================================
    # EXPORT PDF
    # ===================================================
    st.markdown("### 📄 Export PDF")

    if st.button("📄 Export PDF simple"):
        export_simple_pdf(p, table, interpretation, neuro_index)

    if st.button("📘 Rapport clinique complet (PDF)"):
        export_full_pdf(p, table, interpretation, recos, neuro_index)

    if st.button("📄 Générer le PDF Orthoptiste PRO"):
        filename = export_ortho_pro_pdf(
            p, 
            table, 
            interpretation, 
            recos, 
            neuro_index
        )
        with open(filename, "rb") as f:
            st.download_button(
                label="Télécharger le PDF Orthoptiste PRO",
                data=f,
                file_name=filename,
                mime="application/pdf"
            )

    st.markdown("---")

    # ===================================================
    # PRÉDICTION IA
    # ===================================================
    st.markdown("### 🔮 Prédiction IA de la prochaine séance")

    try:
        model = joblib.load("data/model_neuro.pkl")

        features = [
            "mean_rt", "variabilite", "cvrt", "impulsivity",
            "errors_inhibition", "errors", "difficulty", "age"
        ]

        X_last = last[features].values.reshape(1, -1)
        pred_next = model.predict(X_last)[0]

        st.metric("Index neurovisuel prédit", round(pred_next, 1))

    except:
        st.warning("⚠️ Le modèle IA n’est pas encore entraîné. Cliquez sur 'Entraîner l’IA'.")
