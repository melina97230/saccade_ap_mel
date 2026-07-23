import streamlit as st
import random
import time
import pandas as pd
import numpy as np
import os
import hashlib
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestRegressor
from streamlit_autorefresh import st_autorefresh
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
# STRUCTURE DU DATAFRAME
# ===================================================
COLUMNS = [
    "orthoptiste_id", "patient_id", "session_number", "timestamp",
    "age", "tdah", "mode_reeduc",
    "score", "errors",
    "commission_errors", "omissions", "anticipations",
    "targets_presented",
    "impulsivity", "errors_inhibition",
    "mean_rt", "median_rt", "variabilite", "trend", "cvrt",
    "omission_rate", "commission_rate", "anticipation_rate",
    "difficulty", "neuro_index",
    "commentaire"
]

# ===================================================
# CHARGEMENT UNIQUE DU CSV + AJOUT COLONNE
# ===================================================
if os.path.exists(FILE_PATH):
    df = pd.read_csv(FILE_PATH)
else:
    df = pd.DataFrame(columns=COLUMNS)

# Ajout automatique des colonnes absentes pour relire les anciennes séances.
for column in COLUMNS:
    if column not in df.columns:
        df[column] = np.nan

df["orthoptiste_id"] = df["orthoptiste_id"].fillna("").astype(str)
df["commentaire"] = df["commentaire"].fillna("").astype(str)

# Compatibilité avec les anciens noms de compteurs.
df["commission_errors"] = df["commission_errors"].fillna(df["errors_inhibition"])
df["anticipations"] = df["anticipations"].fillna(df["impulsivity"])
df["omissions"] = df["omissions"].fillna(0)
df["median_rt"] = df["median_rt"].fillna(df["mean_rt"])
df["targets_presented"] = df["targets_presented"].fillna(
    df["score"].fillna(0) + df["errors"].fillna(0)
)

targets_denominator = pd.to_numeric(
    df["targets_presented"], errors="coerce"
).fillna(0).clip(lower=1)
df["omission_rate"] = df["omission_rate"].fillna(
    pd.to_numeric(df["omissions"], errors="coerce").fillna(0)
    / targets_denominator
)
df["commission_rate"] = df["commission_rate"].fillna(
    pd.to_numeric(df["commission_errors"], errors="coerce").fillna(0)
    / targets_denominator
)
df["anticipation_rate"] = df["anticipation_rate"].fillna(
    pd.to_numeric(df["anticipations"], errors="coerce").fillna(0)
    / targets_denominator
)

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
# IDENTIFIANT PATIENT (RGPD)
# ===================================================
def hash_patient(pid, birth):
    if not pid or not birth:
        return None
    raw_identifier = (
        f"{orthoptiste.strip().casefold()}|"
        f"{pid.strip().casefold()}|{birth.strftime('%Y%m%d')}"
    )
    digest = hashlib.sha256(raw_identifier.encode("utf-8")).hexdigest()[:16]
    return f"P-{digest.upper()}"

# ===================================================
# ANALYSE DES TEMPS DE RÉACTION
# ===================================================
def analyse_attention(times):
    valid_times = [
        float(value) for value in times
        if value is not None and np.isfinite(value) and value >= 0.20
    ]

    if len(valid_times) < 3:
        return {
            "variabilite": 0.0,
            "trend": 0.0,
            "mean_rt": float(np.mean(valid_times)) if valid_times else 0.0,
            "median_rt": float(np.median(valid_times)) if valid_times else 0.0,
            "n_rt": len(valid_times)
        }

    t = np.array(valid_times)
    return {
        "variabilite": float(np.std(t)),
        "trend": float(np.polyfit(range(len(t)), t, 1)[0]),
        "mean_rt": float(np.mean(t)),
        "median_rt": float(np.median(t)),
        "n_rt": len(valid_times)
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
    return letters + digits+ emojis + shapes + symbols 

# ===================================================
# GESTION DES ERREURS
# ===================================================
def process_click(clicked, start_time):
    rt = time.perf_counter() - start_time

    if clicked == "🟢":
        if rt < 0.20:
            return rt, "anticipation", "⚠️ **Réponse très anticipée !**"
        return rt, None, "🟢 **Bravo !**"

    if rt < 0.5:
        return rt, "commission_anticipation", "⚠️ **Clic anticipé sur un distracteur !**"
    return rt, "commission", "⚠️ **Distracteur sélectionné !**"

def update_error_counters(type_erreur):
    if type_erreur == "anticipation":
        st.session_state.errors += 1
        st.session_state.anticipations += 1
        st.session_state.impulsivity += 1
    elif type_erreur in {"commission", "commission_anticipation"}:
        st.session_state.errors += 1
        st.session_state.errors_inhibition += 1
        st.session_state.commission_errors += 1
        if type_erreur == "commission_anticipation":
            st.session_state.anticipations += 1
            st.session_state.impulsivity += 1


def prepare_stimulus_round(difficulty):
    """Conserve la même cible et les mêmes distracteurs jusqu'au clic."""
    difficulty = int(difficulty)
    current = st.session_state.get("stimuli", [])

    if st.session_state.get("target") is None or len(current) != difficulty:
        target_index = random.randint(0, difficulty - 1)
        pool = generate_stimuli_full()
        st.session_state.stimuli = [
            "🟢" if i == target_index else random.choice(pool)
            for i in range(difficulty)
        ]
        st.session_state.target = target_index
        st.session_state.stimulus_round += 1
        st.session_state.stimulus_start_time = time.perf_counter()


def complete_stimulus_round():
    """Force une nouvelle série de stimuli au prochain rerun."""
    st.session_state.target = None
    st.session_state.stimuli = []
    st.session_state.stimulus_start_time = None


def check_omission(max_response_time=3.0):
    """Enregistre une cible non sélectionnée dans le délai prévu."""
    start_time = st.session_state.get("stimulus_start_time")
    if start_time is None:
        return False

    if time.perf_counter() - start_time < max_response_time:
        return False

    st.session_state.targets_presented += 1
    st.session_state.omissions += 1
    st.session_state.errors += 1
    st.session_state.feedback = "⏳ **Cible non sélectionnée dans le délai.**"
    complete_stimulus_round()
    return True

# ===================================================
# BIOMARQUEURS
# ===================================================
def compute_biomarkers(
    state,
    errors,
    impulsivity,
    inhib_errors,
    correct_responses=0,
    omissions=None,
    commission_errors=None,
    anticipations=None,
    targets_presented=None,
):
    mean_rt = max(state["mean_rt"], 0.01)

    cvrt = state["variabilite"] / mean_rt
    total_responses = max(1, int(correct_responses) + int(errors))
    attention_errors = max(0, int(errors) - int(inhib_errors))
    impuls_ratio = impulsivity / total_responses
    inhib_ratio = inhib_errors / total_responses
    attn_ratio = attention_errors / total_responses

    if omissions is None:
        omissions = st.session_state.get("omissions", 0)
    if commission_errors is None:
        commission_errors = st.session_state.get(
            "commission_errors", inhib_errors
        )
    if anticipations is None:
        anticipations = st.session_state.get("anticipations", impulsivity)
    if targets_presented is None:
        targets_presented = st.session_state.get(
            "targets_presented", total_responses
        )

    denominator = max(1, int(targets_presented))

    return {
        "cvrt": float(cvrt),
        "impuls_ratio": float(impuls_ratio),
        "inhib_ratio": float(inhib_ratio),
        "attn_ratio": float(attn_ratio),
        "omission_rate": float(omissions) / denominator,
        "commission_rate": float(commission_errors) / denominator,
        "anticipation_rate": float(anticipations) / denominator,
    }

# ===================================================
# INDEX NEUROVISUEL
# ===================================================
def compute_neuro_index(
    state,
    biomarkers,
    min_valid_targets=5,
    targets_presented=None,
):
    """
    Indice comportemental exploratoire sur 100.

    Il synthétise quatre dimensions :
    - stabilité des temps de réaction ;
    - vigilance, estimée par les omissions ;
    - contrôle de la réponse, estimé par les commissions ;
    - réponses anticipées.

    Ce score n'est ni diagnostique ni cliniquement validé.
    """

    if targets_presented is None:
        targets_presented = st.session_state.get("targets_presented", 0)
    targets_presented = int(targets_presented)

    if targets_presented < min_valid_targets:
        return np.nan

    stability_score = 1.0 / (
        1.0 + max(float(biomarkers["cvrt"]), 0.0)
    )

    vigilance_score = 1.0 - float(
        biomarkers["omission_rate"]
    )

    response_control_score = 1.0 - float(
        biomarkers["commission_rate"]
    )

    anticipation_control_score = 1.0 - float(
        biomarkers["anticipation_rate"]
    )

    components = np.clip(
        [
            stability_score,
            vigilance_score,
            response_control_score,
            anticipation_control_score
        ],
        0.001,
        1.0
    )

    index = 100.0 * float(
        np.prod(components) ** (1.0 / len(components))
    )

    return round(index, 2)
# ===================================================
# TABLEAU CLINIQUE
# ===================================================
def clinical_table(state, biomarkers, neuro_index):

    table = {
    "Temps de réaction moyen (s)": round(state["mean_rt"], 3),
    "Temps de réaction médian (s)": round(state["median_rt"], 3),
    "Variabilité des RT – IIVRT (s)": round(state["variabilite"], 3),
    "Coefficient de variation – CVRT": round(biomarkers["cvrt"], 3),
    "Tendance des RT": round(state["trend"], 3),

    "Taux d’omissions (%)": round(
        biomarkers["omission_rate"] * 100, 1
    ),
    "Taux de commissions (%)": round(
        biomarkers["commission_rate"] * 100, 1
    ),
    "Taux de réponses anticipées (%)": round(
        biomarkers["anticipation_rate"] * 100, 1
    ),

    "Index comportemental neurovisuel": (
        "Données insuffisantes"
        if np.isnan(neuro_index)
        else round(neuro_index, 1)
    )
}

    interpretation = []

    if state["variabilite"] > 0.35:
        interpretation.append("• Variabilité élevée → instabilité attentionnelle.")
    elif state["variabilite"] < 0.15:
        interpretation.append("• Variabilité faible → bonne stabilité attentionnelle.")

    if biomarkers["omission_rate"] > 0.20:
        interpretation.append(
            "• Proportion élevée de cibles non sélectionnées dans le délai."
        )

    if biomarkers["commission_rate"] > 0.20:
        interpretation.append(
            "• Proportion élevée de clics sur les distracteurs "
            "au cours de cette séance."
        )

    if biomarkers["anticipation_rate"] > 0.10:
        interpretation.append(
            "• Présence de réponses très précoces au cours de cette séance."
        )

    if state["mean_rt"] > 1.2:
        interpretation.append(
            "• Temps de réaction moyen relativement élevé "
            "dans les conditions de cette séance."
        )
    elif state["mean_rt"] < 0.6:
        interpretation.append("• Temps de réaction rapide.")

    if not np.isnan(neuro_index):
        interpretation.append(
            "• Index exploratoire à interpréter avec les indicateurs détaillés "
            "et les observations de l’orthoptiste."
        )

    return table, interpretation

# ===================================================
# RECOMMANDATIONS ORTHOPTIQUES
# ===================================================
def clinical_recommendations(state, biomarkers, neuro_index):
    recos = []

    if biomarkers["omission_rate"] > 0.20:
        recos.append(
            "• Vérifier la compréhension de la consigne, la fatigabilité "
            "et le délai de réponse."
        )

    if biomarkers["commission_rate"] > 0.20:
        recos.append(
            "• Envisager un travail progressif de sélection visuelle "
            "et de résistance aux distracteurs."
        )

    if biomarkers["anticipation_rate"] > 0.10:
        recos.append(
            "• Renforcer la consigne d’attendre l’apparition et "
            "d’identifier la cible avant de cliquer."
        )

    if state["variabilite"] > 0.35:
        recos.append("• Stabilisation attentionnelle : exercices rythmiques.")
        recos.append(
            "• Comparer la stabilité des réponses aux séances précédentes "
            "avant de complexifier."
        )

    if not recos:
        recos.append(
            "• Maintenir les paramètres et confronter ces résultats "
            "aux observations cliniques."
        )

    recos.append(
        "• Ces pistes ne constituent pas une recommandation thérapeutique autonome."
    )

    return recos
# ===================================================
# PROFIL CLINIQUE SYNTHÉTIQUE
# ===================================================
def clinical_profile(df_patient):

    last = df_patient.iloc[-1]
    index_display = (
        "Données insuffisantes"
        if pd.isna(last["neuro_index"])
        else round(last["neuro_index"], 1)
    )

    profile = f"""
    <b>Index neurovisuel :</b> {index_display}<br><br>

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

    same_patient = df[
        (df["patient_id"] == patient_id)
        & (df["orthoptiste_id"] == orthoptiste)
    ]
    session_number = len(same_patient) + 1

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

    "commission_errors": st.session_state.commission_errors,
    "omissions": st.session_state.omissions,
    "anticipations": st.session_state.anticipations,
    "targets_presented": st.session_state.targets_presented,

    # Compatibilité ancienne structure
    "impulsivity": st.session_state.impulsivity,
    "errors_inhibition": st.session_state.errors_inhibition,

    "mean_rt": state["mean_rt"],
    "median_rt": state["median_rt"],
    "variabilite": state["variabilite"],
    "trend": state["trend"],

    "cvrt": biomarkers["cvrt"],
    "omission_rate": biomarkers["omission_rate"],
    "commission_rate": biomarkers["commission_rate"],
    "anticipation_rate": biomarkers["anticipation_rate"],

    "difficulty": difficulty,
    "neuro_index": neuro_index,
    "commentaire": st.session_state.get("commentaire", "")
}

    df = pd.concat([df, pd.DataFrame([session])], ignore_index=True)
    df = df.reindex(columns=COLUMNS)
    df.to_csv(FILE_PATH, index=False)
    
def compute_difficulty(state, biomarkers, current_difficulty):
    """
    Ajustement algorithmique explicite de la difficulté.

    L'algorithme ne pose aucun diagnostic.
    Il modifie uniquement le nombre de stimuli.
    """
    difficulty = int(current_difficulty)

    burden = 0
    favourable = 0

    # Fluctuation importante des temps de réaction
    if biomarkers["cvrt"] > 0.35:
        burden += 1

    # Cibles manquées
    if biomarkers["omission_rate"] > 0.20:
        burden += 1

    # Clics sur distracteurs
    if biomarkers["commission_rate"] > 0.20:
        burden += 1

    # Réponses très anticipées
    if biomarkers["anticipation_rate"] > 0.10:
        burden += 1

    # Performances relativement stables
    if (
        biomarkers["cvrt"] < 0.20
        and biomarkers["omission_rate"] < 0.10
        and biomarkers["commission_rate"] < 0.10
    ):
        favourable += 1

    if burden >= 2:
        difficulty -= 1
    elif favourable >= 1:
        difficulty += 1

    return max(3, min(10, difficulty))
# ===================================================
# ENTRAÎNEMENT DU MODÈLE IA
# ===================================================
MODEL_FEATURES = [
        "median_rt",
        "variabilite",
        "cvrt",
        "omission_rate",
        "commission_rate",
        "anticipation_rate",
        "difficulty",
        "age"
]

def model_file_for_current_orthoptiste():
    digest = hashlib.sha256(
        orthoptiste.strip().casefold().encode("utf-8")
    ).hexdigest()[:12]
    return f"data/model_neuro_{digest}.pkl"


def train_neuro_model(df_train_source):
    """
    Entraîne le modèle sur de vraies transitions :
    indicateurs de la séance n -> index de la séance n+1.
    """
    valid = df_train_source.dropna(
        subset=MODEL_FEATURES + ["neuro_index", "score"]
    ).copy()
    valid = valid[(valid["score"] >= 3) & (valid["mean_rt"] > 0)]

    transition_rows = []
    for _, patient_sessions in valid.groupby("patient_id"):
        patient_sessions = patient_sessions.sort_values(
            ["session_number", "timestamp"]
        )
        for position in range(len(patient_sessions) - 1):
            current = patient_sessions.iloc[position]
            following = patient_sessions.iloc[position + 1]
            row = {
                feature: current[feature]
                for feature in MODEL_FEATURES
            }
            row["next_neuro_index"] = following["neuro_index"]
            transition_rows.append(row)

    transition_df = pd.DataFrame(transition_rows)
    if len(transition_df) < 5:
        raise ValueError(
            "Pas assez de transitions valides pour entraîner le modèle IA "
            "(5 transitions entre deux séances sont nécessaires)."
        )

    X = transition_df[MODEL_FEATURES]
    y = transition_df["next_neuro_index"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42
    )

    model.fit(X, y)

    artifact = {
        "model": model,
        "features": MODEL_FEATURES,
        "transition_count": len(transition_df),
    }
    joblib.dump(artifact, model_file_for_current_orthoptiste())

    return artifact
# ===================================================
# SESSION STATE – INITIALISATION
# ===================================================

defaults = {
    "started": 0,
    "score": 0,
    "errors": 0,
    "impulsivity": 0,
    "errors_inhibition": 0,
    "commission_errors": 0,
    "omissions": 0,
    "anticipations": 0,
    "targets_presented": 0,
    "reaction_times": [],
    "start_time": None,
    "end_time": None,
    "difficulty": 5,
    "target": None,
    "force_end": False,
    "stimuli": [],
    "stimulus_round": 0,
    "stimulus_start_time": None,
    "last_adaptation_round": -1
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
    # Ne charger la difficulté enregistrée qu'avant le démarrage.
    # Pendant la séance, la valeur adaptative en mémoire doit rester prioritaire.
    if patient_id in df["patient_id"].values:
        subset = df[(df["patient_id"] == patient_id) & (df["orthoptiste_id"] == orthoptiste)]
        if len(subset) > 0:   # ← AJOUT SÉCURITÉ
            if not st.session_state.started:
                last = subset.sort_values("session_number").iloc[-1]
                st.session_state.difficulty = int(last["difficulty"])
        else:
            st.session_state.difficulty = 5
    else:
        if not st.session_state.started:
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
    if st.button(
        "🚀 Démarrer la séance",
        disabled=not bool(pid.strip()),
    ):
        st.session_state.started = 1
        st.session_state.start_time = time.time()
        st.session_state.end_time = time.time() + duree_sec
        st.session_state.force_end = False
        st.session_state.score = 0
        st.session_state.errors = 0
        st.session_state.impulsivity = 0
        st.session_state.errors_inhibition = 0
        st.session_state.commission_errors = 0
        st.session_state.omissions = 0
        st.session_state.anticipations = 0
        st.session_state.targets_presented = 0
        st.session_state.reaction_times = []
        st.session_state.feedback = ""
        st.session_state.target = None
        st.session_state.stimuli = []
        st.session_state.stimulus_round = 0
        st.session_state.stimulus_start_time = None
        st.session_state.last_adaptation_round = -1
        st.rerun()

# ===================================================
# LANCEMENT DU JEU
# ===================================================

if st.session_state.started and mode == "Patient":

    # Actualise automatiquement le minuteur et permet de comptabiliser
    # une omission même en l'absence de clic.
    st_autorefresh(interval=250, key="patient_session_refresh")

    # -----------------------------
    # TIMER – TEMPS RESTANT
    # -----------------------------
    temps_restant = int(st.session_state.end_time - time.time())

    if temps_restant <= 0:
        st.session_state.force_end = True
        complete_stimulus_round()
        st.warning("⏳ Temps écoulé !")
    else:
        minutes = temps_restant // 60
        secondes = temps_restant % 60
        st.markdown(f"### ⏳ Temps restant : **{minutes:02d}:{secondes:02d}**")

    if st.session_state.get("force_end", False):
        analysis = analyse_attention(st.session_state.reaction_times)
        state = {
            **analysis,
            "errors": st.session_state.errors,
            "impulsivity": st.session_state.impulsivity,
            "errors_inhibition": st.session_state.errors_inhibition,
        }
        biomarkers = compute_biomarkers(
            state,
            state["errors"],
            state["impulsivity"],
            st.session_state.errors_inhibition,
            st.session_state.score,
        )
        neuro_index = compute_neuro_index(state, biomarkers)
        difficulty = int(st.session_state.difficulty)

        st.info("⏳ Temps écoulé — séance terminée.")
        if st.button("Enregistrer la séance maintenant"):
            save_session(
                patient_id,
                age,
                tdah,
                mode_reeduc,
                state,
                biomarkers,
                difficulty,
                neuro_index,
            )
            st.session_state.started = 0
            st.success("Séance enregistrée avec succès !")
            st.rerun()
        st.stop()

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
        st.session_state.difficulty = int(difficulty)

        # -----------------------------
        # AFFICHAGE DES STIMULI
        # -----------------------------
        prepare_stimulus_round(difficulty)
        if check_omission():
            st.rerun()
        cols = st.columns(difficulty)
        clicked = None
        round_id = st.session_state.stimulus_round

        for i in range(difficulty):
            with cols[i]:
                stimulus = st.session_state.stimuli[i]
                if i == st.session_state.target:
                    if st.button("🟢", key=f"stim_noIA_green_{round_id}_{i}"):
                        clicked = "🟢"
                else:
                    if st.button(stimulus, key=f"stim_noIA_dist_{round_id}_{i}"):
                        clicked = stimulus

        # -----------------------------
        # GESTION DU CLIC
        # -----------------------------
        if clicked is not None:
            st.session_state.targets_presented += 1

            rt, type_erreur, feedback = process_click(
                clicked, st.session_state.stimulus_start_time
            )

            if type_erreur is None:
                st.session_state.reaction_times.append(rt)
                st.session_state.score += 1
            else:
                update_error_counters(type_erreur)

            st.session_state.feedback = feedback
            complete_stimulus_round()
            st.rerun()

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
            st.session_state.errors_inhibition,
            st.session_state.score
)

        neuro_index = compute_neuro_index(state, biomarkers)

        # -----------------------------
        # BOUTON : TERMINER LA SÉANCE
        # -----------------------------
        if st.button("Terminer la séance"):
            save_session(patient_id, age, tdah, mode_reeduc, state, biomarkers, difficulty, neuro_index)
            st.session_state.started = 0
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
            "impulsivity": st.session_state.impulsivity,
            "errors_inhibition": st.session_state.errors_inhibition
        }

        biomarkers = compute_biomarkers(
            state,
            state["errors"],
            state["impulsivity"],
            st.session_state.errors_inhibition,
            st.session_state.score
        )
        neuro_index = compute_neuro_index(state, biomarkers)

        # Calcul de la difficulté adaptative
        # La difficulté ne change qu'entre deux séries, pas à chaque rerun.
        completed_targets = st.session_state.targets_presented
        if (
            st.session_state.target is None
            and completed_targets >= 5
            and completed_targets % 5 == 0
            and st.session_state.last_adaptation_round != completed_targets
        ):
            st.session_state.difficulty = compute_difficulty(
                state, biomarkers, st.session_state.difficulty
            )
            st.session_state.last_adaptation_round = completed_targets
        difficulty = int(st.session_state.difficulty)

        st.caption(f"Difficulté adaptative : **{difficulty}**")

        # -----------------------------
        # AFFICHAGE DES STIMULI
        # -----------------------------
        prepare_stimulus_round(difficulty)
        if check_omission():
            st.rerun()
        cols = st.columns(difficulty)
        clicked = None
        round_id = st.session_state.stimulus_round

        for i in range(difficulty):
            with cols[i]:
                stimulus = st.session_state.stimuli[i]
                if i == st.session_state.target:
                    if st.button("🟢", key=f"stim_IA_green_{round_id}_{i}"):
                        clicked = "🟢"
                else:
                    if st.button(stimulus, key=f"stim_IA_dist_{round_id}_{i}"):
                        clicked = stimulus

        # -----------------------------
        # GESTION DU CLIC
        # -----------------------------
        if clicked is not None:
            st.session_state.targets_presented += 1

            rt, type_erreur, feedback = process_click(
                clicked, st.session_state.stimulus_start_time
            )

            if type_erreur is None:
                st.session_state.reaction_times.append(rt)
                st.session_state.score += 1
            else:
                update_error_counters(type_erreur)

            st.session_state.feedback = feedback
            complete_stimulus_round()
            st.rerun()

        if "feedback" in st.session_state:
            st.markdown(f"### {st.session_state.feedback}")

        if st.button("Terminer la séance"):
            save_session(
                patient_id,
                age,
                tdah,
                mode_reeduc,
                state,
                biomarkers,
                difficulty,
                neuro_index,
            )
            st.session_state.started = 0
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
        omissions = 0
        commissions = 0
        anticipations = 0
        difficulty = 5

        for _ in range(20):

            rt = random.uniform(0.4, 1.2)

            if random.random() < 0.2:
                errors += 1
                commissions += 1
                if rt < 0.5:
                    impulsivity += 1
                    anticipations += 1
                inhib_errors += 1
            else:
                reaction_times.append(rt)

            state = analyse_attention(reaction_times)
            state["errors"] = errors
            state["impulsivity"] = impulsivity
            biomarkers = compute_biomarkers(
                state,
                errors,
                impulsivity,
                inhib_errors,
                len(reaction_times),
                omissions=omissions,
                commission_errors=commissions,
                anticipations=anticipations,
                targets_presented=20,
            )
            difficulty = compute_difficulty(state, biomarkers, difficulty)

        neuro_index = compute_neuro_index(
            state,
            biomarkers,
            targets_presented=20,
        )

        session_number = len(
            df[
                (df["patient_id"] == demo_id)
                & (df["orthoptiste_id"] == orthoptiste)
            ]
        ) + 1

        session = {
            "orthoptiste_id": orthoptiste,
            "patient_id": demo_id,
            "session_number": session_number,
            "timestamp": datetime.now().isoformat(),
            "age": 12,
            "tdah": "Diagnostiqué",
            "mode_reeduc": "Adaptative IA",
            "score": len(reaction_times),
            "errors": errors,
            "commission_errors": commissions,
            "omissions": omissions,
            "anticipations": anticipations,
            "targets_presented": 20,
            "impulsivity": impulsivity,
            "errors_inhibition": inhib_errors,
            "mean_rt": state["mean_rt"],
            "median_rt": state["median_rt"],
            "variabilite": state["variabilite"],
            "trend": state["trend"],
            "cvrt": biomarkers["cvrt"],
            "omission_rate": biomarkers["omission_rate"],
            "commission_rate": biomarkers["commission_rate"],
            "anticipation_rate": biomarkers["anticipation_rate"],
            "difficulty": difficulty,
            "neuro_index": neuro_index,
            "commentaire": "Séance simulée automatiquement.",
        }

        df = pd.concat([df, pd.DataFrame([session])], ignore_index=True)
        df = df.reindex(columns=COLUMNS)
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
        st.session_state.commission_errors = 0
        st.session_state.omissions = 0
        st.session_state.anticipations = 0
        st.session_state.targets_presented = 0

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
            st.session_state.errors_inhibition,
            st.session_state.test_clicks
        )

        st.session_state.difficulty = compute_difficulty(
            state,
            biomarkers,
            st.session_state.difficulty,
        )
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
                        st.session_state.targets_presented += 1

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
                        st.session_state.commission_errors += 1
                        st.session_state.targets_presented += 1
                        rt = time.time() - st.session_state.start_time
                        if rt < 0.5:
                            st.session_state.impulsivity += 1
                            st.session_state.anticipations += 1
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

    # Filtrer les données de l’orthoptiste connecté.
    df_ortho = df[df["orthoptiste_id"] == orthoptiste]

    if len(df_ortho) == 0:
        st.warning("Aucun patient associé à cet orthoptiste.")
        st.stop()

    # -----------------------------
    # ENTRAÎNEMENT IA
    # -----------------------------
    st.markdown("### 🤖 Entraîner / Mettre à jour le modèle IA")

    if st.button("Entraîner l’IA maintenant"):
        try:
            artifact = train_neuro_model(df_ortho)
            st.success(
                "Modèle IA entraîné et sauvegardé sur "
                f"{artifact['transition_count']} transitions !"
            )
        except ValueError as error:
            st.warning(str(error))

    st.markdown("---")

    # -----------------------------
    # SÉLECTION PATIENT
    # -----------------------------
    st.markdown("### 👤 Sélection du patient")
    patient = st.selectbox("Patient", df_ortho["patient_id"].unique())
    p = df_ortho[df_ortho["patient_id"] == patient].sort_values("session_number")

    last = p.iloc[-1]

    st.markdown("---")

    # ===================================================
    # PROFIL CLINIQUE SYNTHÉTIQUE
    # ===================================================
    section_title("Profil clinique synthétique", "🧬")
    green_box(clinical_profile(p))

    index_display = (
        "Données insuffisantes"
        if pd.isna(last["neuro_index"])
        else round(last["neuro_index"], 2)
    )
    st.metric("Index neurovisuel", index_display)
    st.metric("Difficulté actuelle", int(last["difficulty"]))

    st.markdown("---")

    # ===================================================
    # HISTORIQUE + ANALYSES CLINIQUES
    # ===================================================
    st.markdown("### 🕒 Dernière séance")
    st.dataframe(
        pd.DataFrame(
            {
                "Champ": list(last.index),
                "Valeur": [str(value) for value in last.values],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### 📊 Progression de l’index neurovisuel")
    st.line_chart(p.set_index("session_number")["neuro_index"])

    st.markdown("### 🧪 Biomarqueurs attentionnels")
    biom_cols = [
        "median_rt",
        "variabilite",
        "cvrt",
        "omission_rate",
        "commission_rate",
        "anticipation_rate",
    ]
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
        "median_rt": last["median_rt"],
        "n_rt": int(last["score"]) if pd.notna(last["score"]) else 0,
        "errors": last["errors"],
        "impulsivity": last["impulsivity"]
    }

    biomarkers = compute_biomarkers(
        state,
        last["errors"],
        last["impulsivity"],
        last["errors_inhibition"],
        last["score"],
        omissions=last["omissions"],
        commission_errors=last["commission_errors"],
        anticipations=last["anticipations"],
        targets_presented=last["targets_presented"],
    )

    neuro_index = last["neuro_index"]

    table, interpretation = clinical_table(state, biomarkers, neuro_index)

    st.table(
        pd.DataFrame(
            {
                "Indicateur": list(table.keys()),
                "Valeur": [str(value) for value in table.values()],
            }
        )
    )

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
        pdf_data = export_simple_pdf(p, table, interpretation, neuro_index)
        st.download_button(
            "Télécharger le PDF simple",
            data=pdf_data,
            file_name="bilan_simple.pdf",
            mime="application/pdf",
        )

    if st.button("📘 Rapport clinique complet (PDF)"):
        pdf_data = export_full_pdf(
            p, table, interpretation, recos, neuro_index
        )
        st.download_button(
            "Télécharger le rapport clinique complet",
            data=pdf_data,
            file_name="rapport_clinique.pdf",
            mime="application/pdf",
        )

    if st.button("📄 Générer le PDF Orthoptiste PRO"):
        pdf_data = export_ortho_pro_pdf(
            p, 
            table, 
            interpretation, 
            recos, 
            neuro_index
        )
        st.download_button(
            label="Télécharger le PDF Orthoptiste PRO",
            data=pdf_data,
            file_name="bilan_orthoptiste_pro.pdf",
            mime="application/pdf"
        )

    st.markdown("---")

    # ===================================================
    # PRÉDICTION IA
    # ===================================================
    st.markdown("### 🔮 Prédiction IA de la prochaine séance")

    try:
        artifact = joblib.load(model_file_for_current_orthoptiste())
        model = artifact["model"]
        features = artifact["features"]

        X_last = pd.DataFrame(
            [{feature: last[feature] for feature in features}]
        )
        pred_next = model.predict(X_last)[0]

        st.metric("Index neurovisuel prédit", round(pred_next, 1))

    except (FileNotFoundError, KeyError, ValueError):
        st.warning("⚠️ Le modèle IA n’est pas encore entraîné. Cliquez sur 'Entraîner l’IA'.")
