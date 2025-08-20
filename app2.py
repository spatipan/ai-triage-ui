# app.py — ED Front‑Door Triage Assistant (Responsive + Bilingual)
# ---------------------------------------------------
# For triage teams at the hospital entrance (EMS crews, triage nurses, ED doctors)
# - Responsive layout (stacks gracefully on small screens)
# - Language toggle (ไทย / English)
# - Outcome probabilities + 5‑level recommendation with your cutoffs
# - Zone color mapping (per request):
#     L1 Resuscitation → Blue, L2 Emergent → Red, L3 Urgent → Yellow,
#     L4 Less‑urgent → Green, L5 Non‑urgent → White
# - Evidence & team credits shown in the footer
# - Stylish UI with badges, soft cards, and subtle animations
#
# Disclaimer: Decision support only. Follow local protocols and clinical judgment.

from __future__ import annotations
import os
import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Heavy deps
import tensorflow as tf  # noqa: F401
import tensorflow_hub as hub
import tensorflow_text  # Registers custom ops (e.g., SentencepieceOp) for TF‑Hub multilingual models

# Project internals
from src.source import DataPreprocessing, TriageModel

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="ED Triage Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS (responsive, modern look)
st.markdown(
    """
    <style>
      :root {
        --card-bg: rgba(255,255,255,0.65);
        --border: 1px solid rgba(2,6,23,0.08);
        --shadow: 0 8px 24px rgba(2,6,23,0.08);
      }
      .hero {padding: 1.0rem 1.25rem; border-radius: 16px; border:var(--border);
             background: linear-gradient(135deg,#000000 0%, #000000 50%);} 
      .badge {display:inline-block; padding: .35rem .7rem; border-radius: 9999px; font-weight: 700; border:var(--border);} 
      /* Level colors per request */
      .lvl1 {background:#dbeafe; color:#1e40af; border-color:#93c5fd;}  /* Blue */
      .lvl2 {background:#fee2e2; color:#b91c1c; border-color:#fecaca;}  /* Red */
      .lvl3 {background:#fef9c3; color:#854d0e; border-color:#fde68a;}  /* Yellow */
      .lvl4 {background:#dcfce7; color:#166534; border-color:#bbf7d0;}  /* Green */
      .lvl5 {background:#ffffff; color:#334155; border-color:#e2e8f0;}  /* White */

      .metric-card { background: var(--card-bg); border:var(--border); padding: 1rem; border-radius: 16px; box-shadow: var(--shadow); }
      .action-card { background: var(--card-bg); border:var(--border); padding: 1rem; border-radius: 16px; box-shadow: var(--shadow); }

      /* Make buttons full-width and comfy */
      .stButton > button { width: 100%; border-radius: 12px; padding: .6rem 1rem; font-weight: 700; }

      /* Responsive tweaks */
      @media (max-width: 1000px) {
        .hero { padding: .8rem 1rem; }
        .st-emotion-cache-ue6h4q { padding-left: 0 !important; padding-right: 0 !important; }
      }
      @media (max-width: 700px) {
        .stButton > button { padding: .8rem 1.1rem; font-size: 1.05rem; }
      }

      /* Subtle fade-in */
      .fadein { animation: fade 0.5s ease-in-out; }
      @keyframes fade { from {opacity:0; transform: translateY(6px);} to {opacity:1; transform:none;} }

      footer {visibility: hidden}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Constants
# ---------------------------
APP_VERSION = "3.0.0"

DEFAULT_PATHS = {
    "num_preprocessor": "model/num_preprocessor.joblib",
    "keras_model": "model/model.keras",
    "keras_weights": "model/weights.weights.h5",
}

TARGETS = [
    "icu_admission", "or", "7_day_death", "admission", "lab",
    "xray", "et", "inject", "consult"
]

NUM_COLS = ['age','sbp','dbp','temp','pr','rr','o2sat','gcs_e','gcs_v','gcs_m','sex','how_come_er','t_n']
TEXT_COLS = ['cc']

# i18n strings
LANGS = {
  'en': {
    'title': 'ED Triage Assistant',
    'hero': 'Use at the ED entrance to estimate risk and assign the right zone before moving the patient inside. Adjust cutoffs in the sidebar to match your protocol.',
    'patient_details': 'Patient details',
    'age':'Age','gender':'Gender','arrival':'Arrival mode','case_type':'Case type','male':'Male','female':'Female',
    'walkin':'Walk-in','ems':'EMS','referral':'Referral','trauma':'Trauma','nontrauma':'Non-trauma',
    'sbp':'Systolic BP','dbp':'Diastolic BP','temp':'Temperature (°C)','pr':'Pulse rate','rr':'Respiratory rate','o2':'Oxygen saturation (%)',
    'gcs_e':'GCS: Eye','gcs_v':'GCS: Verbal','gcs_m':'GCS: Motor','cc':'Chief complaint',
    'predict':'⚡ Get recommendation',
    'recommendation':'Recommendation & probabilities',
    'triage':'Triage level','zone':'Suggested zone','actions':'Suggested actions now','why':'Why',
    'outcomes':'Outcome probabilities','download':'Download result (CSV)',
    'cutoffs':'Triage cutoffs','cut_l1':'Level 1: Critical risk ≥','cut_l2':'Level 2: Critical risk ≥','cut_l3':'Level 3: Urgent resource risk ≥','cut_l4':'Level 4: Minor resource risk ≥',
    'redflags_toggle':'Apply vital‑sign red‑flags (auto Level 1)','redflags':'Red‑flag thresholds','save_log':'Save prediction to CSV log',
    'advanced':'Advanced (model paths)','num_preproc':'Numeric preprocessor','keras_model':'Keras model','keras_weights':'Keras weights (optional)','embedder':'Text embedder (TF‑Hub)',
    'footer_note':'This tool provides guidance only and does not replace clinical judgment. Follow local protocols.',
    'footer_evidence':'Evidence: internal validation on 163,452 ED visits (2018–2022) at a Thai tertiary center. XGBoost AUROC 0.917; AUPRC 0.629; bootstrapped calibration/stability. Preprint:',
    'footer_team':'Team: Patipan Sitthiprawiat, Borwon Wittayachamnankul, Wachiranun Sirikul, Korsin Laohavisudhi — Chiang Mai University Faculty of Medicine (Emergency Medicine Department & Informatics)',
    'placeholder_cc':'e.g., Sudden chest pain 2 hours, dyspnea',
    'preprint_url':'https://sciety-labs.elifesciences.org/articles/by?article_doi=10.21203/rs.3.rs-6229836/v1',
    'sidebar_language_title':'🌐 Language / ภาษา',
    'language_label':'Language',
    'help_cut_l1':'If max of ICU / ET / OR / 7‑day death ≥ this ⇒ Level 1',
    'rf_label_sbp':'SBP < (mmHg)',
    'rf_label_o2':'SpO₂ < (%)',
    'rf_label_rr':'RR > (/min)',
    'rf_label_temp':'Temp ≥ (°C)',
    'rf_label_gcs':'GCS total ≤',
    'analyzing':'Analyzing...',
    'prediction_failed':'Prediction failed: ',
    'below_cutoffs':'All risks below cutoffs',
    'vital_redflags_prefix':'Vital red‑flags: ',
    'probability':'Probability',
    'actions_l1':["Assign to **Blue zone** (Resuscitation) immediately","Activate resuscitation team; continuous monitoring (ECG, SpO₂, BP)","High‑flow O₂; prepare BVM/advanced airway","2 large‑bore IV/IO; fluids per protocol"],
    'actions_l2':["Assign to **Red zone** (High‑acuity)","Rapid assessment; monitoring as indicated","IV access; protocol‑based treatment"],
    'actions_l3':["Assign to **Yellow zone** (Urgent)","Timely assessment; monitoring as indicated","IV/symptomatic care as needed"],
    'actions_l4':["Assign to **Green zone** (Minor)","Symptomatic relief; basic tests per protocol"],
    'actions_l5':["Assign to **White zone** (Fast‑track/clinic)","Safety‑net instructions and follow‑up advice"],
  },
  'th': {
    'title': 'ตัวช่วยคัดแยกผู้ป่วยฉุกเฉิน (หน้า ER)',
    'hero': 'ใช้ที่หน้าห้องฉุกเฉินเพื่อประเมินความเสี่ยงและกำหนดโซนที่เหมาะสมก่อนพาเข้าพื้นที่ภายใน ปรับค่าตัดสินใจได้ตามแนวทางของโรงพยาบาล',
    'patient_details': 'ข้อมูลผู้ป่วย',
    'age':'อายุ','gender':'เพศ','arrival':'วิธีมา ER','case_type':'ชนิดเคส','male':'ชาย','female':'หญิง',
    'walkin':'เดินมาเอง','ems':'รถพยาบาล','referral':'ส่งต่อ','trauma':'T (อุบัติเหตุ)','nontrauma':'N (ไม่ใช่อุบัติเหตุ)',
    'sbp':'ความดันซิสโตลิก','dbp':'ความดันไดแอสโตลิก','temp':'อุณหภูมิ (°C)','pr':'ชีพจร','rr':'อัตราการหายใจ','o2':'ออกซิเจนปลายนิ้ว (%)',
    'gcs_e':'GCS: ตา','gcs_v':'GCS: พูด','gcs_m':'GCS: เคลื่อนไหว','cc':'อาการสำคัญ',
    'predict':'⚡ ขอคำแนะนำ',
    'recommendation':'ข้อเสนอแนะและความน่าจะเป็น',
    'triage':'ระดับคัดแยก','zone':'โซนที่แนะนำ','actions':'การดำเนินการทันที','why':'เหตุผล',
    'outcomes':'ความน่าจะเป็นของผลลัพธ์','download':'ดาวน์โหลดผลลัพธ์ (CSV)',
    'cutoffs':'ค่าตัดสินใจของระดับคัดแยก','cut_l1':'ระดับ 1: ความเสี่ยงวิกฤต ≥','cut_l2':'ระดับ 2: ความเสี่ยงวิกฤต ≥','cut_l3':'ระดับ 3: ความเสี่ยงทรัพยากรเร่งด่วน ≥','cut_l4':'ระดับ 4: ความเสี่ยงทรัพยากรเล็กน้อย ≥',
    'redflags_toggle':'เปิดใช้สัญญาณเตือนชีพ (ปรับเป็นระดับ 1 อัตโนมัติ)','redflags':'เกณฑ์สัญญาณเตือนชีพ','save_log':'บันทึกผลลง CSV',
    'advanced':'ขั้นสูง (ตำแหน่งไฟล์โมเดล)','num_preproc':'ตัวประมวลผลตัวเลข','keras_model':'ไฟล์โมเดล Keras','keras_weights':'ไฟล์น้ำหนัก (ถ้ามี)','embedder':'ตัวแปลงข้อความ (TF‑Hub)',
    'footer_note':'เครื่องมือนี้ช่วยประกอบการตัดสินใจ ไม่ทดแทนวิจารณญาณทางคลินิก โปรดปฏิบัติตามแนวทางของหน่วยงาน',
    'footer_evidence':'หลักฐาน: ตรวจสอบภายในบนข้อมูล 163,452 เคส (ปี 2018–2022) ที่ รพ.มหาราชเชียงใหม่ XGBoost AUROC 0.917; AUPRC 0.629; ทดสอบความเสถียรด้วย bootstrap และการสอบเทียบ ผลงานพิมพ์ล่วงหน้า:',
    'footer_team':'ทีม: นพ.ปฏิภาณ สิทธิประเวศ, รศ.นพ.บวร วิทยชำนาญกุล, ผศ.ดร.วชิรนันท์ ศิริกุล, อ.นพ.กอสิน เลาหะวิสุทธิ์ — คณะแพทยศาสตร์ มช. (เวชศาสตร์ฉุกเฉิน & อินฟอร์แมติกส์)',
    'placeholder_cc':'เช่น เจ็บหน้าอกเฉียบพลัน 2 ชม. หอบเหนื่อย',
    'preprint_url':'https://sciety-labs.elifesciences.org/articles/by?article_doi=10.21203/rs.3.rs-6229836/v1',
    'sidebar_language_title':'🌐 Language / ภาษา',
    'language_label':'ภาษา',
    'help_cut_l1':'หากค่าสูงสุดของ ICU/ใส่ท่อ/ผ่าตัด/เสียชีวิตภายใน 7 วัน ≥ ค่านี้ ⇒ ระดับ 1',
    'rf_label_sbp':'SBP < (mmHg)',
    'rf_label_o2':'SpO₂ < (%)',
    'rf_label_rr':'RR > (/min)',
    'rf_label_temp':'อุณหภูมิ ≥ (°C)',
    'rf_label_gcs':'GCS รวม ≤',
    'analyzing':'กำลังประมวลผล...',
    'prediction_failed':'ไม่สามารถประมวลผลได้: ',
    'below_cutoffs':'ความเสี่ยงทั้งหมดต่ำกว่าค่าตัดสินใจ',
    'vital_redflags_prefix':'สัญญาณเตือนชีพ: ',
    'probability':'ความน่าจะเป็น',
    'actions_l1':["ส่งเข้าโซนน้ำเงินทันที (พื้นที่กู้ชีพ)", "เปิดทีมกู้ชีพ/monitor ต่อเนื่อง (ECG, SpO₂, BP)", "ให้ออกซิเจน เตรียม BVM/ใส่ท่อ", "เปิดเส้น IV/IO 2 เส้น ให้สารน้ำตามข้อบ่งชี้"],
    'actions_l2':["ส่งเข้าโซนแดง (เฝ้าระวังอาการหนัก)", "ประเมินรวดเร็ว + monitor ตามอาการ", "เปิดเส้น IV และให้การรักษาตาม protocol"],
    'actions_l3':["ส่งเข้าโซนเหลือง (เร่งด่วน)", "ประเมินตามลำดับความเร่งด่วน", "ให้ IV/ยา ตามความจำเป็น"],
    'actions_l4':["ส่งเข้าโซนเขียว (อาการเล็กน้อย)", "ให้การดูแลตามอาการ/พิจารณาตรวจพื้นฐาน"],
    'actions_l5':["ส่งเข้าโซนขาว (Fast‑track/คลินิก)", "แจ้งสัญญาณอันตรายและคำแนะนำกลับมาพบแพทย์"],
  }
}

TARGET_LABELS = {
    "icu_admission": { 'en': "ICU admission", 'th': 'รับเข้าหอผู้ป่วยวิกฤต' },
    "or":            { 'en': "Operating room", 'th': 'ส่งห้องผ่าตัด' },
    "7_day_death":   { 'en': "Death within 7 days", 'th': 'เสียชีวิตภายใน 7 วัน' },
    "admission":     { 'en': "Hospital admission", 'th': 'รับไว้รักษาในโรงพยาบาล' },
    "lab":           { 'en': "Lab needed", 'th': 'ต้องตรวจแลป' },
    "xray":          { 'en': "X‑ray needed", 'th': 'ต้องเอกซเรย์' },
    "et":            { 'en': "Endotracheal tube", 'th': 'ต้องใส่ท่อช่วยหายใจ' },
    "inject":        { 'en': "Injection/IV meds", 'th': 'ต้องให้ยาฉีด/น้ำเกลือ' },
    "consult":       { 'en': "Consultation needed", 'th': 'ต้องปรึกษาแพทย์เฉพาะทาง' },
}

LEVEL_MAP = {
    1: ( { 'en': "Resuscitation", 'th': 'กู้ชีพ' },  "lvl1"),  # Blue
    2: ( { 'en': "Emergent",       'th': 'ฉุกเฉินเร่งด่วน' },  "lvl2"),  # Red
    3: ( { 'en': "Urgent",         'th': 'เร่งด่วน' },        "lvl3"),  # Yellow
    4: ( { 'en': "Less‑urgent",    'th': 'กึ่งเร่งด่วน' },    "lvl4"),  # Green
    5: ( { 'en': "Non‑urgent",     'th': 'ไม่เร่งด่วน' },     "lvl5"),  # White
}

ZONE_MAP = {
    1: ( { 'en': "Blue zone",  'th': 'โซนน้ำเงิน' }, { 'en': "Resuscitation bay", 'th': 'พื้นที่กู้ชีพ' } ),
    2: ( { 'en': "Red zone",   'th': 'โซนแดง' },    { 'en': "High‑acuity / monitored", 'th': 'พื้นที่ฉุกเฉินเร่งด่วน' } ),
    3: ( { 'en': "Yellow zone",'th': 'โซนเหลือง' }, { 'en': "Urgent care", 'th': 'พื้นที่เร่งด่วน' } ),
    4: ( { 'en': "Green zone", 'th': 'โซนเขียว' },  { 'en': "Minor care", 'th': 'พื้นที่กึ่งเร่งด่วน' } ),
    5: ( { 'en': "White zone", 'th': 'โซนขาว' },  { 'en': "Fast‑track / clinic", 'th': 'พื้นที่ไม่เร่งด่วน' } ),
}

# ---------------------------
# Cached loaders (no re-fit)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_preprocessor(num_preprocessor_path: str) -> DataPreprocessing:
    dp = DataPreprocessing()
    dp.num_preprocessor = joblib.load(num_preprocessor_path)
    try:
        dp.x_num_cols = NUM_COLS
        dp.x_text_cols = TEXT_COLS
        dp.y_cols = TARGETS
    except Exception:
        pass
    return dp

# @st.cache_resource(show_spinner=False)
# def load_embedder(embedder_url: str):
#     # tensorflow_text import above ensures custom ops are registered for multilingual USE
#     return hub.load(embedder_url)

@st.cache_resource(show_spinner=False)
def load_embedder(handle: str):
    """Load TF-Hub text embedder.

    - If `handle` points to a local SavedModel dir, load with tf.saved_model.load.
    - Otherwise, load as a TF-Hub module/SavedModel through KerasLayer.
    """
    import tensorflow as tf, os
    # Local SavedModel?
    if os.path.isdir(handle) and (
        os.path.exists(os.path.join(handle, "saved_model.pb")) or
        os.path.exists(os.path.join(handle, "saved_model.pbtxt"))
    ):
        return tf.saved_model.load(handle)

    # TF-Hub handle (module or SavedModel) → KerasLayer
    return hub.KerasLayer(handle, trainable=False)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str, weights_path: str) -> TriageModel:
    tm = TriageModel()
    tm.import_model(model_path)
    if os.path.exists(weights_path):
        tm.load_weights(weights_path)
    return tm

# ---------------------------
# Sidebar — language, cutoffs, red‑flags
# ---------------------------
# Initialize language for first render so T is defined before first use
if 'LANG_KEY' not in st.session_state:
    st.session_state['LANG_KEY'] = 'en'
LANG_KEY = st.session_state['LANG_KEY']
T = LANGS[LANG_KEY]
st.sidebar.title(T['sidebar_language_title'])
lang_choice = st.sidebar.selectbox(T['language_label'], options=["English","ไทย"], index=(0 if LANG_KEY=='en' else 1))
LANG_KEY = 'th' if lang_choice == 'ไทย' else 'en'
st.session_state['LANG_KEY'] = LANG_KEY
T = LANGS[LANG_KEY]

with st.sidebar:
    st.markdown(f"**{T['cutoffs']}**")
    lvl1_cut = st.slider(T['cut_l1'], 0.10, 0.90, 0.50, 0.01,
                         help=T['help_cut_l1'])
    lvl2_cut = st.slider(T['cut_l2'], 0.05, 0.80, 0.30, 0.01)
    lvl3_cut = st.slider(T['cut_l3'], 0.05, 0.80, 0.40, 0.01)
    lvl4_cut = st.slider(T['cut_l4'], 0.05, 0.80, 0.25, 0.01)

    st.markdown("---")
    apply_redflags = st.toggle(T['redflags_toggle'], value=True)
    with st.expander(T['redflags'], expanded=False):
        rf_sbp = st.number_input(T['rf_label_sbp'], value=90, min_value=60, max_value=120)
        rf_o2 = st.number_input(T['rf_label_o2'], value=90, min_value=70, max_value=100)
        rf_rr_hi = st.number_input(T['rf_label_rr'], value=30, min_value=16, max_value=60)
        rf_temp_hi = st.number_input(T['rf_label_temp'], value=39.5, min_value=37.0, max_value=42.0, step=0.1, format="%.1f")
        rf_gcs = st.number_input(T['rf_label_gcs'], value=8, min_value=3, max_value=15)

    st.markdown("---")
    log_predictions = st.toggle(T['save_log'], value=False)

    with st.expander(T['advanced']):
        num_prep_path = st.text_input(T['num_preproc'], value=DEFAULT_PATHS["num_preprocessor"]) 
        keras_model_path = st.text_input(T['keras_model'], value=DEFAULT_PATHS["keras_model"]) 
        keras_weights_path = st.text_input(T['keras_weights'], value=DEFAULT_PATHS["keras_weights"]) 
        embedder_url = st.text_input(T['embedder'], value="https://www.kaggle.com/models/google/universal-sentence-encoder/tensorFlow2/multilingual")

# Load artifacts once
preprocessor = load_preprocessor(num_prep_path)
embedder = load_embedder(embedder_url)
model = load_model(keras_model_path, keras_weights_path)

# ---------------------------
# Helpers
# ---------------------------

def embed_text(text: str) -> np.ndarray:
    return np.array(embedder([str(text)]))


def vital_red_flags(v: dict) -> list[str]:
    flags = []
    if v.get('sbp', 999) < rf_sbp: flags.append(f"SBP < {rf_sbp}")
    if v.get('o2sat', 100) < rf_o2: flags.append(f"SpO₂ < {rf_o2}%")
    if v.get('rr', 0) > rf_rr_hi: flags.append(f"RR > {rf_rr_hi}")
    if v.get('temp', 0) >= rf_temp_hi: flags.append(f"Temp ≥ {rf_temp_hi}°C")
    gcs_total = v.get('gcs_e', 4) + v.get('gcs_v', 5) + v.get('gcs_m', 6)
    if gcs_total <= rf_gcs: flags.append(f"GCS ≤ {rf_gcs}")
    return flags


def triage_decision(preds: dict, vitals: dict) -> tuple[int, str, list[str]]:
    critical = max(preds.get('7_day_death', 0), preds.get('icu_admission', 0), preds.get('et', 0), preds.get('or', 0))
    urgent = max(preds.get('admission', 0), preds.get('inject', 0), preds.get('consult', 0))
    minor  = max(preds.get('lab', 0), preds.get('xray', 0))

    rationale = []

    if apply_redflags:
        flags = vital_red_flags(vitals)
        if flags:
            rationale.append((T['vital_redflags_prefix']) + ", ".join(flags))
            return 1, LEVEL_MAP[1][1], rationale

    if critical >= lvl1_cut:
        rationale.append(("Critical risk " if LANG_KEY=='en' else "ความเสี่ยงวิกฤต ") + f"{critical*100:.1f}% ≥ L1 {lvl1_cut*100:.0f}%")
        return 1, LEVEL_MAP[1][1], rationale
    if critical >= lvl2_cut:
        rationale.append(("Critical risk " if LANG_KEY=='en' else "ความเสี่ยงวิกฤต ") + f"{critical*100:.1f}% ≥ L2 {lvl2_cut*100:.0f}%")
        return 2, LEVEL_MAP[2][1], rationale
    if urgent >= lvl3_cut:
        rationale.append(("Urgent resource risk " if LANG_KEY=='en' else "ความเสี่ยงทรัพยากรเร่งด่วน ") + f"{urgent*100:.1f}% ≥ L3 {lvl3_cut*100:.0f}%")
        return 3, LEVEL_MAP[3][1], rationale
    if minor >= lvl4_cut:
        rationale.append(("Minor resource risk " if LANG_KEY=='en' else "ความเสี่ยงทรัพยากรเล็กน้อย ") + f"{minor*100:.1f}% ≥ L4 {lvl4_cut*100:.0f}%")
        return 4, LEVEL_MAP[4][1], rationale

    rationale.append(T['below_cutoffs'])
    return 5, LEVEL_MAP[5][1], rationale


def zone_for_level(level: int) -> tuple[str, str]:
    name_i18n, area_i18n = ZONE_MAP[level]
    return name_i18n[LANG_KEY], area_i18n[LANG_KEY]


def actions_for_level(level: int) -> list[str]:
    # Centralized in LANGS dict
    try:
        return LANGS[LANG_KEY][f'actions_l{level}']
    except KeyError:
        return []


def _map_gender_to_model_token(g: str) -> str:
    """Map UI gender to model token expected by preprocessor (ช/ญ)."""
    mapping = {
        'ชาย':'ช', 'หญิง':'ญ', 'ช':'ช', 'ญ':'ญ',
        'Male':'ช', 'Female':'ญ', 'M':'ช', 'F':'ญ'
    }
    return mapping.get(g, g)


def _map_case_to_model_token(c: str) -> str:
    """Map UI trauma/non‑trauma to model token (T/N)."""
    mapping = {
        'T':'T', 'N':'N',
        'Trauma':'T', 'Non-trauma':'N',
        'T (อุบัติเหตุ)':'T', 'N (ไม่ใช่อุบัติเหตุ)':'N'
    }
    return mapping.get(c, c)


def predict_single(row_df: pd.DataFrame) -> dict[str, float]:
    num_X = preprocessor.num_preprocessor.transform(row_df[NUM_COLS])
    text_vec = np.array(embedder([row_df.loc[row_df.index[0], 'cc']]))
    preds = model.model.predict([num_X, text_vec], verbose=0)
    flat = preds[0] if isinstance(preds, (list, tuple)) else preds
    flat = np.asarray(flat).reshape(-1)
    return {t: float(p) for t, p in zip(TARGETS, flat)}


def write_log(single_input: dict, preds: dict, level: int):
    if not log_predictions:
        return
    os.makedirs("logs", exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "session": st.session_state.get('session_id'),
        **single_input,
        **{f"pred_{k}": v for k, v in preds.items()},
        "triage_level": level,
    }
    path = os.path.join("logs", "predictions.csv")
    pd.DataFrame([row]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)

# Stable session id
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

# ---------------------------
# UI — Header
# ---------------------------
st.title(T['title'])
st.markdown(f"<div class='hero fadein'>{T['hero']}</div>", unsafe_allow_html=True)

# ---------------------------
# UI — Input & Results (columns collapse on mobile)
# ---------------------------
left, right = st.columns([1.05, 1])

with left:
    st.subheader(T['patient_details'])
    with st.form("patient_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input(T['age'], min_value=18, max_value=110, value=40)
            gender = st.selectbox(T['gender'], options=[T['male'], T['female']], index=0)
            how_come_er = st.selectbox(T['arrival'], options=[T['walkin'],T['ems'],T['referral']], index=1)
            t_n = st.selectbox(T['case_type'], options=[T['trauma'],T['nontrauma']], index=1)
        with c2:
            sbp = st.number_input(T['sbp'], min_value=40, max_value=300, value=110)
            dbp = st.number_input(T['dbp'], min_value=20, max_value=200, value=70)
            temp = st.number_input(T['temp'], min_value=30.0, max_value=43.0, value=37.4, step=0.1, format="%.1f")
            pr = st.number_input(T['pr'], min_value=20, max_value=220, value=98)
        with c3:
            rr = st.number_input(T['rr'], min_value=6, max_value=80, value=22)
            o2sat = st.number_input(T['o2'], min_value=50, max_value=100, value=95)
            gcs_e = st.number_input(T['gcs_e'], min_value=1, max_value=4, value=4)
            gcs_v = st.number_input(T['gcs_v'], min_value=1, max_value=5, value=5)
            gcs_m = st.number_input(T['gcs_m'], min_value=1, max_value=6, value=6)
        cc = st.text_area(T['cc'], placeholder=T['placeholder_cc'])

        submitted = st.form_submit_button(T['predict'], use_container_width=True)

    input_df = None
    if submitted:
        # Map back to raw expected tokens for model
        gender_raw = _map_gender_to_model_token(gender)
        arrival_raw = {'Walk-in':'Walkin','EMS':'EMS','Referral':'Referral',
                       'เดินมาเอง':'Walkin','รถพยาบาล':'EMS','ส่งต่อ':'Referral'}[how_come_er]
        case_raw = _map_case_to_model_token(t_n)

        input_df = pd.DataFrame([[age, sbp, dbp, temp, pr, rr, o2sat, gcs_e, gcs_v, gcs_m, gender_raw, arrival_raw, case_raw, cc]],
                                columns=NUM_COLS + TEXT_COLS)

with right:
    st.subheader(T['recommendation'])
    if submitted and input_df is not None:
        with st.spinner(T['analyzing']):
            try:
                preds = predict_single(input_df)
                vitals = dict(sbp=sbp, o2sat=o2sat, rr=rr, temp=temp, gcs_e=gcs_e, gcs_v=gcs_v, gcs_m=gcs_m)
                level, css, why = triage_decision(preds, vitals)
                lvl_name = LEVEL_MAP[level][0][LANG_KEY]
                zone_name, zone_area = zone_for_level(level)

                st.markdown(f"<span class='badge {css}'>"+T['triage']+f": {level} — {lvl_name}</span>", unsafe_allow_html=True)
                st.markdown(f"<span class='badge {css}'>"+T['zone']+f": {zone_name} — {zone_area}</span>", unsafe_allow_html=True)

                with st.container(border=True):
                    st.markdown("**"+T['actions']+"**")
                    for a in actions_for_level(level):
                        st.write("• ", a)
                    st.caption((T['why']+": ") + "; ".join(why))

                st.markdown("---")
                st.markdown("**"+T['outcomes']+"**")
                cols = st.columns(3)
                for i, t in enumerate(TARGETS):
                    with cols[i % 3]:
                        with st.container(border=True):
                            p = preds[t]
                            label = TARGET_LABELS[t][LANG_KEY]
                            st.markdown(f"**{label}**")
                            st.metric(T['probability'], f"{p*100:.1f}%")
                            st.progress(min(max(p, 0.0), 1.0))

                # Download result
                out_row = {**input_df.iloc[0].to_dict(), **{f"pred_{k}": v for k, v in preds.items()}, "triage_level": level, "zone": zone_name}
                out_df = pd.DataFrame([out_row])
                st.download_button(
                    T['download'],
                    data=out_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"triage_result_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                if log_predictions:
                    write_log(input_df.iloc[0].to_dict(), preds, level)

            except Exception as e:
                st.error((T['prediction_failed']) + f"{type(e).__name__}: {e}")

# ---------------------------
# Footer / Evidence
# ---------------------------
st.markdown("---")
st.markdown("**"+T['footer_note']+"**  ")
st.caption(T['footer_evidence'] + ' ' + T['preprint_url'])
st.caption(T['footer_team'])

# End of file
