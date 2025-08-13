# app.py — ED Front‑Door Triage Assistant (Responsive + Bilingual)
# ---------------------------------------------------
# For triage teams at the hospital entrance (EMS crews, triage nurses, ED doctors)
# - Responsive layout (stacks gracefully on small screens)
# - Language toggle (ไทย / English)
# - Outcome probability + 5‑level recommendation with cutoffs
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
import pickle
import joblib

# Heavy deps (available in your env)
import tensorflow as tf  # noqa: F401
import tensorflow_hub as hub
import tensorflow_text  # IMPORTANT: registers custom ops like SentencepieceOp before loading TF-Hub models

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
             background: linear-gradient(135deg,#eef2ff 0%, #ffffff 100%);} 
      .badge {display:inline-block; padding: .35rem .7rem; border-radius: 9999px; font-weight: 700; border:var(--border);}
      /* Level colors per request */
      .lvl1 {background:#dbeafe; color:#1e40af; border-color:#93c5fd;}  /* Blue */
      .lvl2 {background:#fee2e2; color:#b91c1c; border-color:#fecaca;}  /* Red */
      .lvl3 {background:#fef9c3; color:#854d0e; border-color:#fde68a;}  /* Yellow */
      .lvl4 {background:#dcfce7; color:#166534; border-color:#bbf7d0;}  /* Green */
      .lvl5 {background:#ffffff; color:#334155; border-color:#e2e8f0;}  /* White */

      .metric-card { background: var(--card-bg); border:var(--border); padding: 1rem; border-radius: 16px; box-shadow: var(--shadow); }
      .action-card { background: var(--card-bg); border:var(--border); padding: 1rem; border-radius: 16px; box-shadow: var(--shadow); }

      .stButton > button { width: 100%; border-radius: 12px; padding: .6rem 1rem; font-weight: 700; }

      @media (max-width: 1000px) {
        .hero { padding: .8rem 1rem; }
        .st-emotion-cache-ue6h4q { padding-left: 0 !important; padding-right: 0 !important; }
      }
      @media (max-width: 700px) {
        .stButton > button { padding: .8rem 1.1rem; font-size: 1.05rem; }
      }

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
APP_VERSION = "3.2.0"

MODEL_PATH = "model/xgb_model_calibrated.pkl"
EMBEDDER_URL = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
EXPLAINER_PATH = "model/xgb_explainer.pkl"  # optional

NUM_COLS = ['age','sbp','dbp','temp','pr','rr','o2sat','gcs_e','gcs_v','gcs_m','sex','how_come_er','t_n']
TEXT_COLS = [f'text_{i}' for i in range(512)]  # we generate 512-dim USE then your pipeline may PCA

# i18n strings
LANGS = {
  'en': {
    'title': 'ED Triage Assistant',
    'hero': 'Use at the ED entrance to estimate risk and assign the right zone before moving the patient inside. Adjust cutoffs in the sidebar to match your protocol.',
    'patient_details': 'Patient details',
    'age':'Age','gender':'Gender','arrival':'Arrival mode','case_type':'Case type','male':'M','female':'F',
    'walkin':'Walk-in','ems':'EMS','referral':'Referral','trauma':'Trauma','nontrauma':'Non-trauma',
    'sbp':'Systolic BP','dbp':'Diastolic BP','temp':'Temperature (°C)','pr':'Pulse rate','rr':'Respiratory rate','o2':'Oxygen saturation (%)',
    'gcs_e':'GCS: Eye','gcs_v':'GCS: Verbal','gcs_m':'GCS: Motor','cc':'Chief complaint',
    'predict':'⚡ Get recommendation',
    'recommendation':'Recommendation & probability',
    'triage':'Triage level','zone':'Suggested zone','actions':'Suggested actions now','why':'Why',
    'outcomes':'Outcome probability','download':'Download result (CSV)',
    'cutoffs':'Triage cutoffs','cut_l1':'Level 1: Critical risk ≥','cut_l2':'Level 2: Critical risk ≥','cut_l3':'Level 3: Urgent resource risk ≥','cut_l4':'Level 4: Minor resource risk ≥',
    'redflags_toggle':'Apply vital‑sign red‑flags (auto Level 1)','redflags':'Red‑flag thresholds','save_log':'Save prediction to CSV log',
    'advanced':'Advanced (paths)','footer_note':'Decision support only. Follow local protocols and clinical judgment.',
    'footer_evidence':'Evidence: internal validation (2018–2022) with bootstrapping, calibration, and strong AUROC; see preprint:',
    'footer_team':'Team: Patipan Sitthiprawiat, Borwon Wittayachamnankul, Wachiranun Sirikul, Korsin Laohavisudhi — Chiang Mai University Faculty of Medicine (Emergency Medicine & Informatics)',
    'placeholder_cc':'e.g., Sudden chest pain 2 hours, dyspnea',
    'tabs': ['Tutorial','Evidence','About','Contact & Feedback'],
    'icu':'ICU admission',
  },
  'th': {
    'title': 'ตัวช่วยคัดแยกผู้ป่วยฉุกเฉิน (หน้า ER)',
    'hero': 'ใช้ที่หน้าห้องฉุกเฉินเพื่อประเมินความเสี่ยงและกำหนดโซนที่เหมาะสมก่อนพาเข้าพื้นที่ภายใน ปรับค่าตัดสินใจได้ตามแนวทางของหน่วยงาน',
    'patient_details': 'ข้อมูลผู้ป่วย',
    'age':'อายุ','gender':'เพศ','arrival':'วิธีมา ER','case_type':'ชนิดเคส','male':'ช','female':'ญ',
    'walkin':'เดินมาเอง','ems':'EMS','referral':'ส่งต่อ','trauma':'อุบัติเหตุ','nontrauma':'ไม่ใช่อุบัติเหตุ',
    'sbp':'ความดันซิสโตลิก','dbp':'ความดันไดแอสโตลิก','temp':'อุณหภูมิ (°C)','pr':'ชีพจร','rr':'อัตราการหายใจ','o2':'ออกซิเจนปลายนิ้ว (%)',
    'gcs_e':'GCS: ตา','gcs_v':'GCS: พูด','gcs_m':'GCS: เคลื่อนไหว','cc':'อาการสำคัญ',
    'predict':'⚡ ขอคำแนะนำ',
    'recommendation':'ข้อเสนอแนะและความน่าจะเป็น',
    'triage':'ระดับคัดแยก','zone':'โซนที่แนะนำ','actions':'การดำเนินการทันที','why':'เหตุผล',
    'outcomes':'ความน่าจะเป็นของผลลัพธ์','download':'ดาวน์โหลดผลลัพธ์ (CSV)',
    'cutoffs':'ค่าตัดสินใจของระดับคัดแยก','cut_l1':'ระดับ 1: ความเสี่ยงวิกฤต ≥','cut_l2':'ระดับ 2: ความเสี่ยงวิกฤต ≥','cut_l3':'ระดับ 3: ความเสี่ยงทรัพยากรเร่งด่วน ≥','cut_l4':'ระดับ 4: ความเสี่ยงทรัพยากรเล็กน้อย ≥',
    'redflags_toggle':'เปิดใช้สัญญาณเตือนชีพ (ปรับเป็นระดับ 1 อัตโนมัติ)','redflags':'เกณฑ์สัญญาณเตือนชีพ','save_log':'บันทึกผลลง CSV',
    'advanced':'ขั้นสูง (ตำแหน่งไฟล์)','footer_note':'เป็นเครื่องมือช่วยตัดสินใจ ไม่ทดแทนวิจารณญาณทางคลินิก โปรดปฏิบัติตามแนวทางของหน่วยงาน',
    'footer_evidence':'หลักฐาน: ตรวจสอบภายในพร้อม bootstrap และการสอบเทียบ ค่า AUROC ดีมาก ดู preprint:',
    'footer_team':'ทีม: นพ.ปฏิภาณ สิทธิประเวศ, รศ.นพ.บรวน วิทยาชำนะกุล, ผศ.ดร.วชิรนันท์ ศิริกุล, รศ.นพ.กรศิณ ล้อวิศษฎ์ — คณะแพทยศาสตร์ มช. (ฉุกเฉิน & อินฟอร์แมติกส์)',
    'placeholder_cc':'เช่น เจ็บหน้าอกเฉียบพลัน 2 ชม. หอบเหนื่อย',
    'tabs': ['วิธีใช้งาน','หลักฐานอ้างอิง','เกี่ยวกับผู้พัฒนา','ติดต่อและข้อเสนอแนะ'],
    'icu':'โอกาสเข้าหอผู้ป่วยวิกฤต',
  }
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
    2: ( { 'en': "Red zone",   'th': 'โซนแดง' },    { 'en': "High‑acuity / monitored", 'th': 'พื้นที่เฝ้าระวังอาการหนัก' } ),
    3: ( { 'en': "Yellow zone",'th': 'โซนเหลือง' }, { 'en': "Urgent care", 'th': 'พื้นที่เร่งด่วน' } ),
    4: ( { 'en': "Green zone", 'th': 'โซนเขียว' },  { 'en': "Minor care", 'th': 'พื้นที่อาการเล็กน้อย' } ),
    5: ( { 'en': "White zone", 'th': 'โซนขาว' },   { 'en': "Fast‑track / clinic", 'th': 'เส้นทางเร่งด่วน/คลินิก' } ),
}

# ---------------------------
# Loaders (cached)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_embedder(url: str):
    try:
        # tensorflow_text import above ensures custom ops are registered
        return hub.load(url)
    except Exception as e:
        st.error("Failed to load text embedder. Ensure TensorFlow Text matches your TF version (e.g., tensorflow==2.12.* with tensorflow-text==2.12.*)." 
                 f"Details: {type(e).__name__}: {e}")
        raise

@st.cache_resource(show_spinner=False)
def load_explainer(path: str):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

# Load artifacts once
MODEL = load_model(MODEL_PATH)
EMBEDDER = load_embedder(EMBEDDER_URL)
EXPLAINER = load_explainer(EXPLAINER_PATH)

# Try to detect PCA n_components from pipeline (optional)
try:
    _n_comp = MODEL.get_params().get('preprocessing__col__text__pca__n_components', None)
except Exception:
    _n_comp = None

# ---------------------------
# Sidebar — language, cutoffs, red‑flags
# ---------------------------
st.sidebar.title("🌐 Language / ภาษา")
lang_choice = st.sidebar.selectbox("Language", options=["English","ไทย"], index=0)
LANG_KEY = 'th' if lang_choice == 'ไทย' else 'en'
T = LANGS[LANG_KEY]

with st.sidebar:
    st.markdown(f"**{T['cutoffs']}**")
    lvl1_cut = st.slider(T['cut_l1'], 0.10, 0.90, 0.50, 0.01,
                         help=("If predicted ICU risk ≥ this ⇒ Level 1" if LANG_KEY=='en' else "หากโอกาสเข้า ICU ≥ ค่านี้ ⇒ ระดับ 1"))
    lvl2_cut = st.slider(T['cut_l2'], 0.05, 0.80, 0.30, 0.01)
    lvl3_cut = st.slider(T['cut_l3'], 0.05, 0.80, 0.40, 0.01)
    lvl4_cut = st.slider(T['cut_l4'], 0.05, 0.80, 0.25, 0.01)

    st.markdown("---")
    apply_redflags = st.toggle(T['redflags_toggle'], value=True)
    with st.expander(T['redflags'], expanded=False):
        rf_sbp = st.number_input("SBP < (mmHg)", value=90, min_value=60, max_value=120)
        rf_o2 = st.number_input("SpO₂ < (%)", value=90, min_value=70, max_value=100)
        rf_rr_hi = st.number_input("RR > (/min)", value=30, min_value=16, max_value=60)
        rf_temp_hi = st.number_input("Temp ≥ (°C)", value=39.5, min_value=37.0, max_value=42.0, step=0.1, format="%.1f")
        rf_gcs = st.number_input("GCS total ≤", value=8, min_value=3, max_value=15)

    st.markdown("---")
    log_predictions = st.toggle(T['save_log'], value=False)

    with st.expander(T['advanced']):
        st.caption("Paths are fixed in this build — change constants at top if needed.")

# ---------------------------
# Helpers
# ---------------------------

def embed_text(text: str) -> np.ndarray:
    # Returns 512-dim USE vector
    return np.array(EMBEDDER([str(text)])).reshape(1, -1)


def vital_red_flags(v: dict) -> list[str]:
    flags = []
    if v.get('sbp', 999) < rf_sbp: flags.append(f"SBP < {rf_sbp}")
    if v.get('o2sat', 100) < rf_o2: flags.append(f"SpO₂ < {rf_o2}%")
    if v.get('rr', 0) > rf_rr_hi: flags.append(f"RR > {rf_rr_hi}")
    if v.get('temp', 0) >= rf_temp_hi: flags.append(f"Temp ≥ {rf_temp_hi}°C")
    gcs_total = v.get('gcs_e', 4) + v.get('gcs_v', 5) + v.get('gcs_m', 6)
    if gcs_total <= rf_gcs: flags.append(f"GCS ≤ {rf_gcs}")
    return flags


def triage_decision(icu_prob: float, vitals: dict) -> tuple[int, str, list[str]]:
    rationale = []

    if apply_redflags:
        flags = vital_red_flags(vitals)
        if flags:
            rationale.append(("Vital red‑flags: " if LANG_KEY=='en' else "สัญญาณเตือนชีพ: ") + ", ".join(flags))
            return 1, LEVEL_MAP[1][1], rationale

    # Use ICU risk as proxy for criticality; you can extend to multiple outcomes later
    if icu_prob >= lvl1_cut:
        rationale.append(("ICU risk " if LANG_KEY=='en' else "ความเสี่ยงเข้า ICU ") + f"{icu_prob*100:.1f}% ≥ L1 {lvl1_cut*100:.0f}%")
        return 1, LEVEL_MAP[1][1], rationale
    if icu_prob >= lvl2_cut:
        rationale.append(("ICU risk " if LANG_KEY=='en' else "ความเสี่ยงเข้า ICU ") + f"{icu_prob*100:.1f}% ≥ L2 {lvl2_cut*100:.0f}%")
        return 2, LEVEL_MAP[2][1], rationale

    # If not critical, step down using placeholder resource risk — here we reuse ICU prob for the demo
    if icu_prob >= lvl3_cut:
        rationale.append(("Urgent resource risk " if LANG_KEY=='en' else "ความเสี่ยงทรัพยากรเร่งด่วน ") + f"{icu_prob*100:.1f}% ≥ L3 {lvl3_cut*100:.0f}%")
        return 3, LEVEL_MAP[3][1], rationale
    if icu_prob >= lvl4_cut:
        rationale.append(("Minor resource risk " if LANG_KEY=='en' else "ความเสี่ยงทรัพยากรเล็กน้อย ") + f"{icu_prob*100:.1f}% ≥ L4 {lvl4_cut*100:.0f}%")
        return 4, LEVEL_MAP[4][1], rationale

    rationale.append("All risks below cutoffs" if LANG_KEY=='en' else "ความเสี่ยงทั้งหมดต่ำกว่าค่าตัดสินใจ")
    return 5, LEVEL_MAP[5][1], rationale


def zone_for_level(level: int) -> tuple[str, str]:
    name_i18n, area_i18n = ZONE_MAP[level]
    return name_i18n[LANG_KEY], area_i18n[LANG_KEY]


def actions_for_level(level: int) -> list[str]:
    if LANG_KEY == 'th':
        if level == 1:
            return ["ส่งเข้าโซนน้ำเงินทันที (พื้นที่กู้ชีพ)", "เปิดทีมกู้ชีพ/monitor ต่อเนื่อง (ECG, SpO₂, BP)", "ให้ออกซิเจน เตรียม BVM/ใส่ท่อ", "เปิดเส้น IV/IO 2 เส้น ให้สารน้ำตามข้อบ่งชี้"]
        if level == 2:
            return ["ส่งเข้าโซนแดง (เฝ้าระวังอาการหนัก)", "ประเมินรวดเร็ว + monitor ตามอาการ", "เปิดเส้น IV และให้การรักษาตาม protocol"]
        if level == 3:
            return ["ส่งเข้าโซนเหลือง (เร่งด่วน)", "ประเมินตามลำดับความเร่งด่วน", "ให้ IV/ยา ตามความจำเป็น"]
        if level == 4:
            return ["ส่งเข้าโซนเขียว (อาการเล็กน้อย)", "ให้การดูแลตามอาการ/พิจารณาตรวจพื้นฐาน"]
        return ["ส่งเข้าโซนขาว (Fast‑track/คลินิก)", "แจ้งสัญญาณอันตรายและคำแนะนำกลับมาพบแพทย์"]
    else:
        if level == 1:
            return ["Assign to **Blue zone** (Resuscitation) immediately", "Activate resuscitation team; continuous monitoring (ECG, SpO₂, BP)", "High‑flow O₂; prepare BVM/advanced airway", "2 large‑bore IV/IO; fluids per protocol"]
        if level == 2:
            return ["Assign to **Red zone** (High‑acuity)", "Rapid assessment; monitoring as indicated", "IV access; protocol‑based treatment"]
        if level == 3:
            return ["Assign to **Yellow zone** (Urgent)", "Timely assessment; monitoring as indicated", "IV/symptomatic care as needed"]
        if level == 4:
            return ["Assign to **Green zone** (Minor)", "Symptomatic relief; basic tests per protocol"]
        return ["Assign to **White zone** (Fast‑track/clinic)", "Safety‑net instructions and follow‑up advice"]


def predict_single(input_df: pd.DataFrame) -> float:
    # Your trained sklearn pipeline expects NUM_COLS + TEXT_COLS; we provide both.
    proba = MODEL.predict_proba(input_df)
    # Binary classifier: [p(class0), p(class1)] ⇒ ICU probability is [:,1]
    return float(proba[0][1])


def write_log(single_input: dict, icu_prob: float, level: int):
    if not log_predictions:
        return
    os.makedirs("logs", exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "session": st.session_state.get('session_id'),
        **single_input,
        "pred_icu": icu_prob,
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
            gender = st.selectbox(T['gender'], options=[T['female'] if LANG_KEY=='en' else 'ญ', T['male'] if LANG_KEY=='en' else 'ช'], index=1)
            arrival = st.selectbox(T['arrival'], options=[T['walkin'],T['ems'],T['referral']], index=1)
            case_type = st.selectbox(T['case_type'], options=[T['trauma'],T['nontrauma']], index=1)
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
        gender_raw = gender if LANG_KEY=='th' else ('ญ' if gender=='F' else 'ช')
        arrival_raw = {'Walk-in':'Walkin','EMS':'EMS','Referral':'Referral',
                       'เดินมาเอง':'Walkin','EMS':'EMS','ส่งต่อ':'Referral'}[arrival]
        case_raw = {'Trauma':'T','Non-trauma':'N','อุบัติเหตุ':'T','ไม่ใช่อุบัติเหตุ':'N'}[case_type]

        # Build data frame with numeric/cat + 512-d text embedding
        use_vec = embed_text(cc)
        text_df = pd.DataFrame(use_vec, columns=TEXT_COLS)

        num_df = pd.DataFrame([[age, sbp, dbp, temp, pr, rr, o2sat, gcs_e, gcs_v, gcs_m, gender_raw, arrival_raw, case_raw]],
                              columns=NUM_COLS)
        input_df = pd.concat([num_df, text_df], axis=1)

with right:
    st.subheader(T['recommendation'])
    if submitted and input_df is not None:
        with st.spinner("Analyzing..." if LANG_KEY=='en' else "กำลังประมวลผล..."):
            try:
                icu_prob = predict_single(input_df)
                vitals = dict(sbp=sbp, o2sat=o2sat, rr=rr, temp=temp, gcs_e=gcs_e, gcs_v=gcs_v, gcs_m=gcs_m)
                level, css, why = triage_decision(icu_prob, vitals)
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
                with st.container(border=True):
                    label = T['icu']
                    st.metric("Probability" if LANG_KEY=='en' else 'ความน่าจะเป็น', f"{icu_prob*100:.1f}%")
                    st.progress(min(max(icu_prob, 0.0), 1.0))

                # Download result
                out_row = {**input_df.iloc[0].to_dict(), "pred_icu": icu_prob, "triage_level": level, "zone": zone_name}
                out_df = pd.DataFrame([out_row])
                st.download_button(
                    T['download'],
                    data=out_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"triage_result_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                if log_predictions:
                    write_log(num_df.iloc[0].to_dict() | {"cc": cc}, icu_prob, level)

            except Exception as e:
                st.error(("Prediction failed: " if LANG_KEY=='en' else "ไม่สามารถประมวลผลได้: ") + f"{type(e).__name__}: {e}")

# ---------------------------
# Tabs — Tutorial, Evidence, About, Contact
# ---------------------------
TAB_TITLES = LANGS[LANG_KEY]['tabs']
with st.container():
    t1, t2, t3, t4 = st.tabs(TAB_TITLES)

    with t1:
        if LANG_KEY=='en':
            st.markdown("""
            **How to use**
            1) Fill in patient details and chief complaint at the entrance.
            2) Press **Get recommendation**.
            3) Assign the suggested **zone** (Blue/Red/Yellow/Green/White). Red‑flags auto‑promote to Level 1.
            4) Follow **Suggested actions** and local protocols.
            5) Optionally, download the result or enable CSV logging in the sidebar.
            """)
        else:
            st.markdown("""
            **วิธีใช้งาน**
            1) กรอกข้อมูลผู้ป่วยและอาการสำคัญที่หน้า ER
            2) กด **ขอคำแนะนำ**
            3) จัดผู้ป่วยเข้า **โซน** ที่แนะนำ (น้ำเงิน/แดง/เหลือง/เขียว/ขาว) สัญญาณเตือนชีพจะปรับเป็นระดับ 1 อัตโนมัติ
            4) ปฏิบัติตาม **การดำเนินการทันที** และแนวทางของหน่วยงาน
            5) ดาวน์โหลดผลลัพธ์หรือเปิดบันทึก CSV ได้ทางแถบด้านซ้าย
            """)

    with t2:
        st.markdown("""
        **Model development & validation**  
        - Structured features: demographics, vital signs, GCS, arrival mode, trauma flag.  
        - Free‑text chief complaint encoded with **Multilingual Universal Sentence Encoder (USE)**.  
        - Trained gradient‑boosted model with calibration; evaluated with AUROC/AUPRC and **bootstrap** stability + **calibration** checks.  
        - Intended as **front‑door triage decision support** for EMS crews, triage nurses, and ED doctors.
        
        **Preprint**: https://sciety-labs.elifesciences.org/articles/by?article_doi=10.21203/rs.3.rs-6229836/v1  
        **Team**: Patipan Sitthiprawiat, Borwon Wittayachamnankul, Wachiranun Sirikul, Korsin Laohavisudhi — Chiang Mai University Faculty of Medicine, **Emergency Department & Informatics**.
        """)

    with t3:
        st.markdown("""
        **Patipan Sitthiprawiat, MD**  
        Emergency Medicine, Chiang Mai University.  
        Interests: clinical AI, emergency triage, and health informatics.
        """)

    with t4:
        st.markdown("""
        **Contact / Feedback**  
        - Email: *please insert here*  
        - Issue tracker: *insert link (e.g., GitHub, Google Form)*  
        - Notes: No patient‑identifying data should be uploaded. This tool is not a substitute for clinical judgment.
        """)

# ---------------------------
# Footer / Evidence (fixed)
# ---------------------------
st.markdown("---")
st.caption(LANGS[LANG_KEY]['footer_note'])
st.caption((LANGS[LANG_KEY]['footer_evidence'] + " https://sciety-labs.elifesciences.org/articles/by?article_doi=10.21203/rs.3.rs-6229836/v1"))
st.caption(LANGS[LANG_KEY]['footer_team'])
st.caption(f"Build {APP_VERSION} • Session {st.session_state.get('session_id')}")
