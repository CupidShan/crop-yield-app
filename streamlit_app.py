import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Crop Yield Predictor | University of Sunderland",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.stApp { background-color: #F8FAF9; }
.top-banner {
    background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 50%, #40916C 100%);
    padding: 2rem 3rem; border-radius: 0 0 16px 16px; margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(27,67,50,0.3);
}
.top-banner h1 { color: white; font-size: 2.2rem; font-weight: 700; margin: 0; }
.top-banner p  { color: rgba(255,255,255,0.85); font-size: 1rem; margin: 0.4rem 0 0 0; font-weight: 300; }
.banner-badge {
    display: inline-block; background: rgba(255,255,255,0.15); color: white;
    padding: 4px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 500;
    margin-top: 0.8rem; border: 1px solid rgba(255,255,255,0.25);
}
.how-to-box {
    background: #EBF5FB; border: 1px solid #AED6F1; border-left: 5px solid #2E86C1;
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1.2rem; font-size: 0.88rem; color: #1A5276;
}
.how-to-box h4 { margin: 0 0 0.5rem 0; font-size: 0.95rem; color: #1A5276; }
.how-to-step { padding: 3px 0; line-height: 1.6; }
.step-card {
    background: white; border-radius: 12px; padding: 1.4rem 1.5rem; margin-bottom: 1rem;
    border: 1px solid #E8F0EB; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.step-header { display: flex; align-items: center; gap: 12px; margin-bottom: 0; }
.step-number {
    background: linear-gradient(135deg, #2D6A4F, #40916C); color: white;
    width: 30px; height: 30px; border-radius: 50%; display: flex;
    align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem; flex-shrink: 0;
}
.step-title { font-size: 1rem; font-weight: 600; color: #1B4332; margin: 0; }
.hint-box {
    background: #F0F4F1; border-radius: 8px; padding: 0.6rem 0.9rem;
    font-size: 0.8rem; color: #4A6741; margin-top: 0.4rem; line-height: 1.5;
}
.result-card {
    background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%);
    border-radius: 16px; padding: 2rem; margin: 0.5rem 0;
    box-shadow: 0 8px 32px rgba(27,67,50,0.25);
}
.result-title  { color: rgba(255,255,255,0.75); font-size: 0.8rem; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.4rem; }
.result-value  { color: white; font-size: 3rem; font-weight: 700; line-height: 1; margin-bottom: 0.2rem; }
.result-unit   { color: rgba(255,255,255,0.65); font-size: 0.95rem; }
.unit-explainer { background: rgba(255,255,255,0.12); border-radius: 8px; padding: 0.6rem 0.9rem; margin-top: 0.8rem; font-size: 0.82rem; color: rgba(255,255,255,0.85); line-height: 1.6; }
.metric-box    { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 10px; padding: 0.9rem; text-align: center; }
.metric-label  { color: rgba(255,255,255,0.65); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value  { color: white; font-size: 1.2rem; font-weight: 600; margin-top: 4px; }
.interp-band   { border-radius: 10px; padding: 1.1rem 1.3rem; margin-top: 1rem; font-size: 0.9rem; line-height: 1.65; }
.interp-low    { background: #FFF3CD; border-left: 5px solid #FFC107; color: #664D03; }
.interp-medium { background: #D1ECF1; border-left: 5px solid #17A2B8; color: #0C5460; }
.interp-high   { background: #D4EDDA; border-left: 5px solid #28A745; color: #155724; }
.interp-very   { background: #CCE5FF; border-left: 5px solid #004085; color: #004085; }
.warn-box {
    background: #FFF8E6; border: 1px solid #FFD666; border-left: 4px solid #FFA500;
    border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; font-size: 0.83rem; color: #7A4F00;
}
.disclaimer {
    background: #F1F3F5; border-radius: 10px; padding: 1rem 1.2rem;
    font-size: 0.78rem; color: #6C757D; margin-top: 1rem; line-height: 1.6;
}
.sidebar-section { background: white; border-radius: 10px; padding: 0.9rem 1rem; margin-bottom: 0.8rem; border: 1px solid #E8F0EB; }
.sidebar-title { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: #2D6A4F; margin-bottom: 0.5rem; }
.sidebar-metric { display: flex; justify-content: space-between; align-items: center; padding: 4px 0; border-bottom: 1px solid #F0F4F1; font-size: 0.8rem; }
.sidebar-metric:last-child { border-bottom: none; }
.metric-key { color: #6C757D; } .metric-val { color: #1B4332; font-weight: 600; }
.stButton > button {
    background: linear-gradient(135deg, #1B4332, #40916C) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 0.7rem 2rem !important; font-size: 1rem !important; font-weight: 600 !important;
    width: 100% !important; box-shadow: 0 4px 15px rgba(27,67,50,0.3) !important;
}
.stButton > button:hover { box-shadow: 0 6px 20px rgba(27,67,50,0.45) !important; }
label { font-weight: 500 !important; color: #344054 !important; font-size: 0.87rem !important; }
.footer { text-align: center; padding: 1.5rem; color: #9CA3AF; font-size: 0.77rem; margin-top: 2rem; border-top: 1px solid #E8F0EB; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import gzip
    with gzip.open("rf_model_compressed.pkl.gz", "rb") as f:
        model = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        enc = pickle.load(f)
    return model, enc

model, enc = load_model()
le_area  = enc['le_area']
le_item  = enc['le_item']
features = enc['features']

RANGES = {
    'average_rain_fall_mm_per_year': (51, 3240),
    'pesticides_tonnes':             (0.04, 367778),
    'avg_temp':                      (1.3, 30.6),
}

# ── Varied response messages per interpretation band ─────────────────────────
LOW_MSGS = [
    ("🔴", "Low Yield Estimate",
     "This result suggests limited crop productivity for the conditions you entered. Low yields can occur because of insufficient rainfall, very cold or very hot temperatures, or minimal pesticide use. This does not necessarily mean the crop will fail — it means conditions are less than ideal for high output."),
    ("🔴", "Below Average Yield Expected",
     "The model predicts a below-average yield for these inputs. This could reflect a challenging climate, low farming intensity, or a combination of both. In practical terms, output per hectare would be relatively modest under these conditions."),
    ("🔴", "Challenging Growing Conditions Detected",
     "Based on the values you entered, the model expects a low yield outcome. Environmental conditions appear unfavourable — consider adjusting rainfall, temperature, or pesticide values to explore how yield might change."),
]
MEDIUM_MSGS = [
    ("🟡", "Moderate Yield Estimate",
     "This prediction falls within the typical range for this crop under average growing conditions. A moderate yield suggests the environmental inputs are reasonable but not optimal. In practice, this level of output would be considered acceptable for most smallholder farming contexts."),
    ("🟡", "Average Growing Conditions",
     "The model predicts a moderate yield, which is broadly in line with what you might expect under typical environmental conditions for this crop and country. Neither particularly high nor particularly low — a standard outcome."),
    ("🟡", "Yield Within Normal Range",
     "These inputs produce a prediction in the middle range. The growing conditions you have described are workable — not ideal, but not severely limiting either. This is the kind of result you might see for average seasonal conditions."),
]
HIGH_MSGS = [
    ("🟢", "High Yield Estimate",
     "The model predicts a strong yield for these conditions. High yields are typically associated with favourable climate, adequate rainfall, and higher farming intensity. This result suggests the environmental inputs you have entered are well-suited to growing this crop."),
    ("🟢", "Favourable Growing Conditions",
     "These inputs suggest excellent conditions for crop production. The predicted yield is well above average, which typically reflects a combination of good climate, sufficient water, and active pest management. In agricultural terms, this would be considered a productive season."),
    ("🟢", "Strong Productivity Expected",
     "Based on the conditions entered, the model expects a high crop yield. This is a positive result — the environmental factors appear to be working in favour of good agricultural output for this crop type in this location."),
]
VERY_HIGH_MSGS = [
    ("🔵", "Exceptionally High Estimate",
     "This prediction is very high — above the typical range for most crops. Please double-check that all your inputs are realistic for the crop and country you selected. Exceptionally high results can sometimes occur when inputs are at the extreme end of the scale."),
    ("🔵", "Unusually High Prediction — Please Review Inputs",
     "The model has returned an exceptionally high yield estimate. While this is technically possible under ideal conditions, it is worth reviewing your input values to make sure they accurately reflect the scenario you are modelling."),
    ("🔵", "Very High Output Predicted",
     "This is among the highest yield predictions the model can return. Results at this level are rare and may reflect an unusually productive scenario — or may indicate that one or more inputs is outside the realistic range for this crop and country. Worth reviewing before drawing conclusions."),
]

def pick_msg(pred):
    if pred < 20000:   return random.choice(LOW_MSGS)
    elif pred < 60000: return random.choice(MEDIUM_MSGS)
    elif pred < 150000:return random.choice(HIGH_MSGS)
    else:              return random.choice(VERY_HIGH_MSGS)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 0.5rem;'>
        <div style='font-size:2.5rem;'>🌾</div>
        <div style='font-weight:700; color:#1B4332; font-size:1rem;'>Crop Yield Predictor</div>
        <div style='color:#6C757D; font-size:0.75rem; margin-top:2px;'>University of Sunderland</div>
    </div>
    <hr style='border:none; border-top:1px solid #E8F0EB; margin:0.8rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-metric"><span class="metric-key">Algorithm</span><span class="metric-val">Random Forest</span></div>
        <div class="sidebar-metric"><span class="metric-key">R² Score</span><span class="metric-val">0.9857</span></div>
        <div class="sidebar-metric"><span class="metric-key">Mean Abs. Error</span><span class="metric-val">3,752 hg/ha</span></div>
        <div class="sidebar-metric"><span class="metric-key">Accuracy (±10%)</span><span class="metric-val">80.60%</span></div>
        <div class="sidebar-metric"><span class="metric-key">Training Samples</span><span class="metric-val">22,593</span></div>
        <div class="sidebar-metric"><span class="metric-key">Countries</span><span class="metric-val">101</span></div>
        <div class="sidebar-metric"><span class="metric-key">Crop Types</span><span class="metric-val">10</span></div>
        <div class="sidebar-metric"><span class="metric-key">Data Period</span><span class="metric-val">1990–2013</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">Valid Input Ranges</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-metric"><span class="metric-key">Rainfall</span><span class="metric-val">51–3,240 mm</span></div>
        <div class="sidebar-metric"><span class="metric-key">Temperature</span><span class="metric-val">1.3–30.6 °C</span></div>
        <div class="sidebar-metric"><span class="metric-key">Pesticides</span><span class="metric-val">0.04–367,778 t</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">Important Notes</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-section" style='font-size:0.78rem; color:#6C757D; line-height:1.7;'>
        ⚠️ Soil and irrigation data not included<br>
        ⚠️ Inputs outside training range may be unreliable<br>
        ⚠️ Data covers 1990–2013 only<br>
        ⚠️ For research and planning support only
    </div>
    """, unsafe_allow_html=True)

# ── Banner ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
    <h1>🌾 Crop Yield Predictor</h1>
    <p>Machine learning-based crop yield estimation using environmental and soil data</p>
    <span class="banner-badge">MSc Computing Dissertation &nbsp;·&nbsp; University of Sunderland &nbsp;·&nbsp; 2026</span>
</div>
""", unsafe_allow_html=True)

# ── How to use guide ──────────────────────────────────────────────────────────
st.markdown("""
<div class="how-to-box">
    <h4>📖 How to Use This Tool</h4>
    <div class="how-to-step">① &nbsp;<strong>Step 1</strong> — Choose a crop type and a country from the dropdown menus below.</div>
    <div class="how-to-step">② &nbsp;<strong>Step 2</strong> — Enter the environmental conditions: annual rainfall, average temperature, and pesticide usage. Default values are already filled in — you can change them to explore different scenarios.</div>
    <div class="how-to-step">③ &nbsp;<strong>Step 3</strong> — Click the green <strong>Generate Prediction</strong> button. Your result will appear on the right side of the screen.</div>
    <div class="how-to-step">④ &nbsp;Try different combinations to see how the prediction changes. Use the <strong>Reset</strong> button to start fresh at any time.</div>
</div>
""", unsafe_allow_html=True)

# ── Session state for reset ───────────────────────────────────────────────────
if 'reset' not in st.session_state:
    st.session_state.reset = False

# ── Two-column layout ─────────────────────────────────────────────────────────
col_l, col_r = st.columns([1.1, 0.9], gap="large")

with col_l:
    st.markdown('<div class="step-card"><div class="step-header"><div class="step-number">1</div><p class="step-title">Select Crop Type and Country</p></div></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        crop = st.selectbox(
            "Crop Type",
            sorted(le_item.classes_),
            help="Choose the type of crop you want to predict the yield for. The tool covers 10 crop types including wheat, rice, maize, potatoes, cassava, sorghum, soybeans, sweet potatoes, plantains, and yams."
        )
    with c2:
        country = st.selectbox(
            "Country",
            sorted(le_area.classes_),
            help="Choose the country where the crop is being grown. The tool covers 101 countries worldwide."
        )

    year = st.slider(
        "Harvest Year",
        min_value=1990, max_value=2030, value=2013, step=1,
        help="Select the year of harvest. The model was trained on data from 1990 to 2013 — years beyond 2013 may produce less reliable results."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="step-card"><div class="step-header"><div class="step-number">2</div><p class="step-title">Enter Environmental Conditions</p></div></div>', unsafe_allow_html=True)

    # Rainfall
    rainfall = st.number_input(
        "Annual Rainfall (mm per year)",
        min_value=0.0, max_value=20000.0, value=1016.0, step=10.0,
        help="The total amount of rainfall the country receives in a year, measured in millimetres. For example, the UK receives about 885 mm per year. The valid range for this model is 51 to 3,240 mm."
    )
    st.markdown('<div class="hint-box">💡 <strong>What to enter:</strong> The average yearly rainfall for the country and crop in millimetres. Example: 1,016 mm is close to the dataset average. Try values between 200 and 2,000 mm for realistic results.</div>', unsafe_allow_html=True)

    # Temperature
    temperature = st.number_input(
        "Average Annual Temperature (°C)",
        min_value=-50.0, max_value=60.0, value=17.9, step=0.5,
        help="The average temperature across the whole year, in degrees Celsius. The valid range for this model is 1.3°C to 30.6°C. Temperatures outside this range may produce unreliable predictions."
    )
    st.markdown('<div class="hint-box">💡 <strong>What to enter:</strong> The average yearly temperature in degrees Celsius. Example: 17.9°C is the dataset average. Typical values range from 5°C (cooler countries) to 28°C (tropical countries).</div>', unsafe_allow_html=True)

    # Pesticides
    pesticides = st.number_input(
        "Pesticide Usage (tonnes per year)",
        min_value=0.0, max_value=500000.0, value=36713.0, step=100.0,
        help="The total amount of pesticides used in the country per year, measured in tonnes. Higher values generally indicate more intensive farming systems. The valid range is 0.04 to 367,778 tonnes."
    )
    st.markdown('<div class="hint-box">💡 <strong>What to enter:</strong> How much pesticide is used nationally per year in tonnes. Example: 36,713 tonnes is close to the dataset average. Try 0 to see what happens with no pesticide use, or very high values to simulate intensive farming.</div>', unsafe_allow_html=True)

    # Warnings
    st.markdown("<br>", unsafe_allow_html=True)
    warns = []
    if rainfall == 0:
        warns.append("Rainfall is set to 0 mm. If this represents missing data rather than genuine zero precipitation, the prediction may be unreliable.")
    elif not (51 <= rainfall <= 3240):
        warns.append(f"Rainfall of {rainfall:.0f} mm is outside the model's training range (51–3,240 mm). The prediction may be less reliable.")
    if pesticides == 0:
        warns.append("Pesticide usage is set to 0. The model may significantly underestimate yield for well-managed farming systems.")
    if not (1.3 <= temperature <= 30.6):
        warns.append(f"Temperature of {temperature:.1f}°C is outside the model's training range (1.3–30.6°C). Predictions at extreme temperatures may be misleading.")
    if year > 2013:
        warns.append(f"Year {year} is beyond the training period (1990–2013). The model does not account for changes in farming practices or climate after 2013.")
    for w in warns:
        st.markdown(f'<div class="warn-box">⚠️ {w}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="step-card"><div class="step-header"><div class="step-number">3</div><p class="step-title">Generate Your Prediction</p></div></div>', unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        predict_btn = st.button("🌾 Generate Crop Yield Prediction", type="primary")
    with btn_col2:
        reset_btn = st.button("🔄 Reset", help="Click to reset all inputs back to the default values.")

# ── Right column — Results ─────────────────────────────────────────────────────
with col_r:
    if reset_btn:
        st.rerun()

    elif predict_btn:
        try:
            area_enc = le_area.transform([country])[0]
            item_enc = le_item.transform([crop])[0]
            row  = pd.DataFrame([[area_enc, item_enc, year, rainfall, pesticides, temperature]], columns=features)
            pred = model.predict(row)[0]
            pred_kg = pred / 10
            pred_t  = pred / 100000

            icon, label, desc = pick_msg(pred)

            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">Predicted Crop Yield</div>
                <div class="result-value">{pred:,.0f}</div>
                <div class="result-unit">hectograms per hectare (hg/ha)</div>
                <div class="unit-explainer">
                    📏 <strong>What does hg/ha mean?</strong><br>
                    hg/ha stands for <em>hectograms per hectare</em> — the standard international unit for measuring how much crop is harvested per unit of land.
                    One hectogram = 100 grams, so <strong>{pred:,.0f} hg/ha = {pred_kg:,.0f} kg/ha = {pred_t:.2f} tonnes per hectare</strong>.
                    To put this in context, global average wheat yield is around 34,000 hg/ha.
                </div>
                <br>
                <div style="display:flex; gap:10px; margin-top:4px;">
                    <div class="metric-box" style="flex:1;"><div class="metric-label">Kilograms / ha</div><div class="metric-value">{pred_kg:,.0f}</div></div>
                    <div class="metric-box" style="flex:1;"><div class="metric-label">Tonnes / ha</div><div class="metric-value">{pred_t:.2f}</div></div>
                    <div class="metric-box" style="flex:1;"><div class="metric-label">Crop Selected</div><div class="metric-value" style="font-size:0.82rem;">{crop}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="interp-band interp-{'low' if pred < 20000 else 'medium' if pred < 60000 else 'high' if pred < 150000 else 'very'}">
                <strong>{icon} {label}</strong><br>{desc}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Your Input Summary**")
            st.dataframe(pd.DataFrame({
                'Parameter': ['Crop Type', 'Country', 'Year', 'Rainfall (mm/yr)', 'Temperature (°C)', 'Pesticides (tonnes)'],
                'Value You Entered': [crop, country, str(year), f"{rainfall:,.1f}", f"{temperature:.1f}", f"{pesticides:,.2f}"]
            }), use_container_width=True, hide_index=True)

            st.markdown("""
            <div class="disclaimer">
                <strong>⚠️ Research Disclaimer:</strong> This prediction is generated by a Random Forest machine learning model
                trained on global agricultural data covering 101 countries and 10 crop types (1990–2013).
                It is intended for research and educational purposes only. The model does not incorporate
                soil composition, irrigation, crop variety, or farm management practices.
                Predictions should not be used as the sole basis for agricultural, commercial, or policy decisions.
                Please consult a qualified agronomist for professional agricultural advice.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}. Please check your inputs and try again.")

    else:
        st.markdown("""
        <div style='background:white; border:2px dashed #C8DDD0; border-radius:16px;
                    padding:3rem 2rem; text-align:center; margin-top:0.5rem;'>
            <div style='font-size:3rem; margin-bottom:1rem;'>🌱</div>
            <div style='font-size:1.1rem; font-weight:600; color:#1B4332; margin-bottom:0.5rem;'>
                Your Prediction Will Appear Here
            </div>
            <div style='color:#6C757D; font-size:0.87rem; line-height:1.7;'>
                Follow the three steps on the left to generate<br>
                your crop yield prediction.
            </div>
        </div>
        <br>
        <div class="step-card">
            <div style='font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.8px; color:#2D6A4F; margin-bottom:0.8rem;'>About This Model</div>
            <div style='font-size:0.84rem; color:#495057; line-height:1.9;'>
                🌍 <strong>101 countries</strong> included in the training data<br>
                🌾 <strong>10 crop types</strong> — wheat, rice, maize, potatoes, cassava and more<br>
                📊 <strong>3 environmental inputs</strong>: rainfall, temperature, pesticides<br>
                🤖 <strong>Random Forest algorithm</strong> — 100 decision trees combined<br>
                📈 <strong>R² = 0.9857</strong> — explains 98.6% of yield variation<br>
                🎯 <strong>Accuracy = 80.60%</strong> — 8 in 10 predictions within 10% of actual yield
            </div>
        </div>
        <div class="step-card" style="margin-top:0.5rem;">
            <div style='font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.8px; color:#2D6A4F; margin-bottom:0.8rem;'>Try These Example Scenarios</div>
            <div style='font-size:0.83rem; color:#495057; line-height:1.9;'>
                🇦🇱 <strong>Maize in Albania (average conditions)</strong><br>
                &nbsp;&nbsp;&nbsp;Rainfall: 1,016 mm · Temp: 17.9°C · Pesticides: 36,713 t<br><br>
                🌧️ <strong>Test extreme rainfall</strong><br>
                &nbsp;&nbsp;&nbsp;Try setting Rainfall to 0 or to 9,000 mm<br><br>
                🌡️ <strong>Test extreme temperature</strong><br>
                &nbsp;&nbsp;&nbsp;Try setting Temperature to -20°C or 50°C<br><br>
                🚫 <strong>Test zero pesticide use</strong><br>
                &nbsp;&nbsp;&nbsp;Set Pesticides to 0 and observe the result
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Developed by <strong>Emmanuel Ogbonnaya Egwu</strong> &nbsp;|&nbsp;
    MSc Computing Dissertation &nbsp;|&nbsp; University of Sunderland &nbsp;|&nbsp; 2026 &nbsp;|&nbsp;
    Supervised by <strong>Ian Evans</strong>
</div>
""", unsafe_allow_html=True)
