import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb 
from geopy.distance import geodesic
import plotly.graph_objects as go
import io

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield · Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Exo+2:wght@300;400;600&display=swap');

:root {
    --bg-primary:    #070b14;
    --bg-card:       #0d1526;
    --bg-input:      #101d33;
    --border:        #1a3a6b;
    --border-glow:   #1e5bcc;
    --accent-blue:   #1e7cf5;
    --accent-cyan:   #00d4ff;
    --accent-green:  #00f5a0;
    --accent-red:    #ff3d6b;
    --text-primary:  #e8f0fe;
    --text-secondary:#7a9cc4;
    --text-mono:     #4dd9ff;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    font-family: 'Exo 2', sans-serif;
}
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(30,124,245,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(30,124,245,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}
[data-testid="stMain"] { position: relative; z-index: 1; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080f1f 0%, #0a1628 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

.sidebar-logo {
    text-align: center;
    padding: 1rem 1rem 1.8rem;
    border-bottom: 1px solid rgba(30,124,245,0.2);
    margin-bottom: 1.5rem;
}
.sidebar-logo .icon { font-size: 2.4rem; display: block; filter: drop-shadow(0 0 14px #1e7cf5); }
.sidebar-logo .name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 4px;
    color: var(--text-primary);
    text-transform: uppercase;
}
.sidebar-logo .name span { color: var(--accent-cyan); }
.sidebar-logo .ver {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-secondary);
    letter-spacing: 2px;
    margin-top: 0.2rem;
}

/* ── Header ── */
.fraud-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
    margin-bottom: 0.5rem;
}
.fraud-header .shield-icon {
    font-size: 3.2rem;
    display: block;
    margin-bottom: 0.4rem;
    filter: drop-shadow(0 0 18px #1e7cf5);
    animation: pulse-icon 2.5s ease-in-out infinite;
}
@keyframes pulse-icon {
    0%, 100% { filter: drop-shadow(0 0 18px #1e7cf5); }
    50%       { filter: drop-shadow(0 0 32px #00d4ff); }
}
.fraud-header h1 {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 4px !important;
    color: var(--text-primary) !important;
    text-transform: uppercase;
    margin: 0 !important;
    padding: 0 !important;
}
.fraud-header h1 span { color: var(--accent-cyan); }
.fraud-header .subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-secondary);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* ── Dividers & Labels ── */
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
    position: relative;
}
.section-divider::after {
    content: attr(data-label);
    position: absolute;
    top: -0.6rem;
    left: 50%;
    transform: translateX(-50%);
    background: var(--bg-primary);
    padding: 0 1rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent-blue);
    letter-spacing: 3px;
    text-transform: uppercase;
}
.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 3px;
    color: var(--accent-blue);
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    padding-left: 0.5rem;
    border-left: 2px solid var(--accent-blue);
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div {
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.92rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: var(--border-glow) !important;
    box-shadow: 0 0 0 2px rgba(30,124,245,0.15) !important;
}
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label {
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(30,124,245,0.04) !important;
    border: 1.5px dashed var(--border-glow) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}

/* ── Info Badge ── */
.info-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(30,124,245,0.08);
    border: 1px solid rgba(30,124,245,0.25);
    border-radius: 20px;
    padding: 0.35rem 0.9rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-mono);
    margin-bottom: 1.5rem;
}

/* ── Distance Card ── */
.distance-card {
    background: linear-gradient(135deg, rgba(0,212,255,0.06), rgba(30,124,245,0.04));
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.5rem;
}
.distance-card .label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-secondary);
    letter-spacing: 2px;
    text-transform: uppercase;
}
.distance-card .value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent-cyan);
    line-height: 1;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.2rem 0;
}
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-card.total::before { background: linear-gradient(90deg, #1e7cf5, #00d4ff); }
.kpi-card.fraud::before { background: linear-gradient(90deg, #ff3d6b, #ff8c00); }
.kpi-card.legit::before { background: linear-gradient(90deg, #00f5a0, #00d4ff); }
.kpi-card .kpi-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.kpi-card.total .kpi-value { color: var(--accent-cyan); }
.kpi-card.fraud .kpi-value { color: var(--accent-red); }
.kpi-card.legit .kpi-value { color: var(--accent-green); }
.kpi-card .kpi-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    color: var(--text-secondary);
    letter-spacing: 2px;
    text-transform: uppercase;
}
.kpi-card .kpi-pct {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.78rem;
    margin-top: 0.3rem;
    opacity: 0.7;
}
.kpi-card.fraud .kpi-pct { color: var(--accent-red); }
.kpi-card.legit .kpi-pct { color: var(--accent-green); }

/* ── Buttons ── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #0f3d8a, #1e7cf5) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 4px !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2rem !important;
    margin-top: 1.2rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(30,124,245,0.25) !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #1e7cf5, #00d4ff) !important;
    box-shadow: 0 6px 28px rgba(30,124,245,0.5) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button:active { transform: translateY(0px) !important; }

/* ── Results ── */
.result-fraud {
    background: linear-gradient(135deg, rgba(255,61,107,0.12), rgba(255,61,107,0.05));
    border: 1px solid rgba(255,61,107,0.4);
    border-left: 4px solid var(--accent-red);
    border-radius: 10px;
    padding: 1.5rem 1.8rem;
    margin-top: 1.5rem;
    text-align: center;
}
.result-legit {
    background: linear-gradient(135deg, rgba(0,245,160,0.10), rgba(0,245,160,0.04));
    border: 1px solid rgba(0,245,160,0.4);
    border-left: 4px solid var(--accent-green);
    border-radius: 10px;
    padding: 1.5rem 1.8rem;
    margin-top: 1.5rem;
    text-align: center;
}
.result-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 0.4rem 0 0.2rem;
}
.result-icon { font-size: 2.4rem; display: block; margin-bottom: 0.2rem; }
.result-fraud .result-title { color: var(--accent-red); }
.result-legit .result-title { color: var(--accent-green); }
.result-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-secondary);
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── Error ── */
[data-testid="stAlert"] {
    background: rgba(255,61,107,0.08) !important;
    border: 1px solid rgba(255,61,107,0.3) !important;
    border-radius: 8px !important;
    color: #ffb3c1 !important;
    font-family: 'Exo 2', sans-serif !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    color: #2a4a7a;
    letter-spacing: 2px;
    text-transform: uppercase;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-glow); }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models ────────────────────────────────────────────────────────────────
model   = joblib.load("fraud_detection_model.jb")
encoder = joblib.load("label_encoder.jb")

def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="icon">🛡️</span>
        <div class="name">Fraud<span>Shield</span></div>
        <div class="ver">v2.0 · ML-Powered</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**NAVIGATION**")
    page = st.radio(
        label="",
        options=["🔍  Single Transaction", "📂  Batch CSV Analysis"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#2a4a7a;letter-spacing:2px;text-transform:uppercase;line-height:2.2">
    Model: LightGBM<br>
    Engine: Geodesic<br>
    Status: <span style="color:#00f5a0">●</span> Online
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE TRANSACTION
# ════════════════════════════════════════════════════════════════════════════════
if "Single" in page:

    st.markdown("""
    <div class="fraud-header">
        <span class="shield-icon">🛡️</span>
        <h1>Fraud<span>Shield</span></h1>
        <div class="subtitle">Real-time Transaction Risk Assessment</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="info-badge">⬡ &nbsp;ML-Powered · LightGBM · Geodesic Analysis</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">01 — Merchant Information</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        merchant = st.text_input("Merchant Name")
    with col2:
        category = st.text_input("Category")
    amt = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")

    st.markdown('<hr class="section-divider" data-label="Location Data">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 — Geolocation Data</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        lat  = st.number_input("Latitude",  format="%.6f")
        long = st.number_input("Longitude", format="%.6f")
    with col4:
        merch_lat  = st.number_input("Merchant Latitude",  format="%.6f")
        merch_long = st.number_input("Merchant Longitude", format="%.6f")

    distance = haversine(lat, long, merch_lat, merch_long)

    st.markdown(f"""
    <div class="distance-card">
        <span style="font-size:1.4rem">📡</span>
        <div>
            <div class="label">Computed Distance</div>
            <div class="value">{distance:.2f} <span style="font-size:0.9rem;color:#7a9cc4">km</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider" data-label="Temporal Data">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 — Temporal Details</div>', unsafe_allow_html=True)
    col5, col6, col7 = st.columns(3)
    with col5:
        hour  = st.slider("Transaction Hour",  0,  23, 12)
    with col6:
        day   = st.slider("Transaction Day",   1,  31, 15)
    with col7:
        month = st.slider("Transaction Month", 1,  12,  6)

    st.markdown('<hr class="section-divider" data-label="Cardholder">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 — Cardholder Information</div>', unsafe_allow_html=True)
    col8, col9 = st.columns(2)
    with col8:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col9:
        cc_num = st.text_input("Credit Card Number", type="password")

    if st.button("⚡  Run Fraud Analysis"):
        if merchant and category and cc_num:
            input_data = pd.DataFrame(
                [[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
                columns=['merchant','category','amt','distance','hour','day','month','gender','cc_num']
            )
            categorical_col = ['merchant','category','gender']
            for col in categorical_col:
                try:
                    input_data[col] = encoder[col].transform(input_data[col])
                except ValueError:
                    input_data[col] = -1

            input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.markdown("""
                <div class="result-fraud">
                    <span class="result-icon">🚨</span>
                    <div class="result-title">Fraudulent Transaction</div>
                    <div class="result-sub">High risk detected · Recommend immediate review</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-legit">
                    <span class="result-icon">✅</span>
                    <div class="result-title">Legitimate Transaction</div>
                    <div class="result-sub">No anomalies detected · Transaction cleared</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("⚠️  Please fill all required fields before running the analysis.")

    st.markdown('<div class="footer">FraudShield v2.0 · Powered by LightGBM · All data processed locally</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH CSV ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div class="fraud-header">
        <span class="shield-icon">📂</span>
        <h1>Batch <span>Analysis</span></h1>
        <div class="subtitle">CSV Transaction Report · Fraud Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="info-badge">⬡ &nbsp;Upload a CSV · Auto-Predict · Full Report</div>', unsafe_allow_html=True)

    with st.expander("📋  Expected CSV Columns"):
        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#7a9cc4;line-height:2.5;letter-spacing:1px;">
        <b style="color:#1e7cf5">merchant</b> &nbsp;·&nbsp; <b style="color:#1e7cf5">category</b> &nbsp;·&nbsp;
        <b style="color:#1e7cf5">amt</b> &nbsp;·&nbsp; <b style="color:#1e7cf5">lat</b> &nbsp;·&nbsp;
        <b style="color:#1e7cf5">long</b> &nbsp;·&nbsp; <b style="color:#1e7cf5">merch_lat</b> &nbsp;·&nbsp;
        <b style="color:#1e7cf5">merch_long</b> &nbsp;·&nbsp; <b style="color:#1e7cf5">hour</b> &nbsp;·&nbsp;
        <b style="color:#1e7cf5">day</b> &nbsp;·&nbsp; <b style="color:#1e7cf5">month</b> &nbsp;·&nbsp;
        <b style="color:#1e7cf5">gender</b> &nbsp;·&nbsp; <b style="color:#1e7cf5">cc_num</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Upload Your CSV File</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.markdown('<hr class="section-divider" data-label="Data Preview">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Raw Data Preview (first 5 rows)</div>', unsafe_allow_html=True)
        st.dataframe(df.head(5), use_container_width=True)

        if st.button("⚡  Run Batch Analysis"):
            required_cols = ['merchant','category','amt','lat','long','merch_lat','merch_long','hour','day','month','gender','cc_num']
            missing = [c for c in required_cols if c not in df.columns]

            if missing:
                st.error(f"⚠️  Missing columns: {', '.join(missing)}")
            else:
                with st.spinner("🔄  Analyzing transactions..."):
                    batch = df.copy()
                    batch['distance'] = batch.apply(
                        lambda r: haversine(r['lat'], r['long'], r['merch_lat'], r['merch_long']), axis=1
                    )

                    input_batch = batch[['merchant','category','amt','distance','hour','day','month','gender','cc_num']].copy()

                    for col in ['merchant','category','gender']:
                        def safe_encode(val, col=col):
                            try:    return encoder[col].transform([val])[0]
                            except: return -1
                        input_batch[col] = input_batch[col].apply(safe_encode)

                    input_batch['cc_num'] = input_batch['cc_num'].apply(lambda x: hash(str(x)) % (10 ** 2))
                    predictions = model.predict(input_batch)

                df['prediction']   = predictions
                df['result_label'] = df['prediction'].map({1: '🚨 Fraudulent', 0: '✅ Legitimate'})

                # ── KPIs ──────────────────────────────────────────────────────
                total     = len(df)
                n_fraud   = int(df['prediction'].sum())
                n_legit   = total - n_fraud
                pct_fraud = (n_fraud / total * 100) if total else 0
                pct_legit = (n_legit / total * 100) if total else 0

                st.markdown('<hr class="section-divider" data-label="Report Summary">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">Report Summary</div>', unsafe_allow_html=True)

                st.markdown(f"""
                <div class="kpi-grid">
                    <div class="kpi-card total">
                        <div class="kpi-value">{total:,}</div>
                        <div class="kpi-label">Total Transactions</div>
                        <div class="kpi-pct" style="color:#4dd9ff">100%</div>
                    </div>
                    <div class="kpi-card fraud">
                        <div class="kpi-value">{n_fraud:,}</div>
                        <div class="kpi-label">Fraudulent</div>
                        <div class="kpi-pct">{pct_fraud:.1f}%</div>
                    </div>
                    <div class="kpi-card legit">
                        <div class="kpi-value">{n_legit:,}</div>
                        <div class="kpi-label">Legitimate</div>
                        <div class="kpi-pct">{pct_legit:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Charts ─────────────────────────────────────────────────────
                st.markdown('<hr class="section-divider" data-label="Visual Analysis">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">Visual Analysis</div>', unsafe_allow_html=True)

                c1, c2 = st.columns(2)

                with c1:
                    fig_donut = go.Figure(go.Pie(
                        labels=['Fraudulent', 'Legitimate'],
                        values=[n_fraud, n_legit],
                        hole=0.65,
                        marker=dict(colors=['#ff3d6b','#00f5a0'],
                                    line=dict(color='#070b14', width=3)),
                        textinfo='percent',
                        textfont=dict(family='Rajdhani', size=14, color='#e8f0fe'),
                    ))
                    fig_donut.update_layout(
                        title=dict(text='Transaction Distribution', font=dict(family='Rajdhani', size=16, color='#e8f0fe'), x=0.5),
                        paper_bgcolor='#0d1526', plot_bgcolor='#0d1526',
                        font=dict(family='Exo 2', color='#7a9cc4'),
                        legend=dict(font=dict(family='Exo 2', color='#7a9cc4'), bgcolor='#0d1526'),
                        margin=dict(t=50, b=20, l=20, r=20),
                        annotations=[dict(
                            text=f'<b>{pct_fraud:.1f}%</b><br>Fraud',
                            x=0.5, y=0.5,
                            font=dict(family='Rajdhani', size=18, color='#ff3d6b'),
                            showarrow=False
                        )]
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

                with c2:
                    if 'category' in df.columns:
                        cat_fraud = df[df['prediction']==1].groupby('category').size().reset_index(name='count')
                        cat_fraud = cat_fraud.sort_values('count', ascending=True).tail(8)
                        fig_bar = go.Figure(go.Bar(
                            x=cat_fraud['count'],
                            y=cat_fraud['category'],
                            orientation='h',
                            marker=dict(
                                color=cat_fraud['count'],
                                colorscale=[[0,'#1e7cf5'],[0.5,'#ff8c00'],[1,'#ff3d6b']],
                                line=dict(color='#070b14', width=0.5)
                            ),
                            text=cat_fraud['count'],
                            textposition='outside',
                            textfont=dict(family='Rajdhani', color='#e8f0fe', size=12)
                        ))
                        fig_bar.update_layout(
                            title=dict(text='Fraud by Category', font=dict(family='Rajdhani', size=16, color='#e8f0fe'), x=0.5),
                            paper_bgcolor='#0d1526', plot_bgcolor='#0d1526',
                            xaxis=dict(showgrid=False, color='#7a9cc4', tickfont=dict(family='Share Tech Mono')),
                            yaxis=dict(showgrid=False, color='#7a9cc4', tickfont=dict(family='Share Tech Mono', size=10)),
                            margin=dict(t=50, b=20, l=20, r=50),
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                # Hourly activity
                if 'hour' in df.columns:
                    st.markdown('<div class="section-label">Fraud Activity by Hour</div>', unsafe_allow_html=True)
                    hourly = df.groupby('hour')['prediction'].sum().reset_index()
                    hourly.columns = ['hour','fraud_count']
                    fig_area = go.Figure()
                    fig_area.add_trace(go.Scatter(
                        x=hourly['hour'], y=hourly['fraud_count'],
                        fill='tozeroy',
                        line=dict(color='#ff3d6b', width=2),
                        fillcolor='rgba(255,61,107,0.15)',
                        mode='lines+markers',
                        marker=dict(color='#ff3d6b', size=6),
                        name='Fraud Count'
                    ))
                    fig_area.update_layout(
                        paper_bgcolor='#0d1526', plot_bgcolor='#0d1526',
                        xaxis=dict(showgrid=False, color='#7a9cc4',
                                   tickfont=dict(family='Share Tech Mono'),
                                   title=dict(text='Hour of Day', font=dict(family='Exo 2', color='#7a9cc4')),
                                   dtick=1),
                        yaxis=dict(showgrid=True, gridcolor='rgba(30,58,107,0.4)', color='#7a9cc4',
                                   tickfont=dict(family='Share Tech Mono'),
                                   title=dict(text='Fraudulent Transactions', font=dict(family='Exo 2', color='#7a9cc4'))),
                        margin=dict(t=20, b=40, l=60, r=20),
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_area, use_container_width=True)

                # ── Fraud Table ────────────────────────────────────────────────
                st.markdown('<hr class="section-divider" data-label="Fraud Records">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">Fraudulent Transactions Detail</div>', unsafe_allow_html=True)

                fraud_df = df[df['prediction'] == 1].drop(columns=['prediction'])
                if not fraud_df.empty:
                    st.dataframe(fraud_df, use_container_width=True)
                    csv_out = fraud_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇  Download Fraud Report (CSV)",
                        data=csv_out,
                        file_name="fraud_report.csv",
                        mime="text/csv",
                    )
                else:
                    st.markdown("""
                    <div class="result-legit">
                        <span class="result-icon">✅</span>
                        <div class="result-title">No Fraud Detected</div>
                        <div class="result-sub">All transactions in this file appear legitimate</div>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;border:1.5px dashed #1a3a6b;border-radius:14px;margin-top:2rem;">
            <div style="font-size:3rem;margin-bottom:1rem;opacity:0.5">📁</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:1.3rem;font-weight:600;color:#7a9cc4;letter-spacing:3px;text-transform:uppercase;">
                Upload a CSV File to Begin
            </div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#2a4a7a;letter-spacing:2px;margin-top:0.6rem;text-transform:uppercase;">
                Supports batch prediction · Automatic report generation
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="footer">FraudShield v2.0 · Powered by LightGBM · All data processed locally</div>', unsafe_allow_html=True)
