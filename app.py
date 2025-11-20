# FINSIGHT CORE V2.1 - "LIQUID GLASS" WITH TABS
# Combines the Glassmorphism UI with the multi-tab functionality (Dashboard + Data Engine).

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIGURATION & SETUP ---
st.set_page_config(layout="wide", page_title="FinSight Core | Liquid Glass")

# --- CUSTOM CSS FOR GLASSMORPHISM & TABS ---
st.markdown("""
<style>
    /* 1. Main Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%);
        background-attachment: fixed;
    }
    
    /* 2. Glass Card Style */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 20px;
        margin-bottom: 20px;
        color: white;
    }

    /* 3. Text Styling */
    h1, h2, h3, h4, p, div, span, label {
        color: white !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* 4. Tab Styling (To make them visible on dark background) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px 10px 0 0;
        color: white;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.25);
        border-bottom: 2px solid #4ade80;
    }
</style>
""", unsafe_allow_html=True)

# Gemini API Configuration
GEMINI_MODEL = 'gemini-2.5-flash-preview-09-2025'
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# --- MOCK DATA ---
ANNUAL_DATA = pd.DataFrame([
    # Actuals (Jan-Jun)
    {'month': 'Jan', 'type': 'Actual', 'revenue': 120000, 'direct_cost': 48000, 'indirect_cost': 40000},
    {'month': 'Feb', 'type': 'Actual', 'revenue': 125000, 'direct_cost': 50000, 'indirect_cost': 41000},
    {'month': 'Mar', 'type': 'Actual', 'revenue': 110000, 'direct_cost': 44000, 'indirect_cost': 42000},
    {'month': 'Apr', 'type': 'Actual', 'revenue': 140000, 'direct_cost': 56000, 'indirect_cost': 43000},
    {'month': 'May', 'type': 'Actual', 'revenue': 135000, 'direct_cost': 54000, 'indirect_cost': 42000},
    {'month': 'Jun', 'type': 'Actual', 'revenue': 155000, 'direct_cost': 62000, 'indirect_cost': 45000},
    # Forecasts (Jul-Dec)
    {'month': 'Jul', 'type': 'Forecast', 'revenue': 145000, 'direct_cost': 58000, 'indirect_cost': 44000},
    {'month': 'Aug', 'type': 'Forecast', 'revenue': 148000, 'direct_cost': 59200, 'indirect_cost': 44500},
    {'month': 'Sep', 'type': 'Forecast', 'revenue': 160000, 'direct_cost': 64000, 'indirect_cost': 46000},
    {'month': 'Oct', 'type': 'Forecast', 'revenue': 158000, 'direct_cost': 63200, 'indirect_cost': 45500},
    {'month': 'Nov', 'type': 'Forecast', 'revenue': 165000, 'direct_cost': 66000, 'indirect_cost': 47000},
    {'month': 'Dec', 'type': 'Forecast', 'revenue': 180000, 'direct_cost': 72000, 'indirect_cost': 48000},
])

ANNUAL_DATA['gross_margin'] = ANNUAL_DATA['revenue'] - ANNUAL_DATA['direct_cost']
ANNUAL_DATA['net_profit'] = ANNUAL_DATA['gross_margin'] - ANNUAL_DATA['indirect_cost']

# --- HELPER FUNCTIONS ---

def format_k(val):
    return f"£{val/1000:.1f}k"

def render_glass_metric(label, value, subvalue, is_positive=True):
    arrow = "↑" if is_positive else "↓"
    color = "#4ade80" if is_positive else "#f87171"
    st.markdown(f"""
    <div class="glass-card">
        <div style="font-size: 12px; text-transform: uppercase; opacity: 0.8; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; margin-bottom: 5px;">{value}</div>
        <div style="font-size: 14px; color: {color};">
            {arrow} {subvalue} <span style="color: white; opacity: 0.6;">vs Target</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def generate_glass_narrative(data):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        return "⚠️ API Key Missing. Please add it to Streamlit Secrets."

    current_month = data[data['type'] == 'Actual'].iloc[-1]
    fy_rev = data['revenue'].sum()
    fy_net = data['net_profit'].sum()
    
    context = f"""
    Current Month (Jun): Rev {format_k(current_month['revenue'])}, Direct Cost {format_k(current_month['direct_cost'])}, Indirect {format_k(current_month['indirect_cost'])}.
    Full Year Forecast: Rev {format_k(fy_rev)}, Net Profit {format_k(fy_net)}.
    """
    
    prompt = f"""You are a CFO's AI Assistant. Write a 3-bullet executive summary based on this data: {context}. 
    Focus on the split between Direct Costs (COGS) and Indirect Costs (OpEx). 
    Keep it punchy, professional, and under 100 words. Output raw text with bullet points."""

    payload = {'contents': [{'parts': [{'text': prompt}]}]}
    
    try:
        response = requests.post(f"{API_URL}?key={api_key}", headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"AI Error: {response.status_code}"
    except Exception as e:
        return f"AI Service Unavailable: {str(e)}"

# --- VIEWS ---

def dashboard_view():
    # 1. KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    ytd = ANNUAL_DATA[ANNUAL_DATA['type'] == 'Actual'].sum()

    with c1: render_glass_metric("Total Revenue (YTD)", format_k(ytd['revenue']), "+12%", True)
    with c2: render_glass_metric("Direct Costs (COGS)", format_k(ytd['direct_cost']), "+5% (Vol)", False)
    with c3: render_glass_metric("Indirect Costs (OpEx)", format_k(ytd['indirect_cost']), "-2% (Savings)", True)
    with c4: render_glass_metric("Net Margin", f"£{ytd['net_profit']/1000:.1f}k", "28%", True)

    # 2. Main Charts
    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.markdown('<div class="glass-card" style="height: 450px;">', unsafe_allow_html=True)
        st.markdown("### Predictive Profitability Forecast")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ANNUAL_DATA[ANNUAL_DATA['type']=='Actual']['month'], y=ANNUAL_DATA[ANNUAL_DATA['type']=='Actual']['revenue'], mode='lines+markers', name='Actual Rev', line=dict(color='#ffffff', width=3)))
        fig.add_trace(go.Scatter(x=ANNUAL_DATA[ANNUAL_DATA['type']=='Forecast']['month'], y=ANNUAL_DATA[ANNUAL_DATA['type']=='Forecast']['revenue'], mode='lines', name='Forecast Rev', line=dict(color='#ffffff', width=3, dash='dot')))
        fig.add_trace(go.Scatter(x=ANNUAL_DATA['month'], y=ANNUAL_DATA['direct_cost'] + ANNUAL_DATA['indirect_cost'], mode='lines', name='Total Costs', fill='tozeroy', line=dict(color='rgba(255, 100, 100, 0.5)', width=0), fillcolor='rgba(255, 100, 100, 0.2)'))

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), margin=dict(l=20, r=20, t=20, b=20), height=380, showlegend=True, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_side:
        st.markdown('<div class="glass-card" style="height: 450px;">', unsafe_allow_html=True)
        st.markdown("### Cost Structure")
        fig_donut = go.Figure(data=[go.Pie(labels=['Direct Costs', 'Indirect Costs', 'Net Margin'], values=[ytd['direct_cost'], ytd['indirect_cost'], ytd['net_profit']], hole=.6, marker=dict(colors=['#ff9f43', '#ff6b6b', '#1dd1a1']))])
        fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), margin=dict(l=0, r=0, t=20, b=0), height=300, showlegend=True, legend=dict(orientation="h", y=0))
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. AI Section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    c_ai_head, c_ai_btn = st.columns([3,1])
    with c_ai_head:
        st.markdown("### 🤖 FinSight AI Analyst")
    with c_ai_btn:
        if st.button("Generate Analysis"):
            st.session_state['run_ai'] = True
            st.rerun()

    if st.session_state.get('run_ai'):
        with st.spinner("Analyzing Cost Structures..."):
            analysis = generate_glass_narrative(ANNUAL_DATA)
            st.markdown(f"""<div style="background: rgba(0,0,0,0.2); padding: 15px; border-radius: 10px; border-left: 4px solid #1dd1a1;">{analysis}</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def data_engine_view():
    st.markdown("""
    <div class="glass-card">
        <h3>⚙️ Data Engine & Transformation Layer</h3>
        <p style="opacity:0.8">This is the client-owned backend where raw ERP data is cleaned and structured.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('<div class="glass-card" style="height: 300px;">', unsafe_allow_html=True)
        st.markdown("#### 🔌 System Status")
        st.markdown("✅ **NetSuite Connector:** Active (Latency: 45ms)")
        st.markdown("✅ **Transformation Pipeline:** Healthy")
        st.markdown("✅ **Last Sync:** Today, 09:41 AM")
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(100)
        st.caption("Sync Completeness")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card" style="height: 300px;">', unsafe_allow_html=True)
        st.markdown("#### 🛠️ Transformation Rules (Python)")
        st.code("""
def transform_ledger(df):
    # 1. Standardize Headers
    df.columns = [c.lower() for c in df.columns]
    # 2. Calculate Margins
    df['margin'] = df['rev'] - df['cogs']
    # 3. Tag Direct vs Indirect
    df['type'] = df['acct'].map(acct_map)
    return df
        """, language="python")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📄 Live Data Feed (Sample)")
    st.dataframe(ANNUAL_DATA.head(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN LAYOUT ---

st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
    <div>
        <h1 style="margin:0; font-size: 2.5rem;">FinSight Core</h1>
        <p style="opacity: 0.8;">Liquid Glass Interface • Live Connection</p>
    </div>
    <div class="glass-card" style="padding: 10px 20px; margin:0;">
        User: Aaron M. • Admin
    </div>
</div>
""", unsafe_allow_html=True)

# TABS (The return of the missing features!)
tab1, tab2 = st.tabs(["📊 EXECUTIVE DASHBOARD", "⚙️ DATA ENGINE"])

with tab1:
    dashboard_view()

with tab2:
    data_engine_view()
