# FINSIGHT CORE V3.1 - AUTHENTIC RESTORATION
# Restores the specific Deep Blue/Teal Gradient and Translucent Card style from the approved demo.
# Includes all Enterprise Features: PPTX Export, Waterfall Charts, AI Narrative, and Drill-Downs.

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# --- CONFIGURATION & SETUP ---
st.set_page_config(layout="wide", page_title="FinSight Core | Enterprise")

# --- CUSTOM CSS (RESTORING THE V3.1 VISUAL IDENTITY) ---
st.markdown("""
<style>
    /* 1. The Signature Deep Blue Gradient */
    .stApp {
        background: linear-gradient(180deg, #1e293b 0%, #334155 50%, #0f172a 100%);
        background-attachment: fixed;
    }
    
    /* 2. Translucent Card (The 'Frost' Look) */
    .glass-card {
        background-color: rgba(30, 41, 59, 0.7); /* Deep Blue-Grey Transparency */
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
    }

    /* 3. Typography */
    h1, h2, h3, h4, p, div, span, label, .stMarkdown, .stDataFrame {
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* 4. Clean Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 8px 8px 0 0;
        color: #94a3b8;
        font-weight: 600;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.2); /* Light Blue Selection */
        color: #ffffff;
        border-bottom: 2px solid #38bdf8;
    }
    
    /* 5. Metric Styling */
    .metric-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 5px; font-weight: 600; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #ffffff; }
    .metric-delta-pos { color: #4ade80; font-size: 0.9rem; font-weight: 600; } /* Green */
    .metric-delta-neg { color: #f87171; font-size: 0.9rem; font-weight: 600; } /* Red */
    
</style>
""", unsafe_allow_html=True)

# Gemini API Configuration
GEMINI_MODEL = 'gemini-2.5-flash-preview-09-2025'
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# --- DATA GENERATION ---
def generate_mock_data():
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    data = []
    for i, m in enumerate(months):
        is_actual = i < 6
        rev = 120000 + (i * 5000) + np.random.randint(-5000, 5000)
        direct = rev * 0.4 + np.random.randint(-2000, 2000)
        indirect = 40000 + (i * 1000) + np.random.randint(-1000, 1000)
        cash_flow = (rev - direct - indirect) * 0.9
        
        data.append({
            'month': m, 'type': 'Actual' if is_actual else 'Forecast',
            'revenue': rev, 'direct_cost': direct, 'indirect_cost': indirect, 'cash_flow': cash_flow,
            'budget_revenue': 120000 + (i * 4000), 'budget_direct': (120000 + (i * 4000)) * 0.38, 'budget_indirect': 42000
        })
    return pd.DataFrame(data)

ANNUAL_DATA = generate_mock_data()
ANNUAL_DATA['gross_margin'] = ANNUAL_DATA['revenue'] - ANNUAL_DATA['direct_cost']
ANNUAL_DATA['operating_profit'] = ANNUAL_DATA['gross_margin'] - ANNUAL_DATA['indirect_cost']
ANNUAL_DATA['operating_margin_pct'] = ANNUAL_DATA['operating_profit'] / ANNUAL_DATA['revenue']

DRILL_DATA = pd.DataFrame([
    {'Category': 'Revenue', 'Sub-Category': 'Product A', 'Cost Centre': 'Sales-US', 'Actual': 450000, 'Budget': 420000},
    {'Category': 'Revenue', 'Sub-Category': 'Product B', 'Cost Centre': 'Sales-EU', 'Actual': 320000, 'Budget': 350000},
    {'Category': 'COGS', 'Sub-Category': 'Materials', 'Cost Centre': 'Plant-1', 'Actual': 200000, 'Budget': 190000},
    {'Category': 'OPEX', 'Sub-Category': 'Marketing', 'Cost Centre': 'Global', 'Actual': 80000, 'Budget': 70000},
])

# --- HELPER FUNCTIONS ---
def format_k(val): return f"£{val/1000:.1f}k"
def format_pct(val): return f"{val*100:.1f}%"

def render_glass_metric(label, value, delta, is_good=True):
    delta_class = "metric-delta-pos" if is_good else "metric-delta-neg"
    arrow = "▲" if is_good else "▼"
    st.markdown(f"""
    <div class="glass-card" style="padding: 15px; height: 100%;">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="{delta_class}">{arrow} {delta} <span style="color: #94a3b8; font-size: 0.7rem; font-weight: 400;">vs Budget</span></div>
    </div>
    """, unsafe_allow_html=True)

def apply_glass_style(fig):
    """Applies the V3.1 Style (Transparent BG, White Text)"""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family="Inter"),
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=False, color='#94a3b8'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='#94a3b8'),
        hovermode="x unified"
    )
    return fig

# --- PPTX EXPORT ---
def create_presentation(metrics, ai_text):
    prs = Presentation()
    
    # Define Colors
    BG_COLOR = RGBColor(30, 41, 59)
    ACCENT = RGBColor(56, 189, 248) # Light Blue
    TEXT_MAIN = RGBColor(255, 255, 255)
    
    def set_bg(slide):
        bg = slide.background
        fill = bg.fill
        fill.solid()
        fill.fore_color.rgb = BG_COLOR

    # Slide 1: Title
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    set_bg(slide)
    title = slide.shapes.title
    title.text = "FinSight Executive Summary"
    title.text_frame.paragraphs[0].font.color.rgb = TEXT_MAIN
    
    # Slide 2: Data
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    set_bg(slide)
    t = slide.shapes.title
    t.text = "Key Performance Indicators"
    t.text_frame.paragraphs[0].font.color.rgb = TEXT_MAIN
    
    rows, cols = 7, 3
    left, top, width, height = Inches(1), Inches(2), Inches(8), Inches(3.5)
    table = slide.shapes.add_table(rows, cols, left, top, width, height).table
    
    headers = ['Metric', 'Value', 'Variance']
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = ACCENT
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.color.rgb = BG_COLOR

    data = [
        ['Revenue', metrics['rev_val'], metrics['rev_delta']],
        ['Gross Margin', metrics['gm_val'], metrics['gm_delta']],
        ['Op Profit', metrics['op_val'], metrics['op_delta']],
        ['Op Margin', metrics['op_margin_val'], metrics['op_margin_delta']],
        ['OPEX', metrics['opex_val'], metrics['opex_delta']],
        ['Cash Flow', metrics['cf_val'], metrics['cf_delta']]
    ]
    
    for r_idx, row_data in enumerate(data, 1):
        for c_idx, val in enumerate(row_data):
            cell = table.cell(r_idx, c_idx)
            cell.text = str(val)
            cell.text_frame.paragraphs[0].font.color.rgb = TEXT_MAIN

    # Slide 3: AI Text
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    set_bg(slide)
    t = slide.shapes.title
    t.text = "AI Commentary"
    t.text_frame.paragraphs[0].font.color.rgb = TEXT_MAIN
    b = slide.placeholders[1]
    b.text = ai_text.replace('**', '').replace('###', '')
    for p in b.text_frame.paragraphs: p.font.color.rgb = RGBColor(200,200,200)

    out = BytesIO()
    prs.save(out)
    out.seek(0)
    return out

# --- AI LOGIC ---
def generate_ai_narrative():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError: return "⚠️ API Key Missing"
    
    context = f"YTD Rev: £800k. Op Profit: £220k. Cash: £200k. Accuracy: 94%."
    prompt = f"You are a CFO. Write a board summary based on: {context}. Use headers (###) and bullets."
    payload = {'contents': [{'parts': [{'text': prompt}]}]}
    try:
        r = requests.post(f"{API_URL}?key={api_key}", headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        return r.json()['candidates'][0]['content']['parts'][0]['text'] if r.status_code == 200 else "AI Error"
    except: return "AI Error"

# --- DASHBOARD VIEW ---
def dashboard_view():
    # 1. METRICS
    ytd = ANNUAL_DATA[ANNUAL_DATA['type'] == 'Actual'].sum()
    metrics = {
        'rev_val': format_k(ytd['revenue']), 'rev_delta': "+3.1%",
        'gm_val': format_k(ytd['gross_margin']), 'gm_delta': "-1.2%",
        'op_val': format_k(ytd['operating_profit']), 'op_delta': "-3.4%",
        'op_margin_val': format_pct(ytd['operating_margin_pct']/6), 'op_margin_delta': "-0.5%",
        'opex_val': format_k(ytd['indirect_cost']), 'opex_delta': "+0.8%",
        'cf_val': format_k(ytd['cash_flow']), 'cf_delta': "+5.4%"
    }

    st.markdown("### 🚀 Enterprise Performance")
    cols = st.columns(6)
    render_glass_metric("Revenue", metrics['rev_val'], metrics['rev_delta'], True)
    with cols[1]: render_glass_metric("Gross Margin", metrics['gm_val'], metrics['gm_delta'], False)
    with cols[2]: render_glass_metric("Op Margin", metrics['op_margin_val'], metrics['op_delta'], False)
    with cols[3]: render_glass_metric("OPEX", metrics['opex_val'], metrics['opex_delta'], False)
    with cols[4]: render_glass_metric("Cash Flow", metrics['cf_val'], metrics['cf_delta'], True)
    with cols[5]: render_glass_metric("Accuracy", "94.2%", "-1.1%", False)

    # 2. SCENARIO (V3.1 Feature)
    with st.expander("🎛️ Scenario Simulation", expanded=False):
        c1, c2 = st.columns([1,3])
        with c1:
            growth = st.slider("Growth %", -20, 20, 0)
        with c2:
            fig = go.Figure(go.Bar(y=[10, 20, 30 * (1+growth/100)]))
            fig.update_layout(height=200, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    # 3. CHARTS
    c_main, c_side = st.columns([2, 1])
    with c_main:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ANNUAL_DATA['month'], y=ANNUAL_DATA['revenue'], name='Revenue', marker_color='#38bdf8'))
        fig.add_trace(go.Scatter(x=ANNUAL_DATA['month'], y=ANNUAL_DATA['operating_profit'], name='Op Profit', line=dict(color='#4ade80', width=3), yaxis='y2'))
        fig.update_layout(title="Revenue Trend", height=380, legend=dict(orientation="h", y=1.1), yaxis2=dict(overlaying='y', side='right', showgrid=False))
        fig = apply_glass_style(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_side:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_bridge = go.Figure(go.Waterfall(
            name = "20", orientation = "v", measure = ["relative", "relative", "relative", "total"],
            x = ["Budget", "Vol", "Cost", "Actual"], textposition = "outside", text = ["+10k", "+5k", "-2k", "£68k"],
            y = [55000, 15000, -2000, 0], connector = {"line":{"color":"white"}},
            decreasing = {"marker":{"color":"#f87171"}}, increasing = {"marker":{"color":"#4ade80"}}, totals = {"marker":{"color":"#94a3b8"}}
        ))
        fig_bridge.update_layout(title="Profit Bridge (Jun)", height=380)
        fig_bridge = apply_glass_style(fig_bridge)
        st.plotly_chart(fig_bridge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. DRILL DOWN (FIXED FORMATTING)
    st.markdown("### 🔍 Multi-Layer Drill Down")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.selectbox("Hierarchy Level", ["Category", "Sub-Category", "Cost Centre"])
        st.selectbox("Filter By", ["Revenue", "COGS", "OPEX"])
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        drill_df = DRILL_DATA.copy()
        drill_df['Variance'] = drill_df['Actual'] - drill_df['Budget']
        # CRITICAL FIX: Only format numeric columns to prevent ValueError
        st.dataframe(drill_df.style.format("{:,.0f}", subset=['Actual', 'Budget', 'Variance']), use_container_width=True)

    # 5. AI & EXPORT
    st.markdown("### 🤖 Strategic Intelligence")
    c_ai, c_exp = st.columns([3, 1])
    
    with c_ai:
        if 'ai_analysis' not in st.session_state: st.session_state['ai_analysis'] = "Click 'Generate' to analyze."
        if st.button("⚡ Run Commentary Engine"):
            with st.spinner("Analyzing..."):
                st.session_state['ai_analysis'] = generate_ai_narrative()
        st.markdown(f"""<div style="background: rgba(56, 189, 248, 0.1); padding: 20px; border-radius: 12px; border-left: 4px solid #38bdf8;">{st.session_state['ai_analysis']}</div>""", unsafe_allow_html=True)

    with c_exp:
        st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
        if st.session_state['ai_analysis'] != "Click 'Generate' to analyze.":
            pptx = create_presentation(metrics, st.session_state['ai_analysis'])
            st.download_button("📥 Download Board Pack", pptx, "FinSight_Pack.pptx", type="primary")
        else: st.caption("Run Analysis to Export")
        st.markdown('</div>', unsafe_allow_html=True)

def data_engine_view():
    st.markdown("### ⚙️ Data Engine")
    st.success("Pipeline Active | Latency: 45ms")
    st.dataframe(ANNUAL_DATA.head(), use_container_width=True)

# --- MAIN ---
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
    <div>
        <h1 style="margin:0; font-size: 2.5rem; font-weight: 700;">FinSight Core</h1>
        <p style="color: #94a3b8; margin:0;">Enterprise Edition • Live Connection</p>
    </div>
    <div class="glass-card" style="padding: 10px 20px; margin:0;">Connected: NetSuite OneWorld</div>
</div>
""", unsafe_allow_html=True)

t1, t2 = st.tabs(["Executive Dashboard", "Data Engine"])
with t1: dashboard_view()
with t2: data_engine_view()
