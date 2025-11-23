# FINSIGHT CORE V3.1 - RESTORED VISUALS
# This code strictly uses the V2.2 CSS architecture to guarantee the "Liquid Glass" look.
# It re-integrates the V3.1 Enterprise features (PPTX, Drill-down, New KPIs) into that specific visual shell.

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go
from io import BytesIO
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="FinSight Core | Enterprise")

# --- CUSTOM CSS (THE PROVEN V2.2 STYLE) ---
st.markdown("""
<style>
    /* 1. The Exact Gradient from V2.2 */
    .stApp {
        background: linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%);
        background-attachment: fixed;
    }
    
    /* 2. The Exact Glass Card from V2.2 */
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

    /* 3. Typography */
    h1, h2, h3, h4, p, div, span, label, .stMarkdown {
        color: white !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* 4. Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px 10px 0 0;
        color: #e0e0e0;
        font-weight: bold;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.25);
        color: white;
        border-bottom: 3px solid #4ade80;
    }
    
    /* 5. Metric Styling */
    .metric-label { font-size: 0.8rem; color: #e0e0e0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: white; }
    .metric-delta-pos { color: #4ade80; font-size: 0.9rem; font-weight: 700; }
    .metric-delta-neg { color: #f87171; font-size: 0.9rem; font-weight: 700; }
    
</style>
""", unsafe_allow_html=True)

# Gemini API
GEMINI_MODEL = 'gemini-2.5-flash-preview-09-2025'
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# --- DATA ---
def generate_mock_data():
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    data = []
    for i, m in enumerate(months):
        is_actual = i < 6
        rev = 120000 + (i * 5000) + np.random.randint(-5000, 5000)
        direct = rev * 0.4 + np.random.randint(-2000, 2000)
        indirect = 40000 + (i * 1000) + np.random.randint(-1000, 1000)
        cash = (rev - direct - indirect) * 0.9
        data.append({
            'month': m, 'type': 'Actual' if is_actual else 'Forecast',
            'revenue': rev, 'direct_cost': direct, 'indirect_cost': indirect, 'cash_flow': cash,
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

# --- HELPERS ---
def format_k(val): return f"£{val/1000:.1f}k"
def format_pct(val): return f"{val*100:.1f}%"

def render_glass_metric(label, value, delta, is_good=True):
    delta_class = "metric-delta-pos" if is_good else "metric-delta-neg"
    arrow = "▲" if is_good else "▼"
    st.markdown(f"""
    <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        <div style="margin-top: 12px;">
            <span class="{delta_class}">{arrow} {delta}</span> <span style="color: #e0e0e0; font-size: 0.8rem; margin-left: 8px;">vs Budget</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Key Fix: Ensuring Chart Background matches the V2.2 "Invisible" look
def apply_glass_style(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=False, color='#e0e0e0'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='#e0e0e0'),
        hovermode="x unified"
    )
    return fig

# --- PPTX EXPORT (Professional Dark Mode) ---
def create_pro_pptx(metrics, ai_text):
    prs = Presentation()
    
    BG_COLOR = RGBColor(44, 62, 80)   
    ACCENT = RGBColor(76, 161, 175)   
    TEXT_MAIN = RGBColor(255, 255, 255)
    ROW_ALT = RGBColor(60, 80, 100)

    def set_bg(slide):
        bg = slide.background
        fill = bg.fill
        fill.solid()
        fill.fore_color.rgb = BG_COLOR

    slide = prs.slides.add_slide(prs.slide_layouts[0])
    set_bg(slide)
    title = slide.shapes.title
    title.text = "FinSight Executive Summary"
    title.text_frame.paragraphs[0].font.color.rgb = TEXT_MAIN
    
    sub = slide.placeholders[1]
    sub.text = "Automated Board Pack | June 2024"
    sub.text_frame.paragraphs[0].font.color.rgb = ACCENT

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    set_bg(slide)
    t = slide.shapes.title
    t.text = "Financial Performance (YTD)"
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
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.color.rgb = BG_COLOR
        p.alignment = PP_ALIGN.CENTER

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
            cell.fill.solid()
            cell.fill.fore_color.rgb = ROW_ALT if r_idx % 2 == 0 else BG_COLOR
            p = cell.text_frame.paragraphs[0]
            p.font.color.rgb = TEXT_MAIN
            p.alignment = PP_ALIGN.CENTER

    slide = prs.slides.add_slide(prs.slide_layouts[1])
    set_bg(slide)
    t = slide.shapes.title
    t.text = "Strategic Commentary"
    t.text_frame.paragraphs[0].font.color.rgb = TEXT_MAIN
    
    body = slide.placeholders[1]
    body.text = ai_text.replace('**', '').replace('###', '')
    for p in body.text_frame.paragraphs:
        p.font.color.rgb = RGBColor(200, 200, 200)

    output = BytesIO()
    prs.save(output)
    output.seek(0)
    return output

# --- AI LOGIC ---
def generate_ai_narrative():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError: return "⚠️ API Key Missing"
    
    context = f"YTD Rev: £800k (Budget £780k). Op Profit: £220k. Cash: £200k."
    prompt = f"You are a CFO. Write a board summary based on: {context}. Use clear headers (###) and bullet points."
    
    payload = {'contents': [{'parts': [{'text': prompt}]}]}
    try:
        r = requests.post(f"{API_URL}?key={api_key}", headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        return r.json()['candidates'][0]['content']['parts'][0]['text'] if r.status_code == 200 else "AI Error"
    except: return "AI Error"

# --- DASHBOARD ---
def dashboard_view():
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

    # CHARTS - Using WRAPPERS to enforce the visual container style
    c_main, c_side = st.columns([2, 1])
    with c_main:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ANNUAL_DATA['month'], y=ANNUAL_DATA['revenue'], name='Revenue', marker_color='#ffffff', opacity=0.8)) # White Bars for V2 look
        fig.add_trace(go.Scatter(x=ANNUAL_DATA['month'], y=ANNUAL_DATA['operating_profit'], name='Op Profit', line=dict(color='#4ade80', width=3), yaxis='y2')) # Bright Green line
        fig.update_layout(title="Revenue & Profit Trend", height=380, legend=dict(orientation="h", y=1.1), yaxis2=dict(overlaying='y', side='right', showgrid=False))
        fig = apply_glass_style(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_side:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_bridge = go.Figure(go.Waterfall(
            name = "20", orientation = "v", measure = ["relative", "relative", "relative", "total"],
            x = ["Budget", "Vol", "Cost", "Actual"], textposition = "outside", text = ["+10k", "+5k", "-2k", "£68k"],
            y = [55000, 15000, -2000, 0], connector = {"line":{"color":"white"}},
            decreasing = {"marker":{"color":"#f87171"}}, increasing = {"marker":{"color":"#4ade80"}}, totals = {"marker":{"color":"#ffffff"}}
        ))
        fig_bridge.update_layout(title="Profit Bridge (Jun)", height=380)
        fig_bridge = apply_glass_style(fig_bridge)
        st.plotly_chart(fig_bridge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # DRILL DOWN (FIXED)
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
        # FIX: Explicitly select only numeric columns for formatting to prevent ValueError
        st.dataframe(drill_df.style.format("{:,.0f}", subset=['Actual', 'Budget', 'Variance']), use_container_width=True)

    # AI & EXPORT
    st.markdown("### 🤖 Strategic Intelligence")
    c_ai, c_exp = st.columns([3, 1])
    
    with c_ai:
        if 'ai_analysis' not in st.session_state: st.session_state['ai_analysis'] = "Click 'Generate' to analyze."
        if st.button("⚡ Run Commentary Engine"):
            with st.spinner("Analyzing..."):
                st.session_state['ai_analysis'] = generate_ai_narrative()
        st.markdown(f"""<div style="background: rgba(255, 255, 255, 0.15); padding: 20px; border-radius: 12px; border-left: 4px solid #4ade80;">{st.session_state['ai_analysis']}</div>""", unsafe_allow_html=True)

    with c_exp:
        st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
        if st.session_state['ai_analysis'] != "Click 'Generate' to analyze.":
            pptx = create_pro_pptx(metrics, st.session_state['ai_analysis'])
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
        <h1 style="margin:0; font-size: 2.8rem; font-weight: 800; letter-spacing: -1px;">FinSight Core</h1>
        <p style="color: #e0e0e0; margin:0; font-size: 1.1rem;">Enterprise Edition • Live Connection</p>
    </div>
    <div class="glass-card" style="padding: 12px 24px; margin:0;">
        <span style="color: #4ade80; font-weight: 600;">● Connected: NetSuite OneWorld</span>
    </div>
</div>
""", unsafe_allow_html=True)

t1, t2 = st.tabs(["Executive Dashboard", "Data Engine"])
with t1: dashboard_view()
with t2: data_engine_view()
