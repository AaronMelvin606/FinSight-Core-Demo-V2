# FINSIGHT CORE V3.2 - "LIQUID GLASS" & PRO SLIDES
# Restores the V2 Teal/Slate Gradient.
# Upgrades PPTX Export to use a custom Dark Theme with styled tables.

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
from pptx.enum.shapes import MSO_SHAPE

# --- CONFIGURATION & SETUP ---
st.set_page_config(layout="wide", page_title="FinSight Core | Liquid Glass")

# --- CUSTOM CSS (RESTORED LIQUID GLASS) ---
st.markdown("""
<style>
    /* 1. Restored V2 Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%);
        background-attachment: fixed;
    }
    
    /* 2. Enhanced Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 20px;
        margin-bottom: 20px;
        color: white;
        transition: transform 0.2s;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border: 1px solid rgba(255, 255, 255, 0.4);
    }

    /* 3. Typography */
    h1, h2, h3, h4, p, div, span, label, .stMarkdown, .stDataFrame {
        color: white !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* 4. Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px 10px 0 0;
        color: #e2e8f0;
        font-weight: bold;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.25);
        color: white;
        border-bottom: 3px solid #4ade80;
    }
    
    /* 5. Metrics */
    .metric-label { font-size: 0.8rem; color: #cbd5e1; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.2); }
    
    /* 6. Dataframes (Transparent) */
    [data-testid="stDataFrame"] { background-color: transparent; }
</style>
""", unsafe_allow_html=True)

# Gemini API Configuration
GEMINI_MODEL = 'gemini-2.5-flash-preview-09-2025'
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# --- MOCK DATA GENERATION ---
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
            'month': m,
            'type': 'Actual' if is_actual else 'Forecast',
            'revenue': rev,
            'direct_cost': direct,
            'indirect_cost': indirect,
            'cash_flow': cash,
            'budget_revenue': 120000 + (i * 4000),
            'budget_direct': (120000 + (i * 4000)) * 0.38,
            'budget_indirect': 42000
        })
    return pd.DataFrame(data)

ANNUAL_DATA = generate_mock_data()
ANNUAL_DATA['gross_margin'] = ANNUAL_DATA['revenue'] - ANNUAL_DATA['direct_cost']
ANNUAL_DATA['operating_profit'] = ANNUAL_DATA['gross_margin'] - ANNUAL_DATA['indirect_cost']
ANNUAL_DATA['operating_margin_pct'] = ANNUAL_DATA['operating_profit'] / ANNUAL_DATA['revenue']

# --- HELPER FUNCTIONS ---
def format_k(val): return f"£{val/1000:.1f}k"
def format_pct(val): return f"{val*100:.1f}%"

def render_glass_metric(label, value, delta, is_good=True):
    color = "#4ade80" if is_good else "#f87171" # Green / Red
    arrow = "▲" if is_good else "▼"
    st.markdown(f"""
    <div class="glass-card" style="padding: 15px; height: 100%;">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div style="color: {color}; font-weight: bold; font-size: 0.9rem;">
            {arrow} {delta} <span style="color: #cbd5e1; font-weight: normal; font-size: 0.7rem;">vs Budget</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def apply_glass_style(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=False, color='#cbd5e1'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='#cbd5e1')
    )
    return fig

# --- ADVANCED PPTX GENERATOR (THE FIX) ---
def create_professional_presentation(metrics, ai_text):
    prs = Presentation()
    
    # 1. Define Colors
    DARK_BG = RGBColor(44, 62, 80)   # Matches the Slate in the gradient
    TEAL_ACCENT = RGBColor(76, 161, 175) # Matches the Teal in the gradient
    WHITE = RGBColor(255, 255, 255)
    GREY = RGBColor(200, 200, 200)

    # Helper to style a slide background
    def set_slide_bg(slide):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = DARK_BG

    # --- SLIDE 1: TITLE ---
    slide = prs.slides.add_slide(prs.slide_layouts[0]) # Title Slide
    set_slide_bg(slide)
    
    title = slide.shapes.title
    title.text = "FinSight Executive Summary"
    title.text_frame.paragraphs[0].font.color.rgb = WHITE
    title.text_frame.paragraphs[0].font.bold = True
    
    subtitle = slide.placeholders[1]
    subtitle.text = "Automated Board Pack | Generated by FinSight Core"
    subtitle.text_frame.paragraphs[0].font.color.rgb = TEAL_ACCENT

    # --- SLIDE 2: KPI TABLE ---
    slide = prs.slides.add_slide(prs.slide_layouts[5]) # Blank
    set_slide_bg(slide)
    
    # Add Title
    title_shape = slide.shapes.title
    title_shape.text = "Financial Performance KPIs (YTD)"
    title_shape.text_frame.paragraphs[0].font.color.rgb = WHITE
    
    # Add "Glass" Card Shape behind table
    # shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(1.5), Inches(9), Inches(5))
    # shape.fill.solid()
    # shape.fill.fore_color.rgb = RGBColor(60, 80, 100) # Slightly lighter than BG
    # shape.line.color.rgb = TEAL_ACCENT
    
    # Build Table
    rows, cols = 7, 3
    left = Inches(1)
    top = Inches(2)
    width = Inches(8)
    height = Inches(3.5)
    table = slide.shapes.add_table(rows, cols, left, top, width, height).table
    
    # Headers
    headers = ['Metric', 'Value', 'Variance']
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = TEAL_ACCENT
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.color.rgb = WHITE
        cell.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    # Data Rows
    data = [
        ['Revenue', metrics['rev_val'], metrics['rev_delta']],
        ['Gross Margin', metrics['gm_val'], metrics['gm_delta']],
        ['Operating Profit', metrics['op_val'], metrics['op_delta']],
        ['Operating Margin', metrics['op_margin_val'], metrics['op_margin_delta']],
        ['OPEX', metrics['opex_val'], metrics['opex_delta']],
        ['Cash Flow', metrics['cf_val'], metrics['cf_delta']]
    ]
    
    for row_idx, row_data in enumerate(data, start=1):
        for col_idx, val in enumerate(row_data):
            cell = table.cell(row_idx, col_idx)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(60, 80, 100) # Glass-like row color
            cell.text_frame.paragraphs[0].font.color.rgb = WHITE
            cell.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
            # Add bottom border logic here if strictly needed, but color separation works well

    # --- SLIDE 3: AI NARRATIVE ---
    slide = prs.slides.add_slide(prs.slide_layouts[1]) # Title and Content
    set_slide_bg(slide)
    
    title = slide.shapes.title
    title.text = "Strategic Commentary"
    title.text_frame.paragraphs[0].font.color.rgb = WHITE
    
    body = slide.placeholders[1]
    # Clean up Markdown for PPT text
    clean_text = ai_text.replace('**', '').replace('### ', '').replace('####', '')
    body.text = clean_text
    
    # Style the Body Text
    for paragraph in body.text_frame.paragraphs:
        paragraph.font.color.rgb = GREY
        paragraph.font.size = Pt(18)

    binary_output = BytesIO()
    prs.save(binary_output)
    binary_output.seek(0)
    return binary_output

# --- AI NARRATIVE ENGINE ---
def generate_ai_narrative():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        return "⚠️ API Key Missing."

    ytd = ANNUAL_DATA[ANNUAL_DATA['type'] == 'Actual'].sum()
    budget_rev = ytd['budget_revenue']
    
    context = f"""
    YTD Revenue: {format_k(ytd['revenue'])} (Budget: {format_k(budget_rev)}).
    YTD Op Profit: {format_k(ytd['operating_profit'])}.
    Cash Flow: {format_k(ytd['cash_flow'])}.
    Forecast Accuracy: 94%.
    """
    
    prompt = f"""You are a CFO. Write a board executive summary based on: {context}.
    Structure:
    1. **Headline**: One sentence summary.
    2. **Key Drivers**: Bullet points on Revenue/Margin.
    3. **Strategic Outlook**: One recommendation.
    Keep it professional and concise.
    """

    payload = {'contents': [{'parts': [{'text': prompt}]}]}
    
    try:
        response = requests.post(f"{API_URL}?key={api_key}", headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else: return "AI Error."
    except: return "AI Unavailable."

# --- VIEWS ---

def dashboard_view():
    # 1. KPI STRIP
    ytd_act = ANNUAL_DATA[ANNUAL_DATA['type'] == 'Actual'].sum()
    ytd_bud = ANNUAL_DATA[ANNUAL_DATA['type'] == 'Actual'][['budget_revenue', 'budget_direct', 'budget_indirect']].sum()
    
    rev_var = (ytd_act['revenue'] - ytd_bud['budget_revenue']) / ytd_bud['budget_revenue']
    gm_act = ytd_act['revenue'] - ytd_act['direct_cost']
    gm_bud = ytd_bud['budget_revenue'] - ytd_bud['budget_direct']
    gm_var = (gm_act - gm_bud) / gm_bud
    op_act = ytd_act['operating_profit']
    op_bud = gm_bud - ytd_bud['budget_indirect']
    op_var = (op_act - op_bud) / op_bud
    
    metrics_ppt = {
        'rev_val': format_k(ytd_act['revenue']), 'rev_delta': format_pct(rev_var),
        'gm_val': format_k(gm_act), 'gm_delta': format_pct(gm_var),
        'op_val': format_k(op_act), 'op_delta': format_pct(op_var),
        'op_margin_val': format_pct(ytd_act['operating_margin_pct']/6), 'op_margin_delta': format_pct(op_var),
        'opex_val': format_k(ytd_act['indirect_cost']), 'opex_delta': "+2%",
        'cf_val': format_k(ytd_act['cash_flow']), 'cf_delta': "+5%"
    }

    st.markdown("### 🚀 Enterprise Performance")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: render_glass_metric("Revenue", format_k(ytd_act['revenue']), format_pct(rev_var), rev_var > 0)
    with c2: render_glass_metric("Gross Margin", format_k(gm_act), format_pct(gm_var), gm_var > 0)
    with c3: render_glass_metric("Op Margin", format_pct(ytd_act['operating_margin_pct']/6), format_pct(op_var), op_var > 0)
    with c4: render_glass_metric("OPEX", format_k(ytd_act['indirect_cost']), "+2.1%", False)
    with c5: render_glass_metric("Cash Flow", format_k(ytd_act['cash_flow']), "+5.4%", True)
    with c6: render_glass_metric("Accuracy", "94.2%", "-1.1%", False)

    # 2. MAIN CHARTS
    col_main, col_side = st.columns([2, 1])
    with col_main:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ANNUAL_DATA['month'], y=ANNUAL_DATA['revenue'], name='Revenue', marker_color='#38bdf8', opacity=0.7))
        fig.add_trace(go.Scatter(x=ANNUAL_DATA['month'], y=ANNUAL_DATA['operating_profit'], name='Op Profit', line=dict(color='#4ade80', width=3), yaxis='y2'))
        fig.update_layout(
            title="Revenue & Profit Trend", 
            yaxis=dict(title="Revenue", showgrid=False),
            yaxis2=dict(title="Profit", overlaying='y', side='right', showgrid=False),
            legend=dict(orientation="h", y=1.1), height=350
        )
        fig = apply_glass_style(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_side:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_bridge = go.Figure(go.Waterfall(
            name = "20", orientation = "v",
            measure = ["relative", "relative", "relative", "total"],
            x = ["Budget", "Vol", "Cost", "Actual"],
            textposition = "outside", text = ["+10k", "+5k", "-2k", "£68k"],
            y = [55000, 15000, -2000, 0],
            connector = {"line":{"color":"white"}},
            decreasing = {"marker":{"color":"#f87171"}}, increasing = {"marker":{"color":"#4ade80"}}, totals = {"marker":{"color":"#94a3b8"}}
        ))
        fig_bridge.update_layout(title="Profit Bridge (Jun)", height=350)
        fig_bridge = apply_glass_style(fig_bridge)
        st.plotly_chart(fig_bridge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. DRILL DOWN
    st.markdown("### 🔍 Drill Down Analysis")
    col_d1, col_d2 = st.columns([1,2])
    with col_d1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        cat = st.selectbox("Category", ["Revenue", "COGS", "OPEX"])
        st.markdown('</div>', unsafe_allow_html=True)
    with col_d2:
        drill_data = pd.DataFrame({
            'Sub-Category': ['Product A', 'Product B', 'Service'],
            'Actual': [45000, 32000, 15000],
            'Budget': [42000, 35000, 14000]
        })
        drill_data['Var'] = drill_data['Actual'] - drill_data['Budget']
        st.dataframe(drill_data.style.format("{:,.0f}"), use_container_width=True)

    # 4. AI & EXPORT
    st.markdown("### 🤖 Strategic Intelligence")
    c_ai, c_exp = st.columns([3, 1])
    
    with c_ai:
        if 'ai_analysis' not in st.session_state:
            st.session_state['ai_analysis'] = "Click Generate to analyze."
        
        if st.button("⚡ Run Analysis"):
            with st.spinner("Analyzing..."):
                st.session_state['ai_analysis'] = generate_ai_narrative()
        
        st.markdown(f"""<div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #4ade80;">{st.session_state['ai_analysis']}</div>""", unsafe_allow_html=True)

    with c_exp:
        st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
        if st.session_state['ai_analysis'] != "Click Generate to analyze.":
            pptx = create_professional_presentation(metrics_ppt, st.session_state['ai_analysis'])
            st.download_button("📥 Download Board Pack", pptx, "Board_Pack.pptx", type="primary")
        else: st.caption("Run Analysis to Export")
        st.markdown('</div>', unsafe_allow_html=True)

def data_engine_view():
    st.markdown("### ⚙️ Data Engine")
    st.success("Pipeline Active | Latency: 42ms")
    st.dataframe(ANNUAL_DATA.head(), use_container_width=True)

# --- MAIN ---
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
    <div>
        <h1 style="margin:0; font-size: 2.5rem;">FinSight Core</h1>
        <p style="opacity: 0.8;">Liquid Glass Interface • Live Connection</p>
    </div>
    <div class="glass-card" style="padding: 10px 20px; margin:0;">User: Aaron M.</div>
</div>
""", unsafe_allow_html=True)

t1, t2 = st.tabs(["Executive Dashboard", "Data Engine"])
with t1: dashboard_view()
with t2: data_engine_view()
