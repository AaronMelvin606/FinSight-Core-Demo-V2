# FINSIGHT CORE V1.0 - STREAMLIT DEPLOYMENT
# This script mimics the functionality of the React dashboard in a Streamlit environment,
# including predictive charts and the AI Narrative Engine powered by the Gemini API.

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time # For exponential backoff and loading simulation

# --- CONFIGURATION & SETUP ---
st.set_page_config(layout="wide", page_title="FinSight Core Demo")

# Brand Colors (for visual consistency)
COLORS = {
    'sage': '#8B9D83',
    'forest': '#2C3E2A',
    'forecast': '#EAB308',
    'alert': '#EF4444',
    'success': '#10B981'
}

# Gemini API Configuration
GEMINI_MODEL = 'gemini-2.5-flash-preview-09-2025'
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# --- MOCK DATA ---
# Combined Actual (Jan-Jun) and Forecast (Jul-Dec) data
ANNUAL_DATA = pd.DataFrame([
    # Actuals
    {'month': 'Jan', 'revenue_act': 120000, 'revenue_bud': 115000, 'opex_act': 85000, 'opex_bud': 80000},
    {'month': 'Feb', 'revenue_act': 125000, 'revenue_bud': 118000, 'opex_act': 82000, 'opex_bud': 82000},
    {'month': 'Mar', 'revenue_act': 110000, 'revenue_bud': 122000, 'opex_act': 88000, 'opex_bud': 81000},
    {'month': 'Apr', 'revenue_act': 140000, 'revenue_bud': 125000, 'opex_act': 95000, 'opex_bud': 85000},
    {'month': 'May', 'revenue_act': 135000, 'revenue_bud': 130000, 'opex_act': 89000, 'opex_bud': 86000},
    {'month': 'Jun', 'revenue_act': 155000, 'revenue_bud': 140000, 'opex_act': 102000, 'opex_bud': 90000},
    # Forecasts (Note: Using 'revenue_act' for simplicity in Streamlit plotting)
    {'month': 'Jul', 'revenue_act': 145000, 'revenue_bud': 135000, 'opex_act': 92000, 'opex_bud': 88000},
    {'month': 'Aug', 'revenue_act': 148000, 'revenue_bud': 138000, 'opex_act': 91000, 'opex_bud': 88000},
    {'month': 'Sep', 'revenue_act': 160000, 'revenue_bud': 150000, 'opex_act': 98000, 'opex_bud': 92000},
    {'month': 'Oct', 'revenue_act': 158000, 'revenue_bud': 145000, 'opex_act': 95000, 'opex_bud': 90000},
    {'month': 'Nov', 'revenue_act': 165000, 'revenue_bud': 155000, 'opex_act': 105000, 'opex_bud': 95000},
    {'month': 'Dec', 'revenue_act': 180000, 'revenue_bud': 170000, 'opex_act': 110000, 'opex_bud': 100000},
])

DEPT_DATA = pd.DataFrame([
    {'dept': 'Sales', 'actual': 450000, 'budget': 420000},
    {'dept': 'Marketing', 'actual': 320000, 'budget': 280000},
    {'dept': 'Product', 'actual': 580000, 'budget': 600000},
    {'dept': 'G&A', 'actual': 150000, 'budget': 145000},
    {'dept': 'IT', 'actual': 210000, 'budget': 200000},
])

# --- UTILITIES ---

def format_currency(amount):
    return f"£{amount / 1000:.1f}k"

def calculate_metrics(df, dept_df):
    """Calculates all MTD, YTD, and FY metrics."""
    # Data Split
    ytd_df = df.iloc[0:6]
    fy_forecast_df = df.iloc[6:12]

    # MTD (June)
    mtd_rev_act = ytd_df['revenue_act'].iloc[-1]
    mtd_rev_bud = ytd_df['revenue_bud'].iloc[-1]
    
    # YTD (Jan-Jun)
    ytd_rev_act = ytd_df['revenue_act'].sum()
    ytd_rev_bud = ytd_df['revenue_bud'].sum()
    
    # FY Outlook
    fy_rev_forecast = ytd_rev_act + fy_forecast_df['revenue_act'].sum()
    fy_rev_budget = df['revenue_bud'].sum()

    # OpEx YTD Variance for Narrative
    ytd_opex_act = ytd_df['opex_act'].sum()
    ytd_opex_bud = ytd_df['opex_bud'].sum()
    
    # Departmental Variances for Narrative
    dept_df['variance'] = dept_df['actual'] - dept_df['budget']
    dept_df['pct_var'] = (dept_df['variance'] / dept_df['budget']) * 100
    top_overspends = dept_df[dept_df['variance'] > 0].sort_values(by='variance', ascending=False).head(2)

    return {
        'mtd_rev_act': mtd_rev_act, 'mtd_rev_bud': mtd_rev_bud,
        'ytd_rev_act': ytd_rev_act, 'ytd_rev_bud': ytd_rev_bud,
        'fy_rev_forecast': fy_rev_forecast, 'fy_rev_budget': fy_rev_budget,
        'ytd_opex_act': ytd_opex_act, 'ytd_opex_bud': ytd_opex_bud,
        'top_overspends': top_overspends
    }

def format_variance(actual, budget, is_cost=False):
    """Calculates variance in £ and %."""
    variance_abs = actual - budget
    variance_pct = (variance_abs / budget) * 100 if budget else 0
    
    # Determine the indicator color based on type (Revenue/Cost)
    if is_cost:
        is_good = variance_abs < 0
    else:
        is_good = variance_abs > 0
        
    color = COLORS['success'] if is_good else COLORS['alert']
    
    # Format string for display
    return f"""
        <div style='color: {color}; font-weight: bold;'>
            {('+' if variance_abs > 0 else '')}{format_currency(variance_abs)} 
            ({('+' if variance_pct > 0 else '')}{variance_pct:.1f}%)
        </div>
    """

def metric_card(title, value, budget_value, variance_html, label=""):
    """Renders a visually styled metric card."""
    st.markdown(f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); border: 1px solid #f0f0f0; height: 100%;">
            <p style="color: #6b7280; font-size: 14px; font-weight: bold; text-transform: uppercase; margin-bottom: 10px;">{title} <span style="background-color: #f3f4f6; padding: 2px 8px; border-radius: 4px; font-size: 10px; color: #4b5563;">{label}</span></p>
            <h3 style="font-size: 28px; font-weight: bold; color: #1f2937;">{format_currency(value)}</h3>
            <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #f9fafb; margin-top: 10px; padding-top: 10px;">
                <span style="color: #9ca3af; font-size: 12px;">Budget: {format_currency(budget_value)}</span>
                {variance_html}
            </div>
        </div>
    """, unsafe_allow_html=True)


# --- GEMINI AI NARRATIVE FUNCTION ---
def generate_ai_narrative(metrics, top_overspends):
    """
    Constructs the prompt and calls the Gemini API to get structured narrative.
    """
    fy_var_pct = ((metrics['fy_rev_forecast'] - metrics['fy_rev_budget']) / metrics['fy_rev_budget']) * 100
    ytd_opex_var_pct = ((metrics['ytd_opex_act'] - metrics['ytd_opex_bud']) / metrics['ytd_opex_bud']) * 100

    top_variances_list = [
        f"* **{row['dept']}**: Overspent by **{format_currency(row['variance'])}** ({row['pct_var']:.1f}%) YTD."
        for index, row in top_overspends.iterrows()
    ]
    top_variances_str = "\n".join(top_variances_list) if top_variances_list else "* No significant departmental overspends detected."


    financial_context = f"""
    Key Financial Context (as of June 2024, Month 6/12):
    - Full Year Revenue Forecast: {format_currency(metrics['fy_rev_forecast'])}
    - Full Year Revenue Budget: {format_currency(metrics['fy_rev_budget'])}
    - Full Year Revenue Variance: {fy_var_pct:.1f}% ({'Favourable' if fy_var_pct > 0 else 'Adverse'})
    - YTD OpEx Variance (Actual vs Budget): {ytd_opex_var_pct:.1f}% ({'Adverse' if ytd_opex_var_pct > 0 else 'Favourable'})
    - Top Departmental Overspends YTD:
      {top_variances_str}
    """

    # ENFORCING BOARD-READY STRUCTURE (Step 2. Refinement)
    system_prompt = """
    You are a Senior Financial Analyst creating a board memo for the CFO. Your output MUST be in clear, structured Markdown. 
    - Use '##' for the main section headings.
    - Use markdown bullet points ('*') for all key data points.
    - Bold significant numbers and terms.
    - The tone must be professional, action-oriented, and strategic.
    Generate a summary covering Revenue, Expenses, and a forward-looking Conclusion based on the context.
    """

    user_query = f"Generate the structured executive summary using the following data context. The memo must be ready for immediate copy-paste into a board deck slide. Current Period: June 2024.\n\n{financial_context}"

    payload = {
        'contents': [{'parts': [{'text': user_query}]}],
        'systemInstruction': {'parts': [{'text': system_prompt}]},
        'tools': [{"google_search": {} }]
    }

    # API Key Retrieval (Crucial for Streamlit Cloud Deployment)
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        return "AI Narrative Failed: API Key not found in Streamlit secrets. Please configure it for deployment."

    # Exponential Backoff for Robustness
    max_retries = 3
    for i in range(max_retries):
        try:
            # We must pass the API key as a query parameter in a real environment
            response = requests.post(f"{API_URL}?key={api_key}", headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status() # Raises an exception for HTTP error codes
            
            result = response.json()
            narrative = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "Generation failed to return text.")
            return narrative

        except requests.exceptions.RequestException as e:
            wait_time = 2 ** i
            if i < max_retries - 1:
                time.sleep(wait_time)
                continue
            else:
                return f"Gemini API call failed after {max_retries} retries: {e}"
        except Exception as e:
            return f"AI Narrative Failed: An unexpected error occurred: {e}"
    return "AI Narrative Failed."

# --- STREAMLIT APP LAYOUT ---

def main_dashboard():
    """Renders the main Dashboard and triggers the AI Analysis."""
    st.header("Financial Performance & Outlook", divider='gray')
    st.markdown("Period: June 2024 (Month 6) • Currency: GBP (£)")

    metrics = calculate_metrics(ANNUAL_DATA, DEPT_DATA.copy())

    # --- Header Metrics (MTD, YTD, FY) ---
    st.subheader("Revenue Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        metric_card(
            "Month to Date (MTD)", metrics['mtd_rev_act'], metrics['mtd_rev_bud'],
            format_variance(metrics['mtd_rev_act'], metrics['mtd_rev_bud']), "June"
        )
    with col2:
        metric_card(
            "Year to Date (YTD)", metrics['ytd_rev_act'], metrics['ytd_rev_bud'],
            format_variance(metrics['ytd_rev_act'], metrics['ytd_rev_bud']), "Jan - Jun"
        )
    with col3:
        metric_card(
            "Full Year Outlook (FY)", metrics['fy_rev_forecast'], metrics['fy_rev_budget'],
            format_variance(metrics['fy_rev_forecast'], metrics['fy_rev_budget']), "Actuals + Forecast"
        )

    st.markdown("---")

    # --- Predictive Forecast Chart ---
    st.subheader("Predictive Revenue Forecast")
    
    # Chart data prep: Melt DataFrame to plot Actuals and Budget/Forecast cleanly
    chart_df = ANNUAL_DATA[['month', 'revenue_act', 'revenue_bud']].rename(columns={'revenue_act': 'Revenue', 'revenue_bud': 'Budget'})
    
    # Create a column to distinguish actuals vs forecast for color coding
    chart_df['Type'] = ['Actual'] * 6 + ['Forecast'] * 6

    st.line_chart(
        chart_df, 
        x='month', 
        y=['Revenue', 'Budget'],
        color=[COLORS['sage'], COLORS['forest']],
        height=400
    )
    st.caption("Actuals are solid lines (Jan-Jun). Forecast values (Jul-Dec) are predictive. Budget is the dark line.")

    st.markdown("---")

    # --- Departmental Summary Table ---
    st.subheader("Departmental P&L (YTD)")
    
    dept_display = DEPT_DATA.copy()
    dept_display['Variance (£)'] = dept_display['actual'] - dept_display['budget']
    dept_display['Variance (%)'] = ((dept_display['Variance (£)'] / dept_display['budget']) * 100).round(1).astype(str) + '%'
    dept_display = dept_display[['dept', 'actual', 'budget', 'Variance (£)', 'Variance (%)']].rename(
        columns={'dept': 'Department', 'actual': 'Actual (£)', 'budget': 'Budget (£)'}
    )
    
    st.dataframe(dept_display, use_container_width=True)
    st.caption("Note: Positive Variance (£) indicates an overspend for expense departments.")
    
    # --- AI Narrative Trigger ---
    st.markdown("---")
    st.subheader("AI Narrative Generation")
    
    if st.button("🚀 Run AI Analysis & Generate Board Memo", type="primary"):
        with st.spinner('FinSight AI is analyzing variances, integrating predictive signals, and drafting the Executive Summary...'):
            narrative = generate_ai_narrative(metrics, metrics['top_overspends'])
            st.session_state.ai_narrative = narrative
            st.success("AI Analysis Complete!")
            st.experimental_rerun() # Rerun to refresh the AI Insights tab

def ai_insights():
    """Renders the AI Narrative and simulated insights."""
    st.header("AI Narrative Engine", divider='green')
    st.markdown("### Automated Variance Commentary & Board-Ready Memo")
    
    if 'ai_narrative' not in st.session_state:
        st.session_state.ai_narrative = "Click the 'Run AI Analysis' button on the **Dashboard** tab to generate the first executive summary."

    # --- Executive Summary Output (Renders Markdown from LLM) ---
    st.markdown(f"""
        <div style="background-color: #ffffff; padding: 25px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); border: 2px solid {COLORS['sage']};">
            <h4 style="color: {COLORS['forest']}; margin-bottom: 15px; font-weight: bold;">Executive Summary • June 2024</h4>
            {st.session_state.ai_narrative}
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- Simulated Insights (Fixed for Demo) ---
    st.subheader("Automated Risk & Opportunity Signals")
    st.info("The FinSight Core automatically surfaces anomalies that require investigation:", icon="💡")
    
    col1, col2 = st.columns(2)
    with col1:
        st.error("🚨 **Cost Anomaly Detected**\n\nMarketing spend in June (£22k) is **2.4 standard deviations** above the 6-month average. Primary driver: 'Agency Retainer - Q3 Prep'.")
    with col2:
        st.success("📈 **Revenue Forecast Update**\n\nBased on current run-rates, the ML model projects Q3 Revenue to land at **£480k**, which is £25k ahead of the original January Budget.")
        
    st.markdown("---")
    
    st.subheader("Conversational Analysis (Ask FinSight AI)")
    st.code("Why is OpEx so high in June?", language='python')
    st.markdown("<p style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>June OpEx is £12k adverse to budget. Drilling down into the ledger, 80% of this variance comes from GL Code 6100 (Marketing), specifically vendor 'DigitalReach Agency'. This appears to be a timing difference from May. (Simulated AI Response)</p>", unsafe_allow_html=True)


def data_engine():
    """Renders the Data Engine view."""
    st.header("Data Engine (ETL Blueprint)", divider='gray')
    st.markdown("### Managed Warehouse & Transformation Pipelines")
    
    st.info("This view demonstrates the client's ownership of the data pipeline (your **Integration Sprint** deliverable).", icon="🔑")

    st.markdown("#### System Status")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"<span style='background-color: {COLORS['success']}1A; color: {COLORS['success']}; padding: 5px 10px; border-radius: 5px; font-weight: bold;'>✅ ETL Status: Healthy</span>", unsafe_allow_html=True)
    with col2:
        st.markdown("<span style='color: #4b5563;'>Last Sync: Today, 09:41 AM via ERP Connector</span>", unsafe_allow_html=True)
        
    st.markdown("---")

    st.markdown("#### Transformation Rules (Client-Owned Logic)")
    st.code("""
# Python ETL script used in 'data_transformer.py'
# This logic is deployed locally on client infrastructure.

# 1. Map 'SAP_Export_v2.csv' headers to standard schema
# 2. Filter Department != 'Intercompany'
# 3. Calculate 'Gross Margin %'
    """, language='python')
    
    st.markdown("---")
    st.markdown("#### Raw Data Sample")
    
    raw_data = pd.DataFrame([
        {'Transaction ID': 'TRX-001', 'Date': '2024-06-01', 'Account': '4000-Revenue', 'Dept': 'Sales', 'Amount': 45000, 'Type': 'Actual'},
        {'Transaction ID': 'TRX-002', 'Date': '2024-06-01', 'Account': '6000-Salaries', 'Dept': 'G&A', 'Amount': 12000, 'Type': 'Actual'},
        {'Transaction ID': 'TRX-003', 'Date': '2024-06-02', 'Account': '6100-Marketing', 'Dept': 'Marketing', 'Amount': 5600, 'Type': 'Actual'},
    ])
    st.dataframe(raw_data, use_container_width=True)

# --- MAIN APP EXECUTION ---

# Create tabs for the multi-view application
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "✨ AI Insights", "⚙️ Data Engine"])

with tab1:
    main_dashboard()

with tab2:
    ai_insights()

with tab3:
    data_engine()
