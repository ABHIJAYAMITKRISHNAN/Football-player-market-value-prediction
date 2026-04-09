import streamlit as st

def apply_custom_styles():
    st.markdown("""
    <style>
    /* KPI Card */
    .kpi-card {
        background: linear-gradient(135deg, #161b22, #21262d);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,255,135,0.08);
        transition: transform 0.2s;
        margin-bottom: 20px;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #00ff87; }
    .kpi-delta { font-size: 0.85rem; color: #8b949e; }
    .kpi-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

    /* Section Header */
    .section-header {
        font-size: 1.4rem; font-weight: 600;
        border-left: 4px solid #00ff87;
        padding-left: 12px; margin: 24px 0 16px 0;
        color: #e6edf3;
    }

    /* Risk Badges */
    .badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
    .badge-low    { background: rgba(35,134,54,0.25);  color: #3fb950; border: 1px solid #238636; }
    .badge-medium { background: rgba(158,106,3,0.25);  color: #d29922; border: 1px solid #9e6a03; }
    .badge-high   { background: rgba(218,54,51,0.25);  color: #f85149; border: 1px solid #da3633; }

    /* Valuation Badges */
    .badge-under  { background: rgba(0,255,135,0.15);  color: #00ff87; border: 1px solid #00ff87; }
    .badge-over   { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid #f85149; }
    .badge-fair   { background: rgba(139,148,158,0.15); color: #8b949e; border: 1px solid #8b949e; }

    /* Player Card */
    .player-card {
        background: #161b22; border: 1px solid #30363d; border-radius: 10px;
        padding: 16px; margin: 8px 0;
    }
    .player-name { font-size: 1.1rem; font-weight: 600; color: #e6edf3; }
    .player-meta { font-size: 0.8rem; color: #8b949e; }

    /* Progress Bar (Similarity Score) */
    .progress-bar-bg { background: #21262d; border-radius: 4px; height: 8px; margin: 4px 0; }
    .progress-bar-fill { background: linear-gradient(90deg, #00ff87, #00d4ff); border-radius: 4px; height: 8px; }
    </style>
    """, unsafe_allow_html=True)

def render_kpi(label, value, delta=None):
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ''
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        {delta_html}
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)
