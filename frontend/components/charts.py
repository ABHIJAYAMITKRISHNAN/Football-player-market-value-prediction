import plotly.graph_objects as go
import plotly.express as px

PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e6edf3', family='Inter, sans-serif'),
    xaxis=dict(gridcolor='#21262d', linecolor='#30363d', zerolinecolor='#30363d'),
    yaxis=dict(gridcolor='#21262d', linecolor='#30363d', zerolinecolor='#30363d'),
    colorway=['#00ff87', '#00d4ff', '#ffd700', '#ff6b6b', '#c084fc', '#fb923c'],
    margin=dict(l=40, r=40, t=50, b=40),
    hoverlabel=dict(bgcolor='#161b22', bordercolor='#30363d', font_color='#e6edf3'),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#30363d')
)

def apply_theme(fig):
    fig.update_layout(**PLOTLY_THEME)
    return fig

def create_radar_chart(categories, query_values, match_values=None, query_name="Player A", match_name="Player B"):
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=query_values,
        theta=categories,
        fill='toself',
        name=query_name,
        line_color='#00ff87'
    ))

    if match_values:
        fig.add_trace(go.Scatterpolar(
            r=match_values,
            theta=categories,
            fill='toself',
            name=match_name,
            line_color='#00d4ff'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#30363d'),
            angularaxis=dict(gridcolor='#30363d')
        ),
        showlegend=True
    )
    
    return apply_theme(fig)
