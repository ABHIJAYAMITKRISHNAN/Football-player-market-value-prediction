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

def create_radar_chart(categories, query_values, match_values=None, query_name="Player A", match_name="Player B", query_raws=None, match_raws=None):
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=query_values,
        theta=categories,
        fill='toself',
        opacity=0.6,
        name=query_name,
        line_color='#00ff87',
        customdata=query_raws if query_raws is not None else [],
        hovertemplate="<b>%{theta}</b><br>Value: %{customdata}<extra></extra>" if query_raws is not None else None
    ))

    if match_values:
        fig.add_trace(go.Scatterpolar(
            r=match_values,
            theta=categories,
            fill='toself',
            opacity=0.6,
            name=match_name,
            line_color='#00d4ff',
            customdata=match_raws if match_raws is not None else [],
            hovertemplate="<b>%{theta}</b><br>Value: %{customdata}<extra></extra>" if match_raws is not None else None
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 1], 
                gridcolor='rgba(255, 255, 255, 0.2)',
                tickfont=dict(color='#e6edf3', size=13)
            ),
            angularaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.2)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    
    return apply_theme(fig)
