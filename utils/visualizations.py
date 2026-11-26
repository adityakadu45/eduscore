import plotly.express as px

from utils.data_loader import CLASSES, TARGET

COLORS = {"Low": "#ad304d", "Medium": "#965006", "High": "#09735f"}
ACCENT = "#0b7066"


def plot(figure, height=350):
    figure.update_xaxes(gridcolor="#e6edf5", zerolinecolor="#d0dae8")
    figure.update_yaxes(gridcolor="#e6edf5", zerolinecolor="#d0dae8")
    return figure.update_layout(height=height, margin=dict(l=16, r=18, t=32, b=24),
                                font=dict(color="#17243b"), paper_bgcolor="#ffffff",
                                plot_bgcolor="#ffffff", colorway=[ACCENT, "#4b5caf", "#aa5c22"],
                                hoverlabel=dict(bgcolor="#17243b", font_color="#ffffff"))


def categories(frame, share=False):
    counts = frame[TARGET].value_counts().reindex(CLASSES, fill_value=0)
    values = counts / max(len(frame), 1) if share else counts
    figure = px.bar(x=counts.index, y=values.values, color=counts.index,
                    color_discrete_map=COLORS, text_auto=".1%" if share else True,
                    labels={"x": "Performance category", "y": "Share of records" if share else "Students"})
    figure.update_traces(marker_cornerradius=6)
    if share:
        figure.update_yaxes(tickformat=".0%")
    return plot(figure.update_layout(showlegend=False))

