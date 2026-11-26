import plotly.express as px

from utils.data_loader import CLASSES, TARGET

COLORS = {"Low": "#b42332", "Medium": "#a14d05", "High": "#08754d"}


def plot(figure, height=350):
    return figure.update_layout(height=height, margin=dict(l=15, r=15, t=35, b=20),
                                font=dict(color="#14264a"), paper_bgcolor="#ffffff",
                                plot_bgcolor="#ffffff")


def categories(frame):
    counts = frame[TARGET].value_counts().reindex(CLASSES, fill_value=0)
    return plot(px.bar(x=counts.index, y=counts.values, color=counts.index,
                       color_discrete_map=COLORS, text_auto=True,
                       labels={"x": "Performance category", "y": "Students"})
                .update_layout(showlegend=False))

