"""Results visualisation module."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


class ModelMetricsVisualizer:
    """Class to visualize model metrics."""

    def __init__(self, data):
        self.data = data

    def visualize_bar_plots(self):
        """Visualize metrics using bar plots."""
        models = list(self.data.keys())
        metrics = list(self.data[models[0]].keys())
        fig = go.Figure()
        for metric in metrics:
            metric_values = [self.data[model][metric] for model in models]
            max_value = max(metric_values)
            normalized_values = [value / max_value for value in metric_values]
            fig.add_trace(go.Bar(x=models, y=normalized_values, name=metric, text=metric_values, textposition='auto'))
        fig.update_layout(
            title="Bar Plots of Metrics Across Models",
            xaxis=dict(title="Model"),
            yaxis=dict(title="Normalized Metric Value"),
        )
        return fig

    def visualize_box_plots(self):
        """Visualize metrics using box plots."""
        df = pd.DataFrame(self.data).T.reset_index()
        fig = px.box(
            df, x="index", y=df.columns, title="Box Plot of Metrics Across Models"
        )
        fig.update_layout(xaxis_title="Model", yaxis_title="Metric Value")
        return fig
