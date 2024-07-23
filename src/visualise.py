"""Python module to visualize the dataset."""

import plotly.express as px
import pandas as pd

class DataVisualizer:
    """Data Visualizer class to visualize the dataset."""
    def __init__(self, dataset_path="./data/data.csv"):
        """Initialize DataVisualizer with the dataset."""
        self.data = pd.read_csv(dataset_path)

    def visualize_data(self):
        """Visualize the dataset."""
        plots = []
        for column in self.data.columns:
            if self.data[column].dtype in ["int64", "float64"]:
                if self.data[column].nunique() > 10:
                    plots.append(self.plot_histogram(self.data, column))
                else:
                    plots.append(self.plot_bar_chart(self.data, column))
                if column != "predict":
                    plots.append(self.plot_scatter(self.data, column, "predict"))
            elif self.data[column].dtype == "object":
                plots.append(self.plot_bar_chart(self.data, column))
        return plots

    @staticmethod
    def plot_histogram(data, column):
        """Histogram of a column in the dataset."""
        fig = px.histogram(data, x=column, title=f"Histogram of {column}")
        return fig

    @staticmethod
    def plot_bar_chart(data, column):
        """Bar chart of a column in the dataset."""
        fig = px.bar(
            data[column].value_counts(),
            x=data[column].value_counts().index,
            y=data[column].value_counts().values,
            labels={column: f"{column}", "y": "Frequency"},
            title=f"Bar Chart of {column}",
        )
        return fig

    @staticmethod
    def plot_scatter(data, x_column, y_column):
        """Scatter plot of two columns in the dataset."""
        fig = px.scatter(
            data,
            x=x_column,
            y=y_column,
            title=f"Scatter Plot of {x_column} vs {y_column}",
        )
        return fig
