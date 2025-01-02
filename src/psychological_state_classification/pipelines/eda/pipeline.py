"""
This is a boilerplate pipeline 'eda'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=plot_missing,
            inputs="raw_data",
            outputs="missing_data_plot",
            name="plot_missing_node"
        ),
        node(
            func=plot_dist_hrv,
            inputs="preprocessed_data",
            outputs="dist_hrv_plot",
            name="plot_hrv_distribution"
        )
    ])
