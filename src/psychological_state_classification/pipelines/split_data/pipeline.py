"""
This is a boilerplate pipeline 'split_data'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs=["preprocessed_data", "params:split_options"],
            outputs=["train_data", "test_data"],
            name="split_data"
        )
    ])
