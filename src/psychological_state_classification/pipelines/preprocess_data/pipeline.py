"""
This is a boilerplate pipeline 'preprocess_data'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import preprocess_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_data,
            inputs=["raw_data","params:data_types"],
            outputs="preprocessed_data",
            name="preprocess_data_node"
        )
    ])
