"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=check_outliers,
            inputs="train_data",
            outputs=None,
            name="check_outliers_node",
            namespace='feature_engineering'
        ),
        node(
            func=process_num,
            inputs=["train_data", "test_data",
                    'params:num_transform'],
            outputs=["processed_num_train", "processed_num_test"],
            name="numerical_transform_node",
            namespace='feature_engineering'
        ),
        node(
            func=process_cat,
            inputs=["processed_num_train", "processed_num_test",
                    "params:cat_transform", "params:ordinal_order"],
            outputs=["processed_cat_train", "processed_cat_test"],
            name="categorical_transform_node",
            namespace='feature_engineering'
        ),
        node(
            func=drop_columns,
            inputs=["processed_cat_train", "processed_cat_test",
                    'params:col_drop'],
            outputs=['final_train_data', 'final_test_data'],
            name="drop_columns_node",
            namespace='feature_engineering'
        )
    ])
