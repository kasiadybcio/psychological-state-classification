"""
This is a boilerplate pipeline 'feature_selection'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=get_feat_rank_stat,
            inputs=["final_train_data", "params:fs_params"],
            outputs="feat_rank_stat_results",
            name="get_feat_rank_stat_node",
            namespace='feature_selection'

        ),
        node(
            func=get_feat_rank_corr,
            inputs=["final_train_data", "params:fs_params"],
            outputs="feat_rank_corr_results",
            name="get_feat_rank_corr_node",
            namespace='feature_selection'
        ),
        # node(
        #     func=get_feat_backward,
        #     inputs=["final_train_data", "params:fs_params"],
        #     outputs="feat_backward_results",
        #     name="get_feat_backward_node",
        #     namespace='feature_selection'
        # ),
        # node(
        #     func=get_feat_forward,
        #     inputs=["final_train_data", "params:fs_params"],
        #     outputs="feat_forward_results",
        #     name="get_feat_forward_node",
        #     namespace='feature_selection'
        # ),
        node(
            func=select_features,
            inputs=['final_train_data', 'final_test_data',
                    "feat_rank_stat_results", "feat_rank_corr_results",
                    "params:fs_params"],
            outputs=["train_model","test_model", "selected_features"],
            name="select_features_node",
            namespace='feature_selection'
        ),
    ])
