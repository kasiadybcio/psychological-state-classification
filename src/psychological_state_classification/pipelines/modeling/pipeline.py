"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=get_initial_score,
            inputs=['train_model', 'params:params'],
            outputs='initial_cv_score',
            name='get_initial_score_node',
            namespace='hyperparameter_tuning'
        ),
        node(
            func=optimize_lr_hyperparams,
            inputs=['train_model', 'params:params'],
            outputs=['LR_optuna_params','LR_optuna_params_'],
            name='optimize_lr_hyperparams_node',
            namespace='hyperparameter_tuning'
        ),
        node(
            func=optimize_lgbm_hyperparams,
            inputs=['train_model', 'params:params'],
            outputs=['LGBM_optuna_params','LGBM_optuna_params_'],
            name='optimize_lgbm_hyperparams_node',
            namespace='hyperparameter_tuning'
        ),
        node(
            func=optimize_xgb_hyperparams,
            inputs=['train_model', 'params:params'],
            outputs=['XGB_optuna_params','XGB_optuna_params_'],
            name='optimize_xgb_hyperparams_node',
            namespace='hyperparameter_tuning'
        ),
        node(
            func=optimize_ada_hyperparams,
            inputs=['train_model', 'params:params'],
            outputs=['ADA_optuna_params','ADA_optuna_params_'],
            name='optimize_ada_hyperparams_node',
            namespace='hyperparameter_tuning'
        ),
        node(
            func=eval_best_models,
            inputs=['train_model', 'test_model',
                    'LR_optuna_params_', 'LGBM_optuna_params_', 
                    'XGB_optuna_params_','ADA_optuna_params_',
                    'params:params'],
            outputs=['models_evaluation', 'y_train_preds', 'y_test_preds'],
            name='eval_best_models_node'
        )
    ])
