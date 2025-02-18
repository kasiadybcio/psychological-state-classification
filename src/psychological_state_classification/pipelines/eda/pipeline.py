"""
This is a boilerplate pipeline 'eda'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    plotting1 = pipeline([
        node(
            plot_mood_state_distribution,
            inputs="preprocessed_data",
            outputs="plot_mood_state_distribution_px",
            name="plot_mood_state_distribution",
            namespace='plotting1'
        ),

        node(
            plot_HRV_distribution,
            inputs="preprocessed_data",
            outputs="plot_HRV_distribution_px",
            name="plot_HRV_distribution",
            namespace='plotting1'
        ),

        node(
            plot_GSR_distribution,
            inputs="preprocessed_data",
            outputs="plot_GSR_distribution_px",
            name="plot_GSR_distribution",
            namespace='plotting1'
        ),

        node(
            plot_Age_distribution,
            inputs="preprocessed_data",
            outputs="plot_Age_distribution_px",
            name="plot_Age_distribution",
            namespace='plotting1'
        ),

        node(
            plot_Heart_Rate_distribution,
            inputs="preprocessed_data",
            outputs="plot_Heart_Rate_distribution_px",
            name="plot_Heart_Rate_distribution",
            namespace='plotting1'
        ),

        node(
            plot_Gender_distribution,
            inputs="preprocessed_data",
            outputs="plot_Gender_distribution_px",
            name="plot_Gender_distribution",
            namespace='plotting1'
        ),

        node(
            plot_Task_Type_distribution,
            inputs="preprocessed_data",
            outputs="plot_Task_Type_distribution_px",
            name="plot_Task_Type_distribution",
            namespace='plotting1'
        ),

        node(
            plot_Educational_Level_distribution,
            inputs="preprocessed_data",
            outputs="plot_Educational_Level_distribution_px",
            name="plot_Educational_Level_distribution",
            namespace='plotting1'
        ),        


        node(
            plot_Time,
            inputs="preprocessed_data",
            outputs="plot_Time_px",
            name="plot_Time_px",
            namespace='plotting1'
        )
    ])

    plotting2 = pipeline([
        node(
            plot_Mood_State_by_Gender_distribution,
            inputs="train_data",
            outputs="plot_Mood_State_by_Gender_distribution_px",
            name="plot_Mood_State_by_Gender_distribution",
            namespace='plotting2'
        ),
        node(
            plot_Age_vs_Mood_State,
            inputs="train_data",
            outputs="plot_Age_vs_Mood_State_px",
            name="plot_Age_vs_Mood_State",
            namespace='plotting2'
        ),
        node(
            plot_Task_Type_vs_Educational_Level,
            inputs="train_data",
            outputs="plot_Task_Type_vs_Educational_Level_px",
            name="plot_Task_Type_vs_Educational_Level",
            namespace='plotting2'
        ),
        node(
            plot_Skin_Temp_vs_Mood_State,
            inputs="train_data",
            outputs="plot_Skin_Temp_vs_Mood_State_px",
            name="plot_Skin_Temp_vs_Mood_StateV_px",
            namespace='plotting2'
        ),
        node(
            plot_Heart_Rate_vs_HRV,
            inputs="train_data",
            outputs="plot_Heart_Rate_vs_HRV_px",
            name="plot_Heart_Rate_vs_HRV_px",
            namespace='plotting2'
        )
    ])

    plotting3 = pipeline([
        node(
            plot_confusion_matrix_xgb,
            inputs="y_test_preds",
            outputs="plot_confusion_matrix_xgb_px",
            name="plot_confusion_matrix_xgb_px",
            namespace='plotting_evaluaton'
        ),
        node(
            plot_confusion_matrix_lgbm,
            inputs="y_test_preds",
            outputs="plot_confusion_matrix_lgbm_px",
            name="plot_confusion_matrix_lgbm_px",
            namespace='plotting_evaluaton'
        ),
        node(
            plot_confusion_matrix_lr,
            inputs="y_test_preds",
            outputs="plot_confusion_matrix_lr_px",
            name="plot_confusion_matrix_lr_px",
            namespace='plotting_evaluaton'
        ),
        node(
            plot_confusion_matrix_ada,
            inputs="y_test_preds",
            outputs="plot_confusion_matrix_ada_px",
            name="plot_confusion_matrix_ada_px",
            namespace='plotting_evaluaton'
        ),
        node(
            plot_roc_auc_xgb,
            inputs="y_test_preds",
            outputs="plot_roc_auc_xgb_px",
            name="plot_roc_auc_xgb_px",
            namespace='plotting_evaluaton'
        ),
        node(
            plot_roc_auc_lgbm,
            inputs="y_test_preds",
            outputs="plot_roc_auc_lgbm_px",
            name="plot_roc_auc_lgbm_px",
            namespace='plotting_evaluaton'
        ),
        node(
            plot_roc_auc_lr,
            inputs="y_test_preds",
            outputs="plot_roc_auc_lr_px",
            name="plot_roc_auc_lr_px",
            namespace='plotting_evaluaton'
        ),
        node(
            plot_roc_auc_ada,
            inputs="y_test_preds",
            outputs="plot_roc_auc_ada_px",
            name="plot_roc_auc_ada_px",
            namespace='plotting_evaluaton'
        ),
        node(
            plot_feature_importance_xgb_w,
            inputs=['train_model','XGB_optuna_params_'],
            outputs="plot_fi_xgb_w_px",
            name='plot_fi_xgb_w_px',
            namespace='plotting_evaluaton'
        ),
        node(
            plot_feature_importance_xgb_g,
            inputs=['train_model','XGB_optuna_params_'],
            outputs="plot_fi_xgb_g_px",
            name='plot_fi_xgb_g_px',
            namespace='plotting_evaluaton'
        )

    ])

    return pipeline([plotting1, plotting2, plotting3])
