"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.19.10
"""

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import optuna
import pandas as pd

def _xgb_param_space(trial):
    trial.set_user_attr('objective', 'multi:softprob')
    trial.set_user_attr('num_class', 4)
    return {
        'objective': trial.user_attrs['objective'],
        'num_class': trial.user_attrs['num_class'],
        'max_depth': trial.suggest_int("max_depth", 2, 10),
        'min_child_weight': trial.suggest_int("min_child_weight", 0, 10),
        'gamma': trial.suggest_int("gamma", 0, 10),
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.5)
    }

def _lgbm_param_space(trial):
    trial.set_user_attr('objective', 'multiclass')
    trial.set_user_attr('num_class', 4)
    return {
        'objective': trial.user_attrs['objective'],
        'num_class': trial.user_attrs['num_class'],
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.5),
        'max_depth': trial.suggest_int("max_depth", 0, 10),
        'min_gain_to_split': trial.suggest_float("min_gain_to_split", 0.05, 0.4),
    }

def _logistic_param_space(trial):
    trial.set_user_attr('solver', 'saga')
    return {
        'solver': trial.user_attrs['solver'],
        'C': trial.suggest_float("C", 0.01, 10),
        'penalty': trial.suggest_categorical("penalty", ['l1', 'l2'])
    }

def _ada_param_space(trial):
    # trial.set_user_attr('estimator', DecisionTreeClassifier(max_depth=1))
    return {
        # 'estimator': trial.user_attrs['estimator'],
        'n_estimators': trial.suggest_int("n_estimators", 10, 500),
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 10, log=True)
    }

def get_initial_score(train, params):
    target_column = params['target']
    scoring_method = params['scoring']
    result = {}
    models = {
        'xgb': XGBClassifier(objective='multi:softprob', num_class=4),
        'lr': LogisticRegression(),
        'lgbm': LGBMClassifier(objective='multiclass', num_class=4),
        'ada': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))
    }
    x_train = train.drop(columns=[target_column])
    y_train = train[target_column].cat.codes
    for name, model in models.items():
        res = cross_val_score(model, x_train, y_train,
                        scoring=scoring_method, cv=4)
        result[name] = res.mean()
    return result

def _get_optuna_params(train, model_class, param_space, params):
    """
    Function to optimize hyperparameters for any classification model using Optuna.
    """
    target_column = params['target']
    scoring_method = params['scoring']
    n_trials = params['n_trials']
    if model_class == AdaBoostClassifier:
        params['estimator']=DecisionTreeClassifier(max_depth=1)
    def objective(trial):
        # Generate the hyperparameters using the param_space function
        model_params = param_space(trial)

        # Prepare training data
        x_train = train.drop(columns=[target_column])
        y_train = train[target_column].cat.codes

        # Create the model with the trial's parameters
        model = model_class(**model_params)

        # Perform cross-validation
        score = cross_val_score(model, x_train, y_train,
                                scoring=scoring_method,
                                cv=5).mean()
        return score

    # Optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_params.update(study.best_trial.user_attrs)
    return best_params

def optimize_lr_hyperparams(train, params):
    res = _get_optuna_params(train, LogisticRegression, _logistic_param_space, params)
    return res, res

def optimize_lgbm_hyperparams(train, params):
    res = _get_optuna_params(train, LGBMClassifier, _lgbm_param_space, params)
    return res, res

def optimize_xgb_hyperparams(train, params):
    res = _get_optuna_params(train, XGBClassifier, _xgb_param_space, params)
    return res, res

def optimize_ada_hyperparams(train, params):
    res = _get_optuna_params(train, AdaBoostClassifier, _ada_param_space, params)
    return res, res

def _train_model(train, model, params):
    target_column = params['target']
    x_train = train.drop(columns=[target_column])
    y_train = train[target_column].cat.codes
    model.fit(x_train, y_train)
    return model

def _evaluate_model(model, train, test, params):

    target_column = params['target']

    x_train = train.drop(columns=[target_column])
    y_train = train[target_column].cat.codes
    x_test = test.drop(columns=[target_column])
    y_test = test[target_column].cat.codes

    y_pred_train_proba = model.predict_proba(x_train)
    y_pred_test_proba = model.predict_proba(x_test)
    y_pred_train_label = model.predict(x_train)
    y_pred_test_label = model.predict(x_test)
    res_dict = {
        'train_roc_auc': roc_auc_score(y_train, y_pred_train_proba, multi_class='ovr'),
        'train_accuracy': accuracy_score(y_train, y_pred_train_label),
        'test_roc_auc': roc_auc_score(y_test, y_pred_test_proba, multi_class='ovr'),
        'test_accuracy': accuracy_score(y_test, y_pred_test_label),
    }
    cat_mapping = dict(enumerate(train[target_column].cat.categories))
    y_pred_train_proba = pd.DataFrame(y_pred_train_proba,
                                    columns=[y+'_proba' for x,y in cat_mapping.items()])
    y_pred_train_proba['true_label'] = train[target_column]

    y_pred_test_proba = pd.DataFrame(y_pred_test_proba,
                                    columns=[y+'_proba' for x,y in cat_mapping.items()])
    y_pred_test_proba['true_label'] = test[target_column]

    return res_dict, y_pred_train_proba, y_pred_test_proba

def eval_best_models(train, test, 
                    LR_optuna_params, LGBM_optuna_params, 
                    XGB_optuna_params, ADA_optuna_params,
                    params):
    ADA_optuna_params['estimator']=DecisionTreeClassifier(max_depth=1)
    models = {
        'xgb': XGBClassifier(**XGB_optuna_params),
        'lr': LogisticRegression(**LR_optuna_params),
        'lgbm': LGBMClassifier(**LGBM_optuna_params),
        'ada': AdaBoostClassifier(**ADA_optuna_params)
    }

    results = {}
    y_train_preds = pd.DataFrame()
    y_test_preds = pd.DataFrame()
    for name, model in models.items():
        trained_model = _train_model(train, model, params)
        res,y_train_pred,y_test_pred = _evaluate_model(trained_model, train, test, params)
        y_train_pred['model'] = name
        y_test_pred['model'] = name
        y_train_preds = pd.concat([y_train_preds, y_train_pred], axis=0)
        y_test_preds = pd.concat([y_test_preds, y_test_pred], axis=0)
        results.update({f'{name}_{metric}': value for metric, value in res.items()})
    return results, y_train_preds, y_test_preds
