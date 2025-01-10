"""
This is a boilerplate pipeline 'feature_selection'
generated using Kedro 0.19.10
"""
import logging
import pandas as pd
from scipy.stats import f_oneway, kruskal
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

logger = logging.getLogger(__name__)

def get_feat_rank_stat(train, params):
    """
    Calculate feature ranking statistics using ANOVA and Kruskal-Wallis tests.
    """

    p_an_tests = []
    p_kru_tests = []
    target_col = params['target']
    columns = [x for x in train.columns if x != target_col]

    for col in columns:
        unique_classes = train[target_col].unique()
        group_data = [train[train[target_col] == cls][col] for cls in unique_classes]
        p_kru = kruskal(*group_data).pvalue
        p_anova = f_oneway(*group_data).pvalue
        p_kru_tests.append(float(p_kru))
        p_an_tests.append(float(p_anova))

    return pd.DataFrame({
        "variable": columns,
        "p_kruskal": p_kru_tests,
        "p_anova": p_an_tests,
        'kruskal_rank': pd.Series(p_kru_tests).rank().astype(int).to_list(),
        "anova_rank":  pd.Series(p_an_tests).rank().astype(int).to_list(),
    }).sort_values('kruskal_rank')

def get_feat_rank_corr(train, params):
    """
    Calculate the correlation of each feature with the target variable and rank them.
    """

    target_col = params['target']
    train[target_col] = train[target_col].cat.codes
    columns = [x for x in train.columns if x != target_col]

    corr = train[columns].corrwith(train[target_col])

    return pd.DataFrame({
        "variable": columns,
        "corr": corr,
        "corr_rank": corr.abs().rank(ascending=False).astype(int).to_list()
    }).sort_values('corr_rank')

def get_feat_backward(train, params):
    """
    Perform backward feature selection using multiple classifiers.
    """

    target_col = params['target']
    columns = [x for x in train.columns if x != target_col]

    results = {}
    models = {
        'KNN':KNeighborsClassifier(),
        'GBM':GradientBoostingClassifier(),
        'LogReg':LogisticRegression()
    }
    if n_vars =='all':
        n_vars = len(5)
    else:
        n_vars = params['n_vars']

    for name, model in models.items():
        sfs = SequentialFeatureSelector(model,
                                        n_features_to_select=n_vars,
                                        direction='backward')
        sfs.fit(train[columns], train[target_col])
        results[name] = sfs.get_feature_names_out()
    return pd.DataFrame(results)

def get_feat_forward(train, params):
    """
    Perform forward feature selection using multiple classifiers.
    """

    target_col = params['target']
    columns = [x for x in train.columns if x != target_col]

    if n_vars =='all':
        n_vars = len(5)
    else:
        n_vars = params['n_vars']

    results = {}
    models = {
        'KNN':KNeighborsClassifier(),
        'GBM':GradientBoostingClassifier(),
        'LogReg':LogisticRegression()
    }

    for name, model in models.items():
        sfs = SequentialFeatureSelector(model,
                                        n_features_to_select=n_vars,
                                        direction='forward')
        sfs.fit(train[columns], train[target_col])
        results[name] = sfs.get_feature_names_out()
    return pd.DataFrame(results)

def _get_selected_features(stat_res,corr_res,forward_res,backward_res,params):
    """
    Select the features based on the feature selection method.
    """

    method = params['method']
    n_vars = params['n_vars']
    kwarg = params['kwarg']
    _vars = None

    if method == 'rank_stat':
        _vars =  stat_res.sort_values(f'{kwarg}_rank')['variable'].values[:n_vars].tolist()
    elif method == 'rank_corr':
        _vars = corr_res.sort_values('corr_rank')['variable'].values[:n_vars].tolist()
    elif method == 'forward':
        _vars = forward_res[kwarg].values.tolist()
    elif method == 'backward':
        _vars = backward_res[kwarg].values.tolist()

    if not _vars:
        logger.error("Invalid method name")
    else:
        logger.info("Selecting %d features based on %s%s method", n_vars, method, kwarg)
        return _vars

def select_features(train, tests, stat_res, corr_res, forward_res, backward_res, params):
    """
    Select the features in the train dataframe.
    """
    n_vars = params['n_vars']
    if n_vars =='all':
        selected_features = train.columns
        criteria = 'all'
    else:
        selected_features = _get_selected_features(stat_res,corr_res,
                                                forward_res,
                                                backward_res,
                                                params)
        criteria = {
        'method':params['method'],
        'n_vars':params['n_vars'],
        'kwarg':params['kwarg']
        }
    metrics = {'columns':selected_features, 'criteria':criteria}

    return train[selected_features], tests[selected_features], metrics
