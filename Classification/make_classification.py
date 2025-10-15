import numpy as np
import scipy.stats as stats

import pandas as pd
import random
import time
from typing import Literal

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, LeaveOneOut, GroupKFold, StratifiedKFold, KFold
from sklearn.svm import LinearSVC, SVR, SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.decomposition import PCA

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler

from sklearn.utils import shuffle, resample


def avg_res(res: pd.DataFrame) -> pd.DataFrame:
    return res.groupby(["classifier"]).mean()[["f1-score", "accuracy", "time"]]


def get_model_name(model: BaseEstimator) -> str:
    base_name = model.__class__.__name__
    if base_name == "MLPClassifier":
        hlayers = ",".join(str(elem) for elem in model.hidden_layer_sizes)
        modelname = base_name + (f"_HLs({hlayers})" if hlayers else "")
    else:
        modelname = base_name
    return modelname


def make_pipeline(
    model: BaseEstimator,
    feature_selector: Literal["L1", "RFE", "PCA"] | None = None,
    imputer: TransformerMixin | None = None,
    scaler: TransformerMixin | None = None,
) -> Pipeline:
    if feature_selector == "L1":
        clf = Pipeline(
            [
                ("impute", imputer),
                ("scale", scaler),
                (
                    "feature_selection",
                    SelectFromModel(LogisticRegression(max_iter=5000, C=0.1, penalty="l1", dual=False, solver="saga")),
                ),
                ("classification", model),
            ]
        )
    elif feature_selector == "RFE":
        clf = Pipeline(
            [
                ("impute", imputer),
                ("scale", scaler),
                ("feature_selection", RFECV(RandomForestClassifier(max_depth=5), step=2, cv=2)),
                ("classification", model),
            ]
        )
    elif feature_selector == "PCA":
        clf = Pipeline(
            [
                ("impute", imputer),
                ("scale", scaler),
                ("feature_selection", PCA(n_components=0.95, svd_solver="full")),
                ("classification", model),
            ]
        )
    else:
        clf = Pipeline(
            [
                ("impute", imputer),
                ("scale", scaler),
                ("classification", model),
            ]
        )
    return clf


##########################################################################################


def make_nclassif_random_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 10,
    feature_selector: Literal["L1", "RFE", "PCA"] | None = None,
    list_classifiers: list[BaseEstimator] | None = None,
    impute: bool = True,
    scale: bool = True,
    verbose: bool = False,
    random_seed: int | None = None,
    test_size: float = 0.2,
    useStratification: bool = False,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Pipeline], dict[str, pd.DataFrame]]:
    """
    Perform repeated random train/test splits to evaluate multiple classifiers on a dataset.

    This function splits the data into training and testing sets `n_splits` times using sklearn's
    `train_test_split`, trains each classifier in `list_classifiers` on each split, and collects
    performance metrics for each run. Optionally, feature selection or dimensionality reduction
    can be applied, as well as imputation and scaling.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Ground truth labels.
    n_splits : int, default=10
        Number of random train/test splits to perform.
    feature_selector : str or None, default=None
        Feature selection method to use. Options: 'L1', 'RFE', 'PCA', or None.
    list_classifiers : list, default=None
        List of sklearn classifier instances to evaluate. If None, uses default classifiers.
    impute : bool, default=True
        Whether to apply iterative imputation to missing values.
    scale : bool, default=True
        Whether to apply standard scaling to features.
    verbose : bool, default=True
        Whether to print progress for each split.
    random_seed : int or None, default=None
        Base random seed for reproducibility. Each split increments this seed.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    useStratification : bool, default=False
        Whether to stratify splits by label distribution.

    Returns
    -------
    df_res : pd.DataFrame
        DataFrame containing f1-score, balanced accuracy, classifier name, and timing for each split.
    conf_matrices : list of np.ndarray
        List of confusion matrices for each classifier and split.
    pipelines : dict[str, Pipeline]
        Dictionary mapping classifier names to their fitted sklearn Pipelines from the last split.

    """

    # Dictionnary to store f1-score and accuracy
    df_res = pd.DataFrame({"n": [], "f1-score": [], "accuracy": [], "classifier": [], "time": []})
    conf_matrices: dict[str, np.ndarray] = {}
    pipelines: dict[str, Pipeline] = {}
    probs_dict: dict[str, list[pd.DataFrame]] = {get_model_name(model): [] for model in list_classifiers}

    imputer = IterativeImputer() if impute else None
    scaler = StandardScaler() if scale else None

    # Defaut classifiers tested: Logistic regression, Random Forests, Adaboost
    if not list_classifiers:
        list_classifiers = [
            LogisticRegression(max_iter=2000),
            RandomForestClassifier(max_depth=5),
            AdaBoostClassifier(n_estimators=100),
        ]

    # Make n random splits
    if random_seed:
        random.seed(random_seed)
    for s in range(n_splits):
        stratify = y if useStratification else None
        current_seed = random.randint(0, 1000000) if random_seed else None
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=current_seed, stratify=stratify
        )

        print("Split {0:2d}/{1:2d}".format(s + 1, n_splits))

        # Fit each model
        for model in list_classifiers:
            model_name = get_model_name(model)
            if verbose:
                print(f"Training: {model_name}")
            clf = make_pipeline(model, feature_selector, imputer, scaler)

            tic = time.perf_counter()
            clf.fit(x_train, y_train)
            toc = time.perf_counter()

            # Retrieve accuracy and F1-score
            y_pred = clf.predict(x_test)
            cm = confusion_matrix(y_test, y_pred)
            conf_matrices[model_name] = cm

            probs = clf.predict_proba(x_test)
            _, n_classes = probs.shape
            probs_cols = {}
            for cls_idx in range(n_classes):
                probs_cols[cls_idx] = probs[:, cls_idx]
            fold_pd = pd.DataFrame(probs_cols, index=y_test.index)
            probs_dict[model_name].append(fold_pd)

            new_row = {
                "n": int(s),
                "f1-score": f1_score(y_test, y_pred, average="weighted"),
                "accuracy": balanced_accuracy_score(y_test, y_pred),
                "classifier": model_name,
                "time": toc - tic,
            }
            df_res.loc[len(df_res)] = new_row

    # Training final models
    for model in list_classifiers:
        model_name = get_model_name(model)
        # Concatenate all fold results
        probs_dict[model_name] = pd.concat(probs_dict[model_name])
        clf = make_pipeline(model, feature_selector, imputer, scaler)
        clf.fit(X, y)
        pipelines[model_name] = clf

    return df_res, conf_matrices, pipelines, probs_dict


##########################################################################################


def make_nclassif_kfold(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    feature_selector: Literal["L1", "RFE", "PCA"] | None = None,
    list_classifiers: list[BaseEstimator] | None = None,
    impute=True,
    scale=True,
    verbose=False,
    stratified=True,
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Pipeline], dict[str, pd.DataFrame]]:
    """
    Evaluate classifiers using K-fold cross-validation.
    """
    df_res = pd.DataFrame({"n": [], "f1-score": [], "accuracy": [], "classifier": [], "time": []})
    conf_matrices: dict[str, np.ndarray] = {}
    pipelines: dict[str, Pipeline] = {}
    probs_dict: dict[str, list[pd.DataFrame]] = {get_model_name(model): [] for model in list_classifiers}

    imputer = IterativeImputer() if impute else None
    scaler = StandardScaler() if scale else None

    if not list_classifiers:
        list_classifiers = [
            LogisticRegression(max_iter=2000),
            RandomForestClassifier(max_depth=5),
            AdaBoostClassifier(n_estimators=100),
        ]

    if stratified:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    for s, (train_index, test_index) in enumerate(kf.split(X, y)):
        x_train = X.iloc[train_index]
        x_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        print(f"Fold {s + 1}/{n_splits}")

        for model in list_classifiers:
            model_name = get_model_name(model)
            if verbose:
                print(f"Training: {model_name}")
            clf = make_pipeline(model, feature_selector, imputer, scaler)

            tic = time.perf_counter()
            clf.fit(x_train, y_train)
            toc = time.perf_counter()

            y_pred = clf.predict(x_test)
            cm = confusion_matrix(y_test, y_pred)
            conf_matrices[model_name] = cm

            probs = clf.predict_proba(x_test)
            _, n_classes = probs.shape
            probs_cols = {}
            for cls_idx in range(n_classes):
                probs_cols[cls_idx] = probs[:, cls_idx]
            fold_pd = pd.DataFrame(probs_cols, index=y_test.index)
            probs_dict[model_name].append(fold_pd)

            new_row = {
                "n": int(s),
                "f1-score": f1_score(y_test, y_pred, average="weighted"),
                "accuracy": balanced_accuracy_score(y_test, y_pred),
                "classifier": model_name,
                "time": toc - tic,
            }
            df_res.loc[len(df_res)] = new_row

    # Training final models
    for model in list_classifiers:
        model_name = get_model_name(model)
        # Concatenate all fold results
        probs_dict[model_name] = pd.concat(probs_dict[model_name])
        clf = make_pipeline(model, feature_selector, imputer, scaler)
        clf.fit(X, y)
        pipelines[model_name] = clf

    return df_res, conf_matrices, pipelines, probs_dict
