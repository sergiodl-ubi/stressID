import numpy as np
import scipy.stats as stats

import pandas as pd
import random
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, LeaveOneOut, GroupKFold
from sklearn.svm import LinearSVC, SVR, SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.decomposition import PCA

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.utils import shuffle, resample
from imblearn.over_sampling import SMOTE

from sklearn2pmml import PMMLPipeline


# Compare several classification models over K repetition, using K group splits, grouped by subjects
def make_nclassif(
    X, y, n_splits=10, feature_selector=None, list_classifiers=None, impute=True, scale=True, verbose=True
):
    # Dictionnary to store f1-score and accuracy
    df_res = pd.DataFrame({"n": [], "f1-score": [], "accuracy": [], "classifier": [], "time": []})
    conf_matrices = []

    imputer = IterativeImputer(random_state=0) if impute else None
    scaler = StandardScaler() if scale else None

    # Defaut classifiers tested: Logistic regression, Random Forests, Adaboost
    if not list_classifiers:
        list_classifiers = [
            LogisticRegression(max_iter=2000),
            RandomForestClassifier(max_depth=5),
            AdaBoostClassifier(n_estimators=100),
        ]

    # Make n-group random splits grouped by subjects
    groups = [l.split("_")[0] for l in list(y.index)]

    # rstate = random.randint(0,100)

    X_shuffled, y_shuffled, groups_shuffled = shuffle(X, y, groups)

    group_kfold = GroupKFold(n_splits=n_splits)
    group_kfold.get_n_splits(X_shuffled, y_shuffled, groups_shuffled)

    # SPLITS
    for s, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups)):
        x_train = X.iloc[train_index]
        x_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        if verbose:
            print("Split {0:2d}/{1:2d}".format(s + 1, n_splits))

        # Fit each model
        for model in list_classifiers:
            if feature_selector == "L1":
                clf = Pipeline(
                    [
                        ("impute", imputer),
                        ("scale", scaler),
                        (
                            "feature_selection",
                            SelectFromModel(
                                LogisticRegression(max_iter=5000, C=0.1, penalty="l1", dual=False, solver="saga")
                            ),
                        ),
                        ("classification", model),
                    ]
                )
            elif feature_selector == "RFECV":
                clf = Pipeline(
                    [
                        ("impute", imputer),
                        ("scale", scaler),
                        ("feature_selection", RFECV(RandomForestClassifier(max_depth=5), step=2, cv=2)),
                        ("classification", model),
                    ]
                )
            elif feature_selector == "RFE":
                clf = Pipeline(
                    [
                        ("impute", imputer),
                        ("scale", scaler),
                        (
                            "feature_selection",
                            RFE(RandomForestClassifier(max_depth=5), n_features_to_select=20, step=2),
                        ),
                        ("classification", model),
                    ]
                )
            else:
                clf = Pipeline([("impute", imputer), ("scale", scaler), ("classification", model)])

            tic = time.perf_counter()
            clf.fit(x_train, y_train)
            toc = time.perf_counter()

            # Retrieve accuracy and F1-score
            y_pred = clf.predict(x_test)
            conf_matrices.append(confusion_matrix(y_test, y_pred))

            new_row = {
                "n": int(s),
                "f1-score": f1_score(y_test, y_pred, average="weighted"),
                "accuracy": balanced_accuracy_score(y_test, y_pred),
                "classifier": model.__class__.__name__,
                "time": toc - tic,
            }

            df_res.loc[len(df_res)] = new_row

    return df_res, conf_matrices


##########################################################################################


def avg_res(res):
    return res.groupby(["classifier"]).mean()[["f1-score", "accuracy", "time"]]


##########################################################################################


def make_nclassif_random_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 10,
    feature_selector: str | None = None,
    list_classifiers: list | None = None,
    impute: bool = True,
    scale: bool = True,
    verbose: bool = False,
    random_seed: int | None = None,
    test_size: float = 0.2,
    useStratification: bool = False,
) -> tuple[pd.DataFrame, list[np.ndarray], dict[str, PMMLPipeline]]:
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
    conf_matrices = []
    pipelines = {}

    imputer = SimpleImputer() if impute else None
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
            model_name = model.__class__.__name__
            if verbose:
                print(f"Training: {model_name}")
            if feature_selector == "L1":
                clf = PMMLPipeline(
                    [
                        ("impute", imputer),
                        ("scale", scaler),
                        (
                            "feature_selection",
                            SelectFromModel(
                                LogisticRegression(max_iter=5000, C=0.1, penalty="l1", dual=False, solver="saga")
                            ),
                        ),
                        ("classification", model),
                    ]
                )
            elif feature_selector == "RFE":
                clf = PMMLPipeline(
                    [
                        ("impute", imputer),
                        ("scale", scaler),
                        ("feature_selection", RFECV(RandomForestClassifier(max_depth=5), step=2, cv=2)),
                        ("classification", model),
                    ]
                )
            elif feature_selector == "PCA":
                pca = PCA(n_components=0.95, svd_solver="full")
                clf = PMMLPipeline(
                    [
                        ("impute", imputer),
                        ("scale", scaler),
                        ("pca", pca),
                        ("classification", model),
                    ]
                )
            else:
                clf = PMMLPipeline(
                    [
                        ("impute", imputer),
                        ("scale", scaler),
                        ("classification", model),
                    ]
                )

            tic = time.perf_counter()
            clf.fit(x_train, y_train)
            toc = time.perf_counter()

            # Retrieve accuracy and F1-score
            y_pred = clf.predict(x_test)
            conf_matrices.append(confusion_matrix(y_test, y_pred))
            if model_name == "MLPClassifier":
                suffix = (
                    f"_HLs{len(model.hidden_layer_sizes)}_FL{model.hidden_layer_sizes[0]}"
                    if len(model.hidden_layer_sizes) > 0
                    else ""
                )
                modelname = model_name + suffix
            else:
                modelname = model_name
            pipelines[modelname] = clf

            new_row = {
                "n": int(s),
                "f1-score": f1_score(y_test, y_pred, average="weighted"),
                "accuracy": balanced_accuracy_score(y_test, y_pred),
                "classifier": modelname,
                "time": toc - tic,
            }
            df_res.loc[len(df_res)] = new_row

    return df_res, conf_matrices, pipelines


##########################################################################################


def make_nclassif_random_splits_resample(
    X,
    y,
    n_splits=10,
    resamp="SMOTE",
    feature_selector=None,
    list_classifiers=None,
    impute=True,
    scale=True,
    verbose=True,
):
    # Dictionnary to store f1-score and accuracy
    df_res = pd.DataFrame({"n": [], "f1-score": [], "accuracy": [], "classifier": [], "time": []})

    conf_matrices = []
    y_preds = []
    y_tests = []

    if impute:
        imputer = IterativeImputer()
    else:
        imputer = None

    if scale:
        scaler = StandardScaler()
    else:
        scaler = None

    # Defaut classifiers tested: Logistic regression, Random Forests, Adaboost
    if not list_classifiers:
        list_classifiers = [
            LogisticRegression(max_iter=2000),
            RandomForestClassifier(max_depth=5),
            AdaBoostClassifier(n_estimators=100),
        ]

    # Make n random splits
    for s in range(n_splits):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        y_high = pd.Series(y_train[y_train == 1])
        idx_high = list(x_train.merge(y_high, left_index=True, right_index=True).index)
        x_high = x_train.loc[idx_high]

        y_low = pd.Series(y_train[y_train == 0])
        idx_low = list(x_train.merge(y_low, left_index=True, right_index=True).index)
        x_low = x_train.loc[idx_low]

        if resamp:
            oversample = SMOTE()
            x_train, y_train = oversample.fit_resample(x_train, y_train)

        if verbose:
            print("Split {0:2d}/{1:2d}".format(s + 1, n_splits))

        # Fit each model
        for i, model in enumerate(list_classifiers):
            if feature_selector == "L1":
                clf = Pipeline(
                    [
                        ("impute", imputer),
                        ("scale", scaler),
                        (
                            "feature_selection",
                            SelectFromModel(
                                LogisticRegression(max_iter=5000, C=0.1, penalty="l1", dual=False, solver="saga")
                            ),
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
                pca = PCA(n_components=0.95, svd_solver="full")

                clf = Pipeline([("impute", imputer), ("scale", scaler), ("pca", pca), ("classification", model)])
            else:
                clf = Pipeline([("impute", imputer), ("scale", scaler), ("classification", model)])

            tic = time.perf_counter()
            clf.fit(x_train, y_train)
            toc = time.perf_counter()

            # Retrieve accuracy and F1-score
            y_pred = clf.predict(x_test)
            conf_matrices.append(confusion_matrix(y_test, y_pred))

            y_tests.append(pd.Series(y_test))
            y_preds.append(pd.Series(y_pred, index=y_test.index))

            new_row = {
                "n": int(s),
                "f1-score": f1_score(y_test, y_pred, average="weighted"),
                "accuracy": balanced_accuracy_score(y_test, y_pred),
                "classifier": model.__class__.__name__,
                "time": toc - tic,
            }

            df_res.loc[len(df_res)] = new_row

    return df_res, conf_matrices, y_preds, y_tests
