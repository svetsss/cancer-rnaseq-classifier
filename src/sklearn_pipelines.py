from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import RANDOM_STATE


def build_dummy_classifier(*, random_state: int = RANDOM_STATE) -> DummyClassifier:
    """Create the most-frequent-class baseline."""
    return DummyClassifier(strategy="most_frequent", random_state=random_state)


def build_logistic_pipeline(*, random_state: int = RANDOM_STATE) -> Pipeline:
    """Create the scaled Logistic Regression baseline using all features."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_logistic_feature_selection_pipeline(
    k: int,
    *,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Create the approved Logistic Regression pipeline with fold-local feature selection."""
    classifier = LogisticRegression(max_iter=3000, random_state=random_state)
    return _build_feature_selection_pipeline(k, classifier)


def build_linear_svc_feature_selection_pipeline(
    k: int,
    *,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Create the approved LinearSVC pipeline with fold-local feature selection."""
    classifier = LinearSVC(
        C=1.0,
        dual="auto",
        max_iter=5000,
        random_state=random_state,
    )
    return _build_feature_selection_pipeline(k, classifier)


def _build_feature_selection_pipeline(k: int, classifier: BaseEstimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("selector", SelectKBest(score_func=f_classif, k=k)),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def build_logistic_pca_pipeline(
    n_components: int,
    *,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Create the approved Logistic Regression pipeline with fold-local PCA."""
    classifier = LogisticRegression(max_iter=3000, random_state=random_state)
    return _build_pca_pipeline(n_components, classifier, random_state)


def build_linear_svc_pca_pipeline(
    n_components: int,
    *,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Create the approved LinearSVC pipeline with fold-local PCA."""
    classifier = LinearSVC(
        C=1.0,
        dual="auto",
        max_iter=5000,
        random_state=random_state,
    )
    return _build_pca_pipeline(n_components, classifier, random_state)


def _build_pca_pipeline(
    n_components: int,
    classifier: BaseEstimator,
    random_state: int,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=n_components,
                    svd_solver="randomized",
                    random_state=random_state,
                ),
            ),
            ("classifier", classifier),
        ]
    )
