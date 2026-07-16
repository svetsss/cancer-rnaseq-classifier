import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.evaluation import ExperimentResult, evaluate_classifier
from src.feature_selection import FEATURE_METHOD
from src.splitting import make_stratified_cv

RANDOM_FOREST_EXPERIMENT = ("E16", "RandomForestClassifier", 200)


def build_random_forest_pipeline(*, random_state: int = RANDOM_STATE) -> Pipeline:
    """Create the approved Random Forest pipeline with fold-local feature selection."""
    return Pipeline(
        steps=[
            ("selector", SelectKBest(score_func=f_classif, k=200)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def run_random_forest_experiment(
    features_train: pd.DataFrame,
    target_train: pd.Series,
) -> ExperimentResult:
    """Evaluate E16 on the same train-only cross-validation folds."""
    experiment_id, model_name, n_features = RANDOM_FOREST_EXPERIMENT
    return evaluate_classifier(
        build_random_forest_pipeline(),
        features_train,
        target_train,
        make_stratified_cv(),
        experiment_id=experiment_id,
        model_name=model_name,
        feature_method=FEATURE_METHOD,
        n_features=n_features,
    )
