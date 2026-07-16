from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
