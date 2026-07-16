import random
from typing import TypedDict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10
MIN_DELTA = 1e-4


class TrainingEpoch(TypedDict):
    """Metrics recorded after one MLP training epoch."""

    epoch: int
    train_loss: float
    validation_loss: float
    validation_f1_macro: float
    validation_accuracy: float


class CancerMLP(nn.Module):
    """Small feed-forward classifier for selected RNA-Seq features."""

    def __init__(self, input_features: int = 200, class_count: int = 5) -> None:
        super().__init__()
        self.best_epoch: int | None = None
        self.training_history: list[TrainingEpoch] = []
        self.network = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, class_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def set_reproducible_seed(seed: int) -> None:
    """Set deterministic CPU seeds for one MLP training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_mlp(
    features_train: np.ndarray,
    target_train: np.ndarray,
    features_validation: np.ndarray,
    target_validation: np.ndarray,
    *,
    class_count: int,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    min_delta: float = MIN_DELTA,
    seed: int = 42,
) -> tuple[CancerMLP, int]:
    """Train on inner-train data and stop from inner-validation loss."""
    set_reproducible_seed(seed)
    device = torch.device("cpu")
    model = CancerMLP(features_train.shape[1], class_count).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(
        torch.as_tensor(features_train, dtype=torch.float32),
        torch.tensor(np.array(target_train, dtype=np.int64, copy=True)),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_features = torch.as_tensor(features_validation, dtype=torch.float32)
    validation_target = torch.tensor(np.array(target_validation, dtype=np.int64, copy=True))

    best_validation_loss = float("inf")
    best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    epochs_without_improvement = 0
    epochs_run = 0
    best_epoch = 1

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_train_loss = 0.0
        for batch_features, batch_target in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_features), batch_target)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item()) * len(batch_target)

        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_features)
            validation_loss = float(loss_function(validation_logits, validation_target).item())
            validation_predictions = validation_logits.argmax(dim=1).cpu().numpy()
        model.training_history.append(
            {
                "epoch": epoch,
                "train_loss": total_train_loss / len(train_dataset),
                "validation_loss": validation_loss,
                "validation_f1_macro": float(
                    f1_score(target_validation, validation_predictions, average="macro")
                ),
                "validation_accuracy": float(
                    accuracy_score(target_validation, validation_predictions)
                ),
            }
        )
        epochs_run = epoch

        if validation_loss < best_validation_loss - min_delta:
            best_validation_loss = validation_loss
            best_state = {
                name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            }
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.best_epoch = best_epoch
    model.eval()
    return model, epochs_run


def train_mlp_fixed_epochs(
    features_train: np.ndarray,
    target_train: np.ndarray,
    *,
    class_count: int,
    epochs: int,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    seed: int = 42,
) -> CancerMLP:
    """Refit an MLP on all outer-training rows for a preselected epoch count."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    set_reproducible_seed(seed)
    model = CancerMLP(features_train.shape[1], class_count)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = TensorDataset(
        torch.as_tensor(features_train, dtype=torch.float32),
        torch.tensor(np.array(target_train, dtype=np.int64, copy=True)),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    for _ in range(epochs):
        model.train()
        for batch_features, batch_target in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_features), batch_target)
            loss.backward()
            optimizer.step()

    model.best_epoch = epochs
    model.eval()
    return model


def predict_mlp(model: CancerMLP, features: np.ndarray) -> np.ndarray:
    """Predict integer class indices on CPU."""
    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return model(feature_tensor).argmax(dim=1).cpu().numpy()
