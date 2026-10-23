from pathlib import Path
import csv

import numpy as np

import torch
import torch.nn as nn


STATIC_DATA_URL = "https://arawatabill.org/comp364/"

def configure_torch_determinism(enabled: bool) -> None:
    """Configure PyTorch deterministic behavior.

    When enabled, PyTorch is asked to use deterministic kernels where
    available. This improves reproducibility, especially on GPU.
    """
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled


def seed_torch(seed: int) -> None:
    """Set PyTorch global RNG state using a fixed seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configure_torch_determinism(enabled=True)


def load_simulated_data(
    csv_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load weather-style simulated regression data from CSV into NumPy arrays.

    Expected CSV columns include:
    - ``latitude`` as regression target
    - ``tmax_001``..``tmax_365`` as 365 feature columns

    Returns:
        X: ``(N, 365)`` feature matrix
        y: ``(N,)`` target vector
    """
    resolved_path = csv_path if csv_path is not None else Path("sim_data") / "simulated_data.csv"

    targets: list[float] = []
    instances: list[list[float]] = []
    tmax_cols = [f"tmax_{day:03d}" for day in range(1, 366)]

    with resolved_path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.reader(file_obj)
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV is empty")

        required_cols = ["latitude", "station_id"] + tmax_cols
        missing_cols = [col for col in required_cols if col not in header]
        if missing_cols:
            raise ValueError(
                "CSV missing required columns: "
                + ", ".join(missing_cols[:5])
                + (" ..." if len(missing_cols) > 5 else "")
            )

        target_idx = header.index("latitude")
        tmax_indices = [header.index(col) for col in tmax_cols]

        for row in reader:
            if not row:
                continue
            targets.append(float(row[target_idx]))
            instances.append([float(row[idx]) for idx in tmax_indices])

    if not targets:
        raise ValueError("CSV does not contain any data rows")

    X = np.asarray(instances, dtype=float)
    y = np.asarray(targets, dtype=float)

    return X, y


def maybe_download_static_file(
    target_file: Path,
    static_data_url: str = STATIC_DATA_URL,
    force_download: bool = False,
) -> None:
    """Download static data files if not already present.

    Args:
        target_file: The file to save the downloaded content to.
        static_data_url: Base URL for static data files.
        force_download: If True, re-download even if files exist.
    """
    import requests

    target_file.parent.mkdir(parents=True, exist_ok=True)

    if target_file.exists() and not force_download:
        return

    url = static_data_url + target_file.name
    print(f"Downloading {url} to {target_file}...")
    response = requests.get(url)
    response.raise_for_status()
    with target_file.open("wb") as f:
        f.write(response.content)

def load_station_splits(
    csv_path: Path | None = None,
) -> dict[str, str]:
    """Load station-level split assignments from CSV.

    Expected CSV columns:
    - ``station_id``
    - ``split`` (allowed: ``train``, ``validation``)
    """
    resolved_path = (
        csv_path if csv_path is not None else Path("sim_data") / "simulated_station_splits.csv"
    )

    assignments: dict[str, str] = {}
    with resolved_path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.reader(file_obj)
        header = next(reader, None)
        if header is None:
            raise ValueError("Split CSV is empty")

        required_cols = ["station_id", "split"]
        missing_cols = [col for col in required_cols if col not in header]
        if missing_cols:
            raise ValueError(
                "Split CSV missing required columns: " + ", ".join(missing_cols)
            )

        station_id_idx = header.index("station_id")
        split_idx = header.index("split")

        for row in reader:
            if not row:
                continue

            station_id = row[station_id_idx].strip()
            split = row[split_idx].strip()
            if split not in {"train", "validation"}:
                raise ValueError(
                    f"Unsupported split label '{split}' for station '{station_id}'. "
                    "Allowed labels are: train, validation."
                )

            previous = assignments.get(station_id)
            if previous is not None and previous != split:
                raise ValueError(
                    f"Station '{station_id}' appears with multiple split labels: "
                    f"'{previous}' and '{split}'."
                )
            assignments[station_id] = split

    if not assignments:
        raise ValueError("Split CSV does not contain any data rows")

    return assignments


def load_data_by_station_split(
    data_csv_path: Path | None = None,
    split_csv_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load weather-style data and partition by station split file.

    Returns:
        X_train, y_train, X_validation, y_validation
    """
    data_path = data_csv_path if data_csv_path is not None else Path("sim_data") / "simulated_data.csv"
    split_assignments = load_station_splits(split_csv_path)
    tmax_cols = [f"tmax_{day:03d}" for day in range(1, 366)]

    train_instances: list[list[float]] = []
    train_targets: list[float] = []
    validation_instances: list[list[float]] = []
    validation_targets: list[float] = []

    seen_station_ids: set[str] = set()

    with data_path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.reader(file_obj)
        header = next(reader, None)
        if header is None:
            raise ValueError("Data CSV is empty")

        required_cols = ["station_id", "latitude"] + tmax_cols
        missing_cols = [col for col in required_cols if col not in header]
        if missing_cols:
            raise ValueError(
                "Data CSV missing required columns: "
                + ", ".join(missing_cols[:5])
                + (" ..." if len(missing_cols) > 5 else "")
            )

        station_id_idx = header.index("station_id")
        target_idx = header.index("latitude")
        tmax_indices = [header.index(col) for col in tmax_cols]

        for row in reader:
            if not row:
                continue

            station_id = row[station_id_idx].strip()
            split = split_assignments.get(station_id)
            if split is None:
                raise ValueError(
                    f"Station '{station_id}' in data CSV has no split assignment."
                )

            target = float(row[target_idx])
            features = [float(row[idx]) for idx in tmax_indices]
            seen_station_ids.add(station_id)

            if split == "train":
                train_targets.append(target)
                train_instances.append(features)
            else:
                validation_targets.append(target)
                validation_instances.append(features)

    if not train_targets:
        raise ValueError("No training rows were found after applying split assignments")
    if not validation_targets:
        raise ValueError("No validation rows were found after applying split assignments")

    missing_station_ids = sorted(set(split_assignments.keys()) - seen_station_ids)
    if missing_station_ids:
        raise ValueError(
            "Split CSV contains station_id values not present in data CSV. "
            f"Examples: {', '.join(missing_station_ids[:5])}"
            + (" ..." if len(missing_station_ids) > 5 else "")
        )

    X_train = np.asarray(train_instances, dtype=float)
    y_train = np.asarray(train_targets, dtype=float)
    X_validation = np.asarray(validation_instances, dtype=float)
    y_validation = np.asarray(validation_targets, dtype=float)
    return X_train, y_train, X_validation, y_validation


def instances_to_tensors(instances) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a list of instances into feature and label tensors."""
    X = torch.tensor([[inst.v, inst.p, inst.e]
                     for inst in instances], dtype=torch.float32)
    y = torch.tensor([inst.calc_target_value()
                     for inst in instances], dtype=torch.float32).unsqueeze(1)
    return X, y


def print_dataset_info(X, y) -> None:
    """Print lightweight dataset diagnostics for interactive runs."""
    print(f"Dataset: {X.shape[0]} instances, {X.shape[1]} features")
    print(f"Target range: {y.min().item():.4f} to {y.max().item():.4f}")
    print(f"Target mean: {y.mean().item():.4f}, std: {y.std().item():.4f}")


def print_parameter_summary(model: nn.Module) -> None:
    """Print a summary of model parameters, including trainable counts."""
    total_params = 0
    trainable_params = 0

    print("Parameter summary")
    print("-" * 80)

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params

        print(
            f"{name:40s} "
            f"shape={tuple(param.shape)!s:20s} "
            f"params={num_params:8d} "
            f"trainable={param.requires_grad}"
        )

    print("-" * 80)
    print(f"Total parameters:     {total_params}")
    print(f"Trainable parameters: {trainable_params}")

def generate_diagnostic_report(y_test: torch.Tensor, predictions: torch.Tensor) -> None:
    """Generate regression diagnostics comparing true vs. predicted values."""
    y_true = y_test.detach().reshape(-1).to(torch.float32)
    y_pred = predictions.detach().reshape(-1).to(torch.float32)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Shape mismatch: y_test has {y_true.shape[0]} items, predictions has {y_pred.shape[0]} items."
        )

    n = y_true.shape[0]
    if n == 0:
        print("\nValidation diagnostics (regression): no samples available.")
        return

    residuals = y_pred - y_true
    abs_errors = residuals.abs()
    squared_errors = residuals.pow(2)

    mae = abs_errors.mean().item()
    mse = squared_errors.mean().item()
    rmse = torch.sqrt(squared_errors.mean()).item()
    median_ae = torch.median(abs_errors).item()
    max_ae = abs_errors.max().item()
    bias = residuals.mean().item()

    # R^2 compares model MSE against predicting the mean target value.
    sst = torch.sum((y_true - y_true.mean()).pow(2)).item()
    sse = torch.sum(squared_errors).item()
    r2 = 1.0 - (sse / sst) if sst > 0 else float("nan")

    

    y_true_centered = y_true - y_true.mean()
    y_pred_centered = y_pred - y_pred.mean()
    corr_denom = torch.sqrt(torch.sum(y_true_centered.pow(2)) * torch.sum(y_pred_centered.pow(2))).item()
    pearson_r = (
        torch.sum(y_true_centered * y_pred_centered).item() / corr_denom
        if corr_denom > 0
        else float("nan")
    )

    error_percentiles = torch.quantile(
        abs_errors, torch.tensor([0.50, 0.90, 0.95, 0.99], dtype=abs_errors.dtype)
    )

    print("\nValidation diagnostics (regression):")
    print(f"Samples:        {n}")
    print(f"MAE:            {mae:.6f}")
    print(f"MSE:            {mse:.6f}")
    print(f"RMSE:           {rmse:.6f}")
    print(f"Median AE:      {median_ae:.6f}")
    print(f"Max AE:         {max_ae:.6f}")
    print(f"Bias (mean err): {bias:.6f}")
    print(f"R^2:            {r2:.6f}")
    print(f"Pearson r:      {pearson_r:.6f}")
    
    print("\nAbsolute error percentiles:")
    print(f"p50: {error_percentiles[0].item():.6f}")
    print(f"p90: {error_percentiles[1].item():.6f}")
    print(f"p95: {error_percentiles[2].item():.6f}")
    print(f"p99: {error_percentiles[3].item():.6f}")

    top_k = min(5, n)
    top_indices = torch.argsort(abs_errors, descending=True)[:top_k]
    print("\nWorst predictions (by absolute error):")
    print(f"{'Idx':>6} {'y_true':>12} {'y_pred':>12} {'abs_err':>12} {'residual':>12}")
    for idx in top_indices.tolist():
        print(
            f"{idx:>6} {y_true[idx].item():>12.6f} {y_pred[idx].item():>12.6f} "
            f"{abs_errors[idx].item():>12.6f} {residuals[idx].item():>12.6f}"
        )


def compute_regression_metrics(y_test: torch.Tensor, predictions: torch.Tensor) -> dict[str, float]:
    """Compute core regression metrics for logging/saving."""
    y_true = y_test.detach().reshape(-1).to(torch.float32)
    y_pred = predictions.detach().reshape(-1).to(torch.float32)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Shape mismatch: y_test has {y_true.shape[0]} items, predictions has {y_pred.shape[0]} items."
        )
    if y_true.shape[0] == 0:
        raise ValueError("Cannot compute metrics for empty inputs")

    residuals = y_pred - y_true
    abs_errors = residuals.abs()
    squared_errors = residuals.pow(2)

    mae = abs_errors.mean().item()
    mse = squared_errors.mean().item()
    rmse = torch.sqrt(squared_errors.mean()).item()

    sst = torch.sum((y_true - y_true.mean()).pow(2)).item()
    sse = torch.sum(squared_errors).item()
    r2 = 1.0 - (sse / sst) if sst > 0 else float("nan")

    y_true_centered = y_true - y_true.mean()
    y_pred_centered = y_pred - y_pred.mean()
    corr_denom = torch.sqrt(
        torch.sum(y_true_centered.pow(2)) * torch.sum(y_pred_centered.pow(2))
    ).item()
    pearson_r = (
        torch.sum(y_true_centered * y_pred_centered).item() / corr_denom
        if corr_denom > 0
        else float("nan")
    )

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": pearson_r,
    }


def has_overfitting_signal(
    train_history: list[float],
    val_history: list[float],
    streak_len: int = 3,
) -> bool:
    """Detect overfitting via consecutive train-down / val-up epochs."""
    if len(train_history) != len(val_history) or len(train_history) < streak_len + 1:
        return False

    streak = 0
    for i in range(1, len(train_history)):
        train_down = train_history[i] < train_history[i - 1]
        val_up = val_history[i] > val_history[i - 1]

        if train_down and val_up:
            streak += 1
            if streak >= streak_len:
                return True
        else:
            streak = 0

    return False


def plot_training_comparison(
    training_loss_history: list[float],
    validation_loss_history: list[float] | None = None,
    output_path: Path | None = None,
    show_plot: bool = True,
) -> None:
    """Plot training and optional validation loss curves."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    epoch_history = list(range(1, len(training_loss_history) + 1))
    plt.plot(epoch_history, training_loss_history, linewidth=2, label="Training")

    if validation_loss_history is not None:
        if len(validation_loss_history) != len(training_loss_history):
            raise ValueError(
                "validation_loss_history must match training_loss_history length"
            )
        plt.plot(epoch_history, validation_loss_history, linewidth=2, label="Validation")

    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close()
