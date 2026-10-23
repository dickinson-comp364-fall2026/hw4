"""
One-dimensional Convolutional Neural Network for weather station latitude prediction.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from config_utils import create_run_dir, load_config, save_config_copy, save_metrics_report
from one_d_conv_net_utils import (
    compute_regression_metrics,
    generate_diagnostic_report,
    has_overfitting_signal,
    load_data_by_station_split,
    maybe_download_static_file,
    print_parameter_summary,
    plot_training_comparison,
    print_dataset_info,
    seed_torch,
)

DEFAULT_TORCH_SEED = 54545  # Backward-compatible default reproducibility seed.
SHOW_PLOT = False  # Keep False for tests/headless runs.


class OneDConvnet(nn.Module):
    def __init__(
        self,
        input_len: int,
        num_features: int,
        kernel_size: int,
        stride: int,
        num_conv_layers: int,
        fc_hidden_dim: int,
    ):
        super().__init__()
        self.input_len = input_len
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_conv_layers = num_conv_layers

        # High-level plan:
        # Conv1d → ReLU
        # Conv1d → ReLU
        # ...
        # Conv1d → ReLU
        # Flatten → Linear → ReLU → Linear(…, 1)

        # Convolutional layers: in_channels=out_channels=num_features, except first layer has single feature (temperature)
        conv_layer_list = []
        self.relu = nn.ReLU()
        for conv_layer in range(num_conv_layers):
            # TODO: Set in_channels correctly for the first layer and subsequent layers.
            in_channels = -1
            # TODO: Insert appropriate layers into the sequential container to implement the high-level plan described above.
            this_layer = nn.Sequential(
                # 1. convolution
                # 2. ReLU activation
            )
            conv_layer_list.append(this_layer)
        self.conv_layers = nn.Sequential(*conv_layer_list)

        # TODO: In your own words, explain the purpose of the following block of code and how it works
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_len)
            dummy_output = self.conv_layers(dummy_input)
            self.conv_output_length = dummy_output.flatten(
                start_dim=1).shape[1]

        # In real-world code the following explicit calculation would be unnecessary, but we do it here to improve our understanding of the network structure.
        self.explicit_conv_out_len = self.calc_conv_out_len_explicit()
        assert self.explicit_conv_out_len == self.conv_output_length, (
            "Explicit conv output length does not match dummy-input output length: "
            f"explicit={self.explicit_conv_out_len}, dummy={self.conv_output_length}"
        )

        # Fully connected layer.
        self.fc_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.conv_output_length, fc_hidden_dim),
            nn.ReLU(),
        )

        # Final regression layer.
        self.regressor = nn.Linear(fc_hidden_dim, 1)

    def calc_conv_out_len_explicit(self) -> int:
        """Compute flattened conv output size from layer hyperparameters."""
        length = self.input_len
        for l in range(self.num_conv_layers):
            # TODO: In your own words, briefly explain the following calculation.
            length = (length - self.kernel_size) // self.stride + 1
            if length <= 0:
                raise ValueError(
                        "Invalid conv configuration produced non-positive sequence length "
                    f"after convolution: {length}"
                )
        return length * self.num_features

    def forward(self, x: torch.Tensor):
        # x shape is (N, 1, input_len), where N is batch size.
        assert x.ndim == 3 and x.shape[1] == 1 and x.shape[
            2] == self.input_len, f"Expected input shape (N, 1, {self.input_len}), got {x.shape}"
        # TODO: Uncomment and fill in the blanks for the following three lines. 
        # x = self.___________(x)
        # x = self.___________(x)
        # x = self.___________(x)
        return x.squeeze(-1)


def train_model(
    model: OneDConvnet,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
) -> tuple[list[float], list[float]]:
    """Train the model and return training and validation loss per epoch."""

    # shape of X_train: (num_instances, 1, INPUT_DIM)
    # shape of y_train: (num_instances,)

    criterion = nn.MSELoss()  # Using MSELoss for regression.

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Wrap tensors into a dataset
    dataset = TensorDataset(X_train, y_train)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True)

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        loss_total = 0.0
        num_instances = X_train.shape[0]
        for batch in loader:
            X_batch, y_batch = batch

            # Forward pass: compute predicted values. Under the covers,
            # Pytorch builds a computation graph, which will allow us to compute derivatives later.
            predictions = model(X_batch)

            # Calculate loss for this batch. PyTorch extends the computation graph,
            # which will allow computation of the derivatives of
            # the loss with respect to parameters when we do the backward pass later.
            loss = criterion(predictions, y_batch)

            current_batch_size = X_batch.shape[0]
            loss_total += loss.item() * current_batch_size
            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # average loss per instance
        epoch_train_loss = loss_total / num_instances
        train_loss_history.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            epoch_val_loss = criterion(val_predictions, y_val).item()
        val_loss_history.append(epoch_val_loss)

        print(
            f"  train_loss={epoch_train_loss:.6f}  "
            f"val_loss={epoch_val_loss:.6f}"
        )

    return train_loss_history, val_loss_history


def validate_model(
    model: OneDConvnet,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    show_diagnostics: bool = True,
) -> tuple[float, dict[str, float]]:
    """Evaluate on test data and return MSE plus a metric dictionary."""
    with torch.no_grad():
        predictions = model(X_test)  # shape (num_instances,)
        if show_diagnostics:
            generate_diagnostic_report(y_test, predictions)
        mse = nn.MSELoss()(predictions, y_test).item()
        metrics = compute_regression_metrics(y_test, predictions)
    return mse, metrics


def main(config_path: str | Path) -> Path:
    config = load_config(config_path)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config.get("output", {})
    seed = int(config.get("seed", DEFAULT_TORCH_SEED))

    seed_torch(seed)
    data_csv_path = data_config.get("data_csv_path")
    split_csv_path = data_config.get("split_csv_path")

    for csv_path in [data_csv_path, split_csv_path]:
        if csv_path is not None:
            maybe_download_static_file(Path(csv_path))

    X_train, y_train, X_validation, y_validation = load_data_by_station_split(
        Path(data_csv_path) if data_csv_path is not None else None,
        Path(split_csv_path) if split_csv_path is not None else None,
    )
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    print_dataset_info(X, y)

    input_len = int(data_config["input_len"])
    if X_train.shape[1] != input_len:
        raise ValueError(
            f"Config input_len={input_len} does not match dataset feature count={X_train.shape[1]}"
        )

    run_dir = create_run_dir(config)
    save_config_copy(config, run_dir)

    model = OneDConvnet(
        input_len=input_len,
        num_features=int(model_config["num_features"]),
        kernel_size=int(model_config["kernel_size"]),
        stride=int(model_config["stride"]),
        num_conv_layers=int(model_config["num_conv_layers"]),
        fc_hidden_dim=int(model_config["fc_hidden_dim"]),
    )
    print(model)
    print_parameter_summary(model)
    print(
        "Calculated conv output length "
        f"(explicit={model.explicit_conv_out_len}, dummy={model.conv_output_length})"
    )

    eps = 1e-8
    X_train_scaled = X_train
    X_validation_scaled = X_validation
    y_train_scaled = y_train
    y_validation_scaled = y_validation

    if bool(data_config.get("scale_x", True)):
        X_mean = X_train.mean()
        X_std = X_train.std()
        X_train_scaled = (X_train - X_mean) / (X_std + eps)
        X_validation_scaled = (X_validation - X_mean) / (X_std + eps)

    if bool(data_config.get("scale_y", True)):
        y_mean = y_train.mean()
        y_std = y_train.std()
        y_train_scaled = (y_train - y_mean) / (y_std + eps)
        y_validation_scaled = (y_validation - y_mean) / (y_std + eps)

    X_train_tensor = torch.tensor(
        X_train_scaled, dtype=torch.float32).unsqueeze(1)
    X_validation_tensor = torch.tensor(
        X_validation_scaled, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_validation_tensor = torch.tensor(
        y_validation_scaled, dtype=torch.float32)

    print("\nTraining model...")
    start_time = time.time()
    train_loss_history, val_loss_history = train_model(
        model,
        X_train_tensor,
        y_train_tensor,
        X_validation_tensor,
        y_validation_tensor,
        learning_rate=float(training_config["learning_rate"]),
        num_epochs=int(training_config["num_epochs"]),
        batch_size=int(training_config["batch_size"]),
    )

    elapsed_time = time.time() - start_time

    overfitting_flag = has_overfitting_signal(
        train_loss_history,
        val_loss_history,
        streak_len=3,
    )

    validation_error, metrics = validate_model(
        model,
        X_validation_tensor,
        y_validation_tensor,
    )

    if bool(output_config.get("save_metrics", True)):
        save_metrics_report(metrics, run_dir)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"epochs: {int(training_config['num_epochs'])}")
    print(f"final training loss: {train_loss_history[-1]:.6f}")
    print(f"final validation loss: {val_loss_history[-1]:.6f}")
    print(
        "overfitting flag: "
        + ("YES (3-epoch train-down/val-up streak detected)" if overfitting_flag else "NO")
    )
    print(f"training time: {elapsed_time:.3f} seconds")
    print(f"validation error: {validation_error:.3f}")
    print(f"run directory: {run_dir.as_posix()}")
    print("=" * 60)

    # print(model)
    print_parameter_summary(model)

    save_loss_curve = bool(output_config.get("save_loss_curve", True))
    if save_loss_curve or SHOW_PLOT:
        output_path = run_dir / "loss_curve.png" if save_loss_curve else None
        plot_training_comparison(
            train_loss_history,
            val_loss_history,
            output_path=output_path,
            show_plot=SHOW_PLOT,
        )

    if bool(output_config.get("save_model", True)):
        torch.save(model, run_dir / "model.pt")

    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/3layer-32K-param.json",
        help="Path to JSON configuration file (default: configs/3layer-32K-param.json)",
    )
    args = parser.parse_args()
    main(args.config)
