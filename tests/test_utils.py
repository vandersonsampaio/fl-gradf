import os

from src.utils.config_loader import load_config, load_hyperparameters
from src.utils.logger import get_logger
from src.utils.metrics import (
    accuracy_retention,
    detection_rate,
    fairness_std,
    false_positive_rate,
)
from src.utils.visualization import plot_accuracy_over_rounds, plot_bar_comparison


def test_load_config_has_expected_sections():
    config = load_config()
    assert "federated_learning" in config
    assert "gradf" in config
    assert config["federated_learning"]["n_rounds"] > 0


def test_load_hyperparameters_has_expected_sections():
    hyperparams = load_hyperparameters()
    assert "combiner_nn" in hyperparams
    assert "rl_classifier" in hyperparams
    assert "rl_selector" in hyperparams


def test_get_logger_reuses_handlers():
    log1 = get_logger("gradf.test_utils")
    log2 = get_logger("gradf.test_utils")
    assert log1 is log2
    assert len(log1.handlers) == 1


def test_detection_rate_and_false_positive_rate():
    y_true = [True, True, False, False, True]
    y_pred = [True, False, False, True, True]
    assert detection_rate(y_true, y_pred) == 2 / 3
    assert false_positive_rate(y_true, y_pred) == 0.5


def test_fairness_std_and_accuracy_retention():
    assert fairness_std({"a": 0.9, "b": 0.9}) == 0.0
    assert accuracy_retention(0.9, 0.9) == 1.0
    assert accuracy_retention(0.9, 0.45) == 0.5


def test_plot_helpers_save_files(tmp_path):
    acc_path = str(tmp_path / "acc.png")
    bar_path = str(tmp_path / "bar.png")
    plot_accuracy_over_rounds({"fedavg": [0.5, 0.6, 0.7]}, save_path=acc_path)
    plot_bar_comparison({"fedavg": 0.7, "fltrust": 0.75}, save_path=bar_path)
    assert os.path.exists(acc_path)
    assert os.path.exists(bar_path)
