import pathlib
from typing import List, Union

import click
import GPy
import numpy as np
import pandas as pd
import ray
import scipy.stats
from ray import tune

from .training_datasets import DATASET_PROFILES

TUNE_CONFIG = {
    "training_dataset": tune.grid_search(list(DATASET_PROFILES)),
    "kernels": tune.grid_search(
        [
            ["matern32"],
            ["matern52"],
            ["rbf"],
            ["bias", "matern32"],
            ["bias", "matern52"],
            ["bias", "rbf"],
            ["bias", "linear", "matern32"],
            ["bias", "linear", "matern52"],
            ["bias", "linear", "rbf"],
        ]
    ),
}


def _build_kernel_list(kernels):
    kernel_map = {
        "bias": GPy.kern.Bias,
        "linear": GPy.kern.Linear,
        "matern32": GPy.kern.Matern32,
        "matern52": GPy.kern.Matern52,
        "rbf": GPy.kern.RBF,
    }
    return [kernel_map[kernel](1) for kernel in kernels]


def build_model(
    inputs: List[np.array], outputs: List[np.array], kernels: List[str]
) -> GPy.models.GPRegression:
    """Builds a coregionalization GP model.

    Args:
        inputs (List[np.array]): List of input datasets, must match outputs.
        outputs (List[np.array]): List of output dataset, must match inputs.
        kernels (List[str]): List of kernel names for LCM model.

    Returns:
        GPy.models.GPRegression: A trained coregionalization GP model.
    """
    lcm_kernel = GPy.util.multioutput.LCM(
        input_dim=1,
        num_outputs=2,
        kernels_list=_build_kernel_list(kernels),
    )

    coreg_model = GPy.models.gp_coregionalized_regression.GPCoregionalizedRegression(
        X_list=inputs, Y_list=outputs, kernel=lcm_kernel
    )
    coreg_model[".*noise.*variance"].constrain_fixed(1e-8)

    try:
        coreg_model[".*Mat.*variance.*"].constrain_bounded(30.0, 100.0)
        coreg_model[".*Mat.*lengthscale.*"].constrain_bounded(50.0, 500.0)
    except AttributeError:
        # Kernel does not exist
        pass

    try:
        coreg_model[".*RBF.*variance.*"].constrain_bounded(30.0, 100.0)
        coreg_model[".*RBF.*lengthscale.*"].constrain_bounded(50.0, 500.0)
    except AttributeError:
        # Kernel does not exist
        pass

    return coreg_model


def _get_task_indexers(inputs):
    indexers = [
        inputs["melt_in_partial_cells"] == 0.0,
        inputs["melt_in_partial_cells"] == 1.0,
    ]
    return [indexer.values for indexer in indexers]


def train_model(datasets_folder: pathlib.Path, config: dict) -> GPy.models.GPRegression:
    """Train a single GP model described by the ray config.

    Args:
        datasets_folder (Union[str, pathlib.Path]): Path to the DVC datasets folder.
        config (dict): Ray config
    """
    dataset_file = (
        datasets_folder
        / "prepared"
        / "ensemble_10km_test"
        / "training"
        / f"{config['training_dataset']}.hdf"
    )
    inputs = pd.read_hdf(dataset_file, key="inputs")
    outputs = pd.read_hdf(dataset_file, key="outputs")
    indexers = _get_task_indexers(inputs)

    coreg_model = build_model(
        [inputs.values[indexer][:, [0]] for indexer in indexers],
        [outputs.values[indexer] for indexer in indexers],
        kernels=config["kernels"],
    )

    coreg_model.optimize()

    return {
        "log_likelihood": coreg_model.log_likelihood(),
        **calculate_test_metrics(datasets_folder, coreg_model),
    }


def calculate_test_metrics(
    datasets_folder: pathlib.Path, coreg_model: GPy.models.GPRegression
) -> dict[str, float]:
    """Calculate a suite of test metrics for a coregionalization model.

    Args:
        datasets_folder (pathlib.Path): Path to the DVC datasets folder.
        coreg_model (GPy.models.GPRegression): Trained GP model.

    Returns:
        dict[str, float]: A dictionary containing test metrics.
    """
    test_dataset_file = datasets_folder / "prepared" / "ensemble_10km_test" / "full.hdf"
    inputs = pd.read_hdf(test_dataset_file, key="inputs")
    outputs = pd.read_hdf(test_dataset_file, key="outputs")
    indexers = _get_task_indexers(inputs)

    tasks = [
        (
            inputs.values[indexer][:, [0]],
            outputs.values[indexer],
        )
        for indexer in indexers
    ]

    def _calculate_task_metrics(task_index, inputs, outputs):
        task_indices = np.full([len(inputs), 1], 1.0 * task_index)

        predictions = coreg_model.predict(
            np.hstack([inputs, task_indices]),
            Y_metadata={"output_index": task_indices.astype(int)},
        )

        return {
            "log_likelihoods": scipy.stats.norm.logpdf(
                outputs, loc=predictions[0], scale=np.sqrt(predictions[1])
            ),
            "squared_errors": (outputs - predictions[0]) ** 2,
        }

    task_metrics = [
        _calculate_task_metrics(ix, inputs, outputs)
        for ix, (inputs, outputs) in enumerate(tasks)
    ]

    return {
        "test/log_likelihood_0": np.mean(task_metrics[0]["log_likelihoods"]),
        "test/rmse_0": np.sqrt(np.mean(task_metrics[0]["squared_errors"])),
        "test/log_likelihood_1": np.mean(task_metrics[1]["log_likelihoods"]),
        "test/rmse_1": np.sqrt(np.mean(task_metrics[1]["squared_errors"])),
        "test/log_likelihood": np.mean(
            np.concatenate([task["log_likelihoods"] for task in task_metrics])
        ),
        "test/rmse": np.sqrt(
            np.mean(np.concatenate([task["squared_errors"] for task in task_metrics]))
        ),
    }


def tune_models(datasets_folder: Union[str, pathlib.Path]):
    """Run a full sweep of model trainings.

    Args:
        datasets_folder (Union[str, pathlib.Path]): Path to the DVC datasets folder.
    """
    ray.init()

    datasets_folder = pathlib.Path(datasets_folder).resolve()
    tune.run(
        lambda config: train_model(datasets_folder, config),
        metric="test/log_likelihood",
        mode="max",
        num_samples=2,
        config=TUNE_CONFIG,
    )


@click.command()
@click.argument(
    "datasets_folder",
    type=click.Path(exists=True, file_okay=False, writable=True),
)
def train_models(datasets_folder):
    """Run a full sweep of model trainings."""
    tune_models(datasets_folder)
