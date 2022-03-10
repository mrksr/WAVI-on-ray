import pathlib
from typing import Union

import click
import numpy as np
import pandas as pd

DATASET_PROFILES = {
    "full": (
        ...,
        ...,
    ),
    "symmetric": (
        [0, 3, 20],
        [0, 3, 20],
    ),
    "symmetric_plus_one": (
        [0, 3, 10, 20],
        [0, 3, 10, 20],
    ),
    "asymmetric": (
        [0, 3, 10, 20],
        [0, 7, 11, 18],
    ),
    "transfer": (
        ...,
        [2, 4, 11],
    ),
}


def subselect_indexer(haystack: np.array, needles: np.array) -> np.array:
    """Subselect a binary indexer haystack to only keep the ith needles.

    That is, if passed the haystack [False, True, True, False, True] and the
    needle [0, 2], the result will be [False, True, False, False, True].

    Args:
        haystack (np.array): A 1D binary indexer.
        needles (np.array): A 1D integer indexer specifying which True values to keep.

    Returns:
        np.array: A 1D binary subselected binary indexer.
    """
    to_keep = np.argwhere(haystack)[:, 0][needles]
    indexer = np.zeros_like(haystack, dtype=bool)
    indexer[to_keep] = True
    return indexer


def subsample_dataset(datasets_folder: Union[str, pathlib.Path]):
    """Extracts special-case datasets for model training.

    Args:
        datasets_folder (Union[str, pathlib.Path]): Path to the DVC datasets folder.
    """
    ensemble_folder = pathlib.Path(datasets_folder, "prepared/ensemble_10km_test")
    prepared_dataset = ensemble_folder / "full.hdf"

    inputs = pd.read_hdf(prepared_dataset, key="inputs")
    task_indexers = [
        inputs["melt_in_partial_cells"] == 0.0,
        inputs["melt_in_partial_cells"] == 1.0,
    ]
    task_indexers = [indexer.values for indexer in task_indexers]

    keys = ["inputs", "outputs", "inputs_outputs"]
    (ensemble_folder / "training").mkdir(exist_ok=True)
    for profile, inner_indexers in DATASET_PROFILES.items():
        output_dataset = ensemble_folder / "training" / f"{profile}.hdf"

        full_indexer = np.logical_or(
            *(
                subselect_indexer(task_indexer, inner_indexer)
                for task_indexer, inner_indexer in zip(task_indexers, inner_indexers)
            )
        )
        for key in keys:
            dataframe = pd.read_hdf(prepared_dataset, key=key)
            dataframe.loc[full_indexer].to_hdf(output_dataset, key=key)


@click.command()
@click.argument(
    "datasets_folder",
    type=click.Path(exists=True, file_okay=False, writable=True),
)
def build_datasets(datasets_folder):
    """Create special-case datasets for model training."""
    subsample_dataset(datasets_folder)
