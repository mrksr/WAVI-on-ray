import pathlib
from typing import Union

import click
import pandas as pd
import xarray as xr


def build_hdf_data(datasets_folder: Union[str, pathlib.Path]):
    """Extracts input and output data frames from the ensemble_10km_test.nc dataset.

    Args:
        datasets_folder (Union[str, pathlib.Path]): Path to the DVC datasets folder.
    """
    datasets = pathlib.Path(datasets_folder)
    dataset = xr.open_dataset(datasets / "ensemble_10km_test.nc")

    inputs = pd.DataFrame(
        {
            "melt_rate": dataset["Inputs"][0],
            "melt_in_partial_cells": dataset["Inputs"][1],
        }
    )
    outputs = pd.DataFrame({"SLC": dataset["SLC"][0]})
    inputs_outputs = pd.concat([inputs, outputs], axis=1)

    prepared_folder = datasets / "prepared" / "ensemble_10km_test"
    prepared_folder.mkdir(exist_ok=True, parents=True)

    for key, dataframe in {
        "inputs": inputs,
        "outputs": outputs,
        "inputs_outputs": inputs_outputs,
    }.items():
        dataframe.to_hdf(
            prepared_folder / "full.hdf",
            key=key,
        )


@click.command()
@click.argument(
    "datasets_folder",
    type=click.Path(exists=True, file_okay=False, writable=True),
)
def ingest(datasets_folder):
    """Ingest ensemble_10km_test data for emulator training."""
    build_hdf_data(datasets_folder)
