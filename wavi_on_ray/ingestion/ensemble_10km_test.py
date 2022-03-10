import pathlib
from typing import Union

import click
import pandas as pd
import xarray as xr


def prepare(datasets_folder: Union[str, pathlib.Path]):
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

    prepared_folder = datasets / "prepared"
    prepared_folder.mkdir(exist_ok=True)

    for key, dataframe in {
        "inputs": inputs,
        "outputs": outputs,
        "inputs_outputs": inputs_outputs,
    }.items():
        dataframe.to_hdf(
            prepared_folder / "ensemble_10km_test.hdf",
            key=key,
        )


@click.command()
@click.argument(
    "datasets_folder",
    type=click.Path(exists=True, file_okay=False, writable=True),
)
def cli(datasets_folder):
    """Ingest ensemble_10km_test data for emulator training."""
    prepare(datasets_folder)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli()
