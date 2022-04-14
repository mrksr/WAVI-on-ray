import click

from . import ingestion, training_datasets

cli = click.Group(
    name="ensemble_10km_basin_specific",
    commands=[
        ingestion.ingest,
        training_datasets.build_datasets,
    ],
)
