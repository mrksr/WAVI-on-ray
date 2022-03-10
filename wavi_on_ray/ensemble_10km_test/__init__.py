import click

from . import ingestion, modelling, training_datasets

cli = click.Group(
    name="ensemble_10km_test",
    commands=[
        ingestion.ingest,
        training_datasets.build_datasets,
    ],
)
