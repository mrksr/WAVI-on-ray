import click

from . import ingestion, modelling

cli = click.Group(
    name="ensemble_10km_test",
    commands=[
        ingestion.ingest,
    ],
)
