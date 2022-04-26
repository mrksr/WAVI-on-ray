import click

from . import ingestion

cli = click.Group(
    name="ensemble_10km_5d_medium",
    commands=[
        ingestion.ingest,
    ],
)
