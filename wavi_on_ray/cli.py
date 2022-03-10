import click

import wavi_on_ray

cli = click.Group(
    name="wavi",
    commands=[
        wavi_on_ray.ensemble_10km_test.cli,
    ],
)
