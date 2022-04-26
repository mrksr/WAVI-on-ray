import click

import wavi_on_ray

cli = click.Group(
    name="wavi",
    commands=[
        wavi_on_ray.ensemble_10km_test.cli,
        wavi_on_ray.ensemble_10km_basin_specific.cli,
        wavi_on_ray.ensemble_10km_5d_medium.cli,
    ],
)
