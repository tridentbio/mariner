import click


@click.command("model-capacity", help="Load test the capacity in width and depth")
@click.option(
    "--model-width",
    type=int,
    default=1,
    help="Width of the model to train.",
)
@click.option(
    "--model-depth",
    type=int,
    default=1,
    help="Depth of the model to train.",
)
def load_test_model_capacity(model_width: int = 1, model_depth: int = 5):
    ...
