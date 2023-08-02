import click


@click.command(
    "deployments-prediction",
    help="Load test the number of concurrent prediction requests",
)
def load_test_prediction():
    ...
