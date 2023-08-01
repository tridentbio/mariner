"""
Load testing CLI group.
"""
import click

from cli.load_testing import commands


@click.group("load-testing")
@click.option(
    "--output",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
    help="Path to save report csv.",
    default="load_test_results.csv",
)
@click.option("--timeout", type=int, default=60, help="Time in minutes until timeout")
@click.option(
    "--credentials",
    type=click.File("r", encoding="utf-8"),
    default="credentials.json",
    help="Path to credentials.json file.",
)
@click.option(
    "--url",
    type=str,
    default="http://localhost",
    help="Backend URL to perform the test",
)
def load_testing_cli(
    output: str = "load_test_results.csv",
    credentials: str = "credentials.json",
    timeout: int = 60,
    url: str = "http://localhost:8000",
):
    """
    Load testing CLI group.
    """
    pass


load_testing_cli.add_command(
    commands.load_test_number_of_simulteneous_trainings, "trainings"
)
load_testing_cli.add_command(commands.load_test_trainings, "scale-trainings")
