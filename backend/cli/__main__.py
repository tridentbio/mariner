"""
CLI entry point.
"""
import logging

import click

from cli.admin.group import admin_cli
from cli.load_testing.group import load_testing_cli


@click.group(context_settings={"show_default": True})
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="DEBUG",
    help="Set the logging level.",
)
def _cli(log_level: str = "INFO"):
    logger = logging.getLogger("cli")
    logger.setLevel(log_level)


_cli.add_command(admin_cli)
_cli.add_command(load_testing_cli)

if __name__ == "__main__":
    _cli()
