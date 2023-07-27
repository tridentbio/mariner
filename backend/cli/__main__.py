import click

from cli.admin.group import admin_cli
from cli.load_testing.group import load_testing_cli


@click.group()
def cli():
    pass


cli.add_command(admin_cli)
cli.add_command(load_testing_cli)

if __name__ == "__main__":
    cli()
