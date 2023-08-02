import click

from cli.admin import commands


@click.group("admin")
def admin_cli():
    pass


admin_cli.add_command(commands.get_token, name="get-token")
