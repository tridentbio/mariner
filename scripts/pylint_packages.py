"""
Lints project main packages and check pylint score improvement
"""
import os
import sys

import click
from pylint.lint import Run, load_results, save_results

modules = ["--extension-pkg-whitelist='pydantic'", "mariner", "model_builder", "api"]


HOME = os.getenv("HOME")


if not HOME:
    click.echo("Needs a home variable", err=True)
    sys.exit(1)


PYLINT_HOME = f"{HOME}/.cache/pylint"


@click.command(help="Lints modules")
def main():
    result = Run(modules, do_exit=False)

    previous_results = load_results("global", PYLINT_HOME)
    previous_global_note = previous_results.global_note if previous_results else None
    global_note = result.linter.stats.global_note

    if previous_global_note:
        if global_note < previous_global_note:
            click.echo("Pylint score of main packages regressed", err=True)
            sys.exit(1)
        elif global_note > previous_global_note:
            click.echo("Pylint score was improved!")
            save_results(result.linter.stats, "global", PYLINT_HOME)
            sys.exit(0)
    else:
        click.echo("New Pylint score")
        save_results(result.linter.stats, "global", PYLINT_HOME)
        sys.exit(0)


if __name__ == "__main__":
    main()
