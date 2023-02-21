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
@click.argument("src_branch")
@click.argument("target_branch")
@click.option("--save-src", default=False, is_flag=True, help="Saves src_branch results as target_branch results")
@click.option("--save-new-target", default=False, is_flag=True, help="Saves src_branch results as target_branch results")
def main(src_branch: str, target_branch: str, save_src=False, save_new_target=False):
    result = Run(modules, do_exit=False)

    previous_results = load_results(target_branch, PYLINT_HOME)
    previous_global_note = previous_results.global_note if previous_results else None
    global_note = result.linter.stats.global_note

    if previous_global_note:
        if global_note < previous_global_note:
            click.echo("Pylint score of main packages regressed", err=True)
            if save_src: save_results(result.linter.stats, src_branch, PYLINT_HOME)
            sys.exit(1)
        elif global_note > previous_global_note:
            click.echo("Pylint score was improved!")
            if save_src: save_results(result.linter.stats, src_branch, PYLINT_HOME)
            if save_new_target: save_results(result.linter.stats, target_branch, PYLINT_HOME)
            sys.exit(0)
    else:
        click.echo("New Pylint score")
        if save_src: save_results(result.linter.stats, src_branch, PYLINT_HOME)
        if save_new_target: save_results(result.linter.stats, target_branch, PYLINT_HOME)
        sys.exit(0)


if __name__ == "__main__":
    main()
