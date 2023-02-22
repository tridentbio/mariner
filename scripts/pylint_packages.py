"""
Lints project main packages and check pylint score improvement
"""
import os
import sys

import click
from pylint.lint import Run, load_results, save_results

modules = ["--extension-pkg-whitelist='pydantic'", "mariner", "model_builder", "api"]

PYLINT_HOME = os.path.expanduser("~/.cache/pylint")


@click.command(help="Compares current pylint score to TARGET_BRANCH run")
@click.argument("src_branch")
@click.argument("target_branch")
@click.option(
    "--save-src",
    default=False,
    is_flag=True,
    help="Saves improved results as SRC_BRANCH",
)
@click.option(
    "--save-new-target",
    default=False,
    is_flag=True,
    help="Saves improved results as TARGET_BRANCH",
)
def main(src_branch: str, target_branch: str, save_src=False, save_new_target=False):
    result = Run(modules, do_exit=False)

    previous_results = load_results(target_branch, PYLINT_HOME)
    previous_global_note = previous_results.global_note if previous_results else None
    global_note = result.linter.stats.global_note

    if previous_global_note is not None:
        if global_note < previous_global_note:
            # Don't update if note is lower. Let's try to keep it high
            click.echo(f"Score is less than target ({previous_global_note}). Ignoring save flags")
            sys.exit(1)
        elif global_note > previous_global_note:
            if save_src:
                click.echo(f"Pylint score was improved! Saved as {src_branch}")
                save_results(result.linter.stats, src_branch, PYLINT_HOME)
            if save_new_target:
                click.echo(f"Pylint score was improved! Saved as {target_branch}")
                save_results(result.linter.stats, target_branch, PYLINT_HOME)
        else:
            click.echo("Score is the same")
    else:
        if save_src:
            click.echo(f"New Pylint score saved as {src_branch}")
            save_results(result.linter.stats, src_branch, PYLINT_HOME)
        if save_new_target:
            click.echo(f"New Pylint score saved as {target_branch}")
            save_results(result.linter.stats, target_branch, PYLINT_HOME)


if __name__ == "__main__":
    main()
