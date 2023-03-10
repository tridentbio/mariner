"""
Lints project main packages and check pylint score improvement
"""
import os
import sys

import click
from pylint.lint import Run, load_results, save_results

modules = [
    "--extension-pkg-whitelist='pydantic'",
    "mariner",
    "model_builder",
    "api",
]

PYLINT_HOME = os.getenv("PYLINT_HOME") or os.path.expanduser("~/.cache/pylint")


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

    def update_cache():
        should_save_at = [(save_src, src_branch), (save_new_target, target_branch)]
        for (should_save, at) in should_save_at:
            if not should_save:
                continue
            click.echo(f"New Pylint score saved as {PYLINT_HOME}/{at}")
            save_results(result.linter.stats, at, PYLINT_HOME)

    # Unknown BASE pylint score
    if previous_global_note is None:
        click.echo("New score")
        update_cache()
    # HEAD pylint score higher than BASE pylint score
    elif previous_global_note < global_note:
        click.echo("Setting a new lower bound pylint score")
        update_cache()
    # HEAD pylint score equals BASE pylint score
    elif previous_global_note == global_note:
        click.echo("Score is the same")
    # HEAD pylint score lower than BASE pylint score
    elif previous_global_note > global_note:
        click.echo(
            f"Score is less than target ({previous_global_note}). Ignoring save flags"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
