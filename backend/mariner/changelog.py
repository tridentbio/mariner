"""
Script for taking a RELEASE.md file and generating
events, that will end up as user notifications.
"""


import datetime
import logging
import re
import sys
from typing import List, Literal, Optional, TextIO

import click
import pkg_resources
from pydantic.main import BaseModel

import mariner.entities  # noqa
from mariner import events as events_ctl
from mariner.db.session import SessionLocal
from mariner.stores.event_sql import event_store

version_pattern = re.compile(r"^## \[[\d\.]+] - \d\d\d\d-\d\d-\d\d$")
type_pattern = re.compile(r"^### (Added|Changed|Deprecated|Removed|Fixed|Security)$")
message_pattern = re.compile(r"^- .*$")
ALLOWED_TYPES = ["Added", "Changed", "Security", "Removed", "Fixed", "Deprecated"]


class ReleaseChange(BaseModel):
    """Models change of a release."""

    type: Literal["Added", "Changed", "Removed", "Deprecated", "Fixed", "Security"]
    message: str


class Release(BaseModel):
    """Models a release parsed from RELEASES.md file."""

    version: str
    date: datetime.date
    changes: List[ReleaseChange]


def parseline(
    line: str,
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Parses a line of RELEASES.md file.

    Args:
        line: string line from the RELEASES.md file.

    Returns:
        tuple (new_version, new_date, new_type, new_message returning the
        parsed result.

    """
    new_version, new_date, new_type, new_message = None, None, None, None
    vm = version_pattern.match(line)
    tm = type_pattern.match(line)
    mm = message_pattern.match(line)
    if vm:
        new_version = line[line.find("[") : line.find("]")].strip("[]").strip("\n ")
        new_date = line[line.find("-") + 1 :].strip("\n ")
    elif tm:
        new_type = line[line.find(" ") + 1 :].strip("\n ")
    elif mm:
        new_message = line[2:].strip("\n ")

    return new_version, new_date, new_type, new_message


def parse_text(text: TextIO):
    """Parses the RELEASES.md file

    Parses the contents of a RELEASES.md file where every entry occupies a single string

    Args:
        text: IO object with file contents

    Raises:
        Exception: when a parsed entry is not from a allowed type (see ALLOWED_TYPES constant
                                                                   in this package)
    """
    releases: dict[str, Release] = {}
    current_version: Optional[str] = None
    current_type: Optional[str] = None
    current_messages: set[str] = set()
    for no, line in enumerate(text.readlines()):
        new_version, new_date, new_type, new_message = parseline(line)

        if new_version and new_date:
            if current_version:
                releases[current_version].changes += [
                    ReleaseChange.construct(type=current_type, message=msg)
                    for msg in current_messages
                ]
            current_version = new_version
            releases[new_version] = Release.construct(
                version=new_version,
                date=datetime.date.fromisoformat(new_date.strip(" \n")),
                changes=[],
            )
            current_type = None
            current_messages.clear()
        elif new_type:
            if current_version and current_type:
                if not (ALLOWED_TYPES.index(current_type) >= 0):
                    allowed = repr(ALLOWED_TYPES)
                    raise Exception(
                        f"Type must be one of {allowed} but got {current_type} in {no}"
                    )
                releases[current_version].changes += [
                    ReleaseChange.construct(type=current_type, message=msg)
                    for msg in current_messages
                ]
            current_messages.clear()
            current_type = new_type
        elif new_message:
            assert (
                current_type is not None
            ), f"Parsing message, expected a change type set previously at {no}"
            current_messages.add(new_message)

    if current_type and current_version and len(current_messages):
        releases[current_version].changes += [
            ReleaseChange.construct(type=current_type, message=msg)
            for msg in current_messages
        ]
    return list(releases.values())


def parse_release_file(filepath: str) -> List[Release]:
    """Parses a RELEASES.md file given it's filepath

    Args:
        filepath: string with path to RELEASES.md file.

    Returns:
        A list of releases parsed from the input file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return parse_text(f)


def log_new_release_events(releases: List[Release]):
    """Updates database with releases

    Starts a connection to the database to access the events collecetion
    and insert the releases as new events.

    Args:
        releases: list of releases parsed from RELEASES.md
    """
    db = SessionLocal()
    with SessionLocal() as db:
        # Later something like:
        # events = [ EventCreate(...) for release in releases ]
        # events_ctl.create_events(db, events)
        for release in releases:
            events_ctl.create_event(
                db,
                events_ctl.EventCreate(
                    source="changelog",
                    timestamp=datetime.datetime.now(),
                    payload=release.dict(),
                ),
            )


def version_greater_or_equal(version: str, lower_bound_version: str) -> bool:
    """Compares 2 strings representing versions

    Args:
        version: left hand side of greater or equal check
        lower_bound_version: right hand side of greater or equal check

    Returns:
        True if version is greater or equal lower_bound_version, otherwise False

    Raises:
        ValueError: When input values cannot be parsed as versions
    """

    def split_version(version: str):
        return [int(v) for v in version.split(".")]

    try:
        version_ints = split_version(version)
        boudn_ints = split_version(lower_bound_version)
    except ValueError:
        raise ValueError(
            "Failed to parse versions, expecting semver like 1.0.1, 2.0.21"
        )
    for release_number, bound_number in zip(version_ints, boudn_ints):
        if release_number > bound_number:
            return True
        elif release_number < bound_number:
            return False
    return True


def get_version_from_pyproject() -> str:
    """Gets the version of mariner project as informed by pyproject.toml.

    Returns:
        string with current version.
    """
    return pkg_resources.get_distribution("mariner").version


logger = logging.getLogger("mariner.changelog")
logger.setLevel(logging.INFO)


@click.group()
def cli():
    """Groups cli global options, that will affect all commands."""
    logging.basicConfig(level=logging.INFO)


@cli.command("publish")
@click.option("--from-version", default=get_version_from_pyproject())
@click.option("--force-resend", default=False)
def publish(from_version: str, force_resend: bool):
    """CLI handler for taking a RELEASES.md and publish changes to users.

    Args:
        from_version: Starting version from which to start pushing notifications.
        force_resend: Flag to tell if should send a notification even tho it was
        already send.
    """
    with SessionLocal() as db:
        changelog_events = event_store.get(db, from_source="changelog")
        versions_published = [
            changelog_event.payload["version"] for changelog_event in changelog_events
        ]
        if not force_resend:

            def filter_fn(x):
                return (
                    version_greater_or_equal(x, from_version)
                    and x not in versions_published
                )

        else:

            def filter_fn(x):
                return version_greater_or_equal(x, from_version)

        releases = [
            release for release in parse_text(sys.stdin) if filter_fn(release.version)
        ]
        logger.info("Logging events starting from %s", from_version)
        logger.info("Releases: %s", [release.version for release in releases])
        log_new_release_events(releases)
        logger.info("Success!")


if __name__ == "__main__":
    cli()