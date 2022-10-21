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
from mariner.db.session import SessionLocal
from mariner.stores.event_sql import event_store
from mariner import events as events_ctl
from mariner import events as events_ctl

version_pattern = re.compile(r"^## \[[\d\.]+] - \d\d\d\d-\d\d-\d\d$")
type_pattern = re.compile(r"^### (Added|Changed|Deprecated|Removed|Fixed|Security)$")
message_pattern = re.compile(r"^- .*$")
ALLOWED_TYPES = ["Added", "Changed", "Security", "Removed", "Fixed", "Deprecated"]


class ReleaseChange(BaseModel):
    type: Literal["Added", "Changed", "Removed", "Deprecated", "Fixed", "Security"]
    message: str


class Release(BaseModel):
    version: str
    date: datetime.date
    changes: List[ReleaseChange]


def parseline(
    line: str,
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
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
                assert (
                    ALLOWED_TYPES.index(current_type) >= 0
                ), f"Type must be one of {repr(ALLOWED_TYPES)} but got {current_type} in {no}"
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
    with open(filepath, "r", encoding="utf-8") as f:
        return parse_text(f)


def log_new_release_events(releases: List[Release]):
    db = SessionLocal()
    with SessionLocal() as db:
        # Later something like:
        # events = [ EventCreate(source='changelog', timestamp=datetime.datetime.now(), payload=release.dict()) for release in releases ]
        # events_ctl.create_events(db, events)
        for release in releases:
            events_ctl.create_event(
                db,
                EventCreate(
                    source="changelog",
                    timestamp=datetime.datetime.now(),
                    payload=release.dict(),
                ),
            )


def release_greater_or_equal(version: str, lower_bound_version: str) -> bool:
    def split_version(version: str):
        return [int(v) for v in version.split(".")]

    try:
        version_ints = split_version(version)
        boudn_ints = split_version(lower_bound_version)
    except ValueError:
        raise ValueError("Failed to parse versions, expecting semver liv 1.0.1, 2.0.21")
    for (release_number, bound_number) in zip(version_ints, boudn_ints):
        if release_number > bound_number:
            return True
        elif release_number < bound_number:
            return False
    return True


def get_version_from_pyproject() -> str:
    return pkg_resources.get_distribution("mariner").version


logger = logging.getLogger("mariner.changelog")
logger.setLevel(logging.INFO)


@click.group()
def cli():
    logging.basicConfig(level=logging.INFO)


@cli.command("publish")
@click.option("--from-version", default=get_version_from_pyproject())
@click.option("--force-resend", default=False)
def publish(from_version: str, force_resend: bool):
    try:
        with SessionLocal() as db:
            changelog_events = event_store.get(db, from_source="changelog")
            versions_published = [
                changelog_event.payload["version"]
                for changelog_event in changelog_events
            ]
            if not force_resend:
                filter_fn = (
                    lambda x: release_greater_or_equal(x, from_version)
                    and x not in versions_published
                )
            else:
                filter_fn = lambda x: release_greater_or_equal(x, from_version)
            releases = [
                release
                for release in parse_text(sys.stdin)
                if filter_fn(release.version)
            ]
            logger.info("Logging events starting from %s", from_version)
            logger.info("Releases: %s", [release.version for release in releases])
            log_new_release_events(releases)
            logger.info("Success!")
    except:
        logger.error("Failed to create release events from changelog")
        raise


if __name__ == "__main__":
    cli()
