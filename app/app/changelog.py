"""
Script for taking a RELEASE.md file and generating 
events, that will end up as user notifications.

TODO:
    - A function to merge versions might be useful
    when jumping a release cycle for the user, eg.:
    squashing unreleased versions 0.1.1 and 0.1.2
    into 0.1.1 before releasing
"""


import datetime
import re
from typing import List, Literal, Optional

from pydantic.main import BaseModel

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


def parse_release_file(filepath: str) -> List[Release]:
    with open(filepath, "r", encoding="utf-8") as f:
        releases: dict[str, Release] = {}
        current_version: Optional[str] = None
        current_type: Optional[str] = None
        current_messages: set[str] = set()
        for no, line in enumerate(f.readlines()):
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
