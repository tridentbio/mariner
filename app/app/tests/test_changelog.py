import datetime
from pathlib import Path

import pytest

from app import changelog


@pytest.fixture(scope="function")
def releases_path_fixture(tmp_path: Path) -> str:
    file = (
        "# Changelog\n"
        "All notable changes etc.\n"
        "\n"
        "## [1.0.1] - 2022-09-17\n"
        "### Fixed\n"
        "- Change 13\n"
        "### Security\n"
        "- Change 14\n"
        "## [1.0.0] - 2022-08-16\n"
        "### Added\n"
        "- Change 1\n"
        "- Change 2\n"
        "### Fixed\n"
        "- Change 3\n"
        "- Change 4\n"
        "### Changed\n"
        "- Change 5\n"
        "- Change 6\n"
        "### Removed\n"
        "- Change 7\n"
        "- Change 8\n"
        "### Deprecated\n"
        "- Change 9\n"
        "- Change 10\n"
        "### Security\n"
        "- Change 11\n"
        "- Change 12\n"
    )
    release_path = tmp_path / "release.md"
    release_path.write_bytes(file.encode())
    return str(release_path.absolute())


def test_change_log(releases_path_fixture: str):
    releases = changelog.parse_release_file(releases_path_fixture)
    expected_releases = [
        changelog.Release(
            version="1.0.1",
            date=datetime.date.fromisoformat("2022-09-17"),
            changes=[
                changelog.ReleaseChange(type="Fixed", message="Change 13"),
                changelog.ReleaseChange(type="Security", message="Change 14"),
            ],
        ),
        changelog.Release(
            version="1.0.0",
            date=datetime.date.fromisoformat("2022-08-16"),
            changes=[
                changelog.ReleaseChange(type="Added", message="Change 1"),
                changelog.ReleaseChange(type="Added", message="Change 2"),
                changelog.ReleaseChange(type="Fixed", message="Change 3"),
                changelog.ReleaseChange(type="Fixed", message="Change 4"),
                changelog.ReleaseChange(type="Changed", message="Change 5"),
                changelog.ReleaseChange(type="Changed", message="Change 6"),
                changelog.ReleaseChange(type="Removed", message="Change 7"),
                changelog.ReleaseChange(type="Removed", message="Change 8"),
                changelog.ReleaseChange(type="Deprecated", message="Change 9"),
                changelog.ReleaseChange(type="Deprecated", message="Change 10"),
                changelog.ReleaseChange(type="Security", message="Change 11"),
                changelog.ReleaseChange(type="Security", message="Change 12"),
            ],
        ),
    ]
    for (got, expected) in zip(releases, expected_releases):
        assert got.date == expected.date
        assert got.version == expected.version
        for change in expected.changes:
            assert change.dict() in [c.dict() for c in got.changes]
