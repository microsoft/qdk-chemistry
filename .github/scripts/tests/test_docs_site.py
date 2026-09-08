# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Tests for versioned documentation site assembly."""

import argparse
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).parents[1] / "docs_site.py"
SPEC = importlib.util.spec_from_file_location("docs_site", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Unable to load {SCRIPT}")
docs_site = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(docs_site)


class DocsSiteTest(unittest.TestCase):
    """Exercise versioned documentation publication behavior."""

    def setUp(self) -> None:
        """Create an isolated site root for each test."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.site = self.root / "site"
        self.site.mkdir()

    def tearDown(self) -> None:
        """Remove the isolated site root."""
        self.temporary_directory.cleanup()

    def _html(self, marker: str) -> Path:
        """Create a minimal documentation build containing a marker."""
        html = self.root / f"html-{marker}"
        html.mkdir()
        (html / "index.html").write_text(marker)
        return html

    def _install(
        self,
        target: str,
        version: str,
        package_version: str = "",
        *,
        commit: str = "abcdef123456",
        stable: bool = False,
        docs_commit: str = "",
        docs_ref: str = "",
        repository: Path | None = None,
        source_run_id: str = "",
    ) -> None:
        """Install a minimal documentation build into the test site."""
        marker = source_run_id or docs_commit or package_version or version
        docs_site.install(
            argparse.Namespace(
                html=str(self._html(marker)),
                site=str(self.site),
                target=target,
                version=version,
                package_version=package_version,
                commit=commit,
                ref="main" if target == "dev" else f"v{package_version}",
                docs_commit=docs_commit,
                docs_ref=docs_ref,
                repository=str(repository or self.root),
                stable=stable,
                base_url="https://example.test/docs/",
                source_run_id=source_run_id,
            )
        )

    def _git(self, repository: Path, *arguments: str) -> str:
        """Run Git in a test repository and return standard output."""
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _linear_history(self) -> tuple[Path, str, str]:
        """Create a two-commit Git history for documentation revisions."""
        repository = self.root / "repository"
        repository.mkdir()
        self._git(repository, "init", "--quiet")
        self._git(repository, "config", "user.name", "Docs Test")
        self._git(repository, "config", "user.email", "docs@example.test")

        tracked_file = repository / "docs.txt"
        tracked_file.write_text("first\n")
        self._git(repository, "add", "docs.txt")
        self._git(repository, "commit", "--quiet", "-m", "First revision")
        first_commit = self._git(repository, "rev-parse", "HEAD")

        tracked_file.write_text("second\n")
        self._git(repository, "commit", "--quiet", "-am", "Second revision")
        second_commit = self._git(repository, "rev-parse", "HEAD")
        return repository, first_commit, second_commit

    def _package_version(self, directory: str) -> str:
        """Read the exact package version recorded for a directory."""
        info = json.loads(
            (self.site / directory / docs_site.BUILD_INFO_NAME).read_text()
        )
        return info["package_version"]

    def test_root_prefers_stable_over_dev(self) -> None:
        """Use dev as the fallback until stable documentation exists."""
        self._install("dev", "dev")

        self.assertIn("url=dev/", (self.site / "index.html").read_text())

        self._install("2.1", "2.1", "2.1.0", stable=True)

        self.assertIn("url=stable/", (self.site / "index.html").read_text())

    def test_patch_release_replaces_minor_version_and_stable(self) -> None:
        """Replace a minor directory and stable with a newer patch."""
        self._install("2.1", "2.1", "2.1.0", stable=True)
        self._install("2.1", "2.1", "2.1.1", stable=True)

        self.assertEqual(self._package_version("2.1"), "2.1.1")
        self.assertEqual(self._package_version("stable"), "2.1.1")

    def test_release_metadata_records_documentation_revision(self) -> None:
        """Record release and documentation provenance independently."""
        self._install(
            "2.1",
            "2.1",
            "2.1.3",
            docs_commit="fedcba654321",
            docs_ref="stable/2.1",
        )

        info = docs_site._read_build_info(self.site / "2.1")
        self.assertEqual(info["commit"], "abcdef123456")
        self.assertEqual(info["ref"], "v2.1.3")
        self.assertEqual(info["docs_commit"], "fedcba654321")
        self.assertEqual(info["docs_ref"], "stable/2.1")

    def test_same_package_documentation_revision_moves_forward(self) -> None:
        """Allow a documentation revision descending from the published one."""
        repository, first_commit, second_commit = self._linear_history()
        self._install(
            "2.1",
            "2.1",
            "2.1.3",
            docs_commit=first_commit,
            repository=repository,
        )

        self._install(
            "2.1",
            "2.1",
            "2.1.3",
            docs_commit=second_commit,
            repository=repository,
        )

        info = docs_site._read_build_info(self.site / "2.1")
        self.assertEqual(info["docs_commit"], second_commit)

    def test_same_package_documentation_revision_cannot_move_backward(self) -> None:
        """Reject a documentation revision older than the published one."""
        repository, first_commit, second_commit = self._linear_history()
        self._install(
            "2.1",
            "2.1",
            "2.1.3",
            docs_commit=second_commit,
            repository=repository,
        )

        with self.assertRaises(SystemExit):
            self._install(
                "2.1",
                "2.1",
                "2.1.3",
                docs_commit=first_commit,
                repository=repository,
            )

        info = docs_site._read_build_info(self.site / "2.1")
        self.assertEqual(info["docs_commit"], second_commit)

    def test_legacy_metadata_uses_release_commit_for_history(self) -> None:
        """Use the release commit when older metadata has no docs commit."""
        repository, first_commit, second_commit = self._linear_history()
        self._install(
            "2.1",
            "2.1",
            "2.1.3",
            commit=first_commit,
            docs_commit=first_commit,
            repository=repository,
        )
        info_file = self.site / "2.1" / docs_site.BUILD_INFO_NAME
        info = json.loads(info_file.read_text())
        del info["docs_commit"]
        del info["docs_ref"]
        info_file.write_text(json.dumps(info))

        self._install(
            "2.1",
            "2.1",
            "2.1.3",
            commit=first_commit,
            docs_commit=second_commit,
            repository=repository,
        )

        info = docs_site._read_build_info(self.site / "2.1")
        self.assertEqual(info["docs_commit"], second_commit)

    def test_unavailable_published_documentation_revision_fails_closed(self) -> None:
        """Reject an update when the published revision cannot be verified."""
        repository, _, second_commit = self._linear_history()
        unavailable_commit = "f" * 40
        self._install(
            "2.1",
            "2.1",
            "2.1.3",
            docs_commit=unavailable_commit,
            repository=repository,
        )

        with self.assertRaises(SystemExit):
            self._install(
                "2.1",
                "2.1",
                "2.1.3",
                docs_commit=second_commit,
                repository=repository,
            )

        info = docs_site._read_build_info(self.site / "2.1")
        self.assertEqual(info["docs_commit"], unavailable_commit)

    def test_older_maintenance_release_does_not_downgrade_stable(self) -> None:
        """Keep stable on the newest release across maintenance lines."""
        self._install("2.1", "2.1", "2.1.1", stable=True)
        self._install("1.1", "1.1", "1.1.1", stable=True)

        self.assertEqual(self._package_version("1.1"), "1.1.1")
        self.assertEqual(self._package_version("stable"), "2.1.1")

    def test_older_patch_cannot_replace_newer_minor_docs(self) -> None:
        """Reject replacing a minor directory with an older patch."""
        self._install("2.1", "2.1", "2.1.1", stable=True)

        with self.assertRaises(SystemExit):
            self._install("2.1", "2.1", "2.1.0", stable=True)

    def test_four_component_versions_are_ordered(self) -> None:
        """Accept the tweak component the VERSION file allows."""
        self._install("2.1", "2.1", "2.1.0", stable=True)
        self._install("2.1", "2.1", "2.1.0.1", stable=True)

        self.assertEqual(self._package_version("2.1"), "2.1.0.1")
        self.assertEqual(self._package_version("stable"), "2.1.0.1")

    def test_switcher_lists_dev_and_minor_versions(self) -> None:
        """List dev, stable, and archived minors in expected order."""
        self._install("dev", "dev")
        self._install("1.1", "1.1", "1.1.0")
        self._install("2.1", "2.1", "2.1.0", stable=True)

        entries = json.loads((self.site / "switcher.json").read_text())

        self.assertEqual([entry["version"] for entry in entries], ["dev", "2.1", "1.1"])
        self.assertEqual(
            [entry["name"] for entry in entries],
            ["dev (main)", "2.1 (stable)", "1.1"],
        )
        self.assertEqual(entries[0]["url"], "https://example.test/docs/dev/")

    def test_unversioned_redirect_preserves_query_and_fragment(self) -> None:
        """Keep query strings and fragments when rewriting unversioned URLs."""
        self._install("2.1", "2.1", "2.1.0", stable=True)

        not_found = (self.site / "404.html").read_text()

        self.assertIn("window.location.search + window.location.hash", not_found)

    def test_incomplete_build_metadata_is_rejected(self) -> None:
        """Reject metadata files without all required provenance fields."""
        directory = self.site / "2.1"
        directory.mkdir()
        (directory / docs_site.BUILD_INFO_NAME).write_text('{"version": "2.1"}')

        with self.assertRaises(SystemExit):
            docs_site._read_build_info(directory)

    def test_dev_cannot_update_stable(self) -> None:
        """Reject requests to install development docs as stable."""
        with self.assertRaises(SystemExit):
            self._install("dev", "dev", stable=True)

    def test_older_dev_run_cannot_replace_newer_docs(self) -> None:
        """Ignore delayed workflow runs after newer development docs publish."""
        self._install("dev", "dev", source_run_id="102")
        self._install("dev", "dev", source_run_id="101")

        info = docs_site._read_build_info(self.site / "dev")
        self.assertEqual(info["source_run_id"], "102")
        self.assertEqual((self.site / "dev" / "index.html").read_text(), "102")

    def test_site_over_pages_limit_is_rejected(self) -> None:
        """Refuse to publish an assembled tree over the Pages size limit."""
        with (
            patch.object(docs_site, "_SITE_SIZE_LIMIT_BYTES", 1),
            self.assertRaises(SystemExit),
        ):
            self._install("dev", "dev")


if __name__ == "__main__":
    unittest.main()
