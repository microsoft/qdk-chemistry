# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Tests for versioned documentation site assembly."""

import argparse
import importlib.util
import json
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
        stable: bool = False,
        source_run_id: str = "",
    ) -> None:
        """Install a minimal documentation build into the test site."""
        marker = source_run_id or package_version or version
        docs_site.install(
            argparse.Namespace(
                html=str(self._html(marker)),
                site=str(self.site),
                target=target,
                version=version,
                package_version=package_version,
                commit="abcdef123456",
                ref="main" if target == "dev" else f"v{package_version}",
                stable=stable,
                base_url="https://example.test/docs/",
                source_run_id=source_run_id,
            )
        )

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
