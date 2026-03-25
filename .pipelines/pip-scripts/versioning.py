"""Script for handling release/dev package versioning in the PyPI release pipeline."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
import re
import argparse
import logging

LOG = logging.getLogger(__name__)


def validate_dev_tag(dev_tag: str) -> bool:
    """
        Validate the dev tag against PyPA version specifications. PyPA expects that the final version identifier
        is compliant with the following regex expression: [N!]N(.N)*[{a|b|rc}N][.postN][.devN]. Development tags
        may take the form of devN|rcN|aN|bN|postN, where N is an integer. For example: dev0, rc1, a2, b3, etc.
    Args:
        dev_tag (str): The development tag to validate. This should not include the major/minor version of the package.
    Returns:
        bool: True if the dev tag is valid according to PyPA specifications, False otherwise.
    """
    # Remove erroneous characeters from string
    tag = dev_tag.strip()

    # Pre-release segments
    if re.fullmatch(r"(?:a|b|rc|dev|post)\d+", tag) is None:
        return False
    else:
        return True


def validate_version_string(version_str: str) -> bool:
    """
        Validate the version string against PyPA version specifications. PyPA expects that the final version identifier
        is compliant with the following regex expression: [N!]N(.N)*[{a|b|rc}N][.postN][.devN]. For example, valid version strings include: 1.0.0, 2.1.3.dev1, 0.9.0rc2, etc.
    Args:
        version_str (str): The full version string to validate, including major/minor version and any optional dev tags.
    Returns:
        bool: True if the version string is valid according to PyPA specifications, False otherwise.
    """
    regex_filter = re.compile(
        r"""
        ^
        (?:\d+!)?                      # optional epoch: N!
        \d+(?:\.\d+)*                  # release: N(.N)*
        (?:(?:a|b|rc)\d+)?             # optional pre-release: aN | bN | rcN
        (?:\.post\d+)?                 # optional post-release: .postN
        (?:\.dev\d+)?                  # optional dev-release: .devN
        $
        """,
        re.VERBOSE,
    )
    if regex_filter.match(version_str) is None:
        return False
    else:
        return True


def update_version_file(version_file: str, new_version_string: str) -> None:
    """
    Utility function to update the version file with the new version string.
    Args:
        version_file (str): The path to the version file to update.
        new_version_string (str): The new version string to write to the version file.
    """
    is_valid_version = validate_version_string(new_version_string)
    if not is_valid_version:
        LOG.error(f"""
            Invalid version string: {new_version_string}.\n
            Please ensure that the new version string is compliant with PyPA version specifications.
            The version string should be in the form of X.Y.Z[.devN|rcN|aN|bN|postN], where X, Y, Z, and N are integers. For example: 1.0.0, 2.1.3.dev1, 0.9.0rc2, etc.
        """)
        exit(1)
    else:
        with open(version_file, "w") as f:
            f.write(new_version_string)

    LOG.info(
        f"Updated version file at {version_file} with new version: {new_version_string}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Update version numbers in setup.py and __init__.py"
    )
    parser.add_argument(
        "--dev-tag",
        dest="dev_tag",
        help="Optional dev tag identifier, e.g. 'dev1'",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--version-file",
        dest="version_file",
        help="Path to the 'VERSION' file in the top level directory.",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "VERSION"),
    )

    args = parser.parse_args()

    # Check if dev tag was passed from pipeline and validate against PyPA version specifications
    dev_tag_is_valid = validate_dev_tag(args.dev_tag)
    if not dev_tag_is_valid and args.dev_tag is not None:
        LOG.error(f"""
            Invalid dev tag provided for the wheels: {args.dev_tag}.\n
            Please ensure that the dev tag provided in the parameters section of the pipeline is compliant with PyPA version specifications.
            The dev tag should take the form of devN|rcN|aN|bN|postN, where N is an integer. For example: dev0, rc1, a2, b3, etc.
        """)
        exit(1)
    elif dev_tag_is_valid and args.dev_tag is not None:
        LOG.info(f"Dev tag provided for the wheels: {args.dev_tag} is valid.")
        LOG.info(
            f"New version to be published on PyPI will be suffixed with the dev tag: {args.dev_tag}"
        )

    current_version = open(args.version_file).read().strip()
    LOG.info(f"Current version read from version file: {current_version}")

    if args.dev_tag:
        new_version = f"{current_version}.{args.dev_tag}"
        update_version_file(args.version_file, new_version)

    LOG.info(f"New version to be published on PyPI: {new_version}")


if __name__ == "__main__":
    main()
