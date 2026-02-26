# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Tests to verify no packages are sourced from repo.anaconda.com (defaults channel).

These tests validate that the container image only uses conda-forge packages
and does not include any packages from the Anaconda defaults channel, which
requires a commercial license for organizational use.
"""
import subprocess
import unittest


class TestAnacondaPackages(unittest.TestCase):
    """Verify no packages from repo.anaconda.com are installed."""

    def test_no_anaconda_default_channel_packages(self):
        """Ensure no conda packages are sourced from repo.anaconda.com."""
        result = subprocess.run(
            ["conda", "list", "--explicit"],
            capture_output=True,
            text=True,
        )
        # Skip if conda is not installed
        if result.returncode != 0:
            self.skipTest("conda not available")

        offending = [
            line for line in result.stdout.splitlines() if "repo.anaconda.com" in line
        ]
        self.assertEqual(
            offending,
            [],
            f"Found {len(offending)} packages from repo.anaconda.com "
            f"(defaults channel):\n" + "\n".join(offending[:20]),
        )

    def test_defaults_channel_not_configured(self):
        """Ensure the 'defaults' channel is not in conda config."""
        result = subprocess.run(
            ["conda", "config", "--get", "channels"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.skipTest("conda not available")

        self.assertNotIn(
            "defaults",
            result.stdout,
            "The 'defaults' channel should not be configured in conda. "
            "Run: conda config --remove channels defaults",
        )


if __name__ == "__main__":
    unittest.main()
