"""Tests for CLI entrypoint."""

import pytest

from fraudguard.cli import main


def test_cli_help():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_no_args():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 1
