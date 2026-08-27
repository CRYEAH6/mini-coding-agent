"""Smoke tests for the initial command-line interface."""

from mini_agent.cli import main


def test_main_prints_startup_message(capsys) -> None:
    main()

    captured = capsys.readouterr()
    assert captured.out == "Mini Coding Agent is ready for development.\n"

