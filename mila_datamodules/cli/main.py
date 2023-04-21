from __future__ import annotations

from simple_parsing import ArgumentParser

from .prepare import add_prepare_arguments, prepare


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser()

    command_subparsers = parser.add_subparsers(
        dest="command_name",
        title="Command title",
        description="Description",
        required=True,
    )

    commands = {
        "prepare": (add_prepare_arguments, prepare),
    }

    for command_name, (add_arguments, command) in commands.items():
        subparser = command_subparsers.add_parser(command_name, help=command.__doc__)
        add_arguments(subparser)
        subparser.set_defaults(command=command)

    args = parser.parse_args(argv)
    args.command(args)
