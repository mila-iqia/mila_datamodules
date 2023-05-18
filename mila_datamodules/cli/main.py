from __future__ import annotations

from simple_parsing import ArgumentParser

from .prepare import add_prepare_arguments, prepare


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser()
    add_prepare_arguments(parser)
    args = parser.parse_args(argv)
    prepare(args)

    # More generic code if we add other commands than `prepare`.
    # commands = {
    #     "prepare": (add_prepare_arguments, prepare),
    # }
    # for command_name, (add_arguments, command) in commands.items():
    #     # subparser = command_subparsers.add_parser(command_name, help=command.__doc__)
    #     add_arguments(parser)
    #     parser.set_defaults(command=command)
    # args = parser.parse_args(argv)
    # command = args.command
    # delattr(args, "command")
    # command(args)
