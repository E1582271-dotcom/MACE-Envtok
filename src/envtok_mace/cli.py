"""Command-line entry point: ``envtok-mace <command> ...``."""

from __future__ import annotations

import argparse
import importlib

# command -> (module, help). Each module provides add_arguments(parser) and run(args).
COMMANDS = {
    "extract": ("envtok_mace.data.extract_subset", "pull structures out of OMat24 ASE-LMDB files"),
    "preprocess": ("envtok_mace.data.preprocess", "add neighbour lists to a Frames file"),
    "subset": ("envtok_mace.data.subset", "write a stratified random subset of a Frames file"),
    "ref-eval": ("envtok_mace.eval.reference", "evaluate a reference mace-torch model on a Frames file"),
    "train": ("envtok_mace.train.run", "train a model (one process per GPU)"),
    "eval": ("envtok_mace.eval.run_eval", "score checkpoints / mace-torch models on a Frames file"),
    "bench-speed": ("envtok_mace.bench.speed", "training-step throughput, ours vs mace-torch"),
    "bench-allreduce": ("envtok_mace.bench.allreduce", "all-reduce bandwidth between ranks"),
    "check-backends": ("envtok_mace.bench.backend_check", "e3nn vs cuEquivariance outputs on a GPU"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="envtok-mace")
    sub = parser.add_subparsers(dest="command", required=True)
    modules = {}
    for name, (module, help_text) in COMMANDS.items():
        mod = importlib.import_module(module)
        modules[name] = mod
        mod.add_arguments(sub.add_parser(name, help=help_text))
    args = parser.parse_args(argv)
    modules[args.command].run(args)


if __name__ == "__main__":
    main()
