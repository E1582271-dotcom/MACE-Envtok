"""Command-line entry point: ``envtok-mace <command> ...``."""

from __future__ import annotations

import argparse
import importlib

# command -> (module, help). Each module provides add_arguments(parser) and run(args).
COMMANDS = {
    "extract": ("envtok_mace.data.extract_subset", "pull structures out of OMat24 ASE-LMDB files"),
    "preprocess": ("envtok_mace.data.preprocess", "add neighbour lists to a Frames file"),
    "subset": ("envtok_mace.data.subset", "write a stratified random subset of a Frames file"),
    "matpes-build": ("envtok_mace.data.matpes", "convert the MatPES release into a Frames file"),
    "matpes": ("envtok_mace.bench.matpes", "score a model on MatPES and build the paper's tables"),
    "matcalc": ("envtok_mace.bench.matcalc", "MatCalc-Bench (formation energy, moduli, C_V, softening)"),
    "stability": ("envtok_mace.bench.stability", "MLIP-Arena NPT compression MD stability"),
    "nebdft2k": ("envtok_mace.bench.nebdft2k", "LiTraj nebDFT2k Li-ion migration barriers"),
    "latent": ("envtok_mace.analysis.latent", "latent-space distance from the training data vs force error"),
    "id-ood": ("envtok_mace.analysis.id_ood", "one table of all benchmarks grouped ID / shift / OOD"),
    "learning-curves": ("envtok_mace.analysis.learning", "train vs val loss, and errors per epoch"),
    "accuracy": ("envtok_mace.analysis.accuracy", "parity histograms and error breakdowns from predictions"),
    "figures": ("envtok_mace.plots.figures", "draw every figure from results/ (SVG, PDF, PNG)"),
    "ref-eval": ("envtok_mace.eval.reference", "evaluate a reference mace-torch model on a Frames file"),
    "train": ("envtok_mace.train.run", "train a model (one process per GPU)"),
    "eval": ("envtok_mace.eval.run_eval", "score checkpoints / mace-torch models on a Frames file"),
    "export-weights": ("envtok_mace.eval.export", "write a checkpoint as a portable (e3nn) weights file"),
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
