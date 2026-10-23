"""Experiment logging: CSV files always, Weights & Biases (offline) optionally."""

from __future__ import annotations

import csv
import json
from pathlib import Path


class Tracker:
    def __init__(self, out_dir: str | Path, enabled: bool, config: dict, use_wandb: bool = False,
                 run_name: str = "run"):
        self.enabled = enabled
        self.out_dir = Path(out_dir)
        self._files: dict[str, tuple] = {}
        self._wandb = None
        if not enabled:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
        if use_wandb:
            try:
                import wandb

                self._wandb = wandb.init(project="envtok-mace", name=run_name, config=config,
                                         dir=str(self.out_dir), mode="offline", resume="allow")
            except Exception as exc:  # logging must never stop training
                print(f"wandb disabled: {exc}", flush=True)

    def log(self, table: str, row: dict) -> None:
        if not self.enabled:
            return
        if table not in self._files:
            path = self.out_dir / f"{table}.csv"
            fieldnames = list(row.keys())
            new = not path.exists() or path.stat().st_size == 0
            if not new:
                # appending after a restart: keep the columns of the existing file
                with open(path, newline="") as f:
                    fieldnames = next(csv.reader(f))
            fh = open(path, "a", newline="")
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore", restval="")
            if new:
                writer.writeheader()
            self._files[table] = (fh, writer)
        fh, writer = self._files[table]
        writer.writerow(row)
        fh.flush()
        if self._wandb is not None:
            self._wandb.log({f"{table}/{k}": v for k, v in row.items() if isinstance(v, (int, float))})

    def close(self) -> None:
        for fh, _ in self._files.values():
            fh.close()
        if self._wandb is not None:
            self._wandb.finish()
