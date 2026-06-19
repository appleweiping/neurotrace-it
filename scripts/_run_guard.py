"""Shared do-not-run guard, plan-resolution, and STATUS.json checkpoint helpers.

This module is the single place the BUILD-NOW / RUN-LATER hard guard lives. It is
imported by every ``scripts/run_*.py`` and ``scripts/extract_signatures.py`` /
``scripts/eval_gates.py`` entrypoint. It is intentionally PURE PYTHON / STDLIB at
top level: importing it loads NO model, NO GPU, NO heavy dependency. The heavy
work each entrypoint would perform is lazy-imported *inside* the guarded branch
only, which is reached solely when BOTH conditions hold:

    1. the resolved config has ``server.authorized == true``, AND
    2. the operator passed the explicit ``--i-have-authorization`` CLI flag.

Default behaviour (either condition false) is a DRY RUN: print the resolved plan
JSON and exit 0, having loaded no model and touched no GPU. NOTHING runs now and
``server.authorized`` stays false in every committed config, so this repository
executes no experiment, training, extraction, or model load.

All ``${...}`` placeholders referenced by the plans are defined in
``reports/run_packet.md`` Section 0 and resolved from the frozen configs at run
authorization; this module fabricates NO numeric value.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Frozen config locations (read-only here; never written by an entrypoint).
LATTICE_V5 = ROOT / "configs" / "experiments" / "lattice_v5.yaml"
LATTICE_V4 = ROOT / "configs" / "experiments" / "lattice_v4.yaml"
COMPUTE_BUDGET = ROOT / "configs" / "compute" / "first_gate_budget.yaml"
SEEDS_FILE = ROOT / "configs" / "seeds" / "paper_20.txt"

# Default run-tree (created lazily ONLY inside the authorized branch).
RUNS_ROOT = ROOT / "runs"

PRIMARY_CELLS = (
    # The PRIMARY CONFIRMATORY cells, named identically to the paper / run_packet.
    "R0",  # mechanism certificate (endpoint+NAIT-residualized partial-R^2)
    "R1",  # six-arm masked-LoRA interventional routing IUT
    "G2t",  # R2-target IUT over the baseline set
    "G2r",  # R2 retention-drift non-inferiority
    "G2h",  # R2 hallucination-drift non-inferiority
    "G2c",  # R2 cost-gap non-inferiority
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _shallow_yaml_to_dict(text: str) -> dict[str, Any]:
    """Minimal top-level YAML reader (stdlib only; no PyYAML at import time).

    Only used in the DRY-RUN plan to echo a handful of top-level scalar markers
    (``server.authorized`` in particular). It is deliberately conservative: it
    recognises ``key: value`` pairs and one level of nesting for ``server:``. The
    authoritative parse (PyYAML) happens lazily inside the authorized branch.
    """

    server_authorized = False
    in_server = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("server:"):
            in_server = True
            continue
        if in_server:
            if line.startswith(" ") or line.startswith("\t"):
                stripped = line.strip()
                if stripped.startswith("authorized:"):
                    val = stripped.split(":", 1)[1].strip().split("#", 1)[0].strip()
                    server_authorized = val.lower() == "true"
            else:
                in_server = False
    return {"server": {"authorized": server_authorized}}


def config_authorized(config_path: Path) -> bool:
    """Return ``server.authorized`` from a frozen config, defaulting to False."""

    if not config_path.exists():
        return False
    return bool(_shallow_yaml_to_dict(_read_text(config_path))["server"]["authorized"])


@dataclass
class StatusCheckpoint:
    """A resumable STATUS.json checkpoint for one entrypoint's run.

    The manifest cells are idempotent/restartable: a stage records its name in
    ``completed`` only after its output artifact is durably written. ``--resume``
    reloads this file and skips every stage already in ``completed``.
    """

    entrypoint: str
    run_dir: Path
    completed: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    plan_hash: str = ""
    started_at: str = ""
    updated_at: str = ""
    # Honest provenance of WHY this checkpoint exists. A StatusCheckpoint is only
    # ever constructed inside the authorized branch of ``guard`` (BOTH
    # ``server.authorized == true`` AND ``--i-have-authorization``), so for any
    # checkpoint actually flushed to disk this is ``True``. It is wired through
    # from the resolved config rather than hard-coded, so a future authorized run
    # records ``true`` truthfully instead of a stale literal ``false``.
    server_authorized: bool = True

    @property
    def path(self) -> Path:
        return self.run_dir / "STATUS.json"

    @classmethod
    def load_or_new(
        cls,
        entrypoint: str,
        run_dir: Path,
        stages: list[str],
        plan_hash: str,
        *,
        server_authorized: bool = True,
    ) -> "StatusCheckpoint":
        status_path = run_dir / "STATUS.json"
        now = datetime.now(timezone.utc).isoformat()
        if status_path.exists():
            data = json.loads(status_path.read_text(encoding="utf-8"))
            sc = cls(
                entrypoint=entrypoint,
                run_dir=run_dir,
                completed=list(data.get("completed", [])),
                pending=[s for s in stages if s not in set(data.get("completed", []))],
                plan_hash=data.get("plan_hash", plan_hash),
                started_at=data.get("started_at", now),
                updated_at=now,
                server_authorized=bool(server_authorized),
            )
            if sc.plan_hash != plan_hash:
                raise SystemExit(
                    "STATUS.json plan_hash mismatch: the resolved plan changed since "
                    "this run started; refusing to resume into a different plan. "
                    "Start a fresh --run-dir."
                )
            return sc
        return cls(
            entrypoint=entrypoint,
            run_dir=run_dir,
            completed=[],
            pending=list(stages),
            plan_hash=plan_hash,
            started_at=now,
            updated_at=now,
            server_authorized=bool(server_authorized),
        )

    def mark_done(self, stage: str) -> None:
        if stage not in self.completed:
            self.completed.append(stage)
        self.pending = [s for s in self.pending if s != stage]
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.flush()

    def is_done(self, stage: str) -> bool:
        return stage in self.completed

    def run_resumable(
        self,
        stages: "list[str]",
        stage_runners: "dict[str, Any]",
        *,
        resume: bool = False,
    ) -> "list[str]":
        """Drive ``stages`` in order with crash-safe, idempotent resumability.

        THE REAL RESUMABILITY LOOP (REDESIGN_v5 run_packet §6). For each stage in
        order:

        * if ``resume`` and the stage is already in ``completed`` (durable
          ``STATUS.json``), SKIP it -- no recompute;
        * otherwise call ``stage_runners[stage](self)`` to do the work, and only
          AFTER it returns (its artifact durably written by the runner) call
          :meth:`mark_done`, which appends to ``completed`` and flushes
          ``STATUS.json``.

        Because ``mark_done`` flushes only post-artifact, a crash mid-stage leaves
        that stage OUT of ``completed``, so a ``--resume`` re-entry re-runs exactly
        the unfinished stage and skips every finished one -- the idempotent /
        restartable contract, now exercised end-to-end (not just scaffolding). The
        heavy training/extraction runners are supplied by the caller's authorized
        branch; this driver is the dependency-free control loop they plug into.

        Returns the list of stages actually EXECUTED this call (skipped resumed
        stages are excluded), so a caller/test can assert the skip behaviour.
        """

        executed: list[str] = []
        for stage in stages:
            if resume and self.is_done(stage):
                continue
            runner = stage_runners.get(stage)
            if runner is None:
                raise KeyError(f"no stage runner registered for {stage!r}")
            runner(self)
            self.mark_done(stage)
            executed.append(stage)
        return executed

    def flush(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "entrypoint": self.entrypoint,
            "plan_hash": self.plan_hash,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed": self.completed,
            "pending": self.pending,
            # Truthful provenance, not a stale literal: a checkpoint is only ever
            # constructed/flushed inside the authorized branch of ``guard`` (config
            # server.authorized AND --i-have-authorization), so this is the resolved
            # authorization state of that run, correct for a future authorized run.
            "server_authorized": bool(self.server_authorized),
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the flags every entrypoint shares."""

    parser.add_argument(
        "--config",
        type=Path,
        default=LATTICE_V5,
        help="Path to the frozen v5 routing config (default: configs/experiments/lattice_v5.yaml).",
    )
    parser.add_argument(
        "--i-have-authorization",
        action="store_true",
        help=(
            "Explicit operator authorization flag. REQUIRED (together with "
            "server.authorized==true in the config) to leave dry-run. Omitting it "
            "ALWAYS yields a dry run that prints the plan JSON and exits 0."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the STATUS.json checkpoint in --run-dir; skip completed stages.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Output/checkpoint directory (default: runs/<entrypoint>).",
    )


def plan_hash(plan: dict[str, Any]) -> str:
    import hashlib

    blob = json.dumps(plan, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def resolve_run_dir(args: argparse.Namespace, entrypoint: str) -> Path:
    return args.run_dir if args.run_dir is not None else (RUNS_ROOT / entrypoint)


def emit_dry_run(plan: dict[str, Any]) -> int:
    """Print the resolved plan JSON and exit 0 (loads no model/GPU)."""

    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


def guard(
    args: argparse.Namespace, entrypoint: str, plan: dict[str, Any], stages: list[str]
) -> "StatusCheckpoint | None":
    """The HARD GUARD.

    Returns ``None`` (and the caller must ``return emit_dry_run(plan)``) unless
    BOTH ``server.authorized == true`` in the resolved config AND
    ``--i-have-authorization`` were supplied. Only in that case is a
    :class:`StatusCheckpoint` returned and the caller may enter the guarded,
    heavy-import branch.
    """

    cfg_authorized = config_authorized(args.config)
    plan["server_authorized"] = cfg_authorized
    plan["i_have_authorization_flag"] = bool(args.i_have_authorization)
    plan["will_run"] = bool(cfg_authorized and args.i_have_authorization)

    if not plan["will_run"]:
        reason = []
        if not cfg_authorized:
            reason.append(f"server.authorized is false in {args.config.name}")
        if not args.i_have_authorization:
            reason.append("--i-have-authorization not supplied")
        plan["dry_run_reason"] = "; ".join(reason)
        return None

    run_dir = resolve_run_dir(args, entrypoint)
    ph = plan_hash(plan)
    # ``cfg_authorized`` is necessarily True here (we returned None above otherwise),
    # so the checkpoint records the resolved authorization state truthfully.
    return StatusCheckpoint.load_or_new(
        entrypoint, run_dir, stages, ph, server_authorized=cfg_authorized
    )
