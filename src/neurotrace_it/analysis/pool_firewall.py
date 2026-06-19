"""Three-pool leakage firewall (REDESIGN_v5 §3.1, MF-4, R3-B4) -- ADDITIVE, DO-NOT-RUN.

Partition the candidate corpus by a persisted, stratified hash into three
EXAMPLE-DISJOINT pools, LOCKED for every confirmatory estimand:

* ``P_train`` -- learn ``psi`` (fit the frozen maps ``B_lambda, b_lambda_Y``,
  the cross-fit ``beta_T``, the NAIT directions, the LOCI centroids).
  **``Y_obs`` is computed HERE** for the R0 regression.
* ``P_val``   -- estimate the pool-conditional policy values ``V(pi_arm)`` and
  validate ``J``. ``psi`` on ``P_val`` uses the FROZEN ``B_lambda`` (no refit).
  ``U_train`` lives here (and on ``P_dep`` eval), never ``Y_obs``.
* ``P_dep``   -- held-out deployment / compute-matched comparison. No ``P_dep``
  outcome may enter the ``P_dep`` decision path.

The leakage assertions this module enforces:

1. The three pools are disjoint and cover the corpus (locked partition).
2. ``Y_obs`` is read only on ``P_train``; ``U_train`` only on ``P_val`` /
   ``P_dep`` eval -- never conflated (B4).
3. ``psi`` on ``P_val`` / ``P_dep`` uses the FROZEN ``B_lambda`` (no in-sample
   refit / no in-sample-only residualization).
4. No ``P_dep`` outcome appears in the ``P_dep`` decision path.

An optional ``regenerate_partition(g)`` supports the §3.5a NON-confirmatory
multi-pool generalization probe (with its own Holm accounting), never the locked
confirmatory partition.

DO-NOT-RUN: pure stdlib; no model load, no server call. ``server.authorized``
stays ``false``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = [
    "PoolPartition",
    "LeakageReport",
    "split_pools",
    "assert_no_leakage",
    "assert_frozen_residualization",
    "assert_no_dep_outcome_in_decision",
    "regenerate_partition",
    "POOLS",
]

POOLS: tuple[str, ...] = ("P_train", "P_val", "P_dep")


def _hash_bucket(example_id: str, stratum: str, salt: str, n_buckets: int) -> int:
    """Stable hash bucket in ``[0, n_buckets)`` for a (stratum, example) pair."""

    key = f"{salt}|{stratum}|{example_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % n_buckets


@dataclass(frozen=True)
class PoolPartition:
    """A locked three-pool partition (§3.1)."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    dep: tuple[str, ...]
    salt: str
    fractions: tuple[float, float, float]  # (train, val, dep)
    partition_hash: str

    def pool_of(self, example_id: str) -> str | None:
        if example_id in set(self.train):
            return "P_train"
        if example_id in set(self.val):
            return "P_val"
        if example_id in set(self.dep):
            return "P_dep"
        return None


def split_pools(
    example_ids: Sequence[str],
    strata: Mapping[str, str],
    *,
    salt: str,
    fractions: tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> PoolPartition:
    """Disjoint, stratified, hash-based three-pool split (§3.1, locked partition).

    Each example is assigned to ``P_train`` / ``P_val`` / ``P_dep`` by a stable
    hash bucket within its capability stratum, so the split is reproducible
    (persisted ``salt``) and balanced per stratum. ``fractions`` must sum to 1.

    The partition is LOCKED: ``partition_hash`` recomputes from the sorted
    assignment so any drift is detectable, and every confirmatory estimand is
    conditional on this single fixed partition (R3-B4).
    """

    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("pool fractions must sum to 1")
    f_train, f_val, _ = fractions
    n_buckets = 10_000
    cut_train = int(round(f_train * n_buckets))
    cut_val = cut_train + int(round(f_val * n_buckets))

    train: list[str] = []
    val: list[str] = []
    dep: list[str] = []
    for ex in example_ids:
        stratum = strata.get(ex, "_default")
        bucket = _hash_bucket(ex, stratum, salt, n_buckets)
        if bucket < cut_train:
            train.append(ex)
        elif bucket < cut_val:
            val.append(ex)
        else:
            dep.append(ex)

    train_t, val_t, dep_t = tuple(sorted(train)), tuple(sorted(val)), tuple(sorted(dep))
    payload = "|".join(
        [salt, ",".join(train_t), ",".join(val_t), ",".join(dep_t)]
    ).encode("utf-8")
    phash = hashlib.sha256(payload).hexdigest()
    return PoolPartition(
        train=train_t,
        val=val_t,
        dep=dep_t,
        salt=salt,
        fractions=fractions,
        partition_hash=phash,
    )


@dataclass(frozen=True)
class LeakageReport:
    """Result of the firewall leakage assertions (§3.1)."""

    disjoint: bool
    covers_corpus: bool
    errors: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return self.disjoint and self.covers_corpus and not self.errors


def assert_no_leakage(
    partition: PoolPartition,
    *,
    corpus: Sequence[str] | None = None,
) -> LeakageReport:
    """Assert the three pools are disjoint and (optionally) cover ``corpus``."""

    errors: list[str] = []
    s_train, s_val, s_dep = set(partition.train), set(partition.val), set(partition.dep)
    overlaps = (s_train & s_val) | (s_train & s_dep) | (s_val & s_dep)
    disjoint = not overlaps
    if overlaps:
        errors.append(f"pools overlap on {sorted(overlaps)[:8]}")

    covers = True
    if corpus is not None:
        all_pool = s_train | s_val | s_dep
        missing = set(corpus) - all_pool
        extra = all_pool - set(corpus)
        covers = not missing and not extra
        if missing:
            errors.append(f"corpus examples missing from all pools: {sorted(missing)[:8]}")
        if extra:
            errors.append(f"pool examples not in corpus: {sorted(extra)[:8]}")

    return LeakageReport(disjoint=disjoint, covers_corpus=covers, errors=tuple(errors))


def assert_frozen_residualization(
    *,
    pool: str,
    used_frozen_B_lambda: bool,
    refit_in_sample: bool,
) -> None:
    """Assert ``psi`` on ``P_val`` / ``P_dep`` uses the FROZEN ``B_lambda`` (no refit).

    Raises if an in-sample-only residualization (a refit ``B_lambda``) is used on
    a non-training pool -- that would leak the validation/deployment outcome into
    the nuisance map (§3.1, R2-B2).
    """

    if pool in ("P_val", "P_dep"):
        if refit_in_sample or not used_frozen_B_lambda:
            raise AssertionError(
                f"{pool}: psi must use the FROZEN B_lambda; in-sample refit is forbidden"
            )


def assert_no_dep_outcome_in_decision(
    *,
    decision_inputs: Sequence[str],
    dep_outcome_keys: Sequence[str],
) -> None:
    """Assert no ``P_dep`` outcome feature appears in the ``P_dep`` decision path.

    ``decision_inputs`` are the feature/source keys feeding a ``P_dep`` decision;
    ``dep_outcome_keys`` are the held-out deployment outcomes. Their intersection
    must be empty.
    """

    leak = set(decision_inputs) & set(dep_outcome_keys)
    if leak:
        raise AssertionError(
            f"P_dep outcome(s) {sorted(leak)} leaked into the P_dep decision path"
        )


def assert_outcome_pool_discipline(
    *,
    y_obs_pool: str,
    u_train_pool: str,
) -> None:
    """Assert ``Y_obs`` is read only on ``P_train`` and ``U_train`` only on ``P_val``/``P_dep`` (B4)."""

    if y_obs_pool != "P_train":
        raise AssertionError(f"Y_obs must be computed on P_train, got {y_obs_pool!r}")
    if u_train_pool not in ("P_val", "P_dep"):
        raise AssertionError(
            f"U_train must be scored on P_val/P_dep, got {u_train_pool!r}"
        )


def regenerate_partition(
    g: int,
    example_ids: Sequence[str],
    strata: Mapping[str, str],
    *,
    base_salt: str,
    fractions: tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> PoolPartition:
    """Regenerate a partition for the §3.5a NON-confirmatory multi-pool probe.

    Returns a DIFFERENT partition keyed by ``g`` (a fresh salt), used ONLY by the
    exploratory generalization probe with its own Holm accounting -- never the
    locked confirmatory partition.
    """

    return split_pools(
        example_ids, strata, salt=f"{base_salt}#probe{g}", fractions=fractions
    )


# Re-export for convenience.
__all__.append("assert_outcome_pool_discipline")
