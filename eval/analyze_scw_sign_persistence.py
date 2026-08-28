#!/usr/bin/env python3
"""
Direct analysis of innovation-sign persistence and SCW trigger dynamics.

This script analyzes the packed recurrent models used for the manuscript:

1. P3 with B4 + SCW K4 and theta = 0.
2. Native B4 with B4 + SCW K4 and theta = Delta_B / 8.
3. Native B8 forced to B4 recurrent state with B4 + SCW K4 and
   theta = Delta_B / 8.

For every condition the script:

- loads the exact packed QKeras checkpoint;
- copies the trained recurrent and dense weights into the repository SCW core;
- validates native deterministic equivalence before changing state precision;
- traces deterministic-B4 and SCW decoder trajectories on the complete
  160,000-sample held-out test partition;
- validates recurrent statistics against the manuscript values;
- measures one-step innovation-sign persistence;
- measures same-sign run-length distributions;
- measures SCW trigger probability as a function of current same-sign run;
- measures active votes, trigger visibility, votes since reset, and directional
  consistency at trigger;
- bootstraps sequence-level sign-transition statistics;
- writes JSON, CSV, NPZ, PDF, SVG, and PNG outputs.

No training or weight updates are performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from train_student_vanilla_kd import build_student, find_data_files
from train_student_vanilla_kd_scw import SCWStudentModel


EXPECTED_TEST_SIZE = 160_000
SEQ_LEN = 135
N_OUT = 3
STUDENT_UNITS = 32
Q_ALPHA = 1.0
STATE_BITS = 4
LIVE_STEPS = SEQ_LEN - 1
EQUIVALENCE_TOLERANCE = 5.0e-5
MANUSCRIPT_METRIC_TOLERANCE = 5.0e-5


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    display_name: str
    checkpoint: Path
    source_bits: int
    counter_bits: int
    deadzone_fraction: float
    color_hex: str
    expected_deadband_fraction: float
    expected_deterministic_state_change_fraction: float
    expected_scw_subthreshold_visible_write_fraction: float


class ModeAccumulator:
    """
    Streaming accumulator for one condition and one state operator.

    The run statistic is intentionally stricter than the SCW counter itself.
    A same-sign run requires consecutive decoder write opportunities that both
    cast an eligible vote with the same sign. A no-vote step ends the contiguous
    run, although the actual SCW counter is retained across that no-vote step.
    Counter-segment statistics are tracked separately and reproduce the actual
    SCW semantics: the segment survives no-vote steps and resets only on a
    normal write or an SCW trigger.
    """

    def __init__(
        self,
        condition: ConditionSpec,
        operator_mode: str,
        n_sequences: int,
        live_steps: int,
        units: int,
    ) -> None:
        if operator_mode not in ("deterministic", "scw"):
            raise ValueError(
                f"Unsupported operator_mode={operator_mode!r}"
            )

        self.condition = condition
        self.operator_mode = operator_mode
        self.n_sequences = int(n_sequences)
        self.live_steps = int(live_steps)
        self.units = int(units)

        self.total_elements = 0
        self.subthreshold_elements = 0
        self.state_changes = 0
        self.subthreshold_visible_writes = 0
        self.active_votes = 0
        self.triggers = 0
        self.visible_triggers = 0
        self.rail_blocked_triggers = 0
        self.normal_events = 0

        self.transition_counts = np.zeros(
            (n_sequences, 4),
            dtype=np.int64,
        )

        self.active_votes_by_sequence = np.zeros(
            n_sequences,
            dtype=np.int64,
        )

        self.triggers_by_sequence = np.zeros(
            n_sequences,
            dtype=np.int64,
        )

        self.visible_triggers_by_sequence = np.zeros(
            n_sequences,
            dtype=np.int64,
        )

        self.normal_events_by_sequence = np.zeros(
            n_sequences,
            dtype=np.int64,
        )

        self.subthreshold_by_sequence = np.zeros(
            n_sequences,
            dtype=np.int64,
        )

        self.subthreshold_visible_writes_by_sequence = np.zeros(
            n_sequences,
            dtype=np.int64,
        )

        self.state_changes_by_sequence = np.zeros(
            n_sequences,
            dtype=np.int64,
        )

        self.max_same_sign_run_by_sequence = np.zeros(
            n_sequences,
            dtype=np.int16,
        )

        hist_len = live_steps + 2

        self.completed_run_hist = np.zeros(
            hist_len,
            dtype=np.int64,
        )

        self.active_event_run_hist = np.zeros(
            hist_len,
            dtype=np.int64,
        )

        self.trigger_run_hist = np.zeros(
            hist_len,
            dtype=np.int64,
        )

        self.trigger_visible_run_hist = np.zeros(
            hist_len,
            dtype=np.int64,
        )

        self.trigger_segment_vote_hist = np.zeros(
            hist_len,
            dtype=np.int64,
        )

        self.trigger_consistency_sum = 0.0
        self.trigger_consistency_count = 0
        self.trigger_sign_changes_sum = 0
        self.trigger_segment_votes_sum = 0

        self._next_sequence_offset = 0

    @staticmethod
    def _add_hist_values(
        hist: np.ndarray,
        values: np.ndarray,
    ) -> None:
        values = np.asarray(
            values,
            dtype=np.int64,
        )

        if values.size == 0:
            return

        if np.any(values <= 0):
            raise RuntimeError(
                "Run-length histogram received a non-positive value."
            )

        max_value = int(
            np.max(values)
        )

        if max_value >= hist.size:
            raise RuntimeError(
                f"Run length {max_value} exceeds histogram capacity "
                f"{hist.size - 1}."
            )

        hist += np.bincount(
            values,
            minlength=hist.size,
        )[: hist.size]

    def add_batch(
        self,
        trace: Mapping[str, np.ndarray],
    ) -> None:
        required = (
            "delta_raw",
            "normal",
            "subthreshold",
            "active_vote",
            "vote",
            "trigger",
            "visible_change",
            "counter_before",
            "counter_after",
        )

        missing = [
            key
            for key in required
            if key not in trace
        ]

        if missing:
            raise KeyError(
                f"Missing trace tensors: {missing}"
            )

        delta_raw = np.asarray(
            trace["delta_raw"],
            dtype=np.float32,
        )

        normal = np.asarray(
            trace["normal"],
            dtype=bool,
        )

        subthreshold = np.asarray(
            trace["subthreshold"],
            dtype=bool,
        )

        active_vote = np.asarray(
            trace["active_vote"],
            dtype=bool,
        )

        vote = np.asarray(
            trace["vote"],
            dtype=np.int8,
        )

        trigger = np.asarray(
            trace["trigger"],
            dtype=bool,
        )

        visible_change = np.asarray(
            trace["visible_change"],
            dtype=bool,
        )

        counter_before = np.asarray(
            trace["counter_before"],
            dtype=np.float32,
        )

        counter_after = np.asarray(
            trace["counter_after"],
            dtype=np.float32,
        )

        expected_shape = (
            delta_raw.shape[0],
            self.live_steps,
            self.units,
        )

        for key, value in (
            ("normal", normal),
            ("subthreshold", subthreshold),
            ("active_vote", active_vote),
            ("vote", vote),
            ("trigger", trigger),
            ("visible_change", visible_change),
            ("counter_before", counter_before),
            ("counter_after", counter_after),
        ):
            if value.shape != expected_shape:
                raise RuntimeError(
                    f"{self.condition.key}/{self.operator_mode}: "
                    f"{key} shape {value.shape} does not match "
                    f"{expected_shape}."
                )

        batch_size = expected_shape[0]
        start = self._next_sequence_offset
        stop = start + batch_size

        if stop > self.n_sequences:
            raise RuntimeError(
                f"{self.condition.key}/{self.operator_mode}: "
                "received more sequences than allocated."
            )

        if np.any(
            normal
            & subthreshold
        ):
            raise RuntimeError(
                "normal and subthreshold masks overlap."
            )

        if np.any(
            ~(
                normal
                | subthreshold
            )
        ):
            raise RuntimeError(
                "normal and subthreshold masks are not exhaustive."
            )

        if np.any(
            active_vote
            & ~subthreshold
        ):
            raise RuntimeError(
                "active_vote occurred outside the subthreshold region."
            )

        if np.any(
            trigger
            & ~active_vote
        ):
            raise RuntimeError(
                "SCW trigger occurred without an active vote."
            )

        if np.any(
            active_vote
            & (
                vote
                == 0
            )
        ):
            raise RuntimeError(
                "An active vote has zero sign."
            )

        if (
            self.operator_mode
            == "deterministic"
            and np.any(trigger)
        ):
            raise RuntimeError(
                "A deterministic trace contains SCW triggers."
            )

        self.total_elements += int(
            np.prod(
                expected_shape
            )
        )

        self.subthreshold_elements += int(
            np.count_nonzero(
                subthreshold
            )
        )

        self.state_changes += int(
            np.count_nonzero(
                visible_change
            )
        )

        self.subthreshold_visible_writes += int(
            np.count_nonzero(
                subthreshold
                & visible_change
            )
        )

        self.active_votes += int(
            np.count_nonzero(
                active_vote
            )
        )

        self.triggers += int(
            np.count_nonzero(
                trigger
            )
        )

        self.visible_triggers += int(
            np.count_nonzero(
                trigger
                & visible_change
            )
        )

        self.rail_blocked_triggers += int(
            np.count_nonzero(
                trigger
                & ~visible_change
            )
        )

        self.normal_events += int(
            np.count_nonzero(
                normal
            )
        )

        (
            transition_counts,
            batch_metrics,
        ) = self._analyze_temporal_structure(
            active_vote=active_vote,
            vote=vote,
            normal=normal,
            trigger=trigger,
            visible_change=visible_change,
            counter_before=counter_before,
            counter_after=counter_after,
        )

        self.transition_counts[
            start:stop,
            :,
        ] = transition_counts

        self.active_votes_by_sequence[
            start:stop
        ] = batch_metrics[
            "active_votes"
        ]

        self.triggers_by_sequence[
            start:stop
        ] = batch_metrics[
            "triggers"
        ]

        self.visible_triggers_by_sequence[
            start:stop
        ] = batch_metrics[
            "visible_triggers"
        ]

        self.normal_events_by_sequence[
            start:stop
        ] = batch_metrics[
            "normal_events"
        ]

        self.max_same_sign_run_by_sequence[
            start:stop
        ] = batch_metrics[
            "max_run"
        ]

        self.subthreshold_by_sequence[
            start:stop
        ] = np.sum(
            subthreshold,
            axis=(
                1,
                2,
            ),
            dtype=np.int64,
        )

        self.subthreshold_visible_writes_by_sequence[
            start:stop
        ] = np.sum(
            subthreshold
            & visible_change,
            axis=(
                1,
                2,
            ),
            dtype=np.int64,
        )

        self.state_changes_by_sequence[
            start:stop
        ] = np.sum(
            visible_change,
            axis=(
                1,
                2,
            ),
            dtype=np.int64,
        )

        self._next_sequence_offset = stop

    def _analyze_temporal_structure(
        self,
        active_vote: np.ndarray,
        vote: np.ndarray,
        normal: np.ndarray,
        trigger: np.ndarray,
        visible_change: np.ndarray,
        counter_before: np.ndarray,
        counter_after: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        Dict[str, np.ndarray],
    ]:
        batch_size = active_vote.shape[0]
        units = active_vote.shape[2]

        prev_active = np.zeros(
            (
                batch_size,
                units,
            ),
            dtype=bool,
        )

        prev_vote = np.zeros(
            (
                batch_size,
                units,
            ),
            dtype=np.int8,
        )

        run_len = np.zeros(
            (
                batch_size,
                units,
            ),
            dtype=np.int16,
        )

        segment_vote_count = np.zeros(
            (
                batch_size,
                units,
            ),
            dtype=np.int16,
        )

        segment_signed_sum = np.zeros(
            (
                batch_size,
                units,
            ),
            dtype=np.int16,
        )

        segment_sign_changes = np.zeros(
            (
                batch_size,
                units,
            ),
            dtype=np.int16,
        )

        segment_last_vote = np.zeros(
            (
                batch_size,
                units,
            ),
            dtype=np.int8,
        )

        transitions = np.zeros(
            (
                batch_size,
                4,
            ),
            dtype=np.int64,
        )

        active_by_sequence = np.zeros(
            batch_size,
            dtype=np.int64,
        )

        triggers_by_sequence = np.zeros(
            batch_size,
            dtype=np.int64,
        )

        visible_triggers_by_sequence = np.zeros(
            batch_size,
            dtype=np.int64,
        )

        normal_by_sequence = np.zeros(
            batch_size,
            dtype=np.int64,
        )

        max_run_by_sequence = np.zeros(
            batch_size,
            dtype=np.int16,
        )

        for t in range(
            self.live_steps
        ):
            active_t = active_vote[
                :,
                t,
                :,
            ]

            vote_t = vote[
                :,
                t,
                :,
            ]

            normal_t = normal[
                :,
                t,
                :,
            ]

            trigger_t = trigger[
                :,
                t,
                :,
            ]

            changed_t = visible_change[
                :,
                t,
                :,
            ]

            counter_before_t = counter_before[
                :,
                t,
                :,
            ]

            counter_after_t = counter_after[
                :,
                t,
                :,
            ]

            active_by_sequence += np.sum(
                active_t,
                axis=1,
                dtype=np.int64,
            )

            triggers_by_sequence += np.sum(
                trigger_t,
                axis=1,
                dtype=np.int64,
            )

            visible_triggers_by_sequence += np.sum(
                trigger_t
                & changed_t,
                axis=1,
                dtype=np.int64,
            )

            normal_by_sequence += np.sum(
                normal_t,
                axis=1,
                dtype=np.int64,
            )

            if (
                self.operator_mode
                == "scw"
            ):
                expected_before = (
                    segment_signed_sum.astype(
                        np.float32
                    )
                )

                if not np.allclose(
                    counter_before_t,
                    expected_before,
                    atol=1.0e-6,
                    rtol=0.0,
                ):
                    diff = float(
                        np.max(
                            np.abs(
                                counter_before_t
                                - expected_before
                            )
                        )
                    )

                    raise RuntimeError(
                        f"{self.condition.key}/scw: "
                        f"counter reconstruction mismatch "
                        f"before decoder step {t}, "
                        f"max_abs={diff:.8g}."
                    )

            else:
                if not np.allclose(
                    counter_before_t,
                    0.0,
                    atol=1.0e-6,
                    rtol=0.0,
                ):
                    diff = float(
                        np.max(
                            np.abs(
                                counter_before_t
                            )
                        )
                    )

                    raise RuntimeError(
                        f"{self.condition.key}/deterministic: "
                        f"nonzero counter before decoder "
                        f"step {t}, max_abs={diff:.8g}."
                    )

            pair = (
                active_t
                & prev_active
            )

            same = (
                pair
                & (
                    vote_t
                    == prev_vote
                )
            )

            pp = (
                pair
                & (
                    prev_vote
                    > 0
                )
                & (
                    vote_t
                    > 0
                )
            )

            pn = (
                pair
                & (
                    prev_vote
                    > 0
                )
                & (
                    vote_t
                    < 0
                )
            )

            np_mask = (
                pair
                & (
                    prev_vote
                    < 0
                )
                & (
                    vote_t
                    > 0
                )
            )

            nn = (
                pair
                & (
                    prev_vote
                    < 0
                )
                & (
                    vote_t
                    < 0
                )
            )

            transitions[
                :,
                0,
            ] += np.sum(
                pp,
                axis=1,
                dtype=np.int64,
            )

            transitions[
                :,
                1,
            ] += np.sum(
                pn,
                axis=1,
                dtype=np.int64,
            )

            transitions[
                :,
                2,
            ] += np.sum(
                np_mask,
                axis=1,
                dtype=np.int64,
            )

            transitions[
                :,
                3,
            ] += np.sum(
                nn,
                axis=1,
                dtype=np.int64,
            )

            break_old = (
                (
                    run_len
                    > 0
                )
                & (
                    ~active_t
                    | (
                        active_t
                        & prev_active
                        & (
                            vote_t
                            != prev_vote
                        )
                    )
                )
            )

            self._add_hist_values(
                self.completed_run_hist,
                run_len[
                    break_old
                ],
            )

            continuing = (
                active_t
                & prev_active
                & same
            )

            current_run = np.zeros_like(
                run_len
            )

            current_run[
                continuing
            ] = (
                run_len[
                    continuing
                ]
                + 1
            )

            current_run[
                active_t
                & ~continuing
            ] = 1

            self._add_hist_values(
                self.active_event_run_hist,
                current_run[
                    active_t
                ],
            )

            max_run_by_sequence = np.maximum(
                max_run_by_sequence,
                np.max(
                    current_run,
                    axis=1,
                ),
            )

            if (
                self.operator_mode
                == "scw"
            ):
                if np.any(
                    normal_t
                ):
                    segment_vote_count[
                        normal_t
                    ] = 0

                    segment_signed_sum[
                        normal_t
                    ] = 0

                    segment_sign_changes[
                        normal_t
                    ] = 0

                    segment_last_vote[
                        normal_t
                    ] = 0

                if np.any(
                    active_t
                ):
                    old_last = (
                        segment_last_vote.copy()
                    )

                    old_count = (
                        segment_vote_count.copy()
                    )

                    segment_vote_count[
                        active_t
                    ] += 1

                    segment_signed_sum[
                        active_t
                    ] += vote_t[
                        active_t
                    ]

                    sign_change = (
                        active_t
                        & (
                            old_count
                            > 0
                        )
                        & (
                            old_last
                            != 0
                        )
                        & (
                            vote_t
                            != old_last
                        )
                    )

                    segment_sign_changes[
                        sign_change
                    ] += 1

                    segment_last_vote[
                        active_t
                    ] = vote_t[
                        active_t
                    ]

                if np.any(
                    trigger_t
                ):
                    trigger_runs = current_run[
                        trigger_t
                    ]

                    self._add_hist_values(
                        self.trigger_run_hist,
                        trigger_runs,
                    )

                    self._add_hist_values(
                        self.completed_run_hist,
                        trigger_runs,
                    )

                    self._add_hist_values(
                        self.trigger_visible_run_hist,
                        current_run[
                            trigger_t
                            & changed_t
                        ],
                    )

                    trigger_segment_votes = (
                        segment_vote_count[
                            trigger_t
                        ]
                    )

                    trigger_segment_sum = (
                        segment_signed_sum[
                            trigger_t
                        ]
                    )

                    trigger_segment_changes = (
                        segment_sign_changes[
                            trigger_t
                        ]
                    )

                    self._add_hist_values(
                        self.trigger_segment_vote_hist,
                        trigger_segment_votes,
                    )

                    consistency = (
                        np.abs(
                            trigger_segment_sum.astype(
                                np.float64
                            )
                        )
                        / trigger_segment_votes.astype(
                            np.float64
                        )
                    )

                    self.trigger_consistency_sum += float(
                        np.sum(
                            consistency
                        )
                    )

                    self.trigger_consistency_count += int(
                        consistency.size
                    )

                    self.trigger_sign_changes_sum += int(
                        np.sum(
                            trigger_segment_changes,
                            dtype=np.int64,
                        )
                    )

                    self.trigger_segment_votes_sum += int(
                        np.sum(
                            trigger_segment_votes,
                            dtype=np.int64,
                        )
                    )

                    segment_vote_count[
                        trigger_t
                    ] = 0

                    segment_signed_sum[
                        trigger_t
                    ] = 0

                    segment_sign_changes[
                        trigger_t
                    ] = 0

                    segment_last_vote[
                        trigger_t
                    ] = 0

                expected_after = (
                    segment_signed_sum.astype(
                        np.float32
                    )
                )

            else:
                expected_after = np.zeros_like(
                    counter_after_t
                )

            if not np.allclose(
                counter_after_t,
                expected_after,
                atol=1.0e-6,
                rtol=0.0,
            ):
                diff = float(
                    np.max(
                        np.abs(
                            counter_after_t
                            - expected_after
                        )
                    )
                )

                raise RuntimeError(
                    f"{self.condition.key}/"
                    f"{self.operator_mode}: "
                    f"counter reconstruction mismatch "
                    f"after decoder step {t}, "
                    f"max_abs={diff:.8g}."
                )

            current_run[
                trigger_t
            ] = 0

            prev_active = (
                active_t
                & ~trigger_t
            )

            prev_vote = np.where(
                prev_active,
                vote_t,
                0,
            ).astype(
                np.int8
            )

            run_len = current_run

        self._add_hist_values(
            self.completed_run_hist,
            run_len[
                run_len
                > 0
            ],
        )

        return (
            transitions,
            {
                "active_votes": active_by_sequence,
                "triggers": triggers_by_sequence,
                "visible_triggers": (
                    visible_triggers_by_sequence
                ),
                "normal_events": normal_by_sequence,
                "max_run": max_run_by_sequence,
            },
        )

    def finalize(
        self,
    ) -> None:
        if (
            self._next_sequence_offset
            != self.n_sequences
        ):
            raise RuntimeError(
                f"{self.condition.key}/"
                f"{self.operator_mode}: "
                f"processed "
                f"{self._next_sequence_offset} "
                f"sequences, expected "
                f"{self.n_sequences}."
            )

    def transition_summary(
        self,
    ) -> Dict[str, float]:
        (
            pp,
            pn,
            np_count,
            nn,
        ) = [
            int(x)
            for x in np.sum(
                self.transition_counts,
                axis=0,
                dtype=np.int64,
            ).tolist()
        ]

        total_pairs = (
            pp
            + pn
            + np_count
            + nn
        )

        same_pairs = (
            pp
            + nn
        )

        same_fraction = (
            float(
                same_pairs
                / total_pairs
            )
            if total_pairs
            else float("nan")
        )

        return {
            "pp": pp,
            "pn": pn,
            "np": np_count,
            "nn": nn,
            "adjacent_active_vote_pairs": (
                total_pairs
            ),
            "same_sign_pairs": (
                same_pairs
            ),
            "same_sign_fraction": (
                same_fraction
            ),
            "persistence_odds_ratio": (
                transition_odds_ratio(
                    pp=pp,
                    pn=pn,
                    np_count=np_count,
                    nn=nn,
                )
            ),
        }

    def scalar_summary(
        self,
    ) -> Dict[str, float]:
        if self.total_elements <= 0:
            raise RuntimeError(
                "No traced recurrent elements."
            )

        deadband_fraction = (
            self.subthreshold_elements
            / self.total_elements
        )

        state_change_fraction = (
            self.state_changes
            / self.total_elements
        )

        sub_to_write = (
            self.subthreshold_visible_writes
            / self.subthreshold_elements
            if self.subthreshold_elements
            else float("nan")
        )

        trigger_consistency = (
            self.trigger_consistency_sum
            / self.trigger_consistency_count
            if self.trigger_consistency_count
            else float("nan")
        )

        trigger_votes = (
            self.trigger_segment_votes_sum
            / self.trigger_consistency_count
            if self.trigger_consistency_count
            else float("nan")
        )

        trigger_sign_changes = (
            self.trigger_sign_changes_sum
            / self.trigger_consistency_count
            if self.trigger_consistency_count
            else float("nan")
        )

        return {
            "total_elements": (
                int(
                    self.total_elements
                )
            ),
            "deadband_fraction": (
                float(
                    deadband_fraction
                )
            ),
            "state_change_fraction": (
                float(
                    state_change_fraction
                )
            ),
            "subthreshold_visible_write_fraction": (
                float(
                    sub_to_write
                )
            ),
            "active_votes": (
                int(
                    self.active_votes
                )
            ),
            "normal_events": (
                int(
                    self.normal_events
                )
            ),
            "triggers": (
                int(
                    self.triggers
                )
            ),
            "visible_triggers": (
                int(
                    self.visible_triggers
                )
            ),
            "rail_blocked_triggers": (
                int(
                    self.rail_blocked_triggers
                )
            ),
            "mean_directional_consistency_at_trigger": (
                float(
                    trigger_consistency
                )
            ),
            "mean_active_votes_since_reset_at_trigger": (
                float(
                    trigger_votes
                )
            ),
            "mean_sign_changes_since_reset_at_trigger": (
                float(
                    trigger_sign_changes
                )
            ),
            "median_completed_same_sign_run": (
                histogram_quantile(
                    self.completed_run_hist,
                    0.5,
                )
            ),
            "p90_completed_same_sign_run": (
                histogram_quantile(
                    self.completed_run_hist,
                    0.9,
                )
            ),
            "median_same_sign_run_at_trigger": (
                histogram_quantile(
                    self.trigger_run_hist,
                    0.5,
                )
            ),
            "p90_same_sign_run_at_trigger": (
                histogram_quantile(
                    self.trigger_run_hist,
                    0.9,
                )
            ),
            "median_active_votes_since_reset_at_trigger": (
                histogram_quantile(
                    self.trigger_segment_vote_hist,
                    0.5,
                )
            ),
            "p90_active_votes_since_reset_at_trigger": (
                histogram_quantile(
                    self.trigger_segment_vote_hist,
                    0.9,
                )
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure innovation-sign persistence, "
            "same-sign run lengths, and SCW trigger "
            "dynamics on the complete held-out test partition."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--p3-checkpoint",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--native-b4-checkpoint",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--native-b8-checkpoint",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--batch-size",
        default=512,
        type=int,
    )

    parser.add_argument(
        "--bootstrap-replicates",
        default=2000,
        type=int,
    )

    parser.add_argument(
        "--bootstrap-seed",
        default=42,
        type=int,
    )

    args = parser.parse_args()

    args.data_dir = (
        args.data_dir
        .expanduser()
        .resolve()
    )

    args.p3_checkpoint = (
        args.p3_checkpoint
        .expanduser()
        .resolve()
    )

    args.native_b4_checkpoint = (
        args.native_b4_checkpoint
        .expanduser()
        .resolve()
    )

    args.native_b8_checkpoint = (
        args.native_b8_checkpoint
        .expanduser()
        .resolve()
    )

    args.output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not args.data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory does not exist: "
            f"{args.data_dir}"
        )

    for checkpoint in (
        args.p3_checkpoint,
        args.native_b4_checkpoint,
        args.native_b8_checkpoint,
    ):
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist: "
                f"{checkpoint}"
            )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be > 0."
        )

    if args.bootstrap_replicates <= 0:
        raise ValueError(
            "--bootstrap-replicates must be > 0."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return args


def configure_tensorflow() -> None:
    for gpu in tf.config.list_physical_devices(
        "GPU"
    ):
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )
        except RuntimeError:
            pass

    tf.keras.utils.set_random_seed(
        42
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            block = handle.read(
                1024
                * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def h2rgb(
    hex_color: str,
) -> Tuple[
    float,
    float,
    float,
]:
    value = hex_color.lstrip(
        "#"
    )

    if len(value) != 6:
        raise ValueError(
            f"Invalid hex color "
            f"{hex_color!r}."
        )

    return tuple(
        int(
            value[
                i:
                i + 2
            ],
            16,
        )
        / 255.0
        for i in (
            0,
            2,
            4,
        )
    )


def build_conditions(
    args: argparse.Namespace,
) -> Sequence[
    ConditionSpec
]:
    return (
        ConditionSpec(
            key="p3",
            display_name="P3",
            checkpoint=(
                args.p3_checkpoint
            ),
            source_bits=4,
            counter_bits=4,
            deadzone_fraction=0.0,
            color_hex="#9AA0A6",
            expected_deadband_fraction=(
                0.930550
            ),
            expected_deterministic_state_change_fraction=(
                0.067477
            ),
            expected_scw_subthreshold_visible_write_fraction=(
                0.012188
            ),
        ),
        ConditionSpec(
            key="native_b4",
            display_name="Native B4",
            checkpoint=(
                args.native_b4_checkpoint
            ),
            source_bits=4,
            counter_bits=4,
            deadzone_fraction=0.125,
            color_hex="#5B677A",
            expected_deadband_fraction=(
                0.865438
            ),
            expected_deterministic_state_change_fraction=(
                0.124666
            ),
            expected_scw_subthreshold_visible_write_fraction=(
                0.003950
            ),
        ),
        ConditionSpec(
            key="b8_to_b4",
            display_name="B8 to B4",
            checkpoint=(
                args.native_b8_checkpoint
            ),
            source_bits=8,
            counter_bits=4,
            deadzone_fraction=0.125,
            color_hex="#0072B2",
            expected_deadband_fraction=(
                0.994017
            ),
            expected_deterministic_state_change_fraction=(
                0.005983
            ),
            expected_scw_subthreshold_visible_write_fraction=(
                0.031249
            ),
        ),
    )


def load_test_data(
    data_dir: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    (
        file_input,
        _file_res,
        _file_labels,
        _file_train,
        file_val,
        file_test,
    ) = find_data_files(
        str(
            data_dir
        ),
        SEQ_LEN,
    )

    normalized_input = np.load(
        file_input,
        mmap_mode="r",
    )

    val_idx = np.asarray(
        np.load(
            file_val
        ),
        dtype=np.int64,
    )

    test_idx = np.asarray(
        np.load(
            file_test
        ),
        dtype=np.int64,
    )

    if (
        normalized_input.ndim
        != 3
    ):
        raise RuntimeError(
            f"Expected 3-D normalized input, "
            f"got {normalized_input.shape}."
        )

    if tuple(
        normalized_input.shape[
            1:
        ]
    ) != (
        SEQ_LEN,
        1,
    ):
        raise RuntimeError(
            f"Expected input shape "
            f"(*,{SEQ_LEN},1), "
            f"got {normalized_input.shape}."
        )

    if (
        val_idx.ndim
        != 1
        or len(
            val_idx
        )
        == 0
    ):
        raise RuntimeError(
            "Validation indices must be "
            "a non-empty 1-D array."
        )

    if (
        test_idx.ndim
        != 1
    ):
        raise RuntimeError(
            "Test indices must be a 1-D array."
        )

    if len(
        test_idx
    ) != EXPECTED_TEST_SIZE:
        raise RuntimeError(
            f"Paper analysis requires exactly "
            f"{EXPECTED_TEST_SIZE:,} held-out "
            f"test samples, found "
            f"{len(test_idx):,}."
        )

    n_samples = int(
        normalized_input.shape[
            0
        ]
    )

    if (
        np.any(
            val_idx
            < 0
        )
        or np.any(
            val_idx
            >= n_samples
        )
    ):
        raise RuntimeError(
            "Validation indices are out of bounds."
        )

    if (
        np.any(
            test_idx
            < 0
        )
        or np.any(
            test_idx
            >= n_samples
        )
    ):
        raise RuntimeError(
            "Test indices are out of bounds."
        )

    return (
        normalized_input,
        val_idx,
        test_idx,
    )


def build_reference_model(
    source_bits: int,
    checkpoint: Path,
):
    model = build_student(
        seq_len=SEQ_LEN,
        n_out=N_OUT,
        student_units=STUDENT_UNITS,
        bits_kernel=source_bits,
        bits_recurrent=source_bits,
        bits_bias=source_bits,
        bits_activation=source_bits,
        bits_state=source_bits,
    )

    model.load_weights(
        str(
            checkpoint
        )
    )

    model.trainable = False

    return model


def build_scw_model(
    source_bits: int,
    state_bits: int,
    counter_bits: int,
    deadzone_fraction: float,
    build_encoder_batch: np.ndarray,
) -> SCWStudentModel:
    model = SCWStudentModel(
        seq_len=SEQ_LEN,
        n_out=N_OUT,
        student_units=STUDENT_UNITS,
        bits_kernel=source_bits,
        bits_recurrent=source_bits,
        bits_bias=source_bits,
        bits_activation=source_bits,
        bits_state=state_bits,
        counter_bits=counter_bits,
        deadzone_fraction=(
            deadzone_fraction
        ),
        q_alpha=Q_ALPHA,
    )

    enc = tf.convert_to_tensor(
        np.asarray(
            build_encoder_batch,
            dtype=np.float32,
        ),
        dtype=tf.float32,
    )

    dec = tf.zeros(
        (
            enc.shape[0],
            SEQ_LEN,
            1,
        ),
        dtype=tf.float32,
    )

    _ = model(
        [
            enc,
            dec,
        ],
        training=False,
        operator_mode="deterministic",
    )

    model.trainable = False

    return model


def transfer_standard_weights(
    reference_model,
    scw_model: SCWStudentModel,
) -> None:
    for (
        layer_name,
        target_layer,
    ) in (
        (
            "sencgru",
            scw_model.sencgru,
        ),
        (
            "sdecgru",
            scw_model.sdecgru,
        ),
        (
            "sdec_dense",
            scw_model.sdec_dense,
        ),
    ):
        source_layer = (
            reference_model
            .get_layer(
                layer_name
            )
        )

        source_weights = (
            source_layer
            .get_weights()
        )

        target_weights = (
            target_layer
            .get_weights()
        )

        if len(
            source_weights
        ) != len(
            target_weights
        ):
            raise RuntimeError(
                f"Weight-count mismatch "
                f"for {layer_name}: "
                f"source="
                f"{len(source_weights)}, "
                f"target="
                f"{len(target_weights)}."
            )

        for (
            index,
            (
                source,
                target,
            ),
        ) in enumerate(
            zip(
                source_weights,
                target_weights,
            )
        ):
            if tuple(
                source.shape
            ) != tuple(
                target.shape
            ):
                raise RuntimeError(
                    f"Weight-shape mismatch "
                    f"for {layer_name}"
                    f"[{index}]: "
                    f"source={source.shape}, "
                    f"target={target.shape}."
                )

        target_layer.set_weights(
            source_weights
        )


def predict_model(
    model,
    encoder_batch: np.ndarray,
    operator_mode: Optional[str] = None,
) -> np.ndarray:
    enc = tf.convert_to_tensor(
        np.asarray(
            encoder_batch,
            dtype=np.float32,
        ),
        dtype=tf.float32,
    )

    dec = tf.zeros(
        (
            enc.shape[0],
            SEQ_LEN,
            1,
        ),
        dtype=tf.float32,
    )

    if operator_mode is None:
        pred = model(
            [
                enc,
                dec,
            ],
            training=False,
        )

    else:
        pred = model(
            [
                enc,
                dec,
            ],
            training=False,
            operator_mode=operator_mode,
        )

    return np.asarray(
        pred.numpy(),
        dtype=np.float32,
    )


def validate_native_equivalence(
    condition: ConditionSpec,
    reference_model,
    normalized_input: np.ndarray,
    val_idx: np.ndarray,
) -> Tuple[
    SCWStudentModel,
    float,
    float,
]:
    rows = val_idx[
        :
        min(
            512,
            len(
                val_idx
            ),
        )
    ]

    enc = np.asarray(
        normalized_input[
            rows
        ],
        dtype=np.float32,
    )

    native_custom = (
        build_scw_model(
            source_bits=(
                condition.source_bits
            ),
            state_bits=(
                condition.source_bits
            ),
            counter_bits=(
                condition.counter_bits
            ),
            deadzone_fraction=(
                condition.deadzone_fraction
            ),
            build_encoder_batch=(
                enc[
                    :1
                ]
            ),
        )
    )

    transfer_standard_weights(
        reference_model,
        native_custom,
    )

    reference_pred = (
        predict_model(
            reference_model,
            enc,
            operator_mode=None,
        )
    )

    custom_pred = (
        predict_model(
            native_custom,
            enc,
            operator_mode=(
                "deterministic"
            ),
        )
    )

    diff = np.abs(
        reference_pred.astype(
            np.float64
        )
        - custom_pred.astype(
            np.float64
        )
    )

    max_abs = float(
        np.max(
            diff
        )
    )

    mean_abs = float(
        np.mean(
            diff
        )
    )

    if (
        max_abs
        > EQUIVALENCE_TOLERANCE
        or mean_abs
        > EQUIVALENCE_TOLERANCE
    ):
        raise RuntimeError(
            f"{condition.key}: "
            f"native deterministic equivalence failed: "
            f"max_abs={max_abs:.8g}, "
            f"mean_abs={mean_abs:.8g}, "
            f"tolerance="
            f"{EQUIVALENCE_TOLERANCE:.8g}."
        )

    return (
        native_custom,
        max_abs,
        mean_abs,
    )


def make_trace_function(
    model: SCWStudentModel,
    operator_mode: str,
):
    if operator_mode not in (
        "deterministic",
        "scw",
    ):
        raise ValueError(
            f"Unsupported operator_mode="
            f"{operator_mode!r}."
        )

    seq_len = int(
        model.seq_len
    )

    live_steps = (
        seq_len
        - 1
    )

    units = int(
        model.student_units
    )

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=(
                    None,
                    seq_len,
                    1,
                ),
                dtype=tf.float32,
            )
        ],
        reduce_retracing=True,
    )
    def trace_batch(
        enc_inputs: tf.Tensor,
    ) -> Dict[
        str,
        tf.Tensor,
    ]:
        enc_inputs = tf.cast(
            enc_inputs,
            tf.float32,
        )

        batch = tf.shape(
            enc_inputs
        )[0]

        dec_inputs = tf.zeros(
            (
                batch,
                seq_len,
                1,
            ),
            dtype=tf.float32,
        )

        (
            enc_kernel_q,
            enc_recurrent_q,
            enc_bias_q,
        ) = (
            model.sencgru
            .effective_parameters()
        )

        (
            dec_kernel_q,
            dec_recurrent_q,
            dec_bias_q,
        ) = (
            model.sdecgru
            .effective_parameters()
        )

        zero_raw = tf.zeros(
            (
                batch,
                units,
            ),
            dtype=tf.float32,
        )

        (
            q_enc,
            counter_enc,
            q_enc_hard,
        ) = (
            model.sencgru
            .initialize_state(
                zero_raw,
                operator_mode=(
                    operator_mode
                ),
                use_ste=False,
            )
        )

        raw_enc = zero_raw

        for i in range(
            seq_len
        ):
            (
                raw_enc,
                _,
                _,
                _,
            ) = (
                model.sencgru
                .gru_step(
                    enc_inputs[
                        :,
                        i,
                        :
                    ],
                    q_enc,
                    enc_kernel_q,
                    enc_recurrent_q,
                    enc_bias_q,
                )
            )

            if (
                i
                < live_steps
            ):
                (
                    q_enc,
                    counter_enc,
                    q_enc_hard,
                ) = (
                    model.sencgru
                    .advance_state(
                        raw_state=(
                            raw_enc
                        ),
                        q_prev_hard=(
                            q_enc_hard
                        ),
                        counter_prev=(
                            counter_enc
                        ),
                        operator_mode=(
                            operator_mode
                        ),
                        use_ste=False,
                    )
                )

        (
            q_dec,
            counter_dec,
            q_dec_hard,
        ) = (
            model.sdecgru
            .initialize_state(
                raw_enc,
                operator_mode=(
                    operator_mode
                ),
                use_ste=False,
            )
        )

        hidden_ta = tf.TensorArray(
            tf.float32,
            size=seq_len,
            clear_after_read=False,
            element_shape=(
                tf.TensorShape(
                    [
                        None,
                        units,
                    ]
                )
            ),
        )

        delta_ta = tf.TensorArray(
            tf.float32,
            size=live_steps,
            clear_after_read=False,
            element_shape=(
                tf.TensorShape(
                    [
                        None,
                        units,
                    ]
                )
            ),
        )

        normal_ta = tf.TensorArray(
            tf.bool,
            size=live_steps,
            clear_after_read=False,
            element_shape=(
                tf.TensorShape(
                    [
                        None,
                        units,
                    ]
                )
            ),
        )

        subthreshold_ta = tf.TensorArray(
            tf.bool,
            size=live_steps,
            clear_after_read=False,
            element_shape=(
                tf.TensorShape(
                    [
                        None,
                        units,
                    ]
                )
            ),
        )

        active_vote_ta = tf.TensorArray(
            tf.bool,
            size=live_steps,
            clear_after_read=False,
            element_shape=(
                tf.TensorShape(
                    [
                        None,
                        units,
                    ]
                )
            ),
        )

        vote_ta = tf.TensorArray(
            tf.float32,
            size=live_steps,
            clear_after_read=False,
            element_shape=(
                tf.TensorShape(
                    [
                        None,
                        units,
                    ]
                )
            ),
        )

        trigger_ta = tf.TensorArray(
            tf.bool,
            size=live_steps,
            clear_after_read=False,
            element_shape=(
                tf.TensorShape(
                    [
                        None,
                        units,
                    ]
                )
            ),
        )

        visible_change_ta = tf.TensorArray(
            tf.bool,
            size=live_steps,
            clear_after_read=False,
            element_shape=(
                tf.TensorShape(
                    [
                        None,
                        units,
                    ]
                )
            ),
        )

        counter_before_ta = tf.TensorArray(
            tf.float32,
            size=live_steps,
            clear_after_read=False,
            element_shape=(
                tf.TensorShape(
                    [
                        None,
                        units,
                    ]
                )
            ),
        )

        counter_after_ta = tf.TensorArray(
            tf.float32,
            size=live_steps,
            clear_after_read=False,
            element_shape=(
                tf.TensorShape(
                    [
                        None,
                        units,
                    ]
                )
            ),
        )

        half_step = tf.constant(
            model.sdecgru.half_step,
            dtype=tf.float32,
        )

        deadzone = tf.constant(
            model.sdecgru.deadzone,
            dtype=tf.float32,
        )

        trigger_votes = tf.constant(
            float(
                model.sdecgru
                .trigger_votes
            ),
            dtype=tf.float32,
        )

        for i in range(
            live_steps
        ):
            (
                raw_next,
                _,
                _,
                _,
            ) = (
                model.sdecgru
                .gru_step(
                    dec_inputs[
                        :,
                        i,
                        :
                    ],
                    q_dec,
                    dec_kernel_q,
                    dec_recurrent_q,
                    dec_bias_q,
                )
            )

            hidden_ta = hidden_ta.write(
                i,
                raw_next,
            )

            q_prev = q_dec_hard
            counter_prev = counter_dec

            delta_raw = (
                raw_next
                - q_prev
            )

            abs_delta = tf.abs(
                delta_raw
            )

            normal = (
                abs_delta
                >= half_step
            )

            subthreshold = (
                ~normal
            )

            active_vote = (
                subthreshold
                & (
                    abs_delta
                    > deadzone
                )
            )

            vote = tf.sign(
                delta_raw
            )

            counter_candidate = (
                tf.round(
                    counter_prev
                )
                + vote
            )

            counterfactual_trigger = (
                active_vote
                & (
                    (
                        counter_candidate
                        >= trigger_votes
                    )
                    | (
                        counter_candidate
                        <= -trigger_votes
                    )
                )
            )

            (
                q_next_hard,
                counter_next,
            ) = (
                model.sdecgru
                .hard_advance(
                    raw_state=(
                        raw_next
                    ),
                    q_prev_hard=(
                        q_prev
                    ),
                    counter_prev=(
                        counter_prev
                    ),
                    operator_mode=(
                        operator_mode
                    ),
                )
            )

            if (
                operator_mode
                == "scw"
            ):
                trigger = (
                    counterfactual_trigger
                )

            else:
                trigger = (
                    tf.zeros_like(
                        counterfactual_trigger,
                        dtype=tf.bool,
                    )
                )

            visible_change = (
                tf.not_equal(
                    q_next_hard,
                    q_prev,
                )
            )

            delta_ta = (
                delta_ta.write(
                    i,
                    delta_raw,
                )
            )

            normal_ta = (
                normal_ta.write(
                    i,
                    normal,
                )
            )

            subthreshold_ta = (
                subthreshold_ta.write(
                    i,
                    subthreshold,
                )
            )

            active_vote_ta = (
                active_vote_ta.write(
                    i,
                    active_vote,
                )
            )

            vote_ta = (
                vote_ta.write(
                    i,
                    vote,
                )
            )

            trigger_ta = (
                trigger_ta.write(
                    i,
                    trigger,
                )
            )

            visible_change_ta = (
                visible_change_ta.write(
                    i,
                    visible_change,
                )
            )

            counter_before_ta = (
                counter_before_ta.write(
                    i,
                    tf.round(
                        counter_prev
                    ),
                )
            )

            counter_after_ta = (
                counter_after_ta.write(
                    i,
                    tf.round(
                        counter_next
                    ),
                )
            )

            q_dec = q_next_hard
            q_dec_hard = q_next_hard
            counter_dec = counter_next

        (
            raw_terminal,
            _,
            _,
            _,
        ) = (
            model.sdecgru
            .gru_step(
                dec_inputs[
                    :,
                    live_steps,
                    :
                ],
                q_dec,
                dec_kernel_q,
                dec_recurrent_q,
                dec_bias_q,
            )
        )

        hidden_ta = hidden_ta.write(
            live_steps,
            raw_terminal,
        )

        dec_hidden = tf.transpose(
            hidden_ta.stack(),
            perm=(
                1,
                0,
                2,
            ),
        )

        predictions = (
            model.sdec_dense(
                dec_hidden
            )
        )

        return {
            "predictions": (
                tf.cast(
                    predictions,
                    tf.float32,
                )
            ),
            "delta_raw": (
                tf.transpose(
                    delta_ta.stack(),
                    perm=(
                        1,
                        0,
                        2,
                    ),
                )
            ),
            "normal": (
                tf.transpose(
                    normal_ta.stack(),
                    perm=(
                        1,
                        0,
                        2,
                    ),
                )
            ),
            "subthreshold": (
                tf.transpose(
                    subthreshold_ta.stack(),
                    perm=(
                        1,
                        0,
                        2,
                    ),
                )
            ),
            "active_vote": (
                tf.transpose(
                    active_vote_ta.stack(),
                    perm=(
                        1,
                        0,
                        2,
                    ),
                )
            ),
            "vote": (
                tf.transpose(
                    vote_ta.stack(),
                    perm=(
                        1,
                        0,
                        2,
                    ),
                )
            ),
            "trigger": (
                tf.transpose(
                    trigger_ta.stack(),
                    perm=(
                        1,
                        0,
                        2,
                    ),
                )
            ),
            "visible_change": (
                tf.transpose(
                    visible_change_ta.stack(),
                    perm=(
                        1,
                        0,
                        2,
                    ),
                )
            ),
            "counter_before": (
                tf.transpose(
                    counter_before_ta.stack(),
                    perm=(
                        1,
                        0,
                        2,
                    ),
                )
            ),
            "counter_after": (
                tf.transpose(
                    counter_after_ta.stack(),
                    perm=(
                        1,
                        0,
                        2,
                    ),
                )
            ),
        }

    return trace_batch


def validate_trace_equivalence(
    condition: ConditionSpec,
    model: SCWStudentModel,
    trace_deterministic,
    trace_scw,
    normalized_input: np.ndarray,
    val_idx: np.ndarray,
) -> Tuple[
    float,
    float,
    float,
    float,
]:
    rows = val_idx[
        :
        min(
            128,
            len(
                val_idx
            ),
        )
    ]

    enc_np = np.asarray(
        normalized_input[
            rows
        ],
        dtype=np.float32,
    )

    enc_tf = tf.convert_to_tensor(
        enc_np,
        dtype=tf.float32,
    )

    direct_det = predict_model(
        model,
        enc_np,
        operator_mode=(
            "deterministic"
        ),
    )

    traced_det = np.asarray(
        trace_deterministic(
            enc_tf
        )[
            "predictions"
        ].numpy(),
        dtype=np.float32,
    )

    diff_det = np.abs(
        direct_det.astype(
            np.float64
        )
        - traced_det.astype(
            np.float64
        )
    )

    det_max = float(
        np.max(
            diff_det
        )
    )

    det_mean = float(
        np.mean(
            diff_det
        )
    )

    if (
        det_max
        > EQUIVALENCE_TOLERANCE
        or det_mean
        > EQUIVALENCE_TOLERANCE
    ):
        raise RuntimeError(
            f"{condition.key}: "
            f"deterministic trace equivalence failed: "
            f"max_abs={det_max:.8g}, "
            f"mean_abs={det_mean:.8g}."
        )

    direct_scw = predict_model(
        model,
        enc_np,
        operator_mode="scw",
    )

    traced_scw = np.asarray(
        trace_scw(
            enc_tf
        )[
            "predictions"
        ].numpy(),
        dtype=np.float32,
    )

    diff_scw = np.abs(
        direct_scw.astype(
            np.float64
        )
        - traced_scw.astype(
            np.float64
        )
    )

    scw_max = float(
        np.max(
            diff_scw
        )
    )

    scw_mean = float(
        np.mean(
            diff_scw
        )
    )

    if (
        scw_max
        > EQUIVALENCE_TOLERANCE
        or scw_mean
        > EQUIVALENCE_TOLERANCE
    ):
        raise RuntimeError(
            f"{condition.key}: "
            f"SCW trace equivalence failed: "
            f"max_abs={scw_max:.8g}, "
            f"mean_abs={scw_mean:.8g}."
        )

    return (
        det_max,
        det_mean,
        scw_max,
        scw_mean,
    )


def trace_to_numpy(
    trace: Mapping[
        str,
        tf.Tensor,
    ],
) -> Dict[
    str,
    np.ndarray,
]:
    return {
        key: np.asarray(
            tensor.numpy()
        )
        for (
            key,
            tensor,
        ) in trace.items()
        if key
        != "predictions"
    }


def run_full_test_trace(
    condition: ConditionSpec,
    trace_fn,
    operator_mode: str,
    normalized_input: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
) -> ModeAccumulator:
    accumulator = ModeAccumulator(
        condition=condition,
        operator_mode=operator_mode,
        n_sequences=len(
            test_idx
        ),
        live_steps=LIVE_STEPS,
        units=STUDENT_UNITS,
    )

    n_batches = math.ceil(
        len(
            test_idx
        )
        / batch_size
    )

    for (
        batch_index,
        start,
    ) in enumerate(
        range(
            0,
            len(
                test_idx
            ),
            batch_size,
        ),
        start=1,
    ):
        stop = min(
            start
            + batch_size,
            len(
                test_idx
            ),
        )

        rows = test_idx[
            start:
            stop
        ]

        enc_np = np.asarray(
            normalized_input[
                rows
            ],
            dtype=np.float32,
        )

        enc_tf = (
            tf.convert_to_tensor(
                enc_np,
                dtype=tf.float32,
            )
        )

        accumulator.add_batch(
            trace_to_numpy(
                trace_fn(
                    enc_tf
                )
            )
        )

        if (
            batch_index
            == 1
            or batch_index
            % 25
            == 0
            or batch_index
            == n_batches
        ):
            print(
                f"[{condition.key}/"
                f"{operator_mode}] "
                f"batch "
                f"{batch_index}/"
                f"{n_batches}, "
                f"samples "
                f"{stop:,}/"
                f"{len(test_idx):,}",
                flush=True,
            )

    accumulator.finalize()

    return accumulator


def validate_against_manuscript(
    condition: ConditionSpec,
    deterministic: ModeAccumulator,
    scw: ModeAccumulator,
) -> Dict[
    str,
    Dict[
        str,
        float,
    ],
]:
    det = (
        deterministic
        .scalar_summary()
    )

    scw_summary = (
        scw
        .scalar_summary()
    )

    checks = {
        "deadband_fraction": {
            "observed": (
                det[
                    "deadband_fraction"
                ]
            ),
            "expected": (
                condition
                .expected_deadband_fraction
            ),
        },
        "deterministic_state_change_fraction": {
            "observed": (
                det[
                    "state_change_fraction"
                ]
            ),
            "expected": (
                condition
                .expected_deterministic_state_change_fraction
            ),
        },
        "scw_subthreshold_visible_write_fraction": {
            "observed": (
                scw_summary[
                    "subthreshold_visible_write_fraction"
                ]
            ),
            "expected": (
                condition
                .expected_scw_subthreshold_visible_write_fraction
            ),
        },
    }

    for (
        name,
        payload,
    ) in checks.items():
        observed = float(
            payload[
                "observed"
            ]
        )

        expected = float(
            payload[
                "expected"
            ]
        )

        absolute_error = abs(
            observed
            - expected
        )

        payload[
            "absolute_error"
        ] = absolute_error

        payload[
            "tolerance"
        ] = (
            MANUSCRIPT_METRIC_TOLERANCE
        )

        payload[
            "passed"
        ] = bool(
            absolute_error
            <= MANUSCRIPT_METRIC_TOLERANCE
        )

        if not payload[
            "passed"
        ]:
            raise RuntimeError(
                f"{condition.key}: "
                f"manuscript validation failed "
                f"for {name}: "
                f"observed={observed:.9f}, "
                f"expected={expected:.9f}, "
                f"abs_error="
                f"{absolute_error:.9g}, "
                f"tolerance="
                f"{MANUSCRIPT_METRIC_TOLERANCE:.9g}. "
                f"Use the exact paper checkpoint "
                f"and full fixed test partition."
            )

    return checks


def transition_odds_ratio(
    pp: int,
    pn: int,
    np_count: int,
    nn: int,
) -> float:
    return float(
        (
            (
                pp
                + 0.5
            )
            * (
                nn
                + 0.5
            )
        )
        / (
            (
                pn
                + 0.5
            )
            * (
                np_count
                + 0.5
            )
        )
    )


def bootstrap_transition_statistics(
    per_sequence_counts: np.ndarray,
    replicates: int,
    seed: int,
) -> Dict[
    str,
    float,
]:
    counts = np.asarray(
        per_sequence_counts,
        dtype=np.int64,
    )

    if (
        counts.ndim
        != 2
        or counts.shape[
            1
        ]
        != 4
    ):
        raise ValueError(
            f"Expected transition counts "
            f"shape (N,4), got "
            f"{counts.shape}."
        )

    n_sequences = counts.shape[
        0
    ]

    if n_sequences <= 0:
        raise ValueError(
            "No sequences available for bootstrap."
        )

    rng = np.random.default_rng(
        seed
    )

    same_values = np.empty(
        replicates,
        dtype=np.float64,
    )

    odds_values = np.empty(
        replicates,
        dtype=np.float64,
    )

    chunk_size = 16
    cursor = 0

    while cursor < replicates:
        current = min(
            chunk_size,
            replicates
            - cursor,
        )

        indices = rng.integers(
            0,
            n_sequences,
            size=(
                current,
                n_sequences,
            ),
            dtype=np.int32,
        )

        sampled = np.sum(
            counts[
                indices
            ],
            axis=1,
            dtype=np.int64,
        )

        pp = sampled[
            :,
            0
        ].astype(
            np.float64
        )

        pn = sampled[
            :,
            1
        ].astype(
            np.float64
        )

        np_count = sampled[
            :,
            2
        ].astype(
            np.float64
        )

        nn = sampled[
            :,
            3
        ].astype(
            np.float64
        )

        denominator = (
            pp
            + pn
            + np_count
            + nn
        )

        same = np.divide(
            pp
            + nn,
            denominator,
            out=np.full_like(
                denominator,
                np.nan,
            ),
            where=(
                denominator
                > 0
            ),
        )

        odds = (
            (
                pp
                + 0.5
            )
            * (
                nn
                + 0.5
            )
            / (
                (
                    pn
                    + 0.5
                )
                * (
                    np_count
                    + 0.5
                )
            )
        )

        same_values[
            cursor:
            cursor
            + current
        ] = same

        odds_values[
            cursor:
            cursor
            + current
        ] = odds

        cursor += current

    return {
        "same_sign_fraction_ci_low": (
            float(
                np.nanpercentile(
                    same_values,
                    2.5,
                )
            )
        ),
        "same_sign_fraction_ci_high": (
            float(
                np.nanpercentile(
                    same_values,
                    97.5,
                )
            )
        ),
        "persistence_odds_ratio_ci_low": (
            float(
                np.nanpercentile(
                    odds_values,
                    2.5,
                )
            )
        ),
        "persistence_odds_ratio_ci_high": (
            float(
                np.nanpercentile(
                    odds_values,
                    97.5,
                )
            )
        ),
        "bootstrap_replicates": (
            int(
                replicates
            )
        ),
        "bootstrap_seed": (
            int(
                seed
            )
        ),
    }


def histogram_quantile(
    hist: np.ndarray,
    q: float,
) -> float:
    hist = np.asarray(
        hist,
        dtype=np.int64,
    )

    if not (
        0.0
        <= q
        <= 1.0
    ):
        raise ValueError(
            "q must be in [0,1]."
        )

    total = int(
        np.sum(
            hist
        )
    )

    if total == 0:
        return float(
            "nan"
        )

    cumulative = np.cumsum(
        hist
    )

    target = (
        q
        * total
    )

    return float(
        np.searchsorted(
            cumulative,
            target,
            side="left",
        )
    )


def run_survival(
    completed_run_hist: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    hist = np.asarray(
        completed_run_hist,
        dtype=np.int64,
    )

    total = int(
        np.sum(
            hist[
                1:
            ]
        )
    )

    if total == 0:
        return (
            np.zeros(
                0,
                dtype=np.int64,
            ),
            np.zeros(
                0,
                dtype=np.float64,
            ),
        )

    max_run = int(
        np.max(
            np.flatnonzero(
                hist
            )
        )
    )

    lengths = np.arange(
        1,
        max_run
        + 1,
        dtype=np.int64,
    )

    tail = np.cumsum(
        hist[
            ::-1
        ]
    )[
        ::-1
    ]

    survival = (
        tail[
            lengths
        ].astype(
            np.float64
        )
        / float(
            total
        )
    )

    return (
        lengths,
        survival,
    )


def trigger_probability_by_run(
    active_event_run_hist: np.ndarray,
    trigger_run_hist: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    active = np.asarray(
        active_event_run_hist,
        dtype=np.int64,
    )

    trigger = np.asarray(
        trigger_run_hist,
        dtype=np.int64,
    )

    if active.shape != trigger.shape:
        raise ValueError(
            "Active-event and trigger run "
            "histograms must have identical shapes."
        )

    valid = (
        np.flatnonzero(
            active[
                1:
            ]
            > 0
        )
        + 1
    )

    if valid.size == 0:
        return (
            np.zeros(
                0,
                dtype=np.int64,
            ),
            np.zeros(
                0,
                dtype=np.float64,
            ),
            np.zeros(
                0,
                dtype=np.int64,
            ),
        )

    probability = (
        trigger[
            valid
        ].astype(
            np.float64
        )
        / active[
            valid
        ].astype(
            np.float64
        )
    )

    return (
        valid.astype(
            np.int64
        ),
        probability,
        active[
            valid
        ],
    )


def json_safe(
    value,
):
    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key
            ): json_safe(
                item
            )
            for (
                key,
                item,
            ) in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        value = float(
            value
        )

    if (
        isinstance(
            value,
            float,
        )
        and not math.isfinite(
            value
        )
    ):
        return None

    return value


def write_json(
    path: Path,
    payload: Mapping,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            json_safe(
                payload
            ),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

        handle.write(
            "\n"
        )


def write_summary_csv(
    path: Path,
    summaries: Mapping[
        str,
        Mapping[
            str,
            Mapping,
        ],
    ],
) -> None:
    columns = [
        "condition",
        "operator_mode",
        "deadband_fraction",
        "state_change_fraction",
        "subthreshold_visible_write_fraction",
        "same_sign_fraction",
        "same_sign_fraction_ci_low",
        "same_sign_fraction_ci_high",
        "persistence_odds_ratio",
        "persistence_odds_ratio_ci_low",
        "persistence_odds_ratio_ci_high",
        "median_completed_same_sign_run",
        "p90_completed_same_sign_run",
        "triggers",
        "visible_triggers",
        "rail_blocked_triggers",
        "median_same_sign_run_at_trigger",
        "p90_same_sign_run_at_trigger",
        "mean_active_votes_since_reset_at_trigger",
        "mean_directional_consistency_at_trigger",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        writer.writeheader()

        for (
            condition_key,
            mode_map,
        ) in summaries.items():
            for (
                mode,
                summary,
            ) in mode_map.items():
                scalar = summary[
                    "scalar"
                ]

                transition = summary[
                    "transition"
                ]

                bootstrap = summary[
                    "bootstrap"
                ]

                writer.writerow(
                    {
                        "condition": condition_key,
                        "operator_mode": mode,
                        "deadband_fraction": (
                            scalar[
                                "deadband_fraction"
                            ]
                        ),
                        "state_change_fraction": (
                            scalar[
                                "state_change_fraction"
                            ]
                        ),
                        "subthreshold_visible_write_fraction": (
                            scalar[
                                "subthreshold_visible_write_fraction"
                            ]
                        ),
                        "same_sign_fraction": (
                            transition[
                                "same_sign_fraction"
                            ]
                        ),
                        "same_sign_fraction_ci_low": (
                            bootstrap[
                                "same_sign_fraction_ci_low"
                            ]
                        ),
                        "same_sign_fraction_ci_high": (
                            bootstrap[
                                "same_sign_fraction_ci_high"
                            ]
                        ),
                        "persistence_odds_ratio": (
                            transition[
                                "persistence_odds_ratio"
                            ]
                        ),
                        "persistence_odds_ratio_ci_low": (
                            bootstrap[
                                "persistence_odds_ratio_ci_low"
                            ]
                        ),
                        "persistence_odds_ratio_ci_high": (
                            bootstrap[
                                "persistence_odds_ratio_ci_high"
                            ]
                        ),
                        "median_completed_same_sign_run": (
                            scalar[
                                "median_completed_same_sign_run"
                            ]
                        ),
                        "p90_completed_same_sign_run": (
                            scalar[
                                "p90_completed_same_sign_run"
                            ]
                        ),
                        "triggers": (
                            scalar[
                                "triggers"
                            ]
                        ),
                        "visible_triggers": (
                            scalar[
                                "visible_triggers"
                            ]
                        ),
                        "rail_blocked_triggers": (
                            scalar[
                                "rail_blocked_triggers"
                            ]
                        ),
                        "median_same_sign_run_at_trigger": (
                            scalar[
                                "median_same_sign_run_at_trigger"
                            ]
                        ),
                        "p90_same_sign_run_at_trigger": (
                            scalar[
                                "p90_same_sign_run_at_trigger"
                            ]
                        ),
                        "mean_active_votes_since_reset_at_trigger": (
                            scalar[
                                "mean_active_votes_since_reset_at_trigger"
                            ]
                        ),
                        "mean_directional_consistency_at_trigger": (
                            scalar[
                                "mean_directional_consistency_at_trigger"
                            ]
                        ),
                    }
                )


def write_run_length_csv(
    path: Path,
    accumulators: Mapping[
        str,
        Mapping[
            str,
            ModeAccumulator,
        ],
    ],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "condition",
                "operator_mode",
                "run_length",
                "completed_runs",
                "active_vote_events_at_run_length",
                "triggers_at_run_length",
                "visible_triggers_at_run_length",
                "trigger_probability",
            ]
        )

        for (
            condition_key,
            mode_map,
        ) in accumulators.items():
            for (
                mode,
                acc,
            ) in mode_map.items():
                nonzero_indices = (
                    np.flatnonzero(
                        acc.completed_run_hist
                        + acc.active_event_run_hist
                        + acc.trigger_run_hist
                    )
                )

                if nonzero_indices.size == 0:
                    continue

                max_len = int(
                    np.max(
                        nonzero_indices
                    )
                )

                for run_length in range(
                    1,
                    max_len
                    + 1,
                ):
                    active = int(
                        acc.active_event_run_hist[
                            run_length
                        ]
                    )

                    triggers = int(
                        acc.trigger_run_hist[
                            run_length
                        ]
                    )

                    visible_triggers = int(
                        acc.trigger_visible_run_hist[
                            run_length
                        ]
                    )

                    trigger_probability = (
                        triggers
                        / active
                        if active
                        > 0
                        else float(
                            "nan"
                        )
                    )

                    writer.writerow(
                        [
                            condition_key,
                            mode,
                            run_length,
                            int(
                                acc.completed_run_hist[
                                    run_length
                                ]
                            ),
                            active,
                            triggers,
                            visible_triggers,
                            (
                                f"{trigger_probability:.12g}"
                                if math.isfinite(
                                    trigger_probability
                                )
                                else ""
                            ),
                        ]
                    )


def write_trigger_segment_csv(
    path: Path,
    accumulators: Mapping[
        str,
        Mapping[
            str,
            ModeAccumulator,
        ],
    ],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "condition",
                "operator_mode",
                "active_votes_since_reset_at_trigger",
                "trigger_count",
            ]
        )

        for (
            condition_key,
            mode_map,
        ) in accumulators.items():
            for (
                mode,
                acc,
            ) in mode_map.items():
                nonzero = np.flatnonzero(
                    acc.trigger_segment_vote_hist
                )

                if nonzero.size == 0:
                    continue

                max_value = int(
                    np.max(
                        nonzero
                    )
                )

                for vote_count in range(
                    1,
                    max_value
                    + 1,
                ):
                    count = int(
                        acc.trigger_segment_vote_hist[
                            vote_count
                        ]
                    )

                    if count == 0:
                        continue

                    writer.writerow(
                        [
                            condition_key,
                            mode,
                            vote_count,
                            count,
                        ]
                    )


def save_per_sequence_npz(
    output_dir: Path,
    condition_key: str,
    mode: str,
    acc: ModeAccumulator,
) -> None:
    np.savez_compressed(
        output_dir
        / (
            f"{condition_key}_"
            f"{mode}_"
            f"per_sequence_metrics.npz"
        ),
        transition_counts=(
            acc.transition_counts
        ),
        active_votes=(
            acc.active_votes_by_sequence
        ),
        triggers=(
            acc.triggers_by_sequence
        ),
        visible_triggers=(
            acc.visible_triggers_by_sequence
        ),
        normal_events=(
            acc.normal_events_by_sequence
        ),
        subthreshold_events=(
            acc.subthreshold_by_sequence
        ),
        subthreshold_visible_writes=(
            acc.subthreshold_visible_writes_by_sequence
        ),
        state_changes=(
            acc.state_changes_by_sequence
        ),
        max_same_sign_run=(
            acc.max_same_sign_run_by_sequence
        ),
    )


def make_figure(
    output_dir: Path,
    conditions: Sequence[
        ConditionSpec
    ],
    accumulators: Mapping[
        str,
        Mapping[
            str,
            ModeAccumulator,
        ],
    ],
    summaries: Mapping[
        str,
        Mapping[
            str,
            Mapping,
        ],
    ],
) -> None:
    colors = {
        condition.key: h2rgb(
            condition.color_hex
        )
        for condition in conditions
    }

    graphite = h2rgb(
        "#263238"
    )

    divider = h2rgb(
        "#D9DEE1"
    )

    threshold = h2rgb(
        "#AEB8BD"
    )

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    (
        fig,
        axes,
    ) = plt.subplots(
        1,
        3,
        figsize=(
            12.8,
            3.8,
        ),
        constrained_layout=True,
    )

    ax = axes[
        0
    ]

    for (
        idx,
        condition,
    ) in enumerate(
        conditions
    ):
        summary = summaries[
            condition.key
        ][
            "deterministic"
        ]

        estimate = summary[
            "transition"
        ][
            "persistence_odds_ratio"
        ]

        low = summary[
            "bootstrap"
        ][
            "persistence_odds_ratio_ci_low"
        ]

        high = summary[
            "bootstrap"
        ][
            "persistence_odds_ratio_ci_high"
        ]

        ax.errorbar(
            idx,
            estimate,
            yerr=np.array(
                [
                    [
                        max(
                            0.0,
                            estimate
                            - low,
                        )
                    ],
                    [
                        max(
                            0.0,
                            high
                            - estimate,
                        )
                    ],
                ]
            ),
            fmt="o",
            color=colors[
                condition.key
            ],
            markeredgecolor=graphite,
            markeredgewidth=0.8,
            markersize=7.0,
            linewidth=1.5,
            capsize=3.0,
        )

    ax.axhline(
        1.0,
        color=threshold,
        linewidth=1.2,
        linestyle="--",
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xticks(
        np.arange(
            len(
                conditions
            )
        )
    )

    ax.set_xticklabels(
        [
            condition.display_name
            for condition in conditions
        ]
    )

    ax.set_ylabel(
        "Same-sign transition odds ratio"
    )

    ax.set_title(
        "Deterministic B4 innovation-sign persistence"
    )

    ax.grid(
        axis="y",
        color=divider,
        linewidth=0.7,
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.text(
        -0.16,
        1.04,
        "a",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=graphite,
    )

    ax = axes[
        1
    ]

    for condition in conditions:
        acc = accumulators[
            condition.key
        ][
            "deterministic"
        ]

        (
            lengths,
            survival,
        ) = run_survival(
            acc.completed_run_hist
        )

        if lengths.size == 0:
            continue

        ax.plot(
            lengths,
            survival,
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            color=colors[
                condition.key
            ],
            label=(
                condition.display_name
            ),
        )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Completed same-sign run length"
    )

    ax.set_ylabel(
        "P(run length ≥ L)"
    )

    ax.set_title(
        "Eligible sub-threshold vote runs"
    )

    ax.grid(
        axis="y",
        color=divider,
        linewidth=0.7,
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.legend(
        frameon=False,
        fontsize=8,
    )

    ax.text(
        -0.16,
        1.04,
        "b",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=graphite,
    )

    ax = axes[
        2
    ]

    for condition in conditions:
        acc = accumulators[
            condition.key
        ][
            "scw"
        ]

        (
            lengths,
            probability,
            _denominator,
        ) = (
            trigger_probability_by_run(
                acc.active_event_run_hist,
                acc.trigger_run_hist,
            )
        )

        if lengths.size == 0:
            continue

        ax.plot(
            lengths,
            100.0
            * probability,
            marker="o",
            markersize=4.0,
            linewidth=1.6,
            color=colors[
                condition.key
            ],
            label=(
                condition.display_name
            ),
        )

    ax.set_xlabel(
        "Current same-sign run length"
    )

    ax.set_ylabel(
        "SCW trigger probability (%)"
    )

    ax.set_title(
        "Directional runs and SCW triggering"
    )

    ax.grid(
        axis="y",
        color=divider,
        linewidth=0.7,
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.legend(
        frameon=False,
        fontsize=8,
    )

    ax.text(
        -0.16,
        1.04,
        "c",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=graphite,
    )

    for axis in axes:
        axis.tick_params(
            direction="out",
            colors=graphite,
        )

        axis.xaxis.label.set_color(
            graphite
        )

        axis.yaxis.label.set_color(
            graphite
        )

        axis.title.set_color(
            graphite
        )

        axis.spines[
            "bottom"
        ].set_color(
            graphite
        )

        axis.spines[
            "left"
        ].set_color(
            graphite
        )

    base = (
        output_dir
        / "scw_sign_persistence"
    )

    fig.savefig(
        base.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        base.with_suffix(
            ".svg"
        ),
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        base.with_suffix(
            ".png"
        ),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )


def main() -> None:
    args = parse_args()

    configure_tensorflow()

    (
        normalized_input,
        val_idx,
        test_idx,
    ) = load_test_data(
        args.data_dir
    )

    conditions = build_conditions(
        args
    )

    write_json(
        args.output_dir
        / "analysis_manifest.json",
        {
            "analysis": (
                "SCW innovation-sign persistence "
                "and run-length analysis"
            ),
            "repository": (
                "https://github.com/"
                "ismailerbas/"
                "Seq2SeqLite-kd"
            ),
            "test_samples": (
                int(
                    len(
                        test_idx
                    )
                )
            ),
            "sequence_length": (
                SEQ_LEN
            ),
            "live_write_steps_per_sequence": (
                LIVE_STEPS
            ),
            "hidden_units": (
                STUDENT_UNITS
            ),
            "batch_size": (
                int(
                    args.batch_size
                )
            ),
            "bootstrap_replicates": (
                int(
                    args.bootstrap_replicates
                )
            ),
            "bootstrap_seed": (
                int(
                    args.bootstrap_seed
                )
            ),
            "checkpoints": {
                condition.key: {
                    "path": (
                        str(
                            condition.checkpoint
                        )
                    ),
                    "sha256": (
                        sha256_file(
                            condition.checkpoint
                        )
                    ),
                    "source_bits": (
                        condition.source_bits
                    ),
                    "analysis_state_bits": (
                        STATE_BITS
                    ),
                    "counter_bits": (
                        condition.counter_bits
                    ),
                    "deadzone_fraction_of_delta": (
                        condition.deadzone_fraction
                    ),
                }
                for condition in conditions
            },
        },
    )

    all_accumulators: Dict[
        str,
        Dict[
            str,
            ModeAccumulator,
        ],
    ] = {}

    all_summaries: Dict[
        str,
        Dict[
            str,
            Dict,
        ],
    ] = {}

    validation_payload: Dict[
        str,
        Dict,
    ] = {}

    for (
        condition_index,
        condition,
    ) in enumerate(
        conditions
    ):
        print(
            f"\n[CONDITION] "
            f"{condition.display_name}",
            flush=True,
        )

        print(
            f"[CHECKPOINT] "
            f"{condition.checkpoint}",
            flush=True,
        )

        reference_model = (
            build_reference_model(
                source_bits=(
                    condition.source_bits
                ),
                checkpoint=(
                    condition.checkpoint
                ),
            )
        )

        (
            native_custom,
            native_max_abs,
            native_mean_abs,
        ) = (
            validate_native_equivalence(
                condition=condition,
                reference_model=(
                    reference_model
                ),
                normalized_input=(
                    normalized_input
                ),
                val_idx=(
                    val_idx
                ),
            )
        )

        print(
            "[VALIDATION] native deterministic "
            "equivalence passed: "
            f"max_abs={native_max_abs:.3e}, "
            f"mean_abs={native_mean_abs:.3e}",
            flush=True,
        )

        analysis_model = (
            build_scw_model(
                source_bits=(
                    condition.source_bits
                ),
                state_bits=(
                    STATE_BITS
                ),
                counter_bits=(
                    condition.counter_bits
                ),
                deadzone_fraction=(
                    condition.deadzone_fraction
                ),
                build_encoder_batch=(
                    np.asarray(
                        normalized_input[
                            val_idx[
                                :1
                            ]
                        ],
                        dtype=np.float32,
                    )
                ),
            )
        )

        transfer_standard_weights(
            reference_model,
            analysis_model,
        )

        trace_deterministic = (
            make_trace_function(
                analysis_model,
                "deterministic",
            )
        )

        trace_scw = (
            make_trace_function(
                analysis_model,
                "scw",
            )
        )

        (
            det_trace_max_abs,
            det_trace_mean_abs,
            scw_trace_max_abs,
            scw_trace_mean_abs,
        ) = (
            validate_trace_equivalence(
                condition=condition,
                model=(
                    analysis_model
                ),
                trace_deterministic=(
                    trace_deterministic
                ),
                trace_scw=(
                    trace_scw
                ),
                normalized_input=(
                    normalized_input
                ),
                val_idx=(
                    val_idx
                ),
            )
        )

        print(
            "[VALIDATION] trace equivalence passed: "
            f"det_max="
            f"{det_trace_max_abs:.3e}, "
            f"scw_max="
            f"{scw_trace_max_abs:.3e}",
            flush=True,
        )

        deterministic_acc = (
            run_full_test_trace(
                condition=condition,
                trace_fn=(
                    trace_deterministic
                ),
                operator_mode=(
                    "deterministic"
                ),
                normalized_input=(
                    normalized_input
                ),
                test_idx=(
                    test_idx
                ),
                batch_size=(
                    args.batch_size
                ),
            )
        )

        scw_acc = (
            run_full_test_trace(
                condition=condition,
                trace_fn=(
                    trace_scw
                ),
                operator_mode=(
                    "scw"
                ),
                normalized_input=(
                    normalized_input
                ),
                test_idx=(
                    test_idx
                ),
                batch_size=(
                    args.batch_size
                ),
            )
        )

        manuscript_checks = (
            validate_against_manuscript(
                condition=condition,
                deterministic=(
                    deterministic_acc
                ),
                scw=(
                    scw_acc
                ),
            )
        )

        print(
            "[VALIDATION] manuscript recurrent "
            "metrics passed for "
            f"{condition.display_name}",
            flush=True,
        )

        all_accumulators[
            condition.key
        ] = {
            "deterministic": (
                deterministic_acc
            ),
            "scw": (
                scw_acc
            ),
        }

        all_summaries[
            condition.key
        ] = {}

        for (
            mode_index,
            (
                mode,
                acc,
            ),
        ) in enumerate(
            all_accumulators[
                condition.key
            ].items()
        ):
            all_summaries[
                condition.key
            ][
                mode
            ] = {
                "scalar": (
                    acc.scalar_summary()
                ),
                "transition": (
                    acc.transition_summary()
                ),
                "bootstrap": (
                    bootstrap_transition_statistics(
                        per_sequence_counts=(
                            acc.transition_counts
                        ),
                        replicates=(
                            args.bootstrap_replicates
                        ),
                        seed=(
                            args.bootstrap_seed
                            + 1000
                            * condition_index
                            + 100
                            * mode_index
                        ),
                    )
                ),
            }

            save_per_sequence_npz(
                output_dir=(
                    args.output_dir
                ),
                condition_key=(
                    condition.key
                ),
                mode=(
                    mode
                ),
                acc=(
                    acc
                ),
            )

        validation_payload[
            condition.key
        ] = {
            "native_deterministic_equivalence": {
                "max_abs": (
                    native_max_abs
                ),
                "mean_abs": (
                    native_mean_abs
                ),
                "tolerance": (
                    EQUIVALENCE_TOLERANCE
                ),
                "passed": True,
            },
            "deterministic_trace_equivalence": {
                "max_abs": (
                    det_trace_max_abs
                ),
                "mean_abs": (
                    det_trace_mean_abs
                ),
                "tolerance": (
                    EQUIVALENCE_TOLERANCE
                ),
                "passed": True,
            },
            "scw_trace_equivalence": {
                "max_abs": (
                    scw_trace_max_abs
                ),
                "mean_abs": (
                    scw_trace_mean_abs
                ),
                "tolerance": (
                    EQUIVALENCE_TOLERANCE
                ),
                "passed": True,
            },
            "manuscript_metric_checks": (
                manuscript_checks
            ),
        }

        del native_custom
        del analysis_model
        del reference_model

        tf.keras.backend.clear_session()

    write_json(
        args.output_dir
        / "validation.json",
        validation_payload,
    )

    write_json(
        args.output_dir
        / "sign_persistence_summary.json",
        all_summaries,
    )

    write_summary_csv(
        args.output_dir
        / "sign_persistence_summary.csv",
        all_summaries,
    )

    write_run_length_csv(
        args.output_dir
        / "run_length_distributions.csv",
        all_accumulators,
    )

    write_trigger_segment_csv(
        args.output_dir
        / "trigger_segment_vote_distribution.csv",
        all_accumulators,
    )

    make_figure(
        output_dir=(
            args.output_dir
        ),
        conditions=(
            conditions
        ),
        accumulators=(
            all_accumulators
        ),
        summaries=(
            all_summaries
        ),
    )

    print(
        "\n[DONE] SCW sign-persistence "
        "analysis completed.",
        flush=True,
    )

    print(
        f"[DONE] Outputs written to "
        f"{args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()