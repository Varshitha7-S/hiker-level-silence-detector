"""
analysis.py
===========
Core analytical engine for Capstone Phase-1 Review:
"Event-Driven SNN-GNN Framework for Terrain-Aware Silence Attribution and Shared-Fate Detection in Offline Hiking Groups"

ARCHITECTURAL ORGANIZATION:
1. Feature Preparation (temporal and motion aggregation)
2. Individual Silence Analysis (heuristic proxy representing the proposed SNN stage)
3. Group-Level Pattern Analysis (heuristic proxy representing the proposed GNN stage)
4. Attribution & Explanation Generation (transparent rule-based inference)

DISCLAIMER:
This module implements a transparent, rule-based prototype using simulated data.
It does NOT use a trained SNN, GNN, or real-world sensor streams.
Results are illustrative demonstrations for Capstone Phase-1 and are NOT validated safety predictions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


@dataclass
class PrototypeConfig:
    """Configurable prototype thresholds (illustrative, not validated safety limits)."""
    low_link_threshold: float = 0.35          # Below this, link quality is considered degraded
    consecutive_miss_threshold: int = 2       # Number of consecutive misses to flag persistent silence
    group_silence_min_hikers: int = 2         # Minimum number of silent hikers to constitute group silence
    movement_halt_threshold: float = 0.8      # Speed in km/h below which a hiker is considered stationary
    movement_drop_pct_threshold: float = 60.0 # Percentage speed reduction considered a significant halt


@dataclass
class IndividualHikerAnalysis:
    """Individual hiker evaluation (Stage 2 proxy for proposed SNN stage)."""
    hiker_id: str
    total_observations: int
    total_misses: int
    consecutive_misses: int
    is_currently_silent: bool
    latest_speed: float
    baseline_speed: float
    speed_drop_pct: float
    is_movement_stopped: bool
    latest_link_quality: float
    latest_terrain: str
    latest_segment: str
    individual_anomaly_score: float
    notes: List[str] = field(default_factory=list)


@dataclass
class GroupPatternAnalysis:
    """Group-level relational evaluation (Stage 3 proxy for proposed GNN stage)."""
    total_hikers: int
    silent_hikers_count: int
    silent_hiker_ids: List[str]
    reporting_hiker_ids: List[str]
    is_simultaneous_silence: bool
    mean_group_link_quality: float
    is_group_link_degraded: bool
    mean_group_speed: float
    is_group_halted: bool
    dominant_terrain: str
    active_segment: str
    isolation_detected: bool
    isolated_hiker_id: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class AttributionResult:
    """Synthesized final output and reasoning trace (Stage 4)."""
    classification: str
    badge_color: str
    confidence_descriptor: str
    summary: str
    reasoning_points: List[str]
    rule_criteria_met: List[Dict[str, Any]]
    individual_summary: Dict[str, IndividualHikerAnalysis]
    group_summary: GroupPatternAnalysis
    config_snapshot: Dict[str, Any]


# ============================================================================
# STAGE 1: FEATURE PREPARATION
# ============================================================================

def prepare_features(df: pd.DataFrame, at_timestamp: Optional[str] = None) -> pd.DataFrame:
    """
    Stage 1: Feature preparation
    Prepares temporal sequences, calculates consecutive silence counters,
    baseline speeds, and velocity deltas per hiker up to the given timestamp.
    """
    if df is None or df.empty:
        raise ValueError("Cannot prepare features on empty DataFrame.")

    df_prep = df.copy()

    # Filter up to selected timestamp if requested
    if at_timestamp is not None:
        df_prep = df_prep[df_prep["timestamp"] <= at_timestamp].copy()
        if df_prep.empty:
            raise ValueError(f"No records found up to timestamp '{at_timestamp}'.")

    # Ensure consistent sorting by hiker and timestamp
    df_prep = df_prep.sort_values(by=["hiker_id", "timestamp"]).reset_index(drop=True)

    # Boolean checkin flag
    df_prep["missed_checkin"] = (~df_prep["checkin_received"]).astype(int)

    # Compute consecutive missed check-ins per hiker using cumulative grouping
    def calc_consecutive_misses(group: pd.DataFrame) -> pd.Series:
        # Resets counter whenever checkin_received is True
        missed = (~group["checkin_received"]).astype(int)
        cumsum = (group["checkin_received"]).cumsum()
        return missed.groupby(cumsum).cumsum()

    df_prep["consecutive_misses"] = df_prep.groupby("hiker_id", group_keys=False).apply(
        calc_consecutive_misses
    )

    # Compute baseline movement speed (first 2 observations or overall mean when moving)
    def calc_baseline_speed(group: pd.DataFrame) -> float:
        initial = group.head(2)["movement_speed"]
        if len(initial) > 0 and initial.mean() > 0:
            return float(initial.mean())
        return float(group["movement_speed"].mean())

    baseline_speeds = df_prep.groupby("hiker_id").apply(calc_baseline_speed).to_dict()
    df_prep["baseline_speed"] = df_prep["hiker_id"].map(baseline_speeds)

    # Speed change ratio
    df_prep["speed_drop_pct"] = np.where(
        df_prep["baseline_speed"] > 0,
        np.maximum(0.0, (df_prep["baseline_speed"] - df_prep["movement_speed"]) / df_prep["baseline_speed"] * 100.0),
        0.0
    )

    return df_prep


# ============================================================================
# STAGE 2: INDIVIDUAL SILENCE ANALYSIS (Proposed SNN Stage Proxy)
# ============================================================================

def analyze_individual_silence(
    df_features: pd.DataFrame,
    config: PrototypeConfig
) -> Dict[str, IndividualHikerAnalysis]:
    """
    Stage 2: Individual silence & mobility analysis
    In the proposed framework, an Event-Driven Spiking Neural Network (SNN) processes
    asynchronous spike trains of missing check-in events and inertial movement changes.
    Here, a rule-based heuristic proxy evaluates each hiker's local temporal profile.
    """
    individual_results: Dict[str, IndividualHikerAnalysis] = {}
    hikers = df_features["hiker_id"].unique()

    for hid in sorted(hikers):
        h_data = df_features[df_features["hiker_id"] == hid].sort_values("timestamp")
        latest = h_data.iloc[-1]

        consecutive_misses = int(latest["consecutive_misses"])
        total_misses = int(h_data["missed_checkin"].sum())
        is_currently_silent = not bool(latest["checkin_received"])
        latest_speed = float(latest["movement_speed"])
        baseline_speed = float(latest["baseline_speed"])
        speed_drop_pct = float(latest["speed_drop_pct"])
        latest_link = float(latest["link_quality"])
        latest_terrain = str(latest["terrain_type"])
        latest_segment = str(latest["route_segment"])

        is_movement_stopped = (
            latest_speed <= config.movement_halt_threshold or
            speed_drop_pct >= config.movement_drop_pct_threshold
        )

        notes = []
        score = 0.0

        if is_currently_silent:
            notes.append(f"Currently silent at latest interval ({latest['timestamp']}).")
            score += 0.3
        if consecutive_misses >= config.consecutive_miss_threshold:
            notes.append(f"Repeated silence: {consecutive_misses} consecutive missed check-ins (threshold >= {config.consecutive_miss_threshold}).")
            score += 0.4
        if is_movement_stopped:
            notes.append(f"Movement halt detected: speed {latest_speed:.1f} km/h (baseline {baseline_speed:.1f} km/h, -{speed_drop_pct:.0f}%).")
            score += 0.3
        else:
            notes.append(f"Movement active: speed {latest_speed:.1f} km/h (consistent with baseline {baseline_speed:.1f} km/h).")

        individual_results[hid] = IndividualHikerAnalysis(
            hiker_id=hid,
            total_observations=len(h_data),
            total_misses=total_misses,
            consecutive_misses=consecutive_misses,
            is_currently_silent=is_currently_silent,
            latest_speed=latest_speed,
            baseline_speed=baseline_speed,
            speed_drop_pct=speed_drop_pct,
            is_movement_stopped=is_movement_stopped,
            latest_link_quality=latest_link,
            latest_terrain=latest_terrain,
            latest_segment=latest_segment,
            individual_anomaly_score=round(score, 2),
            notes=notes,
        )

    return individual_results


# ============================================================================
# STAGE 3: GROUP-LEVEL PATTERN ANALYSIS (Proposed GNN Stage Proxy)
# ============================================================================

def analyze_group_patterns(
    df_features: pd.DataFrame,
    ind_results: Dict[str, IndividualHikerAnalysis],
    config: PrototypeConfig
) -> GroupPatternAnalysis:
    """
    Stage 3: Group-level spatial & topological pattern analysis
    In the proposed framework, a Graph Neural Network (GNN) models the group topology,
    inter-agent distances, and shared terrain edges to capture 'shared fate'.
    Here, a rule-based heuristic proxy checks for simultaneous co-occurrence,
    peer isolation, and collective mobility states.
    """
    total_hikers = len(ind_results)
    silent_hikers = [hid for hid, h in ind_results.items() if h.is_currently_silent]
    reporting_hikers = [hid for hid, h in ind_results.items() if not h.is_currently_silent]
    silent_count = len(silent_hikers)

    # Check latest timestamp across the group
    latest_ts = df_features["timestamp"].max()
    latest_slice = df_features[df_features["timestamp"] == latest_ts]

    mean_group_link = float(latest_slice["link_quality"].mean())
    is_group_link_degraded = mean_group_link <= config.low_link_threshold
    mean_group_speed = float(latest_slice["movement_speed"].mean())
    is_group_halted = mean_group_speed <= config.movement_halt_threshold

    dominant_terrain = str(latest_slice["terrain_type"].mode().iloc[0])
    active_segment = str(latest_slice["route_segment"].mode().iloc[0])

    is_simultaneous_silence = silent_count >= config.group_silence_min_hikers

    # Isolation check: exactly 1 hiker silent with persistent misses while all peers report normally
    isolation_detected = False
    isolated_hiker_id = None
    if silent_count == 1:
        lone_silent = silent_hikers[0]
        if ind_results[lone_silent].consecutive_misses >= config.consecutive_miss_threshold:
            isolation_detected = True
            isolated_hiker_id = lone_silent

    notes = []
    if is_simultaneous_silence:
        notes.append(
            f"Simultaneous silence: {silent_count}/{total_hikers} hikers silent ({', '.join(silent_hikers)})."
        )
    elif silent_count == 0:
        notes.append("Nominal group state: All hikers actively checking in.")
    else:
        notes.append(f"Isolated silence: {silent_count}/{total_hikers} hiker silent ({', '.join(silent_hikers)}).")

    if is_group_link_degraded:
        notes.append(
            f"RF link degraded: group mean link quality {mean_group_link:.2f} <= threshold {config.low_link_threshold:.2f} in '{dominant_terrain}'."
        )
    else:
        notes.append(
            f"RF link adequate: group mean link quality {mean_group_link:.2f} > threshold {config.low_link_threshold:.2f}."
        )

    if is_group_halted:
        notes.append(f"Group halted: average speed {mean_group_speed:.2f} km/h <= threshold {config.movement_halt_threshold:.2f} km/h.")
    else:
        notes.append(f"Group mobile: average speed {mean_group_speed:.2f} km/h.")

    return GroupPatternAnalysis(
        total_hikers=total_hikers,
        silent_hikers_count=silent_count,
        silent_hiker_ids=silent_hikers,
        reporting_hiker_ids=reporting_hikers,
        is_simultaneous_silence=is_simultaneous_silence,
        mean_group_link_quality=round(mean_group_link, 2),
        is_group_link_degraded=is_group_link_degraded,
        mean_group_speed=round(mean_group_speed, 2),
        is_group_halted=is_group_halted,
        dominant_terrain=dominant_terrain,
        active_segment=active_segment,
        isolation_detected=isolation_detected,
        isolated_hiker_id=isolated_hiker_id,
        notes=notes,
    )


# ============================================================================
# STAGE 4: STATUS & EXPLANATION GENERATION
# ============================================================================

def generate_attribution_and_explanation(
    df_features: pd.DataFrame,
    ind_results: Dict[str, IndividualHikerAnalysis],
    grp_result: GroupPatternAnalysis,
    config: PrototypeConfig
) -> AttributionResult:
    """
    Stage 4: Status and explanation generation
    Applies the transparent, deterministic prototype rules to synthesize individual
    and group patterns into one of the four required outcomes:
    - Likely shared connectivity issue
    - Individual anomaly — check required
    - Group risk — follow-up required
    - Insufficient evidence
    """
    criteria_trace = []
    classification = "Insufficient evidence"
    badge_color = "#6c757d"  # Grey default
    confidence_descriptor = "Low / Inconclusive"
    summary = ""
    reasons = []

    # Count how many silent hikers also experienced movement halt
    silent_and_halted = [
        hid for hid in grp_result.silent_hiker_ids
        if ind_results[hid].is_movement_stopped
    ]
    silent_and_moving = [
        hid for hid in grp_result.silent_hiker_ids
        if not ind_results[hid].is_movement_stopped
    ]

    # Check Rule 3: Group risk — follow-up required
    # Several hikers miss repeated check-ins AND movement stops or changes unexpectedly
    rule3_cond_silence = grp_result.is_simultaneous_silence
    rule3_cond_repeated = any(
        ind_results[h].consecutive_misses >= config.consecutive_miss_threshold
        for h in grp_result.silent_hiker_ids
    ) if grp_result.silent_hiker_ids else False
    rule3_cond_movement_halt = grp_result.is_group_halted or (len(silent_and_halted) >= config.group_silence_min_hikers)
    # Group risk is reinforced when link quality is NOT degraded (cannot be explained by RF blackout alone)
    rule3_cond_not_just_rf = not grp_result.is_group_link_degraded

    criteria_trace.append({
        "rule": "Group Risk Rule",
        "conditions": {
            f"Simultaneous silence (>= {config.group_silence_min_hikers} hikers)": rule3_cond_silence,
            f"Repeated misses (>= {config.consecutive_miss_threshold} intervals)": rule3_cond_repeated,
            f"Group movement halt (speed <= {config.movement_halt_threshold} km/h)": rule3_cond_movement_halt,
            f"Adequate link quality (> {config.low_link_threshold})": rule3_cond_not_just_rf,
        },
        "matched": rule3_cond_silence and rule3_cond_repeated and rule3_cond_movement_halt
    })

    # Check Rule 1: Likely shared connectivity issue
    # Several hikers miss check-ins together in a low-link-quality segment, while movement continues
    rule1_cond_silence = grp_result.is_simultaneous_silence
    rule1_cond_low_link = grp_result.is_group_link_degraded
    rule1_cond_moving = not grp_result.is_group_halted and (len(silent_and_moving) >= config.group_silence_min_hikers)

    criteria_trace.append({
        "rule": "Shared Connectivity Gap Rule",
        "conditions": {
            f"Simultaneous silence (>= {config.group_silence_min_hikers} hikers)": rule1_cond_silence,
            f"Low link quality (mean <= {config.low_link_threshold})": rule1_cond_low_link,
            f"Movement continues (> {config.movement_halt_threshold} km/h)": rule1_cond_moving,
        },
        "matched": rule1_cond_silence and rule1_cond_low_link and rule1_cond_moving
    })

    # Check Rule 2: Individual anomaly — check required
    # One hiker misses repeated check-ins and shows a movement change, while other hikers continue reporting
    rule2_cond_isolated = grp_result.isolation_detected
    rule2_isolated_hiker = grp_result.isolated_hiker_id
    rule2_cond_movement_change = (
        ind_results[rule2_isolated_hiker].is_movement_stopped
        if rule2_isolated_hiker else False
    )
    rule2_cond_peers_reporting = len(grp_result.reporting_hiker_ids) >= 1

    criteria_trace.append({
        "rule": "Individual Anomaly Rule",
        "conditions": {
            "Isolated silence (exactly 1 hiker silent repeatedly)": rule2_cond_isolated,
            f"Target hiker movement stopped/slowed (speed <= {config.movement_halt_threshold} km/h)": rule2_cond_movement_change,
            "Peers continuing normal check-ins": rule2_cond_peers_reporting,
        },
        "matched": rule2_cond_isolated and rule2_cond_movement_change and rule2_cond_peers_reporting
    })

    # Rule precedence:
    # 1. Group Risk (high safety priority)
    # 2. Shared Connectivity Issue (shared environmental explanation)
    # 3. Individual Anomaly (isolated individual event)
    # 4. Insufficient Evidence (default fallback)

    if rule3_cond_silence and rule3_cond_repeated and rule3_cond_movement_halt:
        classification = "Group risk — follow-up required"
        badge_color = "#dc3545"  # Danger Red
        confidence_descriptor = "High (Correlated multi-node silence + collective halt)"
        summary = (
            f"Multiple hikers ({', '.join(grp_result.silent_hiker_ids)}) have missed repeated check-ins "
            f"accompanied by an unexpected collective halt (group speed {grp_result.mean_group_speed:.1f} km/h) "
            f"in {grp_result.active_segment} ({grp_result.dominant_terrain})."
        )
        reasons = [
            f"Coordinated silence: {grp_result.silent_hikers_count} of {grp_result.total_hikers} hikers failed to report simultaneously.",
            f"Persistence: Hiker(s) exceeded the repeated missed check-in threshold (>= {config.consecutive_miss_threshold} cycles).",
            f"Kinematic signature: Collective movement dropped to {grp_result.mean_group_speed:.1f} km/h (below threshold of {config.movement_halt_threshold:.1f} km/h).",
            f"Connectivity context: Average link quality is {grp_result.mean_group_link_quality:.2f} (above RF blackout threshold {config.low_link_threshold:.2f}), making radio shadowing alone an insufficient explanation.",
            "Terrain context: Located in high-gradient/complex terrain segment where collective hazards (rockfall, route blockage) warrant immediate attention.",
        ]

    elif rule1_cond_silence and rule1_cond_low_link and rule1_cond_moving:
        classification = "Likely shared connectivity issue"
        badge_color = "#ffc107"  # Warning Amber
        confidence_descriptor = "High (Correlated multi-node silence + low RF + continuous movement)"
        summary = (
            f"Multiple hikers ({', '.join(grp_result.silent_hiker_ids)}) missed check-ins simultaneously "
            f"inside {grp_result.active_segment} ({grp_result.dominant_terrain}), where link quality is severely degraded "
            f"({grp_result.mean_group_link_quality:.2f} <= {config.low_link_threshold:.2f}). Hiking pace remains normal."
        )
        reasons = [
            f"Simultaneous silence: {grp_result.silent_hikers_count} of {grp_result.total_hikers} hikers missed check-ins concurrently.",
            f"Environmental RF degradation: Segment link quality is {grp_result.mean_group_link_quality:.2f}, crossing the low-link threshold ({config.low_link_threshold:.2f}).",
            f"Topographical correlation: Segment terrain is '{grp_result.dominant_terrain}', known for RF line-of-sight obstruction.",
            f"Normal mobility: Group continues moving at {grp_result.mean_group_speed:.1f} km/h (well above halt threshold {config.movement_halt_threshold:.1f} km/h), indicating progress along trail.",
            "Shared fate inference: The simultaneous silence aligns with route geography rather than individual distress.",
        ]

    elif rule2_cond_isolated and rule2_cond_movement_change and rule2_cond_peers_reporting:
        iso_id = grp_result.isolated_hiker_id
        iso_info = ind_results[iso_id]
        classification = "Individual anomaly — check required"
        badge_color = "#fd7e14"  # Orange Alert
        confidence_descriptor = "High (Single-node silence + halt while peer network operates normally)"
        summary = (
            f"Hiker {iso_id} has missed {iso_info.consecutive_misses} consecutive check-ins and stopped moving "
            f"(speed: {iso_info.latest_speed:.1f} km/h), while peer hikers ({', '.join(grp_result.reporting_hiker_ids)}) "
            f"continue active check-ins and steady pace in {grp_result.active_segment}."
        )
        reasons = [
            f"Isolated node silence: Only {iso_id} is non-reporting; {len(grp_result.reporting_hiker_ids)} peers ({', '.join(grp_result.reporting_hiker_ids)}) are communicating successfully.",
            f"Persistence: {iso_id} has missed {iso_info.consecutive_misses} consecutive check-in cycles (threshold: >= {config.consecutive_miss_threshold}).",
            f"Kinematic arrest: {iso_id}'s speed dropped from baseline {iso_info.baseline_speed:.1f} km/h to {iso_info.latest_speed:.1f} km/h (-{iso_info.speed_drop_pct:.0f}%).",
            f"Good environmental connectivity: Segment link quality is {grp_result.mean_group_link_quality:.2f} (> threshold {config.low_link_threshold:.2f}), ruling out a shared radio gap.",
            "Attribution: Divergence between peer health and single-hiker cessation suggests individual separation or local impediment.",
        ]

    else:
        # Fallback / Insufficient evidence
        classification = "Insufficient evidence"
        badge_color = "#6c757d"  # Grey
        confidence_descriptor = "Inconclusive / Mixed indicators"
        silent_str = ", ".join(grp_result.silent_hiker_ids) if grp_result.silent_hiker_ids else "None"
        summary = (
            f"The observed data pattern does not decisively match defined prototype rules. "
            f"Silent hikers: {silent_str}. Group link quality: {grp_result.mean_group_link_quality:.2f}. "
            f"Group speed: {grp_result.mean_group_speed:.1f} km/h."
        )
        reasons = [
            "Silence criteria: Observed missed check-ins do not meet the minimum simultaneous group count or persistence thresholds.",
            f"Link quality value: Segment link quality is {grp_result.mean_group_link_quality:.2f} (boundary: {config.low_link_threshold:.2f}).",
            f"Kinematics: Movement speed of {grp_result.mean_group_speed:.1f} km/h does not show uniform cessation or normal traversal consistency.",
            "Evaluation note: Thresholds may need adjustment, or additional time intervals are required to distinguish transient latency from sustained anomalies.",
        ]

    config_snapshot = {
        "low_link_threshold": config.low_link_threshold,
        "consecutive_miss_threshold": config.consecutive_miss_threshold,
        "group_silence_min_hikers": config.group_silence_min_hikers,
        "movement_halt_threshold": config.movement_halt_threshold,
        "movement_drop_pct_threshold": config.movement_drop_pct_threshold,
    }

    return AttributionResult(
        classification=classification,
        badge_color=badge_color,
        confidence_descriptor=confidence_descriptor,
        summary=summary,
        reasoning_points=reasons,
        rule_criteria_met=criteria_trace,
        individual_summary=ind_results,
        group_summary=grp_result,
        config_snapshot=config_snapshot,
    )


# ============================================================================
# HIGH LEVEL PIPELINE ENTRY POINT
# ============================================================================

def run_pipeline(
    df: pd.DataFrame,
    config: Optional[PrototypeConfig] = None,
    at_timestamp: Optional[str] = None
) -> AttributionResult:
    """
    Executes the 4-stage analytical pipeline:
    1. Feature Preparation
    2. Individual Silence Analysis (Proposed SNN proxy)
    3. Group-Level Pattern Analysis (Proposed GNN proxy)
    4. Attribution and Explanation Generation
    """
    if config is None:
        config = PrototypeConfig()

    # Stage 1: Feature preparation
    df_features = prepare_features(df, at_timestamp=at_timestamp)

    # Stage 2: Individual silence & mobility analysis
    individual_results = analyze_individual_silence(df_features, config)

    # Stage 3: Group-level pattern analysis
    group_results = analyze_group_patterns(df_features, individual_results, config)

    # Stage 4: Status and explanation generation
    result = generate_attribution_and_explanation(df_features, individual_results, group_results, config)

    return result
