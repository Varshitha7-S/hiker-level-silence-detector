"""
data_generator.py
=================
Deterministic simulated data generator and loader for Capstone Phase-1 Review:
"Event-Driven SNN-GNN Framework for Terrain-Aware Silence Attribution and Shared-Fate Detection in Offline Hiking Groups"

NOTICE:
- All data generated here is SYNTHETIC / SIMULATED for demonstration purposes.
- This is NOT real sensor data, and no real-world emergency predictions are made.
"""

import os
from typing import Dict, List, Tuple
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp",
    "scenario",
    "group_id",
    "hiker_id",
    "route_segment",
    "terrain_type",
    "link_quality",
    "checkin_received",
    "movement_speed",
]

SCENARIO_METADATA = {
    "scenario_1": {
        "id": "scenario_1",
        "name": "1. Shared Coverage Gap",
        "filename": "scenario_1_shared_coverage_gap.csv",
        "description": (
            "Several hikers miss check-ins simultaneously in a deep canyon with low link quality, "
            "while their movement speeds remain normal (~3.5–4.2 km/h). "
            "Expected outcome: Likely shared connectivity issue."
        ),
        "expected_label": "Likely shared connectivity issue",
    },
    "scenario_2": {
        "id": "scenario_2",
        "name": "2. Individual Anomaly",
        "filename": "scenario_2_individual_anomaly.csv",
        "description": (
            "One hiker (H4 - Sweep) misses repeated check-ins and stops moving (0.0 km/h) in a segment "
            "with good link quality, while peer hikers continue checking in and moving normally. "
            "Expected outcome: Individual anomaly — check required."
        ),
        "expected_label": "Individual anomaly — check required",
    },
    "scenario_3": {
        "id": "scenario_3",
        "name": "3. Possible Group Risk",
        "filename": "scenario_3_possible_group_risk.csv",
        "description": (
            "All hikers miss repeated check-ins simultaneously in a moderate/good link quality segment, "
            "and all movement drops to 0.0 km/h (abrupt group halt). "
            "Expected outcome: Group risk — follow-up required."
        ),
        "expected_label": "Group risk — follow-up required",
    },
}

TIMESTAMPS = [
    "09:00",
    "09:05",
    "09:10",
    "09:15",
    "09:20",
    "09:25",
    "09:30",
    "09:35",
]

HIKERS = [
    {"id": "H1", "role": "Hiker 1 (Lead)"},
    {"id": "H2", "role": "Hiker 2 (Navigator)"},
    {"id": "H3", "role": "Hiker 3 (Pacer)"},
    {"id": "H4", "role": "Hiker 4 (Sweep)"},
]


def generate_scenario_1_data() -> pd.DataFrame:
    """
    Scenario 1: Shared coverage gap
    - Route transitions from Open Ridge -> Canyon Entry -> Deep Canyon (Shadow Gorge).
    - In Deep Canyon (09:15 onwards), link quality drops to 0.12 - 0.22.
    - All 4 hikers miss check-ins simultaneously due to RF obstruction in the canyon.
    - Movement speed continues at normal steady hiking pace (3.6 - 4.1 km/h).
    - Expected classification: 'Likely shared connectivity issue'
    """
    rows = []
    group_id = "GROUP-ALPHA"
    scenario_label = "Shared coverage gap"

    timeline = [
        ("09:00", "Seg-1: West Ridge", "Open Ridge", 0.86, True, [3.9, 4.0, 3.8, 3.7]),
        ("09:05", "Seg-1: West Ridge", "Open Ridge", 0.83, True, [4.0, 4.1, 3.9, 3.8]),
        ("09:10", "Seg-2: Canyon Rim", "Rocky Slope", 0.68, True, [3.7, 3.8, 3.6, 3.5]),
        ("09:15", "Seg-3: Shadow Gorge", "Deep Canyon", 0.22, False, [3.8, 3.9, 3.7, 3.6]),
        ("09:20", "Seg-3: Shadow Gorge", "Deep Canyon", 0.16, False, [3.7, 3.8, 3.6, 3.5]),
        ("09:25", "Seg-3: Shadow Gorge", "Deep Canyon", 0.14, False, [3.9, 4.0, 3.7, 3.6]),
        ("09:30", "Seg-3: Shadow Gorge", "Deep Canyon", 0.15, False, [3.8, 3.9, 3.8, 3.7]),
        ("09:35", "Seg-3: Shadow Gorge", "Deep Canyon", 0.18, False, [4.0, 4.1, 3.9, 3.8]),
    ]

    for t, seg, terrain, base_link, checkin, speeds in timeline:
        for idx, hiker in enumerate(HIKERS):
            link = round(base_link + (idx * 0.01 - 0.015), 2)
            link = max(0.05, min(0.98, link))
            spd = speeds[idx]
            rows.append({
                "timestamp": t,
                "scenario": scenario_label,
                "group_id": group_id,
                "hiker_id": hiker["id"],
                "route_segment": seg,
                "terrain_type": terrain,
                "link_quality": link,
                "checkin_received": checkin,
                "movement_speed": spd,
            })

    return pd.DataFrame(rows)


def generate_scenario_2_data() -> pd.DataFrame:
    """
    Scenario 2: Individual anomaly
    - Route stays in moderate/good terrain (Open Ridge / Pine Forest).
    - Link quality stays consistently strong (0.75 - 0.88).
    - Hikers H1, H2, H3 continue regular check-ins and normal speed (~3.8 - 4.2 km/h).
    - Hiker H4 (Sweep) stops checking in from 09:15 onwards (repeated misses: 09:15, 09:20, 09:25, 09:30, 09:35).
    - Hiker H4's speed drops abruptly from 3.6 km/h at 09:10 to 0.0 km/h at 09:15+.
    """
    rows = []
    group_id = "GROUP-ALPHA"
    scenario_label = "Individual anomaly"

    timeline = [
        ("09:00", "Seg-1: West Ridge", "Open Ridge", 0.88),
        ("09:05", "Seg-1: West Ridge", "Open Ridge", 0.85),
        ("09:10", "Seg-2: Pine Plateau", "Sparse Forest", 0.82),
        ("09:15", "Seg-2: Pine Plateau", "Sparse Forest", 0.80),
        ("09:20", "Seg-3: High Meadow", "Open Meadow", 0.84),
        ("09:25", "Seg-3: High Meadow", "Open Meadow", 0.86),
        ("09:30", "Seg-4: Lookout Spur", "Open Ridge", 0.85),
        ("09:35", "Seg-4: Lookout Spur", "Open Ridge", 0.87),
    ]

    for t_idx, (t, seg, terrain, base_link) in enumerate(timeline):
        for idx, hiker in enumerate(HIKERS):
            hid = hiker["id"]
            link = round(base_link + (idx * 0.01 - 0.01), 2)

            if hid == "H4":
                # H4 starts missing at 09:15 (t_idx >= 3) and speed drops to 0.0
                if t_idx >= 3:
                    checkin = False
                    spd = 0.0
                else:
                    checkin = True
                    spd = 3.6
            else:
                # Peers are unaffected
                checkin = True
                spd = round(3.8 + (idx * 0.15) + (0.05 if t_idx % 2 == 0 else -0.05), 1)

            rows.append({
                "timestamp": t,
                "scenario": scenario_label,
                "group_id": group_id,
                "hiker_id": hid,
                "route_segment": seg,
                "terrain_type": terrain,
                "link_quality": link,
                "checkin_received": checkin,
                "movement_speed": spd,
            })

    return pd.DataFrame(rows)


def generate_scenario_3_data() -> pd.DataFrame:
    """
    Scenario 3: Possible group risk
    - Group hikes through Granite Pass / Talus Slope.
    - Terrain has reasonable link quality (0.68 - 0.76) — NOT a radio blackout canyon.
    - At 09:15 onwards, ALL hikers abruptly miss check-ins (09:15, 09:20, 09:25, 09:30, 09:35 all False).
    - Crucially, movement speed for ALL hikers drops to 0.0 km/h (group halted unexpectedly).
    """
    rows = []
    group_id = "GROUP-ALPHA"
    scenario_label = "Possible group risk"

    timeline = [
        ("09:00", "Seg-1: Talus Approach", "Rocky Trail", 0.78),
        ("09:05", "Seg-1: Talus Approach", "Rocky Trail", 0.75),
        ("09:10", "Seg-2: Granite Pass", "Boulder Field", 0.72),
        ("09:15", "Seg-2: Granite Pass", "Boulder Field", 0.70),
        ("09:20", "Seg-2: Granite Pass", "Boulder Field", 0.69),
        ("09:25", "Seg-2: Granite Pass", "Boulder Field", 0.68),
        ("09:30", "Seg-2: Granite Pass", "Boulder Field", 0.71),
        ("09:35", "Seg-2: Granite Pass", "Boulder Field", 0.70),
    ]

    for t_idx, (t, seg, terrain, base_link) in enumerate(timeline):
        for idx, hiker in enumerate(HIKERS):
            hid = hiker["id"]
            link = round(base_link + (idx * 0.01 - 0.015), 2)

            if t_idx >= 3:  # 09:15 onwards: simultaneous failure + complete halt
                checkin = False
                spd = 0.0
            else:
                checkin = True
                spd = round(3.5 + (idx * 0.2), 1)

            rows.append({
                "timestamp": t,
                "scenario": scenario_label,
                "group_id": group_id,
                "hiker_id": hid,
                "route_segment": seg,
                "terrain_type": terrain,
                "link_quality": link,
                "checkin_received": checkin,
                "movement_speed": spd,
            })

    return pd.DataFrame(rows)


def validate_scenario_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates that a dataset matches all required schema and value constraints.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    if df is None or df.empty:
        return False, ["Dataset is empty or None."]

    # Check required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if errors:
        return False, errors

    # Check nulls
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        errors.append(f"Null/NaN values detected in columns: {null_cols.to_dict()}")

    # Check link_quality ranges
    if not df["link_quality"].between(0.0, 1.0).all():
        invalid_links = df[~df["link_quality"].between(0.0, 1.0)]
        errors.append(
            f"link_quality values must be between 0.0 and 1.0. Found invalid values: {invalid_links['link_quality'].tolist()}"
        )

    # Check movement_speed
    if (df["movement_speed"] < 0).any():
        invalid_speeds = df[df["movement_speed"] < 0]
        errors.append(f"movement_speed cannot be negative. Found: {invalid_speeds['movement_speed'].tolist()}")

    # Check checkin_received is boolean-compatible
    if not df["checkin_received"].isin([True, False, 1, 0, "True", "False"]).all():
        errors.append("checkin_received must contain boolean values (True/False).")

    is_valid = len(errors) == 0
    return is_valid, errors


def export_all_scenarios(data_dir: str = "data") -> Dict[str, str]:
    """Generates and writes CSV files for all 3 scenarios into the data_dir."""
    os.makedirs(data_dir, exist_ok=True)
    paths = {}

    scenarios = {
        "scenario_1": generate_scenario_1_data(),
        "scenario_2": generate_scenario_2_data(),
        "scenario_3": generate_scenario_3_data(),
    }

    for key, df in scenarios.items():
        meta = SCENARIO_METADATA[key]
        filepath = os.path.join(data_dir, meta["filename"])
        df.to_csv(filepath, index=False)
        paths[key] = filepath

    return paths


def load_scenario(scenario_key: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Loads a scenario DataFrame from CSV, or generates it on the fly if CSV doesn't exist.
    Validates data integrity before returning.
    """
    generators = {
        "scenario_1": generate_scenario_1_data,
        "scenario_2": generate_scenario_2_data,
        "scenario_3": generate_scenario_3_data,
    }

    if scenario_key not in SCENARIO_METADATA:
        raise ValueError(f"Unknown scenario key: '{scenario_key}'. Available: {list(SCENARIO_METADATA.keys())}")

    filename = SCENARIO_METADATA[scenario_key]["filename"]
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            # Ensure boolean type for checkin_received
            df["checkin_received"] = df["checkin_received"].astype(bool)
            df["link_quality"] = df["link_quality"].astype(float)
            df["movement_speed"] = df["movement_speed"].astype(float)
        except Exception as e:
            # Fallback to generator if file is corrupt
            df = generators[scenario_key]()
    else:
        df = generators[scenario_key]()
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(filepath, index=False)

    valid, errors = validate_scenario_data(df)
    if not valid:
        raise ValueError(f"Scenario data validation failed for {scenario_key}: {'; '.join(errors)}")

    return df


if __name__ == "__main__":
    saved = export_all_scenarios()
    print("Exported all simulated scenario CSV files successfully:")
    for k, v in saved.items():
        print(f" - {k}: {v}")
