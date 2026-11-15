"""
test_prototype.py
=================
Unit test suite for Capstone Phase-1 Demonstration Prototype.
Tests data generation, schema validation, 4-stage pipeline execution,
deterministic reproducibility, and edge-case handling.
"""

import unittest
import pandas as pd
import numpy as np

import data_generator
from analysis import (
    PrototypeConfig,
    prepare_features,
    analyze_individual_silence,
    analyze_group_patterns,
    generate_attribution_and_explanation,
    run_pipeline,
)


class TestDataGenerator(unittest.TestCase):
    """Tests synthetic dataset creation, schema adherence, and constraints."""

    def test_scenario_1_schema_and_ranges(self):
        df = data_generator.generate_scenario_1_data()
        valid, errors = data_generator.validate_scenario_data(df)
        self.assertTrue(valid, f"Scenario 1 invalid: {errors}")
        self.assertEqual(len(df["hiker_id"].unique()), 4)
        self.assertTrue((df["link_quality"] >= 0.0).all() and (df["link_quality"] <= 1.0).all())
        self.assertTrue((df["movement_speed"] >= 0.0).all())

    def test_scenario_2_schema_and_ranges(self):
        df = data_generator.generate_scenario_2_data()
        valid, errors = data_generator.validate_scenario_data(df)
        self.assertTrue(valid, f"Scenario 2 invalid: {errors}")
        self.assertEqual(len(df["hiker_id"].unique()), 4)
        self.assertTrue((df["link_quality"] >= 0.0).all() and (df["link_quality"] <= 1.0).all())

    def test_scenario_3_schema_and_ranges(self):
        df = data_generator.generate_scenario_3_data()
        valid, errors = data_generator.validate_scenario_data(df)
        self.assertTrue(valid, f"Scenario 3 invalid: {errors}")
        self.assertEqual(len(df["hiker_id"].unique()), 4)

    def test_malformed_data_validation(self):
        # Missing columns
        df_bad = pd.DataFrame({"dummy": [1, 2, 3]})
        valid, errors = data_generator.validate_scenario_data(df_bad)
        self.assertFalse(valid)
        self.assertTrue(any("Missing required columns" in e for e in errors))

        # Invalid link quality (> 1.0)
        df_valid = data_generator.generate_scenario_1_data()
        df_invalid_link = df_valid.copy()
        df_invalid_link.loc[0, "link_quality"] = 1.85
        valid2, errors2 = data_generator.validate_scenario_data(df_invalid_link)
        self.assertFalse(valid2)
        self.assertTrue(any("link_quality" in e for e in errors2))


class TestAnalysisPipeline(unittest.TestCase):
    """Tests the 4-stage rule-based analysis pipeline and deterministic attributions."""

    def setUp(self):
        self.config = PrototypeConfig()

    def test_scenario_1_attribution(self):
        """Scenario 1: Low link quality + sustained movement -> Likely shared connectivity issue"""
        df = data_generator.generate_scenario_1_data()
        res = run_pipeline(df, self.config)
        self.assertEqual(res.classification, "Likely shared connectivity issue")
        self.assertTrue(res.group_summary.is_simultaneous_silence)
        self.assertTrue(res.group_summary.is_group_link_degraded)
        self.assertFalse(res.group_summary.is_group_halted)

    def test_scenario_2_attribution(self):
        """Scenario 2: Single hiker silent + halted while peers report -> Individual anomaly — check required"""
        df = data_generator.generate_scenario_2_data()
        res = run_pipeline(df, self.config)
        self.assertEqual(res.classification, "Individual anomaly — check required")
        self.assertTrue(res.group_summary.isolation_detected)
        self.assertEqual(res.group_summary.isolated_hiker_id, "H4")
        self.assertTrue(res.individual_summary["H4"].is_movement_stopped)

    def test_scenario_3_attribution(self):
        """Scenario 3: Multiple silent hikers + group movement halted + adequate link quality -> Group risk — follow-up required"""
        df = data_generator.generate_scenario_3_data()
        res = run_pipeline(df, self.config)
        self.assertEqual(res.classification, "Group risk — follow-up required")
        self.assertTrue(res.group_summary.is_simultaneous_silence)
        self.assertTrue(res.group_summary.is_group_halted)
        self.assertFalse(res.group_summary.is_group_link_degraded)
        # Verify Group Risk Rule in trace matched and all 4 conditions were True
        grp_rule_trace = next(r for r in res.rule_criteria_met if r["rule"] == "Group Risk Rule")
        self.assertTrue(grp_rule_trace["matched"])
        self.assertTrue(all(grp_rule_trace["conditions"].values()))

    def test_group_risk_requires_adequate_link_quality(self):
        """If group is silent and halted but link quality is severely degraded (RF blackout), Group Risk Rule must NOT match."""
        df = data_generator.generate_scenario_3_data()
        # Simulate degraded RF link across all hikers in latest intervals
        df_degraded = df.copy()
        df_degraded["link_quality"] = 0.15
        res = run_pipeline(df_degraded, self.config)
        grp_rule_trace = next(r for r in res.rule_criteria_met if r["rule"] == "Group Risk Rule")
        self.assertFalse(grp_rule_trace["matched"])
        self.assertNotEqual(res.classification, "Group risk — follow-up required")

    def test_insufficient_evidence_at_early_timestamp(self):
        """At 09:05 when everyone is reporting normally, outcome should be Insufficient evidence"""
        df = data_generator.generate_scenario_1_data()
        res = run_pipeline(df, self.config, at_timestamp="09:05")
        self.assertEqual(res.classification, "Insufficient evidence")

    def test_deterministic_repeatability(self):
        """Repeated runs on the same scenario must produce identical outputs."""
        df = data_generator.generate_scenario_2_data()
        res1 = run_pipeline(df, self.config)
        res2 = run_pipeline(df, self.config)
        self.assertEqual(res1.classification, res2.classification)
        self.assertEqual(res1.summary, res2.summary)
        self.assertEqual(res1.reasoning_points, res2.reasoning_points)


if __name__ == "__main__":
    unittest.main()
