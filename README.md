# Event-Driven SNN-GNN Framework for Terrain-Aware Silence Attribution and Shared-Fate Detection in Offline Hiking Groups

**Capstone Phase-1 Review Demonstration Prototype**

> ⚠️ **IMPORTANT PROTOTYPE NOTICE & DISCLAIMER**  
> **Prototype demo — simulated data, rule-based analysis; not a validated safety system.**  
> This prototype demonstrates how missed check-ins in an offline wilderness hiking group can be interpreted using terrain, connectivity, movement, and group-level relational patterns. It uses **synthetic simulated data and transparent rule-based heuristics**. It does **NOT** use a trained Spiking Neural Network (SNN), a trained Graph Neural Network (GNN), real-world sensor streams, field tests, or validated emergency predictions. Do not present any result as proof that an emergency has or has not occurred.

---

## 1. Project Overview

In offline wilderness group hiking, hikers traverse complex terrain where radio frequency (RF) signals frequently drop due to line-of-sight obstruction (canyons, dense forest canopies, ravines). When an offline peer misses periodic check-ins, a fundamental attribution challenge arises:

* **Is the silence benign?** (e.g., several hikers walking through an RF-shadowed gorge while continuing normal movement)
* **Is it an individual incident?** (e.g., one hiker falls behind, stops moving, or gets injured while peers report normally)
* **Is it a collective hazard?** (e.g., multiple hikers simultaneously stop moving in a hazardous talus/avalanche zone where radio coverage is actually fine)

This Capstone Phase-1 prototype provides an interactive, presentation-ready web dashboard demonstrating how terrain-aware RF context, local kinematic velocity, and group co-occurrence patterns can be synthesized to attribute missing check-ins accurately.

---

## 2. Decoupled 4-Stage Architectural Pipeline

The codebase is organized into four decoupled stages designed for modular evolution into Phase-2 machine learning models:

```
[Simulated Sensor & RF Stream]
              │
              ▼
   Stage 1: Feature Preparation
   (Cumulative silence runs, velocity deltas, baseline hiking speeds)
              │
              ├───► Stage 2: Individual Silence & Mobility Analysis
              │     (Local temporal & kinematic profile per hiker)
              │     *Heuristic proxy for proposed on-node neuromorphic SNN*
              │
              └───► Stage 3: Group-Level Pattern Analysis
                    (Spatial co-occurrence, peer isolation, collective mobility)
                    *Heuristic proxy for proposed group-topology GNN*
              │
              ▼
   Stage 4: Status Attribution & Explanation Generation
   (Transparent rule synthesis & natural language justification)
```

1. **Stage 1: Feature Preparation (`prepare_features`)**:  
   Computes consecutive missed check-in runs, baseline velocities, and percentage speed drops for each hiker over time.
2. **Stage 2: Individual Silence Analysis (`analyze_individual_silence`)**:  
   *Proposed Future Model:* On-node event-driven Spiking Neural Network (SNN) processing asynchronous spike trains of missed heartbeats and inertial measurement unit (IMU) telemetry.  
   *Current Prototype:* Local rule-based heuristic evaluating consecutive misses and velocity cessation.
3. **Stage 3: Group-Level Pattern Analysis (`analyze_group_patterns`)**:  
   *Proposed Future Model:* Graph Neural Network (GNN) aggregating spatial mesh edges, inter-hiker distances, and terrain obstruction matrices.  
   *Current Prototype:* Spatial-relational rules assessing simultaneous silence, peer isolation, and group movement states.
4. **Stage 4: Status Attribution & Explanation (`generate_attribution_and_explanation`)**:  
   Synthesizes findings into one of four deterministic outcomes with an explicit reason trace:
   - **“Likely shared connectivity issue”**
   - **“Individual anomaly — check required”**
   - **“Group risk — follow-up required”**
   - **“Insufficient evidence”**

---

## 3. Demo Scenarios

The prototype includes three deterministic scenarios featuring 4 hikers:
- `H1`: Lead
- `H2`: Navigator
- `H3`: Pacer
- `H4`: Sweep

| Scenario | Route & Terrain | Link Quality (RF) | Check-In Status | Movement Speeds | Attributed Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Shared Coverage Gap** | Shadow Gorge *(Deep Canyon)* | Low (0.12 – 0.22) | All 4 hikers miss check-ins simultaneously | Normal & steady (3.6 – 4.1 km/h) | **“Likely shared connectivity issue”** |
| **2. Individual Anomaly** | Lookout Spur *(Open Ridge)* | High (0.80 – 0.88) | H4 (Sweep) misses 5 consecutive check-ins; H1–H3 report normally | H4 drops to 0.0 km/h; H1–H3 continue (3.8 – 4.2 km/h) | **“Individual anomaly — check required”** |
| **3. Possible Group Risk** | Granite Pass *(Boulder Field)* | Adequate (0.68 – 0.76) | All 4 hikers miss check-ins simultaneously | Entire group abruptly drops to 0.0 km/h | **“Group risk — follow-up required”** |

*Deterministic Guarantee:* Selecting the same scenario and settings consistently produces the exact same outcome.

---

## 4. Quick Start & Run Instructions

### Prerequisites
- Python 3.9+ (Python 3.11 recommended)

### Installation
1. Clone or open the repository directory:
   ```bash
   cd Capstone_prototype-1
   ```
2. Install minimal dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Launch the Application
Run the Streamlit application locally:
```bash
streamlit run app.py
```
Streamlit will automatically open the dashboard in your default browser at `http://localhost:8501`.

---

## 5. Two-Minute Capstone Review Presentation Walkthrough

Use this concise, high-impact guide during your Phase-1 review:

```
⏱️ 0:00 – 0:30 | Problem Context & Framework Overview
------------------------------------------------------
• Open app.py in browser.
• Point to the top banner: emphasize that Phase-1 uses simulated data and rule-based
  proxies to establish the architectural pipeline for our event-driven SNN-GNN framework.
• Problem statement: "In offline hiking, missed check-ins shouldn't automatically trigger false
  rescue alarms or be ignored. We interpret silence by correlating terrain, RF link, and movement."

⏱️ 0:30 – 1:00 | Scenario 1: Shared Coverage Gap
------------------------------------------------------
1. Select "1. Shared Coverage Gap" in the sidebar.
2. Slide the observation scrubber to 09:25.
3. Show the Amber Banner: "Likely shared connectivity issue".
4. Point to the metrics:
   - RF Link Quality is degraded (0.16 in "Deep Canyon").
   - Check-ins: 4/4 hikers silent simultaneously.
   - Group speed: 3.8 km/h (normal hiking pace).
5. Reviewer takeaway: "Because movement continues normally despite silence, the framework
   correctly infers a shared environmental dead zone rather than distress."

⏱️ 1:00 – 1:30 | Scenario 2: Individual Anomaly
------------------------------------------------------
1. Select "2. Individual Anomaly" in the sidebar.
2. Slide the scrubber to 09:25.
3. Show the Orange Banner: "Individual anomaly — check required".
4. Point to the metrics & charts:
   - Link Quality is strong (0.86 in Open Meadow).
   - Hikers H1, H2, H3 are actively reporting and moving.
   - Only Hiker H4 (Sweep) is silent with speed dropped to 0.0 km/h.
5. Reviewer takeaway: "The peer graph isolates H4 as an anomalous node requiring a localized check."

⏱️ 1:30 – 2:00 | Scenario 3: Possible Group Risk & Phase-2 Roadmap
-------------------------------------------------------------------
1. Select "3. Possible Group Risk" in the sidebar.
2. Slide the scrubber to 09:25.
3. Show the Red Banner: "Group risk — follow-up required".
4. Explain the key distinction:
   - RF link is 0.70 (sufficient for packets). Radio shadowing CANNOT explain the silence.
   - Entire group speed dropped to 0.0 km/h in a boulder field.
5. Reviewer takeaway: "Correlated multi-node silence + collective halt in clear RF terrain
   signals potential shared-fate hazard (rockfall, route blockage)."
6. Conclude: "In Phase-2, Stage 2 will be powered by on-node neuromorphic SNNs, and Stage 3 by
   GNN group topology convolutions."
```

---

## 6. Configurable Prototype Thresholds

The sidebar enables live tuning of the rule boundaries:

- **Low Link Quality Threshold (default: 0.35)**: RF attenuation cut-off.
- **Consecutive Miss Threshold (default: 2 cycles)**: Number of sequential missed heartbeats to flag persistent silence.
- **Min Hikers for Group Silence (default: 2 hikers)**: Minimum concurrent silent nodes for collective pattern matching.
- **Movement Halt Threshold (default: 0.8 km/h)**: Kinematic cut-off below which a hiker is considered stationary.
- **Speed Drop Anomaly (default: 60%)**: Percentage deceleration relative to baseline velocity.

*Note:* A **"Reset Demo to Defaults"** button is provided to restore initial parameters at any time.

---

## 7. Repository File Structure

```
Capstone_prototype-1/
├── app.py                   # Streamlit web application & presentation dashboard
├── analysis.py              # Modular 4-stage analytical engine (SNN/GNN proxy logic)
├── data_generator.py        # Deterministic simulation generator & schema validator
├── requirements.txt         # Minimal Python dependencies (streamlit, pandas, plotly)
├── README.md                # Comprehensive documentation & 2-minute review script
└── data/                    # Generated deterministic CSV scenario datasets
    ├── scenario_1_shared_coverage_gap.csv
    ├── scenario_2_individual_anomaly.csv
    └── scenario_3_possible_group_risk.csv
```
