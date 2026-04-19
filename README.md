# RAN Network KPI Anomaly Detection

Unsupervised anomaly detection on telecom network KPIs — finds failing, congested, and interference-affected cells before users start complaining, and tells you exactly which KPI caused the problem.

**Live demo:** [telecom-network-anomaly-detection.streamlit.app](https://telecom-network-anomaly-detection-cwwlsv4y937rfsbgghvl8c.streamlit.app/)

---

## The Problem

A telecom network has hundreds of cells reporting KPI data every few minutes. Something goes wrong — a cell starts dropping packets, signal strength collapses, or a tower gets congested. The old way to catch this is manual thresholds: if RSRP drops below X, send an alert.

That breaks down fast. You get flooded with false alarms, miss problems that only show up across multiple KPIs together, and have no idea whether to send a field team or a capacity upgrade team.

This project skips the threshold approach entirely: learn what normal looks like, flag anything that doesn't fit, and explain why.

---

## What's Inside

```
telecom-network-anomaly-detection/
├── data/
│   └── ran_kpi_data.csv              # synthetic RAN KPI dataset (5000 records)
├── notebooks/
│   ├── 01_eda.ipynb                  # KPI distributions and anomaly patterns
│   ├── 02_isolation_forest.ipynb     # Isolation Forest training and evaluation
│   ├── 03_autoencoder.ipynb          # Autoencoder training and evaluation
│   └── 04_comparison.ipynb           # Model comparison + root cause attribution
├── src/
│   ├── generate_data.py              # synthetic data generator with 3 anomaly types
│   ├── preprocess.py                 # scaling and feature prep
│   ├── isolation_forest.py           # IF model logic
│   └── autoencoder.py                # sklearn MLP autoencoder
├── models/
│   ├── isolation_forest.pkl
│   ├── autoencoder.pkl
│   ├── scaler.pkl
│   └── ae_threshold.npy
├── app/
│   └── streamlit_app.py              # live monitoring dashboard
└── requirements.txt
```

---

## Dataset

Synthetic RAN KPI data across 20 cells with 5000 records and 5% injected anomalies. Each record has 6 KPIs:

| KPI | Normal Range | What it measures |
|-----|-------------|-----------------|
| RSRP | -70 to -100 dBm | Signal strength at the device |
| SINR | 5 to 30 dB | Signal quality after interference |
| PRB Utilization | 10–75% | How much cell capacity is in use |
| Throughput | 10–45 Mbps | Actual data speed |
| Packet Loss | 0–1.5% | Data packets being dropped |
| Latency | 8–35 ms | Response time |

Three anomaly types were injected with distinct KPI signatures:

- **Cell Failure** — RSRP collapses to -115 dBm, packet loss spikes to 6–10%
- **Interference** — SINR drops to -2 to 5 dB while everything else stays normal
- **Congestion** — PRB utilization hits 90%+, throughput crashes, latency spikes to 85ms

---

## Models

### Isolation Forest
Anomalies are easier to isolate than normal points. The algorithm builds random decision trees and counts how many splits it takes to isolate each data point. Normal points sit in dense clusters and take many splits. Anomalies sit far from the crowd and get isolated quickly.

Trained on all 5000 samples with 5% contamination assumption.

### Autoencoder (sklearn MLP)
Trained only on normal data. Learns to compress 6 KPI values down to 4 and reconstruct them. When it sees an anomaly — something outside its training distribution — reconstruction error is high. Threshold set at 95th percentile of normal reconstruction errors.

---

## Results

| Model | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Isolation Forest | 0.82 | 0.82 | 0.82 |
| Autoencoder | 0.51 | 0.98 | 0.67 |

Isolation Forest won on F1. The Autoencoder surprised us on recall — 98% is hard to beat for a network that can't afford missed failures. The right answer depends on what's more expensive: a false alarm or a missed outage.

- **Isolation Forest** is better when false alarms are costly. Each alert might trigger a field engineer dispatch. You want precision.
- **Autoencoder** is better when missing a failure is unacceptable — SLA violations, outages, customer complaints. You want recall. It catches 98% of real anomalies.

The dashboard lets you switch between both models and see the difference live.

---

## Root Cause Attribution

Detecting an anomaly is only half the job. The root cause layer inverse-transforms each flagged sample back to original KPI units and compares against normal thresholds. Whichever KPIs are out of range get flagged as the cause.

Example output for a detected anomaly:
```
rsrp = -117.2 dBm  → Too low (< -100)   # cell failure
packet_loss = 6.0% → Too high (> 1.5%)  # data dropping
sinr = 14.2 dB     → Normal
prb_utilization = 63% → Normal
```

A network engineer sees this and knows immediately: signal problem, not congestion. Different fix, different team.

---

## Run It Locally

```bash
git clone git@github.com:Sai-manohar695/telecom-network-anomaly-detection.git
cd telecom-network-anomaly-detection

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

Generate the dataset and train models by running notebooks in order (`01` → `02` → `03` → `04`), then launch the dashboard:

```bash
cd app
streamlit run streamlit_app.py
```

---

## Tech Stack

- **Anomaly Detection:** scikit-learn (Isolation Forest, MLP Autoencoder)
- **Dashboard:** Streamlit
- **Data:** Synthetic RAN KPI data generated with numpy

---

