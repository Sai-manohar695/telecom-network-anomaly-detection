import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow import keras
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Load models and data
@st.cache_resource
def load_models():
    scaler = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl'))
    iso_forest = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'isolation_forest.pkl'))
    autoencoder = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'autoencoder.pkl'))
    ae_threshold = np.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'ae_threshold.npy'))
    return scaler, iso_forest, autoencoder, ae_threshold

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'ran_kpi_data.csv'))
    return df

def get_root_cause(sample, scaler, kpis):
    original = scaler.inverse_transform(sample.reshape(1, -1))[0]
    kpi_values = dict(zip(kpis, original))
    thresholds = {
        'rsrp':            (-100, -70),
        'sinr':            (5, 30),
        'prb_utilization': (10, 75),
        'throughput_mbps': (10, 45),
        'packet_loss':     (0, 1.5),
        'latency_ms':      (8, 35)
    }
    violations = []
    for kpi, value in kpi_values.items():
        low, high = thresholds[kpi]
        if value < low:
            violations.append((kpi, value, f'Too low (< {low})'))
        elif value > high:
            violations.append((kpi, value, f'Too high (> {high})'))
    return kpi_values, violations

# Page config
st.set_page_config(page_title="RAN KPI Anomaly Monitor", page_icon="📡", layout="wide")
st.title("📡 RAN Network KPI Anomaly Monitor")
st.markdown("Real-time anomaly detection on telecom network KPIs using Isolation Forest and Autoencoder.")

# Load
scaler, iso_forest, autoencoder, ae_threshold = load_models()
df = load_data()

kpis = ['rsrp', 'sinr', 'prb_utilization', 'throughput_mbps', 'packet_loss', 'latency_ms']
X = df[kpis].values
X_scaled = scaler.transform(X)

# Run predictions
preds_if = (iso_forest.predict(X_scaled) == -1).astype(int)
X_reconstructed = autoencoder.predict(X_scaled, verbose=0)
reconstruction_errors = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)
preds_ae = (reconstruction_errors > ae_threshold).astype(int)

df['anomaly_if'] = preds_if
df['anomaly_ae'] = preds_ae
df['reconstruction_error'] = reconstruction_errors

# Sidebar
st.sidebar.header("Controls")
selected_cell = st.sidebar.selectbox("Select Cell", sorted(df['cell_id'].unique()))
selected_model = st.sidebar.radio("Detection Model", ["Isolation Forest", "Autoencoder"])
selected_kpi = st.sidebar.selectbox("KPI to Plot", kpis)

anomaly_col = 'anomaly_if' if selected_model == "Isolation Forest" else 'anomaly_ae'
cell_df = df[df['cell_id'] == selected_cell].copy()
cell_df['timestamp'] = pd.to_datetime(cell_df['timestamp'])
cell_df = cell_df.sort_values('timestamp')

# Top metrics
col1, col2, col3, col4 = st.columns(4)
total = len(cell_df)
anomalies = cell_df[anomaly_col].sum()
col1.metric("Total Records", total)
col2.metric("Anomalies Detected", int(anomalies))
col3.metric("Anomaly Rate", f"{anomalies/total:.1%}")
col4.metric("Model", selected_model)

st.markdown("---")

# KPI time series plot
st.subheader(f"{selected_kpi.upper()} over Time — {selected_cell}")
fig, ax = plt.subplots(figsize=(12, 4))
normal_data = cell_df[cell_df[anomaly_col] == 0]
anomaly_data = cell_df[cell_df[anomaly_col] == 1]
ax.plot(cell_df['timestamp'], cell_df[selected_kpi],
        color='steelblue', linewidth=0.8, alpha=0.7, label='Normal')
ax.scatter(anomaly_data['timestamp'], anomaly_data[selected_kpi],
           color='red', s=40, zorder=5, label='Anomaly')
ax.set_xlabel('Timestamp')
ax.set_ylabel(selected_kpi)
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# Anomaly table with root cause
st.subheader("Detected Anomalies with Root Cause")
anomaly_rows = cell_df[cell_df[anomaly_col] == 1]

if len(anomaly_rows) == 0:
    st.success("No anomalies detected for this cell.")
else:
    for _, row in anomaly_rows.head(10).iterrows():
        sample = scaler.transform([[row[k] for k in kpis]])[0]
        kpi_values, violations = get_root_cause(sample, scaler, kpis)

        with st.expander(f"🔴 {row['timestamp']} — {len(violations)} KPI violation(s)"):
            vcol1, vcol2 = st.columns(2)
            with vcol1:
                st.markdown("**KPI Values:**")
                for k, v in kpi_values.items():
                    st.write(f"- {k}: `{v:.3f}`")
            with vcol2:
                st.markdown("**Root Cause:**")
                if violations:
                    for kpi, val, reason in violations:
                        st.error(f"{kpi} = {val:.3f} → {reason}")
                else:
                    st.info("No single KPI threshold violated — multi-KPI pattern anomaly")

st.markdown("---")

# Network wide anomaly rate
st.subheader("Network-Wide Anomaly Rate per Cell")
cell_anomaly_rate = df.groupby('cell_id')[anomaly_col].mean().sort_values(ascending=False)
fig2, ax2 = plt.subplots(figsize=(14, 4))
cell_anomaly_rate.plot(kind='bar', ax=ax2, color='steelblue')
ax2.set_title('Anomaly Rate per Cell')
ax2.set_xlabel('Cell ID')
ax2.set_ylabel('Anomaly Rate')
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.markdown("---")
st.caption("Built with Isolation Forest + Autoencoder | Telecom RAN KPI Anomaly Detection")