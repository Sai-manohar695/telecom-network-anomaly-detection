import numpy as np
import pandas as pd

def generate_ran_kpi_data(n_samples=5000, anomaly_fraction=0.05, random_state=42):
    np.random.seed(random_state)
    n_anomalies = int(n_samples * anomaly_fraction)
    n_normal = n_samples - n_anomalies

    # Normal network conditions
    normal = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_normal, freq='5min'),
        'cell_id': np.random.choice([f'CELL_{i:03d}' for i in range(1, 21)], n_normal),
        'rsrp': np.random.normal(-85, 8, n_normal),        # Reference Signal Received Power (dBm)
        'sinr': np.random.normal(15, 4, n_normal),          # Signal to Interference Noise Ratio (dB)
        'prb_utilization': np.random.normal(45, 12, n_normal), # Physical Resource Block utilization (%)
        'throughput_mbps': np.random.normal(25, 6, n_normal),  # Throughput (Mbps)
        'packet_loss': np.random.normal(0.5, 0.2, n_normal),   # Packet loss (%)
        'latency_ms': np.random.normal(20, 4, n_normal),        # Latency (ms)
        'is_anomaly': 0
    })

    # Anomalous network conditions — inject 3 types
    anomaly_types = np.random.choice(['interference', 'congestion', 'failure'], n_anomalies)
    anomalies = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_anomalies, freq='47min'),
        'cell_id': np.random.choice([f'CELL_{i:03d}' for i in range(1, 21)], n_anomalies),
        'rsrp': np.where(anomaly_types == 'failure',
                         np.random.normal(-115, 5, n_anomalies),
                         np.random.normal(-85, 8, n_anomalies)),
        'sinr': np.where(anomaly_types == 'interference',
                         np.random.normal(-2, 3, n_anomalies),
                         np.random.normal(15, 4, n_anomalies)),
        'prb_utilization': np.where(anomaly_types == 'congestion',
                                    np.random.normal(92, 4, n_anomalies),
                                    np.random.normal(45, 12, n_anomalies)),
        'throughput_mbps': np.where(anomaly_types == 'congestion',
                                    np.random.normal(4, 1.5, n_anomalies),
                                    np.random.normal(25, 6, n_anomalies)),
        'packet_loss': np.where(anomaly_types == 'failure',
                                np.random.normal(8, 2, n_anomalies),
                                np.random.normal(0.5, 0.2, n_anomalies)),
        'latency_ms': np.where(anomaly_types == 'congestion',
                               np.random.normal(85, 15, n_anomalies),
                               np.random.normal(20, 4, n_anomalies)),
        'is_anomaly': 1
    })
    anomalies['anomaly_type'] = anomaly_types

    # Combine and clip to realistic ranges
    df = pd.concat([normal, anomalies], ignore_index=True)
    df['anomaly_type'] = df['anomaly_type'].fillna('normal')
    df['rsrp'] = df['rsrp'].clip(-140, -40)
    df['sinr'] = df['sinr'].clip(-20, 40)
    df['prb_utilization'] = df['prb_utilization'].clip(0, 100)
    df['throughput_mbps'] = df['throughput_mbps'].clip(0, 100)
    df['packet_loss'] = df['packet_loss'].clip(0, 100)
    df['latency_ms'] = df['latency_ms'].clip(1, 500)

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_ran_kpi_data()
    df.to_csv('../data/ran_kpi_data.csv', index=False)
    print(f"Dataset saved: {df.shape}")
    print(f"Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean():.1%})")
    print(f"Anomaly types:\n{df['anomaly_type'].value_counts()}")