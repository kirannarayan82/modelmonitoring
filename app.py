# Install required packages
# !pip install streamlit nannyml pandas scikit-learn

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import nannyML as nml

# Simulate some data for demonstration purposes
def generate_data(n_samples):
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(loc=0, scale=1, size=n_samples),
        'feature2': np.random.normal(loc=5, scale=2, size=n_samples),
        'target': np.random.randint(0, 2, size=n_samples)
    })
    return data

reference_data = generate_data(1000)
new_data = generate_data(1000)

# Title of the app
st.title("Data and Model Drift Monitoring App")
st.header("Upload Your Datasets")

# Display simulated reference data
st.subheader("Simulated Reference (Training) Data")
st.write(reference_data)

# Display simulated new data
st.subheader("Simulated New Data")
st.write(new_data)

# Train a simple model on reference data
model = RandomForestClassifier()
model.fit(reference_data.drop(columns='target'), reference_data['target'])

# Initialize NannyML monitor
monitor = nml.Monitor(
    reference_data=reference_data,
    analysis_data=new_data,
    model=model,
    metrics=['performance', 'drift']
)

# Calculate metrics
results = monitor.calculate()

# Separate data drift and model drift
data_drift = results[results['metric'] == 'data_drift']
model_drift = results[results['metric'] == 'model_drift']

# Display results
st.header("Data Drift Metrics")
st.write(data_drift)

st.header("Model Drift Metrics")
st.write(model_drift)

# Plot results
st.line_chart(data_drift[['timestamp', 'value']])
st.line_chart(model_drift[['timestamp', 'value']])

# Use KS statistic and chi-square test for drift detection
ks_statistic, ks_p_value = nml.drift_tests.ks_test(reference_data, new_data)
chi2_statistic, chi2_p_value = nml.drift_tests.chi_square_test(reference_data, new_data)

st.subheader("KS-Statistic and Chi-Square Test Results")
st.write(f"KS Statistic: {ks_statistic}, P-value: {ks_p_value}")
st.write(f"Chi-Square Statistic: {chi2_statistic}, P-value: {chi2_p_value}")

if ks_p_value < 0.05:
    st.write("Warning: Significant data drift detected using KS test.")

if chi2_p_value < 0.05:
    st.write("Warning: Significant data drift detected using Chi-Square test.")
