# Belt Conveyor Health Monitoring System

A Python and Streamlit-based dashboard for predictive maintenance and health monitoring of belt conveyor equipment, using sensor data and machine learning for real-time condition classification, visualization, and maintenance planning[1][2].

---

## **Features**

- **Sensor Data Ingestion**
  - Upload CSV files with sensor readings (`timestamp`, `tension`, `vibration`, `remaining_life`)
  - Manual entry of sensor features for quick predictions
  - Demo and real-time data simulation modes

- **Preprocessing & Feature Engineering**
  - Time-based interpolation and resampling
  - Rolling window statistics (mean, std) for tension
  - FFT-based vibration feature extraction
  - Automatic health condition labeling: `Critical`, `Warning`, `Healthy`

- **Machine Learning Model**
  - Random Forest classifier for predicting equipment condition
  - Model training, evaluation, and persistence (`equipment_fault_predictor.pkl`)
  - Predicts health condition from engineered features

- **Interactive Dashboard (Streamlit)**
  - Data preview and prediction results table
  - Multiple visualization options:
    - Matplotlib, Plotly, and Seaborn trend plots
    - Interactive custom trend explorer
    - Vibration spectrogram and 3D vibration analysis
  - Anomaly detection using Isolation Forest
  - Statistical health metrics (MTBF, failure probability)
  - Feature importance visualization

- **Maintenance & Operations Tools**
  - Equipment profile manager (length, load)
  - Sensor calibration log with expiry warnings
  - Maintenance scheduler and overdue alerts
  - Spare parts inventory tracking
  - Maintenance history timeline
  - PDF report generation (placeholder)

- **User Access Control**
  - Role-based sidebar (Operator, Engineer, Admin)
  - Read-only and advanced access modes

---

## **Installation**

### **Requirements**

- Python 3.8+
- pip

### **Install Dependencies**

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn plotly streamlit joblib
```

---

## **How to Run**

### **1. Train the Model (Optional)**

If you want to retrain the model with your own data:

```bash
python your_script_name.py
```

- Expects a file named `sensor_maintenance_dataset.csv` in the same directory.
- Outputs `equipment_fault_predictor.pkl` after training.

### **2. Launch the Dashboard**

```bash
streamlit run your_streamlit_dashboard.py
```

- The dashboard will open in your browser.

---

## **Usage**

### **A. Data Upload Workflow**

1. **Select "Upload CSV File"** in the sidebar.
2. **Upload your sensor data file** (must include columns: `timestamp`, `tension`, `vibration`, `remaining_life`).
3. **View data preview, predictions, and visualizations**.
4. **Explore trends**: Tension/vibration over time, rolling means, spectrograms, and custom charts.
5. **Review anomaly detection, maintenance schedules, inventory, and history**.

### **B. Manual Input Workflow**

1. **Select "Manual Sensor Input"** in the sidebar.
2. **Enter values** for `tension_mean`, `tension_std`, and `vibration_fft_max`.
3. **Click "Predict Health Condition"** to get an instant classification and recommended action.

### **C. Demo & Simulation**

- Use sidebar buttons to load demo data or simulate real-time sensor streams.
- Useful for testing and demonstration without real sensor input.

---

## **File Structure**

| File                             | Purpose                                 |
|----------------------------------|-----------------------------------------|
| `predictive_maintenance.py`      |  Model training and CLI prediction      |
| `belt_dashboard.py`              | Streamlit dashboard app                 |
| `equipment_fault_predictor.pkl`  | Trained ML model (auto-generated)       |
| `sensor_maintenance_dataset.csv` | Example training data                   |

---

## **Notes**

- The dashboard expects preprocessed features: `tension_mean`, `tension_std`, `vibration_fft_max`.
- Health condition labels are derived from `remaining_life`:
  - `Critical`: ≤ 50
  - `Warning`: 51–100
  - `Healthy`: >100
- Some features (PDF report, advanced filtering) are placeholders for future extension.

---

## **Example**

**Manual Prediction:**

```python
import pandas as pd
import joblib

model = joblib.load('equipment_fault_predictor.pkl')
new_data = pd.DataFrame({
    'tension_mean': [1200.0],
    'tension_std': [20.0],
    'vibration_fft_max': [85.0]
})
prediction = model.predict(new_data)
print(f"Predicted Condition: {prediction[0]}")
```

---

## **Developed with ❤️ using Streamlit**
