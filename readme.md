# 🌍 Data-Driven Spatio-Temporal Inference of Air Pollution Transport Using Multi-Constraint Validation

A research-oriented framework for identifying and validating inter-station air pollution transport pathways using multi-year air quality and meteorological observations.

Unlike traditional air quality projects that focus solely on predicting pollutant concentrations, this work investigates whether pollution physically travels between monitoring locations and how such transport can be reliably identified using observational data.

---

## 📌 Project Overview

Air pollution monitoring studies typically focus on forecasting pollutant concentrations such as PM2.5 and PM10. While forecasting is useful, it does not explain how pollution propagates across regions.

This project addresses a different question:

> Can pollution transport between locations be identified using only monitoring station data and meteorological observations?

To answer this, a six-stage validation framework was developed to distinguish genuine pollution transport from coincidental statistical correlations.

The framework combines:

- Time-lag correlation analysis
- Randomized baseline validation
- Multi-pollutant consistency checks
- Wind alignment verification
- Distance-lag plausibility testing
- Event-based spike tracking

---

## 🎯 Objectives

- Identify inter-station pollution transport pathways
- Distinguish real transport from spurious correlations
- Validate transport relationships using atmospheric constraints
- Quantify transport delays between monitoring stations
- Construct an interpretable pollution transport network

---

## 📊 Dataset

### Monitoring Stations

- AIIMS Raipur
- Bhatagaon
- IGKV
- Siltara

### Time Coverage

- 2021 – 2026
- Hourly observations
- 15-minute (quarter-hourly) observations

### Pollutant Variables

- PM2.5
- PM10
- NO
- NO₂
- NOx
- SO₂
- O₃
- CO
- NH₃
- Benzene

### Meteorological Variables

- Wind Speed
- Wind Direction
- Temperature
- Humidity
- Solar Radiation
- Rainfall

---

## 🔬 Methodology

### Stage 1 – Lag Correlation Analysis

For every ordered station pair, lagged Pearson correlations are computed across multiple time delays to identify candidate transport pathways.

### Stage 2 – Randomized Baseline Validation

Source station time series are randomized to generate a null distribution and eliminate relationships caused by chance.

### Stage 3 – Multi-Pollutant Consistency Filtering

Transport signatures are validated across multiple pollutants including:

- PM2.5
- NO₂
- SO₂

### Stage 4 – Wind Alignment Validation

Candidate pathways are retained only when observed wind conditions support physical transport.

### Stage 5 – Distance-Lag Plausibility Check

Transport delays are compared against realistic atmospheric transport speeds.

### Stage 6 – Event-Based Spike Validation

Individual pollution spikes are tracked from source to target stations to confirm real transport events.

Only pathways that pass all six validation stages are accepted.

---

## 🏗 Framework Pipeline

```text
Raw Monitoring Data
        │
        ▼
Data Cleaning & Alignment
        │
        ▼
Lag Correlation Analysis
        │
        ▼
Randomized Baseline Validation
        │
        ▼
Multi-Pollutant Consistency
        │
        ▼
Wind Alignment Validation
        │
        ▼
Distance-Lag Plausibility Check
        │
        ▼
Event-Based Spike Tracking
        │
        ▼
Validated Pollution Transport Network
```

---

## 📈 Key Results

### Validated Transport Pathway

```text
AIIMS ─────────▶ Bhatagaon
      Lag = 1 Hour
      Correlation ≈ 0.563
```

### Transport Validation Metrics

| Metric | Value |
|----------|----------|
| Optimal Lag | 1 Hour |
| Lagged Correlation | 0.563 |
| Randomized Baseline | 0.055 |
| Correlation Surplus | +0.508 |
| Total PM2.5 Spikes | 82 |
| Successful Propagations | 34 |
| Event Success Rate | 41.5% |

### Key Finding

Several station pairs exhibited high raw correlations but failed physical validation constraints, demonstrating that:

> High correlation alone is not sufficient evidence of pollution transport.

---

## 💡 Novelty

Most existing studies focus on:

- Air quality forecasting
- Single-station prediction
- PM2.5 concentration estimation

This work instead focuses on:

- Inter-station transport inference
- Multi-constraint validation
- Event-level transport verification
- Interpretable pollution pathway discovery

without relying on external atmospheric trajectory simulators such as HYSPLIT.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- TensorFlow / Keras
- Matplotlib
- Seaborn
- NetworkX
- Jupyter Notebook

---

## 📂 Project Structure

```text
├── data/
│   ├── hourly/
│   └── quarterly/
│
├── notebooks/
│
├── src/
│
├── results/
│   ├── transport_networks/
│   ├── lag_analysis/
│   ├── validation_plots/
│   └── figures/
│
├── report/
│
└── README.md
```

---

## 📖 Research Significance

Understanding pollution transport is critical for:

- Urban air quality management
- Environmental policy planning
- Emission control strategies
- Early warning systems
- Regional pollution mitigation

The proposed framework demonstrates that meaningful transport insights can be extracted directly from monitoring data without requiring expensive atmospheric simulation models.

---

## 👨‍💻 Authors

### Tenneti Vineel Krishna
Department of Computer Science and Engineering  
International Institute of Information Technology, Naya Raipur  
📧 tenneti24100@iiitnr.edu.in

### Arkapriya Das
Department of Computer Science and Engineering  
International Institute of Information Technology, Naya Raipur  
📧 arkapriya24100@iiitnr.edu.in

### Annam Hema Kumar
Department of Computer Science and Engineering  
International Institute of Information Technology, Naya Raipur  
📧 annam24100@iiitnr.edu.in

---

## 📜 License

This project is intended for academic and research purposes.

---

⭐ If you found this project useful, consider giving it a star.
