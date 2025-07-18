﻿# 💡 Data Center Optimization using Machine Learning

> Predicting and reducing energy consumption in data centers using ensemble models, time-series deep learning, and smart optimization strategies.

---

## 📊 Dataset

- **Source**: [IEEE DataPort - Data Server Energy Consumption Dataset](https://ieee-dataport.org/open-access/data-server-energy-consumption-dataset)
- The original dataset had fragmented time rows, so a cleaned and consolidated version was created: `cleaned_dataset.csv`.

---

## 🧪 Model Comparison Results

| Model             | R² Score   | RMSE     | MAE      |
| ----------------- | ---------- | -------- | -------- |
| Random Forest     | 0.9998     | 0.63     | 0.03     |
| Gradient Boosting | 0.9936     | 3.86     | 2.27     |
| Linear Regression | 0.6502     | 28.62    | 21.16    |
| SVR               | 0.3808     | 38.08    | 18.95    |
| **XGBoost**       | **0.9999** | **0.40** | **0.20** |
| LSTM (PyTorch)    | 0.9949     | 3.50     | 0.55     |

> ✅ **XGBoost and Random Forest performed the best**, with XGBoost slightly outperforming all others in terms of both R² and error metrics.
>  
> 🧠 LSTM also achieved high accuracy due to its ability to learn sequential patterns over time.

---

## ⚙️ Energy Optimization Strategies

After training the models, several strategies were applied to reduce predicted energy usage. These were simulated by modifying the feature inputs and re-running predictions with the XGBoost model.

### 🔧 Techniques Applied

| Strategy                                 | Description                                                                 | Avg. Energy Saving |
|------------------------------------------|-----------------------------------------------------------------------------|--------------------|
| Reschedule to Off‑Peak Hours             | Shifted high-activity hours (9 AM–5 PM) to 2 AM                             | +0.15%             |
| CPU Throttling                           | Reduced CPU usage across all time by 20%                                    | **−8.71%**         |
| Power Factor Improvement                 | Increased power factor by 10% (capped at 1)                                 | **−15.13%**        |
| Dynamic Scaling (Conditional Throttle)   | Applied CPU throttling **only** when predicted energy was in top 25%       | **+0.47%**         |

> ❗ Not all strategies are beneficial. Some — like raw CPU throttling or increasing power factor — led to **higher** predicted energy, likely due to underlying data patterns learned by the model.


## 📌 Insights & Conclusion

- **Ensemble models like XGBoost** provide powerful and accurate predictions for energy consumption.
- **Smart interventions** (like dynamic throttling or workload rescheduling) can help **save energy without hurting performance**.
- Even though some intuitive actions (like reducing CPU or increasing power factor) might seem efficient, their effects can be counterintuitive based on real-world data.
- These predictive models can form the **foundation for automated, energy-aware data center management systems**, optimizing resources in real-time.

---

