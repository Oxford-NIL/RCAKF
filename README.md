# RCAKF: Core Body Temperature Estimation from Heart Rate

This repository contains the **official implementation** for the paper:
> *"Estimating Core Body Temperature from Heart Rate Using a Residual-Compensated Adaptive Kalman Filter"*

It includes both the **research code** used in the published study and a **simplified tutorial** for practical understanding.

---

### 🧠 Repository Structure

This repository contains **two complementary components**:

1. **Research Code (RCAKF.py)**  
   Implements the full **nested 3-fold cross-validation** pipeline described in the paper.  
   Designed for **reproducible scientific experiments**, these scripts perform:
   - Model training and evaluation across multiple folds
   - Adaptive parameter tuning and early stopping
   - Statistical performance analysis and visualization
   

2. **Tutorial Code (This README)**  
   Provides a **conceptual and educational example** for practitioners who want to understand or reimplement the RCAKF workflow.  
   It demonstrates how to construct and combine:
   - A standard **Extended Kalman Filter (EKF)**
   - A lightweight **LSTM-based residual compensator**
   - An **(optional) adaptive observation‑noise estimator** (sliding‑window update of the observation noise to adapt the Kalman gain in real time)

---

## 🚀 Quickstart / Mini-Tutorial for Practitioners

This guide follows a reviewer's suggestion to provide an accessible tutorial for researchers who are not machine learning experts. The goal is to show how to **augment a standard Extended Kalman Filter (EKF)** with our proposed **LSTM-based residual compensation**.

---

### 🧩 Step 1: Prepare Your Data

You will need a dataset (e.g., a CSV or Excel file) with time-aligned columns for:

| Column       | Description                                         |
|--------------|-----------------------------------------------------|
| `Participant`| A unique identifier for each subject                |
| `Time`       | A timestamp or sequence index                       |
| `HR`         | Heart rate measurements                             |
| `Core`       | Ground-truth core body temperature (for training)   |

Example:
```text
Participant,Time,HR,Core
P01,0,72,37.0
P01,1,73,37.1
P01,2,95,37.3
...
```

---

### ⚙️ Step 2: Train a Base EKF Model & Calculate Residuals

First, train a **standard EKF** on your training data. This involves fitting two models:

1) **State Transition Model** — predicts the next core temperature from the previous one.

2) **Observation Model** — predicts heart rate from a given core temperature (e.g., 2nd-order polynomial or ECTemp-S sigmoid).

After fitting the base model, produce the EKF heart-rate prediction and compute residuals:
```text
r_t = HR_actual - HR_EKF_hat
```
These residuals (`r_t`) become the **training targets** for the LSTM.

---

### 🤖 Step 3: Train a Simple LSTM to Predict the Residuals

The core innovation of RCAKF is to predict systematic residuals with a lightweight **LSTM**. Below is a minimal **TensorFlow/Keras** example:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# === Define the model ===
# Use a window of the last 10 HR values to predict the next residual.
window_size = 10
model = Sequential()
model.add(LSTM(32, input_shape=(window_size, 1)))  # 32 hidden units
model.add(Dense(1))  # Output: predicted residual

# === Compile and Train ===
model.compile(optimizer='adam', loss='mean_squared_error')

# X_train: HR sequences with shape (samples, 10, 1)
# Y_train: residual targets with shape (samples,)
model.fit(X_train, Y_train, epochs=50, validation_split=0.2)

# Save trained model
model.save("residual_lstm_model.h5")
```

---

### 🔁 Step 4: Integrate Both Components for the Full RCAKF Loop

In your **real-time prediction loop**, combine the base EKF with the trained LSTM. At each time step `t`:

1) **EKF prior:**
```text
T_hat_t, v_hat_t = ekf_predict(T_{t-1}, v_{t-1})
```

2) **Residual compensation:**
```text
r_hat_t = LSTM(HR_{t-9:t-1})
HR_comp_hat_t = HR_EKF_hat_t + r_hat_t
```

3) **Innovation:**
```text
e_t = HR_actual_t - HR_comp_hat_t
```

4) **Adaptive observation noise (optional):** estimate from a sliding window of recent innovations (window size = `w`):
```text
sigma_t_sq = max(0, mean(e_{t-w+1}^2, ..., e_t^2) - m_t^2 * v_hat_t)
```
where `m_t` is the derivative of the observation model evaluated at `T_hat_t`.

5) **Kalman update:**
```text
k_t      = (v_hat_t * m_t) / (m_t^2 * v_hat_t + sigma_t_sq)
T_t      = T_hat_t + k_t * e_t
v_t      = (1 - k_t * m_t) * v_hat_t
```

This modular approach lets RCAKF **boost the performance and robustness** of nearly any existing Kalman Filter for physiological monitoring.

---


## 📘 About the Source Code

To run the source code:

1. **Install dependencies**
```bash
pip install pandas numpy torch scikit-learn matplotlib openpyxl
```

2. **Place your data file** (example)
```text
Data.xlsx
```

3. **Run the RCAKF script**
```bash
python RCAKF.py
```

---


