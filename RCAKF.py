# ========== Imports and Data Preprocessing ==========
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import copy

# ========== GPU Device Setup ==========
# Check for available GPU and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== Data Loading ==========
# Please ensure the Excel file path is correct
file_path = "Data.xlsx"  #
try:
    data = pd.read_excel(file_path)
except FileNotFoundError:
    print(
        f"Error: File not found '{file_path}'. Please ensure the path is correct and the file is in the same directory as the script, or provide a full path.")
    # Create an empty DataFrame to avoid subsequent errors
    data = pd.DataFrame(columns=['Participant', 'Time', 'HR', 'Core'])

# ========== Basic Data Cleaning and Preprocessing ==========
if not data.empty:
    data.dropna(subset=['HR', 'Core'], inplace=True)
    data.sort_values(by=['Participant', 'Time'], inplace=True)
    data['Core_shift'] = data.groupby('Participant')['Core'].shift(1)
    data = data.dropna(subset=['Core_shift'])

    participants = data['Participant'].unique()
    # Outer cross-validation for final model evaluation
    outer_kf = KFold(n_splits=3, shuffle=True, random_state=42)
else:
    participants = []
    outer_kf = KFold(n_splits=3)


# ========== Model and Dataset Definitions for RCAKF ==========

# RCAKF LSTM Network (for predicting residuals)
class ObservationResidualLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


# RCAKF Dataset (target is the residual)
class ObservationDataset(Dataset):
    def __init__(self, hr_seq, core_seq, b2, b1, b0, window):
        self.X = hr_seq[:-1].reshape(-1, 1)
        self.y = hr_seq[1:] - (b2 * core_seq[1:] ** 2 + b1 * core_seq[1:] + b0)
        self.window = window

    def __len__(self):
        return len(self.X) - self.window + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.window]
        y_seq = self.y[idx:idx + self.window]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32).unsqueeze(-1)


# ========== Training and Hyperparameter Search ==========

# LSTM training function with early stopping mechanism
def train_lstm_with_early_stopping(train_loader, val_loader, model_class, device, patience=5, epochs=200):
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad();
            loss.backward();
            optimizer.step()

        # Validation loop
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    val_loss += loss_fn(pred, yb).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break
        else:  # If no validation loader, just train for max epochs
            best_model_state = copy.deepcopy(model.state_dict())

    if epochs_no_improve >= patience:
        print(f"  LSTM training stopped early.")

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model


# ========== Global Parameters for RCAKF ==========
LSTM_WINDOW = 10  # Fixed LSTM window size
ADAPTIVE_WINDOW_CANDIDATES = [5, 10, 20, 30, 50]

# Store results from all folds
true_vals, rekf_pred, participant_ids_all = [], [], []

# ========== Nested Cross-Validation Main Loop for RCAKF ==========
if participants.size > 0:
    for fold_idx, (train_idx, test_idx) in enumerate(outer_kf.split(participants)):
        print(f"====================\n=== Outer Fold {fold_idx + 1} ===\n====================")
        train_parts, test_parts = participants[train_idx], participants[test_idx]
        outer_train_df, test_df = data[data['Participant'].isin(train_parts)].copy(), data[
            data['Participant'].isin(test_parts)].copy()

        # Split outer training set into training and validation sets (80/20) for early stopping and hyperparameter search
        if len(train_parts) >= 5:
            val_participants = np.random.choice(train_parts, size=int(len(train_parts) * 0.2), replace=False)
        else:
            val_participants = np.array([])
        train_participants_split = [p for p in train_parts if p not in val_participants]
        train_df_split = outer_train_df[outer_train_df['Participant'].isin(train_participants_split)]
        val_df_split = outer_train_df[outer_train_df['Participant'].isin(val_participants)]

        # --- Base EKF Parameter Training (on the entire outer training set) ---
        print("\n--- Training Base EKF Parameters ---")
        reg_time = LinearRegression().fit(outer_train_df['Core_shift'].values.reshape(-1, 1),
                                          outer_train_df['Core'].values)
        a1, a0 = reg_time.coef_[0], reg_time.intercept_
        gamma = np.std(outer_train_df['Core'] - reg_time.predict(outer_train_df['Core_shift'].values.reshape(-1, 1)))

        poly = PolynomialFeatures(2, include_bias=False)
        X_obs_poly = poly.fit_transform(outer_train_df['Core'].values.reshape(-1, 1))
        reg_obs = LinearRegression().fit(X_obs_poly, outer_train_df['HR'])
        b1, b2, b0 = *reg_obs.coef_, reg_obs.intercept_
        sigma_initial = np.std(outer_train_df['HR'] - reg_obs.predict(X_obs_poly))
        print("  Base EKF parameters fitted.")

        # --- Training RCAKF's LSTM Model with Early Stopping ---
        print("\n--- Training RCAKF's LSTM model with Early Stopping (Window=10) ---")
        rekf_train_ds = ObservationDataset(train_df_split['HR'].values, train_df_split['Core'].values, b2, b1, b0,
                                           LSTM_WINDOW)
        rekf_val_ds = ObservationDataset(val_df_split['HR'].values, val_df_split['Core'].values, b2, b1, b0,
                                         LSTM_WINDOW) if not val_df_split.empty else []
        rekf_train_loader = DataLoader(rekf_train_ds, batch_size=32, shuffle=True)
        rekf_val_loader = DataLoader(rekf_val_ds, batch_size=32, shuffle=False) if len(rekf_val_ds) > 0 else None
        rekf_model = train_lstm_with_early_stopping(rekf_train_loader, rekf_val_loader, ObservationResidualLSTM, device)
        print("  RCAKF LSTM model trained.")

        # --- Searching for the best adaptive_window_size on the validation set ---
        print("\n--- Searching for best RCAKF adaptive_window_size ---")
        best_adaptive_window = 10  # Default value
        best_rmse = float('inf')
        if not val_df_split.empty:
            rekf_model.eval()
            with torch.no_grad():
                for ad_win in ADAPTIVE_WINDOW_CANDIDATES:
                    all_val_preds, all_val_true = [], []
                    for pid in val_participants:
                        df_p_val = val_df_split[val_df_split['Participant'] == pid].reset_index(drop=True)
                        if len(df_p_val) <= 1: continue

                        core_rekf_val, var_rekf_val = [37.0], [0.01]
                        innovations_sq = [sigma_initial ** 2] * ad_win
                        for t in range(1, len(df_p_val)):
                            h_seq = [df_p_val['HR'].iloc[max(0, t - i - 1)] for i in range(LSTM_WINDOW)][::-1]
                            input_seq = torch.tensor(h_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                            xi = rekf_model(input_seq).squeeze()[-1].cpu().item()

                            CT_r, v_r = a1 * core_rekf_val[-1] + a0, var_rekf_val[-1] + gamma ** 2
                            m_r = 2 * b2 * CT_r + b1
                            HR_exp_r = b2 * CT_r ** 2 + b1 * CT_r + b0 + xi
                            innovation = df_p_val['HR'].iloc[t] - HR_exp_r

                            innovations_sq.pop(0)
                            innovations_sq.append(innovation ** 2)
                            sigma_sq = max(0, np.mean(innovations_sq) - m_r ** 2 * v_r)

                            k_r = (v_r * m_r) / (m_r ** 2 * v_r + sigma_sq + 1e-9)
                            core_rekf_val.append(CT_r + k_r * innovation)
                            var_rekf_val.append((1 - k_r * m_r) * v_r)

                        all_val_preds.extend(core_rekf_val[1:])
                        all_val_true.extend(df_p_val['Core'][1:])

                    if not all_val_true: continue
                    current_rmse = np.sqrt(mean_squared_error(all_val_true, all_val_preds))
                    if current_rmse < best_rmse:
                        best_rmse = current_rmse
                        best_adaptive_window = ad_win

        print(f"  Best Adaptive Window for Fold {fold_idx + 1}: {best_adaptive_window}")

        # ========== Predicting on the Outer Test Set with RCAKF ==========
        print("\n--- Predicting on Test Set with RCAKF ---")
        rekf_model.eval()
        for pid in test_parts:
            df_p = test_df[test_df['Participant'] == pid].reset_index(drop=True)
            if len(df_p) <= 1: continue

            initial_core, initial_var = 37.0, 0.01
            core_rekf, var_rekf = [initial_core], [initial_var]
            innovations_rekf_squared = [sigma_initial ** 2] * best_adaptive_window

            with torch.no_grad():
                for t in range(1, len(df_p)):
                    HR_t = df_p['HR'].iloc[t]

                    h_seq_rekf = [df_p['HR'].iloc[max(0, t - i)] for i in range(LSTM_WINDOW)][::-1]
                    input_seq_rekf = torch.tensor(h_seq_rekf, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                    xi = rekf_model(input_seq_rekf).squeeze()[-1].cpu().item()

                    CT_r, v_r = a1 * core_rekf[-1] + a0, var_rekf[-1] + gamma ** 2
                    m_r = 2 * b2 * CT_r + b1
                    HR_exp_r = b2 * CT_r ** 2 + b1 * CT_r + b0 + xi
                    innovation = HR_t - HR_exp_r

                    innovations_rekf_squared.pop(0)
                    innovations_rekf_squared.append(innovation ** 2)
                    sigma_rekf_adaptive_squared = max(0, np.mean(innovations_rekf_squared) - m_r ** 2 * v_r)

                    k_r = (v_r * m_r) / (m_r ** 2 * v_r + sigma_rekf_adaptive_squared + 1e-9)
                    core_rekf.append(CT_r + k_r * innovation)
                    var_rekf.append((1 - k_r * m_r) * v_r)

            # Record results
            true_vals.extend(df_p['Core'])
            rekf_pred.extend(core_rekf)
            participant_ids_all.extend([pid] * len(df_p))
else:
    print("No data loaded, skipping training and evaluation.")

# ========== Aggregating All Results and Visualization ==========
if len(true_vals) > 0:
    all_true = np.array(true_vals)
    all_rekf = np.array(rekf_pred)
    all_participants = np.array(participant_ids_all)

    # 1. Overall Prediction Curve Plot
    plt.figure(figsize=(18, 9))
    plt.plot(all_true, label='True Core Temperature', color='black', linewidth=2.5, zorder=10)
    plt.plot(all_rekf, label='RCAKF Prediction', linestyle='-', color='green', linewidth=2.0, zorder=6)
    plt.xlabel('Time Index (All Folds Combined)')
    plt.ylabel('Core Temp (°C)')
    plt.title('RCAKF: Core Temperature Estimation vs. Ground Truth')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


    # 2. Evaluation Metrics Output
    def evaluate_and_print(true, pred, label):
        if len(true) == 0 or len(pred) == 0:
            print(f"{label: <45} → No data to evaluate.")
            return
        rmse = np.sqrt(mean_squared_error(true, pred))
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        pearson_corr = np.corrcoef(true, pred)[0, 1] if len(true) > 1 else np.nan
        print(f"{label: <45} → RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Pearson Corr: {pearson_corr:.4f}")


    print("\n--- Overall Performance Evaluation ---")
    evaluate_and_print(all_true, all_rekf, "RCAKF")


    # 3. Bland-Altman Plot
    def bland_altman(true, pred, label):
        if len(true) < 2:
            print(f"Skipping Bland-Altman for {label} due to insufficient data.")
            return

        valid_indices = ~np.isnan(true) & ~np.isnan(pred)
        true, pred = true[valid_indices], pred[valid_indices]
        if len(true) < 2: return

        mean_temp, diff_temp = (true + pred) / 2, pred - true
        mean_diff, std_diff = np.mean(diff_temp), np.std(diff_temp)
        upper_limit, lower_limit = mean_diff + 1.96 * std_diff, mean_diff - 1.96 * std_diff

        plt.figure(figsize=(10, 5))
        plt.scatter(mean_temp, diff_temp, alpha=0.3, s=15, edgecolors='k', linewidths=0.5)
        plt.axhline(mean_diff, color='blue', linestyle='--', label=f'Mean Diff: {mean_diff:.2f}')
        plt.axhline(upper_limit, color='red', linestyle='--', label=f'±1.96 SD [{lower_limit:.2f}, {upper_limit:.2f}]')
        plt.axhline(lower_limit, color='red', linestyle='--')
        plt.xlabel('Mean of True and Estimated Temp (°C)')
        plt.ylabel(f'Difference ({label} - True) (°C)')
        plt.title(f'Bland-Altman Plot - {label}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    bland_altman(all_true, all_rekf, "RCAKF")

    # 4. Save Results to Excel
    min_len = min(len(all_true), len(all_rekf), len(all_participants))
    results_df = pd.DataFrame({
        'Participant': all_participants[:min_len],
        'Core': all_true[:min_len],
        'RCAKF': all_rekf[:min_len]
    })
    results_df.to_excel("Archived_RCAKF_results.xlsx", index=False)
    print("\nRCAKF results saved to Archived_RCAKF_results.xlsx")
