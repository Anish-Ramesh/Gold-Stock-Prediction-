import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import Huber


def parse_volume(v):
    if isinstance(v, str):
        v = v.replace(",", "").strip()
        if v.endswith("M"):
            return float(v[:-1]) * 1e6
        if v.endswith("K"):
            return float(v[:-1]) * 1e3
    return float(v)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    csv_path = "TataPower_2005_2025.csv"
    train_end_date = "2023-01-01"
    window = 14

    df = pd.read_csv(csv_path)

    if "Close" in df.columns and "Price" not in df.columns:
        df = df.rename(columns={"Close": "Price"})

    if "Volume" in df.columns and "Vol." not in df.columns:
        df = df.rename(columns={"Volume": "Vol."})

    if "Date" not in df.columns:
        raise ValueError("CSV must contain a 'Date' column")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if "Vol." in df.columns:
        df["Vol."] = df["Vol."].apply(parse_volume)

    # base features
    base_features = ["Price", "Open", "High", "Low", "Vol."]
    missing = [c for c in base_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # ==========================================
    # MANUAL INDICATOR CALCULATION
    # ==========================================

    # 1. Volatility (Standard Deviation of last 5 days)
    df["Volatility"] = df["Price"].rolling(window=5).std()

    # 2. RSI-14 (Relative Strength Index)
    delta = df["Price"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # 3. Momentum (Price today - Price 5 days ago)
    df["Momentum"] = df["Price"] - df["Price"].shift(5)

    # Fill NaNs from rolling windows
    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)

    # log-return target for t -> t+1
    df["Price_next"] = df["Price"].shift(-1)
    df["LogRet_t1"] = np.log(df["Price_next"] / df["Price"])

    df = df.dropna(subset=["Price_next", "LogRet_t1"]).reset_index(drop=True)

    features = base_features + ["Volatility", "RSI", "Momentum"]

    X_all = df[features].values.astype(float)
    y_all = df["LogRet_t1"].values.astype(float)
    date_t = df["Date"].values
    price_t = df["Price"].values.astype(float)
    price_t1 = df["Price_next"].values.astype(float)

    if len(df) <= window + 5:
        raise RuntimeError("Not enough rows to create sequences.")

    # build sequences: each sample uses last `window` days up to t to predict t+1 log-return
    X_seq, y_seq, d_t_seq, price_t_seq, price_t1_seq = [], [], [], [], []
    for i in range(window - 1, len(df) - 1):
        X_seq.append(X_all[i - window + 1 : i + 1])
        y_seq.append(y_all[i])
        d_t_seq.append(date_t[i])
        price_t_seq.append(price_t[i])
        price_t1_seq.append(price_t1[i])

    X_seq = np.asarray(X_seq)
    y_seq = np.asarray(y_seq)
    d_t_seq = pd.to_datetime(np.asarray(d_t_seq))
    price_t_seq = np.asarray(price_t_seq)
    price_t1_seq = np.asarray(price_t1_seq)

    # time-based split by t date (info available at t)
    train_mask = d_t_seq < pd.to_datetime(train_end_date)
    test_mask = ~train_mask

    X_train, y_train = X_seq[train_mask], y_seq[train_mask]
    X_test, y_test = X_seq[test_mask], y_seq[test_mask]
    d_t_test = d_t_seq[test_mask]
    price_t_test = price_t_seq[test_mask]
    price_t1_test = price_t1_seq[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        raise RuntimeError("Train/test split produced empty set. Check train_end_date.")

    # scalers: StandardScaler for X, RobustScaler for y (log-returns)
    scaler_X = StandardScaler()
    scaler_y = RobustScaler()

    # fit on training only
    n_features = X_train.shape[2]
    X_train_flat = X_train.reshape(-1, n_features)
    scaler_X.fit(X_train_flat)
    scaler_y.fit(y_train.reshape(-1, 1))

    # transform all sequences
    X_train_scaled = scaler_X.transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(window, n_features)),
        Dropout(0.25),
        LSTM(64),
        Dropout(0.25),
        Dense(1),
    ])

    model.compile(optimizer="adam", loss=Huber(delta=1.0))

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    checkpoint_path = "v3_delta_savedmodel"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True)

    model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_split=0.1,
        epochs=80,
        batch_size=32,
        callbacks=[early_stop, checkpoint],
        verbose=1,
    )

    best_model = load_model(checkpoint_path)

    # predictions (log-returns)
    pred_y_scaled = best_model.predict(X_test_scaled, verbose=0).reshape(-1, 1)
    pred_logret = scaler_y.inverse_transform(pred_y_scaled).flatten()
    actual_logret = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    # reconstruct next-day prices
    pred_price_t1 = price_t_test * np.exp(pred_logret)
    actual_price_t1 = price_t1_test

    abs_err = np.abs(pred_price_t1 - actual_price_t1)

    print("==== v3_delta next-day Close evaluation (log-return, aligned on t+1) ====")
    print(f"Test samples: {len(actual_price_t1)}")
    print(f"MAE : {mae(pred_price_t1, actual_price_t1):.4f}")
    print(f"RMSE: {rmse(pred_price_t1, actual_price_t1):.4f}")
    print(f"Within ±50: {float(np.mean(abs_err <= 50)) * 100:.2f}%")

    out = pd.DataFrame({
        "t_date": d_t_test,
        "t1_price_actual": actual_price_t1,
        "t1_price_pred": pred_price_t1,
        "abs_err": abs_err,
        "logret_actual": actual_logret,
        "logret_pred": pred_logret,
    })

    print("\nLast 15 predictions:")
    print(out.sort_values("t_date").tail(15).to_string(index=False))

    with open("v3_delta_scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)

    with open("v3_delta_scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

    plt.figure(figsize=(15, 6))
    out_sorted = out.sort_values("t_date")
    dates = out_sorted["t_date"]
    plt.plot(dates, out_sorted["t1_price_actual"], label="Actual Close(t+1)", alpha=0.7)
    plt.plot(dates, out_sorted["t1_price_pred"], label="Predicted Close(t+1)", alpha=0.9)
    plt.title("Next-Day Close Prediction (v3_delta, log-return)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
