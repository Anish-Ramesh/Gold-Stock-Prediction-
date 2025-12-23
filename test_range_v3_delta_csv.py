import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import load_model


def _to_ts(d: str) -> pd.Timestamp:
    return pd.to_datetime(d).normalize()


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def parse_volume(v):
    if isinstance(v, str):
        v = v.replace(",", "").strip()
        if v.endswith("M"):
            return float(v[:-1]) * 1e6
        if v.endswith("K"):
            return float(v[:-1]) * 1e3
    return float(v)


def main():
    csv_path = "TataPower_2005_2025.csv"
    window = 14
    start_date = _to_ts("2025-11-01")
    end_date = _to_ts("2025-12-31")

    # load trained artifacts from v3_delta
    model = load_model("v3_delta.keras")
    scaler_X = pickle.load(open("v3_delta_scaler_X.pkl", "rb"))
    scaler_y = pickle.load(open("v3_delta_scaler_y.pkl", "rb"))

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

    # indicators (must match v3_delta.py)
    df["Volatility"] = df["Price"].rolling(window=5).std()

    delta = df["Price"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Momentum"] = df["Price"] - df["Price"].shift(5)

    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)

    df["Price_next"] = df["Price"].shift(-1)
    df["LogRet_t1"] = np.log(df["Price_next"] / df["Price"])

    df = df.dropna(subset=["Price_next", "LogRet_t1"]).reset_index(drop=True)

    features = base_features + ["Volatility", "RSI", "Momentum"]

    X_all = df[features].values.astype(float)
    y_all = df["LogRet_t1"].values.astype(float)
    date_t = df["Date"].values
    price_t = df["Price"].values.astype(float)
    price_t1 = df["Price_next"].values.astype(float)

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

    mask_range = (d_t_seq >= start_date) & (d_t_seq <= end_date)
    X_test = X_seq[mask_range]
    y_test = y_seq[mask_range]
    d_t_test = d_t_seq[mask_range]
    price_t_test = price_t_seq[mask_range]
    price_t1_test = price_t1_seq[mask_range]

    if len(X_test) == 0:
        raise RuntimeError("No samples found in requested date range.")

    n_features = X_test.shape[2]
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    pred_outputs = model.predict(X_test_scaled, verbose=0)

    if isinstance(pred_outputs, dict):
        pred_y_scaled = pred_outputs["price"].reshape(-1, 1)
    else:
        pred_y_scaled = np.asarray(pred_outputs).reshape(-1, 1)

    pred_logret = scaler_y.inverse_transform(pred_y_scaled).flatten()
    actual_logret = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    print("Min predicted logret:", float(np.min(pred_logret)))
    print("Max predicted logret:", float(np.max(pred_logret)))

    pred_price_t1 = price_t_test * np.exp(pred_logret)
    actual_price_t1 = price_t1_test

    abs_err = np.abs(pred_price_t1 - actual_price_t1)

    out = pd.DataFrame({
        "t_date": d_t_test,
        "t_price_actual": price_t_test,
        "t1_price_actual": actual_price_t1,
        "t1_price_pred": pred_price_t1,
        "abs_err": abs_err,
        "logret_actual": actual_logret,
        "logret_pred": pred_logret,
    }).sort_values("t_date").reset_index(drop=True)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)

    print(out.to_string(index=False))

    print("\n==== Range metrics (Nov 1 .. Dec 31, 2025) ====")
    print(f"MAE : {mae(pred_price_t1, actual_price_t1):.4f}")
    print(f"RMSE: {rmse(pred_price_t1, actual_price_t1):.4f}")
    print(f"Within ±50: {float(np.mean(abs_err <= 50)) * 100:.2f}%")

    plt.figure(figsize=(15, 6))
    plt.plot(out["t_date"], out["t1_price_actual"], label="Actual Close(t+1)", alpha=0.7)
    plt.plot(out["t_date"], out["t1_price_pred"], label="Predicted Close(t+1)", alpha=0.9)
    plt.title("v3_delta – Next-Day Close (CSV, Nov 1..Dec 31, 2025)")
    plt.xlabel("t date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Check how many times the model predicted a drop (negative log-return)
    df_test = out.copy()
    df_test['pred_log_ret'] = np.log(df_test['t1_price_pred'] / df_test['t_price_actual'])
    negative_predictions = df_test[df_test['pred_log_ret'] < 0]

    print(f"\n--- Drop Prediction Analysis ---")
    print(f"Total Test Days: {len(df_test)}")
    print(f"Days Model Predicted a DROP: {len(negative_predictions)}")

    if len(negative_predictions) > 0:
        print("\nExample days where model predicted a drop:")
        print(negative_predictions[['t_date', 't_price_actual', 'pred_log_ret']].head())
    else:
        print("\nWARNING: The model NEVER predicted a price drop.")


if __name__ == "__main__":
    main()
