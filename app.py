import base64
import io
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Fix for running on servers without a display
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from keras.models import load_model

app = Flask(__name__)

# --- Global Configuration ---
CSV_PATH = "TataPower_2005_2025.csv"
MODEL_PATH = "v3_delta_savedmodel"
SCALER_X_PATH = "v3_delta_scaler_X.pkl"
SCALER_Y_PATH = "v3_delta_scaler_y.pkl"
WINDOW_SIZE = 14

# --- Helper Functions ---
def _to_ts(d: str) -> pd.Timestamp:
    return pd.to_datetime(d).normalize()

def parse_volume(v):
    if isinstance(v, str):
        v = v.replace(",", "").strip()
        if v.endswith("M"):
            return float(v[:-1]) * 1e6
        if v.endswith("K"):
            return float(v[:-1]) * 1e3
    return float(v)

def mae(a, b): return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

# --- Core Logic ---
def get_predictions(start_date_str, end_date_str):
    # Load Artifacts
    try:
        model = load_model(MODEL_PATH)
        with open(SCALER_X_PATH, "rb") as f: scaler_X = pickle.load(f)
        with open(SCALER_Y_PATH, "rb") as f: scaler_y = pickle.load(f)
    except Exception as e:
        return None, f"Error loading model/scalers: {str(e)}"

    # Load & Preprocess Data
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        return None, "CSV file not found."

    # Rename columns if needed
    col_map = {"Close": "Price", "Volume": "Vol."}
    df = df.rename(columns=col_map)
    
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if "Vol." in df.columns:
        df["Vol."] = df["Vol."].apply(parse_volume)

    # Indicator Calculation
    df["Volatility"] = df["Price"].rolling(window=5).std()
    
    delta = df["Price"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Momentum"] = df["Price"] - df["Price"].shift(5)
    
    # Fill NaNs
    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)

    # Targets
    df["Price_next"] = df["Price"].shift(-1)
    df["LogRet_t1"] = np.log(df["Price_next"] / df["Price"])
    df = df.dropna(subset=["Price_next", "LogRet_t1"]).reset_index(drop=True)

    # Features
    base_features = ["Price", "Open", "High", "Low", "Vol."]
    features = base_features + ["Volatility", "RSI", "Momentum"]

    # Prepare Sequences
    X_all = df[features].values.astype(float)
    y_all = df["LogRet_t1"].values.astype(float)
    dates = df["Date"].values
    prices = df["Price"].values.astype(float)
    prices_next = df["Price_next"].values.astype(float)

    X_seq, y_seq, d_seq, p_seq, p_next_seq = [], [], [], [], []
    
    for i in range(WINDOW_SIZE - 1, len(df) - 1):
        X_seq.append(X_all[i - WINDOW_SIZE + 1 : i + 1])
        y_seq.append(y_all[i])
        d_seq.append(dates[i])
        p_seq.append(prices[i])
        p_next_seq.append(prices_next[i])

    # Convert to Numpy
    X_seq = np.asarray(X_seq)
    y_seq = np.asarray(y_seq)
    d_seq = pd.to_datetime(np.asarray(d_seq))
    p_seq = np.asarray(p_seq)
    p_next_seq = np.asarray(p_next_seq)

    # Filter by Date Range
    start_ts = _to_ts(start_date_str)
    end_ts = _to_ts(end_date_str)
    mask = (d_seq >= start_ts) & (d_seq <= end_ts)

    if not np.any(mask):
        return None, "No data found for the selected date range."

    X_test = X_seq[mask]
    y_test = y_seq[mask]
    d_test = d_seq[mask]
    p_test = p_seq[mask]
    p_next_test = p_next_seq[mask]

    # Scaling & Prediction
    n_features = X_test.shape[2]
    X_scaled = scaler_X.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    y_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    pred_scaled = model.predict(X_scaled, verbose=0).reshape(-1, 1)
    pred_logret = scaler_y.inverse_transform(pred_scaled).flatten()
    
    # Reconstruct Price
    pred_price = p_test * np.exp(pred_logret)
    abs_err = np.abs(pred_price - p_next_test)

    # DataFrame for Result
    results = pd.DataFrame({
        "Date": d_test,
        "Actual_Next_Close": p_next_test,
        "Predicted_Next_Close": pred_price,
        "Error": abs_err,
        "LogRet_Pred": pred_logret
    })

    # Metrics
    metrics = {
        "MAE": round(mae(pred_price, p_next_test), 4),
        "RMSE": round(rmse(pred_price, p_next_test), 4),
        "Accuracy_50": f"{float(np.mean(abs_err <= 50)) * 100:.2f}%",
        "Drop_Predictions": int(np.sum(pred_logret < 0))
    }

    return results, metrics

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    # Defaults
    start_date = "2025-11-01"
    end_date = "2025-12-31"
    
    if request.method == "POST":
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")

    results, info = get_predictions(start_date, end_date)
    
    if results is None:
        return render_template("index.html", error=info)

    # Generate Plot
    plt.figure(figsize=(10, 5))
    plt.plot(results["Date"], results["Actual_Next_Close"], label="Actual", color="#2c3e50")
    plt.plot(results["Date"], results["Predicted_Next_Close"], label="Predicted", color="#e74c3c", linestyle="--")
    plt.title(f"Stock Price Prediction ({start_date} to {end_date})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Convert dataframe to list of dicts for template
    table_data = results.sort_values("Date").to_dict(orient="records")

    return render_template("index.html", 
                           plot_url=plot_url, 
                           metrics=info, 
                           table_data=table_data,
                           start_date=start_date,
                           end_date=end_date)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)