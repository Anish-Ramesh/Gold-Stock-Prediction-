# Tata Power Next-Day Close Price Prediction

This project trains and tests an LSTM model to predict **next-day closing prices** for equities using daily OHLCV data and technical indicators.

The pipeline is **generic for any company/stock** as long as you provide a similar daily CSV (with Date, Open, High, Low, Close/Price, Volume). In this repository we have **tested it on Tata stocks (Tata Power / Tata Motors)** and obtained good results.

## Files
- **v3_delta.py**  
  Trains the LSTM model on `TataPower_2005_2025.csv` using:
  - Window size: **14** previous trading days
  - Features: `Price`, `Open`, `High`, `Low`, `Vol.`, `Volatility`, `RSI`, `Momentum`
  - Target: next-day **log return** of `Price`, later converted back to price.
  - Outputs:
    - `v3_delta_savedmodel/` (TensorFlow SavedModel)
    - `v3_delta_scaler_X.pkl` (StandardScaler for features)
    - `v3_delta_scaler_y.pkl` (RobustScaler for target)

- **test_range_v3_delta_csv.py**  
  Loads the saved model and scalers, then evaluates predictions over a chosen date range (currently **2025-11-01 .. 2025-12-31**) using the CSV file. It prints:
  - Per-day actual vs predicted next-day close
  - MAE, RMSE, and `% within ±50` points
  - How many days the model predicted a **drop** (negative log return)
  - A matplotlib plot of actual vs predicted next-day close.

- **TataPower_2005_2025.csv**  
  Historical Tata Power price data used for training and testing.

- **v3_delta_savedmodel/**  
  Trained TensorFlow model directory.

- **v3_delta_scaler_X.pkl**, **v3_delta_scaler_y.pkl**  
  Scalers used for input features and target, saved after training.

- **Figure_1.png**  
  Example plot of **Actual vs Predicted next-day Close** from a test run.

## Example Figure

Below is an embedded example plot from the project:

![Actual vs Predicted Next-Day Close](Figure_1.png)

## How to Run

1. **Environment**
   - Python 3.11
   - TensorFlow 2.15.x
   - Required Python packages: `tensorflow`, `keras`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`.

2. **Train / Retrain the Model**

```bash
python v3_delta.py
```

This:
- Reads `TataPower_2005_2025.csv`
- Trains the LSTM with a 14-day window
- Saves the model and scalers.

3. **Run the Test Script**

```bash
python test_range_v3_delta_csv.py
```

This:
- Loads `v3_delta_savedmodel/` and the scalers
- Evaluates from **2025-11-01** to **2025-12-31**
- Prints metrics and shows the **Actual vs Predicted** plot
- Prints how many days the model predicted a **price drop**.

## Notes
- The main goal is to keep next-day close predictions within **±50** points of the actual close.
- The shorter **14-day window** makes the model more sensitive to recent market moves and better at predicting both **up** and **down** days.
