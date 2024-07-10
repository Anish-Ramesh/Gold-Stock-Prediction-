# Gold-Stock-Prediction:

 * This work aims to predict the next day’s opening value will be high or low, the result will be a boolean number either 0 or 1.
 * This work uses single dataset gold.csv which is split into training and testing dataset later in coding. So the scores are high than usual, normally the model will have unknown testing data to work on.
 * It is a simple work that provides efficient result to predict next day opening value of market either is going to  go up(1) or down(0). The parameter it needs is the present day closing value.

# Data Retrieval: 

* It checks if a CSV file (goldstock.csv) exists. If it does, it reads the data from the file; otherwise, it retrieves historical stock data for the S&P 500 index 

# Data Exploration: 

* It displays the retrieved stock data, focusing on the "Close" prices, and plots a bar chart of the closing prices.

# Feature Engineering: 

* It creates a new column ("Tomorrow") by shifting the "Close" prices one day into the future to simulate predicting future stock movements. It then creates a binary target variable ("Target") based on whether the closing price increases the next day.

![image](https://github.com/Anish-Ramesh/Gold-Stock-Prediction-/assets/140417012/cdaa0b85-bd4a-42a7-836e-b34b21ed44e2)




![image](https://github.com/Anish-Ramesh/Gold-Stock-Prediction-/assets/140417012/118305df-774b-4d64-9086-ebd2d52ebec3)

# Data Visualization: 

* It defines a function _plot_series() to plot time series data and visualizes the closing prices over time.

# Model Training: 

* It trains a Random Forest classifier using historical stock data features ("Close", "Volume", "Open", "High", "Low") to predict the binary target variable ("Target").

# Model Evaluation: 

* It evaluates the precision score of the trained model on a test dataset.

# Backtesting: 

* It performs backtesting by simulating predictions over time and evaluates the precision score over multiple periods.


![image](https://github.com/Anish-Ramesh/Gold-Stock-Prediction-/assets/140417012/b23c133a-4cdf-4455-a900-04705b3c4b2a)







