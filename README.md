# StockPredictions_MLmodule
## 📜 Project Overview
This project is an **interactive stock price prediction tool** using machine learning models:  
- **Linear Regression**
- **Random Forest**
- **XGBoost**

It fetches stock data, analyzes it, and predicts future stock prices using trained models. The tool also visualizes historical data and predicted prices with interactive candlestick charts.

---

## ⚙️ Features
- **Fetch Real-Time Stock Data:** Uses `yfinance` to retrieve stock data.
- **OHLC Data Display:** Displays Open, High, Low, Close data in a tabular format for the last 7 days.
- **Market Capitalization:** Shows the market cap of the selected stock.
- **Volatility Analysis:** Analyzes the stock's short-term volatility.
- **Price Prediction:** Predicts the stock's price for the next two days using:
  - Linear Regression
  - Random Forest
  - XGBoost
- **Interactive Charts:** Plots:
  - Short-term predictions (7 days)
  - Long-term predictions (1 month)
- **Future Predictions Markers:** Displays predicted future prices as markers on the charts.

---

## 🛠️ How It Works
1. **Data Collection:**  
   Fetches historical stock data from Yahoo Finance using the `yfinance` library.
   
2. **Model Training:**  
   Trains three models:
   - **Linear Regression:** Finds the best-fit linear relationship between stock features and the target price.
   - **Random Forest:** Uses multiple decision trees to enhance prediction accuracy.
   - **XGBoost:** An optimized gradient boosting algorithm for high accuracy.

3. **Prediction:**  
   Predicts future prices based on the trained models.

4. **Visualization:**  
   Plots candlestick charts along with predicted price lines and markers for future prices.

---

## 🖥️ Requirements
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - yfinance
  - scikit-learn
  - xgboost
  - plotly
  - tabulate

To install the required libraries, run:
```bash
pip install pandas numpy yfinance scikit-learn xgboost plotly tabulate
```

---

## 🚀 Usage
1. Clone the repository:
   ``bash
   gh repo clone Goral-5/StockPredictions_MLmodule
   ```
3. Run the script:
   ```bash
   main(1).py
   ```

4. Enter a stock ticker symbol (e.g., RELIANCE.BO, TCS.BO) when prompted.
5. View the predictions and interactive charts.

---

## 📊 Example Output
1. **Next 2 Days Predicted Prices:**  
   - Linear Regression: ₹1234.56  
   - Random Forest: ₹1240.78  
   - XGBoost: ₹1238.90  

2. **Interactive Candlestick Chart:**  
   Displays the candlestick chart with predicted prices for short-term and long-term periods.

---

## 🤔 Why This Project?
This project showcases:
- **Practical application of machine learning** for financial analysis.
- **Real-time data handling** and interactive visualization.
- A comprehensive approach to **model comparison** and evaluation.

---

## 🙌 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

---
