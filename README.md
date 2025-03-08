# **Time Series Forecasting Model for Portfolio Management**

This project involves the exploration, analysis, and forecasting of financial data for three assets: TSLA, BND, and SPY. It integrates advanced time series modeling techniques to predict market trends and optimize portfolio management strategies. The primary objective is to provide actionable insights for balancing risks and maximizing returns, aligning with GMF Investments' goal of leveraging data-driven approaches for client portfolio optimization.

---

## **Project Structure**

```
Time_Series_Forecasting_Model_for_Portfolio_Management/
├── notebooks/
│   ├── EDA_and_Preprocessing.ipynb
│   ├── Model_Forecasting.ipynb
│   ├── Portfolio_Optimization.ipynb
│   ├── __init__.py
├── data/
│   ├── financial_data.csv
├── .gitignore
├── requirements.txt
├── README.md
├── LICENSE
```

---

## **Getting Started**

### **Prerequisites**

Ensure you have the following libraries installed:

- `yfinance`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `scikit-learn`
- `tensorflow`
- `mplfinance`
- `logging`

You can install all required libraries using the command:

```bash
pip install -r requirements.txt
```

### **Project Setup**

1. Clone the repository:
   ```bash
   git clone https://github.com/Abenezer-Baheru/Time_Series_Forecasting_Model_for_Portfolio_Management.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Time_Series_Forecasting_Model_for_Portfolio_Management
   ```

3. Ensure that data files (e.g., `financial_data.csv`) are located in the `data` directory.

4. Run Jupyter notebooks or Python scripts as required for each step of the project.

---

## **Usage**

### **Exploratory Data Analysis (EDA) and Preprocessing**

1. Use `EDA_and_Preprocessing.ipynb` to explore, clean, and prepare the financial data for modeling.
2. Visualize key insights through this notebook:
   ```bash
   jupyter notebook notebooks/EDA_and_Preprocessing.ipynb
   ```

### **Develop Time Series Forecasting Models**

1. Use `Model_Forecasting.ipynb` to develop and evaluate ARIMA, SARIMA, and LSTM models.
2. Run the notebook for step-by-step guidance on model development and evaluation.

### **Forecast Future Market Trends**

1. Leverage trained models to forecast future prices for TSLA, BND, and SPY.
2. Use these forecasts to guide portfolio optimization decisions:
   ```bash
   jupyter notebook notebooks/Model_Forecasting.ipynb
   ```

### **Optimize Portfolio Based on Forecast**

1. Use `Portfolio_Optimization.ipynb` to compute annual returns, covariance matrices, and optimized portfolio weights.
2. Analyze the results, including metrics like Sharpe Ratio and Value at Risk (VaR), through this notebook.

---

## **Project Highlights**

### **1. Forecasting Models**
- Developed and evaluated ARIMA, SARIMA, and LSTM models for forecasting.
- The LSTM model achieved the best performance:
  - MAE: 9.53
  - RMSE: 13.68
  - MAPE: 3.85%

### **2. Portfolio Optimization**
- Optimized a portfolio of TSLA, BND, and SPY using forecasted trends.
- Key metrics:
  - Annual Returns: TSLA (-8.66%), BND (-17.29%), SPY (0.05%)
  - Sharpe Ratio: -12.61 (reflecting poor risk-adjusted returns in the current market scenario).
  - Value at Risk (VaR): TSLA (-0.36), BND (-0.03), SPY (-0.09).

---

## **Business Alignment**

### **Objective**
Guide Me in Finance (GMF) Investments specializes in personalized, data-driven portfolio management strategies. This project demonstrates the use of cutting-edge time series forecasting models to predict market trends and guide asset allocation, directly supporting GMF's mission.

### **Key Achievements**
1. **Data-Driven Insights**:
   - Leveraged LSTM to predict future market trends, outperforming traditional models like ARIMA and SARIMA.
   - Forecasted trends for TSLA, BND, and SPY highlighted critical risk-return tradeoffs.

2. **Actionable Recommendations**:
   - Enhanced portfolio performance by rebalancing allocations, emphasizing stability from BND and SPY while managing Tesla’s high risk.

3. **Risk Management**:
   - Used metrics like Value at Risk (VaR) to quantify downside exposure and improve decision-making.

---

## **Author**
- **Abenezer Baheru** - [Abenezer-Baheru](https://github.com/Abenezer-Baheru)

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

