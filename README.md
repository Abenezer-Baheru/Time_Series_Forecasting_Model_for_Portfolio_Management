# Time Series Forecasting Model for Portfolio Management

This project involves the exploration and preprocessing of financial data for three assets: TSLA, BND, and SPY. The primary objective is to perform Exploratory Data Analysis (EDA) and preprocessing, followed by statistical modeling and forecasting. The project is implemented using Python and various data science libraries.

## Project Structure

```
Time_Series_Forecasting_Model_for_Portfolio_Management/

├── scripts/
│   ├── EDA_and_Preprocessing.py
│   ├── __init__.py
├── tests/
│   ├── test_EDA_and_Preprocessing.py
│   ├── __init__.py
├── notebooks/
│   ├── EDA_and_Preprocessing.ipynb
│   ├── __init__.py
├── .gitignore
├── requirements.txt
├── README.md
```

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:

- yfinance
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn
- mplfinance
- logging

You can install the required libraries using the following command:

```bash
pip install yfinance pandas numpy matplotlib seaborn statsmodels scikit-learn mplfinance
```

### Project Setup

1. Clone the repository:

```bash
git clone https://github.com/Abenezer-Baheru/Time_Series_Forecasting_Model_for_Portfolio_Management.git
cd Time_Series_Forecasting_Model_for_Portfolio_Management
```

2. Navigate to the project directory:

```bash
cd Time_Series_Forecasting_Model_for_Portfolio_Management
```

3. Ensure the data files (`financial_data.csv`) are located in the `src/data` directory.

## Usage

### EDA and Preprocessing

1. The `EDA_and_Preprocessing.py` script contains the `EDAAndPreprocessing` class, which includes methods for various EDA and preprocessing tasks.
2. The `EDA_and_Preprocessing.ipynb` notebook provides a step-by-step guide to using the `EDAAndPreprocessing` class and visualizing the data.

To run the notebook:

```bash
jupyter notebook notebooks/EDA_and_Preprocessing.ipynb
```

### Testing

Unit tests for the `EDAAndPreprocessing` class are provided in the `test_EDA_and_Preprocessing.py` script in the `tests` directory. To run the tests:

```bash
python -m unittest discover tests
```

## Project Files

- `scripts/EDA_and_Preprocessing.py`: The script containing the `EDAAndPreprocessing` class.
- `tests/test_EDA_and_Preprocessing.py`: The script containing unit tests for the `EDAAndPreprocessing` class.
- `notebooks/EDA_and_Preprocessing.ipynb`: The Jupyter notebook for EDA and preprocessing.
- `README.md`: The project documentation.

## Develop Time Series Forecasting Models

### Choose Between ARIMA, SARIMA, or LSTM Models

### Model Evaluation

## Forecast Future Market Trends

### Forecasting with Trained Models

### Interpret Results

## Optimize Portfolio Based on Forecast

### Use Forecasted Data for Optimization

### Summarize Portfolio Adjustments


## Author

- **Abenezer Baheru** - [Abenezer-Baheru](https://github.com/Abenezer-Baheru)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.