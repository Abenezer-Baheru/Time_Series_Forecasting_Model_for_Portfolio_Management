import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mplfinance as mpf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EDAAndPreprocessing:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def fetch_data(self):
        try:
            logging.info("Fetching data from Yahoo Finance...")
            self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)
            logging.info("Data fetching completed.")
            logging.info(f"First few rows of the data:\n{self.data.head()}")
        except Exception as e:
            logging.error(f"Error fetching data: {e}")

    def check_missing_values(self):
        try:
            logging.info("Checking for missing values...")
            missing_values = self.data.isnull().sum()
            logging.info(f"Missing values:\n{missing_values}")
        except Exception as e:
            logging.error(f"Error checking for missing values: {e}")

    def check_duplicate_rows(self):
        try:
            logging.info("Checking for duplicate rows...")
            duplicate_rows = self.data.duplicated().sum()
            logging.info(f"Number of duplicate rows: {duplicate_rows}")
        except Exception as e:
            logging.error(f"Error checking for duplicate rows: {e}")

    def check_white_spaces(self):
        try:
            logging.info("Checking for white spaces in column names and data...")
            logging.info(f"Column names before stripping white spaces:\n{self.data.columns}")
            white_spaces = self.data.apply(lambda x: x.str.contains(' ', na=False) if x.dtype == "object" else False).sum()
            logging.info(f"White spaces in data columns:\n{white_spaces}")
        except Exception as e:
            logging.error(f"Error checking for white spaces: {e}")

    def display_dataset_size(self):
        try:
            logging.info("Checking dataset size...")
            dataset_size = self.data.shape
            logging.info(f"Dataset size: {dataset_size}")
        except Exception as e:
            logging.error(f"Error checking dataset size: {e}")

    def display_data_types(self):
        try:
            logging.info("Checking data types of each column...")
            data_types = self.data.dtypes
            logging.info(f"Data types of each column:\n{data_types}")
        except Exception as e:
            logging.error(f"Error checking data types: {e}")

    def display_basic_statistics(self):
        try:
            logging.info("Displaying basic statistics...")
            stats = self.data.describe()
            logging.info(f"Basic statistics:\n{stats.to_frame()}")
        except Exception as e:
            logging.error(f"Error displaying basic statistics: {e}")

    def plot_closing_prices(self):
        try:
            logging.info("Plotting closing prices...")
            plt.figure(figsize=(14, 7))
            for ticker in self.tickers:
                plt.plot(self.data['Close'][ticker], label=ticker)
            plt.title('Closing Prices Over Time')
            plt.xlabel('Date')
            plt.ylabel('Closing Price')
            plt.legend()
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting closing prices: {e}")

    def calculate_daily_pct_change(self):
        try:
            logging.info("Calculating daily percentage change...")
            daily_pct_change = self.data['Close'].pct_change()
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
            for i, ticker in enumerate(self.tickers):
                sns.lineplot(data=daily_pct_change[ticker], ax=axes[i])
                axes[i].set_title(f'Daily Percentage Change - {ticker}')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel('Percentage Change')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error calculating daily percentage change: {e}")

    def calculate_rolling_mean_std(self):
        try:
            logging.info("Calculating rolling mean and standard deviation...")
            rolling_mean = self.data['Close'].rolling(window=20).mean()
            rolling_std = self.data['Close'].rolling(window=20).std()
            plt.figure(figsize=(14, 7))
            for ticker in self.tickers:
                plt.plot(rolling_mean[ticker], label=f'{ticker} Rolling Mean')
                plt.plot(rolling_std[ticker], label=f'{ticker} Rolling Std')
            plt.title('Rolling Mean and Standard Deviation')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.show()
        except Exception as e:
            logging.error(f"Error calculating rolling mean and standard deviation: {e}")

    def calculate_outliers(self, series):
        try:
            logging.info("Calculating outliers using IQR...")
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            return outliers
        except Exception as e:
            logging.error(f"Error calculating outliers: {e}")

    def analyze_outliers(self):
        try:
            logging.info("Analyzing outliers for daily percentage change...")
            daily_pct_change = self.data['Close'].pct_change()
            for ticker in self.tickers:
                outliers = self.calculate_outliers(daily_pct_change[ticker])
                logging.info(f"{ticker} - Number of outliers in daily percentage change: {outliers.shape[0]}")
            plt.figure(figsize=(14, 7))
            sns.boxplot(data=daily_pct_change)
            plt.title('Outlier Detection in Daily Percentage Change')
            plt.ylabel('Percentage Change')
            plt.show()
        except Exception as e:
            logging.error(f"Error analyzing outliers: {e}")

    def outlier_analysis(self):
        try:
            logging.info("Analyzing outliers for all columns...")
            for ticker in self.tickers:
                logging.info(f"Outlier analysis for {ticker}:")
                for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    outliers = self.calculate_outliers(self.data[column][ticker])
                    logging.info(f"{column} - Number of outliers: {outliers.shape[0]}")
                    # Optionally, print the outlier values
                    # logging.info(outliers)
        except Exception as e:
            logging.error(f"Error in outlier analysis: {e}")

    def plot_outliers_in_volume(self):
        try:
            logging.info("Plotting outliers in volume...")
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
            for i, ticker in enumerate(self.tickers):
                sns.boxplot(data['Volume'][ticker], ax=axes[i])
                axes[i].set_title(f'Outliers in Volume for {ticker}')
                axes[i].set_xlabel('Volume')
                axes[i].set_ylabel('Frequency')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting outliers in volume: {e}")

    def plot_closing_price_distribution(self):
        try:
            logging.info("Plotting distribution of closing prices...")
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
            for i, ticker in enumerate(self.tickers):
                sns.histplot(data['Close'][ticker], kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of Closing Prices - {ticker}')
                axes[i].set_xlabel('Closing Price')
                axes[i].set_ylabel('Frequency')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting closing price distribution: {e}")

    def identify_unusual_returns(self):
        try:
            logging.info("Identifying days with unusually high or low returns...")
            daily_pct_change = self.data['Close'].pct_change()
            high_returns = daily_pct_change[daily_pct_change > daily_pct_change.quantile(0.95)]
            low_returns = daily_pct_change[daily_pct_change < daily_pct_change.quantile(0.05)]
            logging.info("Days with unusually high returns:")
            logging.info(high_returns.dropna())
            logging.info("\nDays with unusually low returns:")
            logging.info(low_returns.dropna())
        except Exception as e:
            logging.error(f"Error identifying unusual returns: {e}")

    def seasonal_decomposition(self):
        try:
            logging.info("Performing seasonal decomposition...")
            fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
            for i, ticker in enumerate(self.tickers):
                result = seasonal_decompose(self.data['Close'][ticker], model='multiplicative', period=252)
                result.observed.plot(ax=axes[i, 0], legend=False)
                axes[i, 0].set_title(f'{ticker} - Observed', fontsize=12, fontweight='bold')
                axes[i, 0].set_ylabel('Value', fontsize=10, fontweight='bold')
                result.trend.plot(ax=axes[i, 1], legend=False)
                axes[i, 1].set_title(f'{ticker} - Trend', fontsize=12, fontweight='bold')
                result.seasonal.plot(ax=axes[i, 2], legend=False)
                axes[i, 2].set_title(f'{ticker} - Seasonal', fontsize=12, fontweight='bold')
                result.resid.plot(ax=axes[i, 3], legend=False)
                axes[i, 3].set_title(f'{ticker} - Residual', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error performing seasonal decomposition: {e}")

    def plot_correlation_matrix(self):
        try:
            logging.info("Calculating and plotting correlation matrix...")
            correlation_matrix = self.data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Correlation Matrix')
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting correlation matrix: {e}")

    def calculate_macd_fit(self, price, slow=26, fast=12, signal=9):
        try:
            logging.info(f"Calculating MACD and fit metrics for {price.name}...")
            exp1 = price.ewm(span=fast, adjust=False).mean()
            exp2 = price.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            mse = mean_squared_error(macd, signal_line)
            mae = mean_absolute_error(macd, signal_line)
            r2 = r2_score(macd, signal_line)
            return macd, signal_line, mse, mae, r2
        except Exception as e:
            logging.error(f"Error calculating MACD and fit metrics: {e}")

    def plot_macd(self):
        try:
            logging.info("Plotting MACD and Signal Line...")
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
            for i, ticker in enumerate(self.tickers):
                macd, signal_line, mse, mae, r2 = self.calculate_macd_fit(self.data['Close'][ticker])
                axes[i].plot(self.data.index, macd, label='MACD')
                axes[i].plot(self.data.index, signal_line, label='Signal Line')
                axes[i].set_title(f'MACD for {ticker}', fontsize=14, fontweight='bold')
                axes[i].set_xlabel('Date', fontsize=12, fontweight='bold')
                axes[i].set_ylabel('MACD', fontsize=12, fontweight='bold')
                axes[i].legend(fontsize=10, frameon=True)
                axes[i].text(0.02, 0.95, f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}", transform=axes[i].transAxes, 
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5), fontsize=10, fontweight='bold')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting MACD: {e}")

    def calculate_bollinger_bands(self, price, window=20, num_std=2):
        try:
            logging.info(f"Calculating Bollinger Bands for {price.name}...")
            rolling_mean = price.rolling(window).mean()
            rolling_std = price.rolling(window).std()
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            return rolling_mean, upper_band, lower_band
        except Exception as e:
            logging.error(f"Error calculating Bollinger Bands: {e}")

    def plot_bollinger_bands(self):
        try:
            logging.info("Plotting Bollinger Bands...")
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
            for i, ticker in enumerate(self.tickers):
                rolling_mean, upper_band, lower_band = self.calculate_bollinger_bands(self.data['Close'][ticker])
                axes[i].plot(self.data.index, self.data['Close'][ticker], label='Closing Price')
                axes[i].plot(self.data.index, rolling_mean, label='20-Day MA')
                axes[i].plot(self.data.index, upper_band, label='Upper Band')
                axes[i].plot(self.data.index, lower_band, label='Lower Band')
                axes[i].set_title(f'Bollinger Bands for {ticker}', fontsize=14, fontweight='bold')
                axes[i].set_xlabel('Date', fontsize=12, fontweight='bold')
                axes[i].set_ylabel('Price', fontsize=12, fontweight='bold')
                axes[i].legend(fontsize=10, frameon=True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting Bollinger Bands: {e}")

    def plot_acf_pacf(self):
        try:
            logging.info("Plotting ACF and PACF...")
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 14))
            for i, ticker in enumerate(self.tickers):
                plot_acf(self.data['Close'][ticker].dropna(), lags=40, ax=axes[0, i])
                axes[0, i].set_title(f'Autocorrelation for {ticker}')
                plot_pacf(self.data['Close'][ticker].dropna(), lags=40, ax=axes[1, i])
                axes[1, i].set_title(f'Partial Autocorrelation for {ticker}')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting ACF and PACF: {e}")

    def plot_candlestick_chart(self):
        try:
            logging.info("Plotting candlestick charts...")
            for ticker in self.tickers:
                df = self.data['Close'][ticker].to_frame().join(self.data['Open'][ticker], rsuffix='_Open').join(
                    self.data['High'][ticker], rsuffix='_High').join(self.data['Low'][ticker], rsuffix='_Low').join(
                    self.data['Volume'][ticker], rsuffix='_Volume')
                df.columns = ['Close', 'Open', 'High', 'Low', 'Volume']
                df.index.name = 'Date'
                mpf.plot(df, type='candle', volume=True, title=f'Candlestick Chart for {ticker}', style='yahoo')
        except Exception as e:
            logging.error(f"Error plotting candlestick charts: {e}")

    def save_cleaned_data(self):
        try:
            logging.info("Saving cleaned data to CSV...")
            self.data.to_csv('../src/data/cleaned_data.csv')
            logging.info("Cleaned data saved to '../src/data/cleaned_data.csv'")
        except Exception as e:
            logging.error(f"Error saving cleaned data: {e}")

if __name__ == "__main__":
    # Define the ticker symbols
    tickers = ['TSLA', 'BND', 'SPY']
    
    # Set the date range
    start_date = '2015-01-01'
    end_date = '2025-01-31'
    
    # Create an instance of the EDAAndPreprocessing class
    eda = EDAAndPreprocessing(tickers, start_date, end_date)
    
    # Perform EDA and preprocessing steps
    eda.fetch_data()
    eda.check_missing_values()
    eda.check_duplicate_rows()
    eda.check_white_spaces()
    eda.display_dataset_size()
    eda.display_data_types()
    eda.display_basic_statistics()
    eda.plot_closing_prices()
    eda.calculate_daily_pct_change()
    eda.calculate_rolling_mean_std()
    eda.analyze_outliers()
    eda.outlier_analysis()
    eda.plot_outliers_in_volume()
    eda.plot_closing_price_distribution()
    eda.identify_unusual_returns()
    eda.seasonal_decomposition()
    eda.plot_correlation_matrix()
    eda.plot_macd()
    eda.plot_bollinger_bands()
    eda.plot_acf_pacf()
    eda.plot_candlestick_chart()
    eda.save_cleaned_data()