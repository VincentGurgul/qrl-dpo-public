'''This module contains utility functions for calculating performance metrics
and evaluating portfolio optimization models.

- softmax: Calculate the softmax of a given tensor.
- sharpe_ratio: Calculate the Sharpe ratio for a given mean return and standard
    deviation.
- sharpe_ratio_series: Calculate the annualized Sharpe ratio from a series of
    daily returns.
- calculate_test_performance: Calculate the average yearly profit and Sharpe
    ratio for a given dataset.
- RLEvaluator: Class to evaluate a PyTorch Actor model using SPO and DPO
    strategies.
'''
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from utilities.wrappers import progress_bar
from utilities.data_processing import arima_forecast
from config import tickers

          
def sharpe_ratio(mean_return: float, std: float) -> float:
    '''Calculate the Sharpe ratio.

    Args:
        mean_return (float): Mean return of the portfolio.
        std (float): Standard deviation of the returns.

    Returns:
        float: Sharpe ratio.
    '''
    rf = 0.0418 # setting the risk-free rate to U.S. 10 Year Treasury Note
    epsilon = 10e-8 # constant to ensure numerical stability
    return (mean_return - rf) / (std + epsilon) if std != 0 else 0


def sharpe_ratio_series(daily_returns: pd.Series) -> float:
    '''Calculate the annualized Sharpe ratio from a series of daily returns.

    Args:
        daily_returns (pd.Series): Series of daily returns.

    Returns:
        float: Annualized Sharpe ratio.
    '''
    mean_return = np.mean(daily_returns) * 252
    std_dev_returns = np.std(daily_returns) * np.sqrt(252)
    return sharpe_ratio(mean_return, std_dev_returns)


def calculate_test_performance(data: np.array, weights: list = None) -> tuple:
    '''
    Calculate the average yearly profit and Sharpe ratio for a given dataset.

    Args:
        data (pd.DataFrame): A DataFrame containing asset or portfolio returns.
            If asset returns, each column represents an asset, each row
            represents a daily return, and weights must be specified. If
            portfolio returns, no weights and a 1-d array must be passed.
        weights (list, optional): A list of weights for the assets. If provided,
            the function calculates the weighted portfolio return. Defaults to
            None, i.e. data is provided weighted.

    Returns:
        tuple: A tuple containing:
            - avg_profit_pa (float): The average yearly profit calculated based
                on daily returns.
            - sharpe_ratio (float): The Sharpe ratio of the portfolio or assets.

    Raises:
        ValueError: If weights are provided but do not match the number of columns
            in `data`.

    Notes:
        - Assumes 252 trading days in a year for annualizing the profit.
    '''
    if weights is not None:
        # Sum the weighted returns across assets to get portfolio return per day
        data = np.sum(data.values * weights, axis=1)

    # Calculate average yearly profit
    avg_profit_pa = ((1 + np.mean(data)) ** 252) - 1
    
    # Calculate the Sharpe ratio
    sharpe_ratio = sharpe_ratio_series(pd.Series(data))

    return avg_profit_pa, sharpe_ratio


class RLEvaluator:
    '''Class to evaluate a PyTorch Actor model using SPO and DPO strategies.

    Args:
        actor (nn.Module or QNN): Trained Actor network, can be written in torch
            or qiskit.
        train_data (pd.DataFrame)): Train data for first step of evaluation.
        test_data (pd.DataFrame): Test data for evaluation and results computation.
    '''
    def __init__(
        self,
        actor: nn.Module,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        forecast_size: int,
        reduce_negatives: bool = True,
    ):
        self.actor = actor
        self.train_data = train_data
        self.test_data = test_data
        self.window_size = int(actor.input_size / len(tickers)) - forecast_size
        self.forecast_size = forecast_size
        self.reduce_negatives = reduce_negatives

    def evaluate_spo(self, verbose: int = 0) -> tuple:
        '''Evaluate using Static Portfolio Optimization (SPO).

        Args:
            verbose (int): Verbosity level for printing details.

        Returns:
            tuple: Total profit and Sharpe ratio from SPO strategy.
        '''
        self.actor.eval()  # putting actor in evaluation mode  

        # Use the last training batch to set initial portfolio allocation
        input_data = arima_forecast(
            self.train_data.tail(self.window_size).values,
            self.forecast_size)

        with torch.no_grad():   
            state = torch.tensor(input_data, dtype=torch.float32).flatten()
            portfolio_allocation = self.actor(state).numpy().squeeze()
        if self.reduce_negatives:
            portfolio_allocation = reduce_negatives(portfolio_allocation)

        # Calculate performance on test set
        avg_profit_pa, sharpe = calculate_test_performance(self.test_data,
                                                           portfolio_allocation)

        if verbose > 0:
            print('\nPortfolio Allocation (SPO):')
            for i in range(len(tickers)):
                print(f'{tickers[i]:<10} {(portfolio_allocation[i]*100):.2f} %')
            print(f'\nProfit p.a. (SPO): {avg_profit_pa*100:.4f} %')
            print(f'Sharpe Ratio (SPO): {sharpe:.4f}\n')

        return avg_profit_pa, sharpe

    def evaluate_dpo(self, interval: int, verbose: int = 0) -> tuple:
        '''Evaluate using Dynamic Portfolio Optimization (DPO).

        Args:
            verbose (int): Verbosity level for printing details.

        Returns:
            tuple: Total profit and Sharpe ratio from DPO strategy.

        Raises:
            ValueError if interval is shorter than actor input window.
        '''
        if interval < self.window_size:
            raise ValueError('DPO interval must be larger than actor input window size.')

        # Merge last batch of training data with the test data to ensure continuity
        test_data = pd.concat((self.train_data.tail(self.window_size), self.test_data))

        self.actor.eval()  # putting actor in evaluation mode

        # setting number of intervals for the dynamic optimization loop
        # adding +1 to ensure entire dataset is covered
        num_intervals = len(self.test_data) // interval + 1

        # empty array for daily portfolio returns of all intervals
        all_daily_portfolio_returns = np.array([])

        if verbose > 0:
            print(f'Performing dynamic portfolio optimization over {num_intervals} intervals...\n')
            wrapper = progress_bar
        else:
            wrapper = lambda x: x

        for i in wrapper(range(num_intervals)):

            # initialize rolling train and test data as consequtive chunks of window_size
            train_start_idx = i * interval
            test_start_idx = train_start_idx + interval
            test_end_idx = test_start_idx + interval

            rolling_train_data = test_data[train_start_idx:test_start_idx]
            rolling_test_data = test_data[test_start_idx:test_end_idx]

            # get portfolio allocation for the given interval based on rolling train dataset
            with torch.no_grad():
                input_data = torch.tensor(
                    arima_forecast(
                        rolling_train_data.tail(self.window_size).values,
                        self.forecast_size,
                    ), dtype=torch.float32)
                portfolio_allocation = self.actor(input_data.flatten()).numpy().squeeze()
            if self.reduce_negatives:
                portfolio_allocation = reduce_negatives(portfolio_allocation)

            if verbose > 1:
                print(f'\nPeriod {i+1} Portfolio Allocations:')
                for i in range(len(tickers)):
                    print(f'{tickers[i]:<10} {(portfolio_allocation[i]*100):.2f} %')
 
            # Get daily returns based on rolling test dataset
            daily_portfolio_returns = np.sum(rolling_test_data.values * portfolio_allocation, axis=1)
            all_daily_portfolio_returns = np.append(all_daily_portfolio_returns, daily_portfolio_returns)

        # Calculate profit and sharpe ratio
        profit, sharpe_ratio = calculate_test_performance(all_daily_portfolio_returns)

        # Summarize overall performance
        if verbose > 0:
            print(f'\nProfit p.a. (DPO): {profit * 100:.4f} %')
            print(f'Sharpe Ratio (DPO): {sharpe_ratio:.4f}\n')

        return profit, sharpe_ratio


def reduce_negatives(vec: np.ndarray, clamp_min: float = -1.0) -> np.ndarray:
    '''Reduces large negative values in a vector while preserving sum to 1.
    
    Args:
        vec (np.ndarray): Input vector (sums to 1).
        clamp_min (float, optional): Minimum allowed value for clamping.
            Defaults to -1.0.

    Returns:
        np.ndarray: Adjusted vector (sums to 1) with reduced large negatives.
    '''
    clamped = np.maximum(vec, clamp_min)
    adjusted = clamped / np.sum(clamped)
    return adjusted
