'''Functions for Mean-Variance Optimization'''

import tqdm
import pandas as pd
import numpy as np
import plotly.graph_objs as go

from pypfopt import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov
from pypfopt import objective_functions

from config import tickers
from utilities.metrics import sharpe_ratio, calculate_test_performance


def get_optimal_portfolio(
    train_data: pd.DataFrame,
    short_selling: bool = False,
    gamma: float = None,
    risk_aversion: float = None,
    risk_free_rate: float = 0,
) -> dict:
    '''Function to calculate the optimal portfolio using mean-variance
    optimization.

    Args:
        train_data (pd.DataFrame): DataFrame containing training data (daily
            log returns).
        short_selling (bool, optional): Whether short selling is allowed.
            Defaults to False.
        gamma (float, optional): Regularization parameter for L2 regularization.
            Reduces allocation of zero weight to assets. Defaults to None, i.e.
            no regularization of portfolio weights.
        risk_aversion (float, optional): Risk aversion parameter for quadratic
            utility optimization. Defaults to None, i.e. maximizing Sharpe ratio.
        risk_free_rate (float, optional): Risk-free rate for Sharpe ratio
            calculation. Defaults to 0.
    '''
    if short_selling:
        bounds = (-1, 1)
    else:
        bounds = (0, 1)

    # Calculate mean historical returns and sample covariance matrix
    mean_returns = mean_historical_return(
        prices=train_data,
        returns_data=True,
        log_returns=True,
    )
    covariance_matrix = sample_cov(
        prices=train_data,
        returns_data=True,
        log_returns=True,
    )

    # Optimize for maximal Sharpe ratio
    efficient_frontier = EfficientFrontier(
        expected_returns=mean_returns,
        cov_matrix=covariance_matrix,
        weight_bounds=bounds,
    )
    if gamma:
        efficient_frontier.add_objective(objective_functions.L2_reg, gamma=1)

    if risk_aversion:
        optimal_portfolio = efficient_frontier.max_quadratic_utility(risk_aversion=risk_aversion)
    else:
        optimal_portfolio = efficient_frontier.max_sharpe(
            risk_free_rate=risk_free_rate,
        )

    return optimal_portfolio


class MeanVarianceOptimization:
    '''Class to evaluate the MVO model using SPO and DPO strategies.

    Args:
        train_data (pd.DataFrame): DataFrame containing training data (daily returns).
        test_data (pd.DataFrame): DataFrame containing test data (daily returns).
        kwargs: Additional keyword arguments for the optimization, e.g. gamma,
            risk_aversion, or risk_free_rate. Note: risk_aversion parameter
            switches maximization approach from maximizing Sharpe ratio to
            maximizing quadratic utility.
    '''
    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        **kwargs,
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.kwargs = kwargs

    def evaluate_spo(
        self,
        verbose: int = 0,
    ) -> tuple:
        '''
        Evaluate the static portfolio optimization (SPO) strategy.

        Runs mean-variance optimization on the entire training data, evaluates
        performance on the test data.

        Args:
            verbose (int): Verbosity level. Defaults to 1.

        Returns:
            tuple: A tuple containing:
                - profit (float): The average yearly profit calculated based on daily returns.
                - sharpe_ratio (float): The Sharpe ratio of the portfolio.
        '''
        # Get optimal portfolio
        optimal_portfolio = get_optimal_portfolio(self.train_data, **self.kwargs)
        optimal_weights = list(optimal_portfolio.values())

        # Calculate profit and sharpe ratio
        profit, sharpe = calculate_test_performance(self.test_data, optimal_weights)

        if verbose > 0:
            print('\nPortfolio Allocation (SPO):')
            for asset, weight in optimal_portfolio.items():
                print(f'{asset:<10} {(weight*100):.2f} %')
            print(f'\nProfit p.a. (SPO): {profit*100:.4f} %')
            print(f'Sharpe Ratio (SPO): {sharpe:.4f}\n')

        return profit, sharpe

    def evaluate_dpo(
        self,
        interval: int = 30,
        verbose: int = 0,
    ) -> tuple:
        '''
        Evaluate the dynamic portfolio optimization (DPO) strategy.

        Runs mean-variance optimization in rolling intervals. For each interval,
        adds `n` days of test data to the training data and evaluates the next `n`
        days of test data. Computes cumulative profit and average Sharpe ratio.

        Args:
            interval (int): Number `n` of days in each rolling window.
                Defaults to 30.
            verbose (int): Verbosity level. Defaults to 1.

        Returns:
            tuple: A tuple containing:
                - profit (float): The average yearly profit calculated based on daily returns.
                - sharpe_ratio (float): The Sharpe ratio of the portfolio.
        '''
        # setting number of intervals for the dynamic optimization loop
        # adding +1 to ensure entire dataset is covered
        num_intervals = len(self.test_data) // interval + 1 
        
        # empty array for daily portfolio returns of all intervals
        all_daily_portfolio_returns = np.array([])

        if verbose > 0:
            print(f'Performing dynamic portfolio optimization over {num_intervals} intervals...\n')
            wrapper = tqdm.tqdm
        else:
            wrapper = lambda x: x

        for i in wrapper(range(num_intervals)):
            test_start_idx = i * interval
            test_end_idx = test_start_idx + interval

            rolling_train_data = pd.concat((self.train_data, self.test_data[:test_start_idx]))
            rolling_test_data = self.test_data[test_start_idx:test_end_idx]

            # Get optimal portfolio
            optimal_portfolio = get_optimal_portfolio(rolling_train_data, **self.kwargs)

            # Get daily returns and append them to final array
            daily_portfolio_returns = np.sum(rolling_test_data.values * list(optimal_portfolio.values()), axis=1)
            all_daily_portfolio_returns = np.append(all_daily_portfolio_returns, daily_portfolio_returns)

            if verbose > 1:
                print(f'\nPeriod {i+1} Portfolio Allocations:')
                for asset, weight in optimal_portfolio.items():
                    print(f'{asset:<10} {(weight*100):.2f} %')

        # Calculate profit and sharpe ratio
        profit, sharpe_ratio = calculate_test_performance(all_daily_portfolio_returns)

        # Summarize overall performance
        if verbose > 0:
            print(f'\nProfit p.a. (DPO): {profit * 100:.4f} %')
            print(f'Sharpe Ratio (DPO): {sharpe_ratio:.4f}\n')

        return profit, sharpe_ratio
