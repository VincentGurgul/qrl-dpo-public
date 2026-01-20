'''Naive Benchmark Module'''

import pandas as pd

from config import tickers
from utilities.metrics import calculate_test_performance


class EqualWeights():
    '''Implements a simple portfolio optimization model that assigns equal
    weights to all assets. It evaluates the performance of the equal-weight
    strategy using average profit and Sharpe ratio.

    Args:
        None

    Methods:
        evaluate: Evaluates the performance of the equal-weight portfolio on test data.
    '''
    def __init__(self):

        # Assign equal weights to each asset
        self.portfolio_allocation = [1/len(tickers)]*len(tickers)

    def evaluate(self, test_data: pd.DataFrame) -> tuple:
            
        # Calculate and return test performance
        avg_profit, sharpe_ratio = calculate_test_performance(
            test_data,
            self.portfolio_allocation,
        )
        return avg_profit, sharpe_ratio


# Example usage
if __name__ == '__main__':

    price_data_test = pd.read_parquet('./data/price_data_test.parquet.gzip')

    model = EqualWeights()
    avg_profit, sharpe_ratio = model.evaluate(price_data_test)
        
    print(f'Profit p.a.:\t{avg_profit*100:.4f} %')
    print(f'Sharpe Ratio:\t{sharpe_ratio:.4f}\n')
