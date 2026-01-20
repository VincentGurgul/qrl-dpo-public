'''This module contains functions and classes for data processing and manipulation.

- log_difference_dataframe: Log-difference transformation for time series data.
- TimeSeriesCrossValidation: Custom cross-validation class for time series data.
- RLDataLoader: Custom DataLoader class for reinforcement learning applications.
'''
import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from statsforecast import StatsForecast, models

from utilities.wrappers import progress_bar, np_cache


def log_difference_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    '''Returns dataframe where all variables are log-differenced once.'''
    return df.replace(0, np.nan).apply(lambda x: np.log(x).diff(), axis=0)


class TimeSeriesCrossValidation():
    '''Performs time series cross-validation with a custom progress bar.

    Attributes:
        data (pd.DataFrame): The time series data to be split.
        n_splits (int): The total number of splits for cross-validation.
        max_train_size (int): The maximum size of the training set in each split
            for a contant train window time series cross-validation. Defaults
            to None, i.e. an increasing window time series cross-validation.
    '''
    def __init__(
        self,
        data: pd.DataFrame,
        n_splits: int,
        max_train_size: int = None,
    ):
        # Add one split because the first one will be skipped due to
        # insufficient train data size.
        self.n_splits = n_splits + 1 
        self.data = data
        self.max_train_size = max_train_size

    def __call__(self):
        '''Generates train-test splits for cross-validation.

        This method initializes the `TimeSeriesSplit` generator and wraps it
        with a custom progress bar. The first fold is skipped to ensure
        sufficient training data in the splits.

        Yields:
            tuple: A tuple containing:
                - fold (int): The current fold number.
                - train_index (np.ndarray): The indices of the training set for
                    the fold.
                - test_index (np.ndarray): The indices of the testing set for
                    the fold.
        '''
        # Initialize TimeSeriesSplit
        self.tss = TimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=self.max_train_size,
        )
        self.tss = self.tss.split(self.data)

        # Iterate through the splits with a progress bar
        generator = progress_bar(
            enumerate(self.tss), length=self.n_splits, print_end='\n\n')

        for fold, (train_index, test_index) in generator:

            # Skip the first fold
            if fold == 0:
                continue
            
            # Yield train and test indices
            yield train_index, test_index


class RLDataLoader:
    '''Loads custom PyTorch DataLoaders for Reinforcement Learning (RL) applications.

    Args:
        data_train (pd.DataFrame): DataFrame with train dataset in standard
            tabular format.
        data_test (pd.DataFrame): DataFrame with validation dataset.
        shuffle (bool, optional): set to True to have the data reshuffled at
            every epoch. Defaults to False.

    Returns:
        tuple: A tuple containing two DataLoader objects:
            - train_loader (DataLoader): DataLoader for the training dataset with
                pairs of sequential states of length `window_size`.
            - val_loader (DataLoader): DataLoader for the val dataset, which
                includes the last `window_size` days of training data
                concatenated with the val data.

    Notes:
        - The training dataset contains sequences derived solely from the
            training data.
        - The val dataset includes the last `window_size` days of training data
            to ensure continuity for the first sample.
    '''
    def __init__(
        self,
        data_train: pd.DataFrame,
        data_val: pd.DataFrame,
        shuffle: bool = False,
    ):
        self.data_train = data_train
        self.data_val = data_val
        self.shuffle = shuffle

    class RLDataset(torch.utils.data.Dataset):
        '''Custom PyTorch Dataset for Reinforcement Learning (RL) applications.

        This dataset generates samples consisting of pairs of sequential states: 
        - The `current state` represented by a window of historical data.
        - The `next state` represented by the subsequent window of data. 

        Each entry in the dataset corresponds to two consecutive windows of size `window_size`. 

        Args:
            df (pd.DataFrame): Input DataFrame with each column being a feature
                and each row being an instance or timepoint.
            window_size (int): Number of consecutive data points to consider for each state window.
        '''
        def __init__(self, df: pd.DataFrame, window_size: int, forecast_size: int):
            self.data = df.values
            self.window_size = window_size
            self.forecast_size = forecast_size

        def __len__(self) -> int:
            # -1 to skip last window which isn't full length
            return len(self.data) // (self.window_size + self.forecast_size) - 1

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            train_start = idx*self.window_size
            train_end = train_start + self.window_size
            test_start = train_end
            test_end = train_end + self.window_size + self.forecast_size
            
            input_data = self.data[train_start:train_end]
            if self.forecast_size > 0:
                input_data = arima_forecast(input_data, self.forecast_size)
            input_window = torch.tensor(
                input_data,
                dtype=torch.float32,
            )
            target_window = torch.tensor(
                self.data[test_start:test_end],
                dtype=torch.float32,
            )
            return input_window, target_window

    def __call__(
        self,
        batch_size: int,
        window_size: int,
        forecast_size: int = 0,
    ):
        '''Returns DataLoader objects for the training and validation datasets.
        
        Args:
            batch_size (int): Number of samples per batch.
            window_size (int): Number of consecutive data points to consider for
                each state window.
            forecast_size (int, optional): Number of days to forecast using ARIMA.
                Defaults to 0.
        '''
        train_dataset = self.RLDataset(
            self.data_train,
            window_size=window_size,
            forecast_size=forecast_size,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=self.shuffle,
        )
        
        val_data = pd.concat((
            self.data_train[-window_size:],
            self.data_val
        )) # add last window from the train data to ensure continuity
        val_dataset = self.RLDataset(
            val_data,
            window_size=window_size,
            forecast_size=forecast_size,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=self.shuffle,
        )
        
        return train_loader, val_loader
    
    @property
    def number_of_assets(self):
        '''Returns the number of financial assets in the dataset.'''
        return self.data_train.shape[1]


@np_cache(maxsize=512)
def arima_forecast(values, forecast_size):
    '''Transforms a given NumPy array, applies AutoARIMA forecasting, 
    and returns an extended NumPy array.
    
    Args:
        values (np.ndarray): Input data (2D array where each column is a time series).
        forecast_size (int): Number of forecast steps.
        
    Returns:
        np.ndarray: Extended NumPy array with forecasts appended.
    '''
    if forecast_size == 0:
        return values
    
    num_rows, num_cols = values.shape  # Get original shape
    forecasts = []  # Store forecasted values for each column

    for col in range(num_cols):
        # Create unique_id column
        unique_id = 1.0  

        # Generate a date range (starting from a fixed date)
        start_date = pd.to_datetime('1949-01-01')
        ds = pd.date_range(start=start_date, periods=num_rows, freq='D')

        # Create transformed DataFrame
        transformed_df = pd.DataFrame({
            'unique_id': unique_id,
            'ds': ds,
            'y': values[:, col]  # Use column values directly
        })

        # Initialize and fit AutoARIMA
        model = models.AutoARIMA(
            d=0, D=0, max_p=1, max_q=1, max_P=1, max_Q=1)
        pipeline = StatsForecast(models=[model], freq='D')
        pipeline.fit(transformed_df)
        forecast = pipeline.predict(h=forecast_size)

        # Extract forecasted values as a NumPy array
        forecast_values = forecast['AutoARIMA'].values.reshape(-1, 1)
        forecasts.append(forecast_values)

    # Stack all forecasts horizontally
    forecast_array = np.hstack(forecasts)

    # Append forecasts to the original values
    output_array = np.vstack([values, forecast_array])

    return output_array
