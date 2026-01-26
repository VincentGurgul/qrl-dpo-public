'''Fetch financial data from yfinance and preprocess.'''

import yfinance as yf

from config import tickers, start_date, end_date
from utilities.data_processing import log_difference_dataframe

import requests


def main():

    # Create a session
    session = requests.Session()

    # Fetch yfinance
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        session=session,
    )

    # Select data
    price_data = data['Adj Close']
    volume_data = data['Volume']

    # Log difference data to ensure stationarity and fill NaNs with 0 since
    # if the asset didn't exist its price and volume change is 0 
    price_data_differenced = log_difference_dataframe(price_data)[1:].fillna(0)
    volume_data_differenced = log_difference_dataframe(volume_data)[1:].fillna(0)

    # Train-val-test split
    train_split_price = int(len(price_data_differenced) * 0.6)
    val_split_price = int(len(price_data_differenced) * 0.8)

    price_data_train = price_data_differenced[:train_split_price]
    price_data_val = price_data_differenced[train_split_price:val_split_price]
    price_data_test = price_data_differenced[val_split_price:]

    train_split_volume = int(len(volume_data_differenced) * 0.6)
    val_split_volume = int(len(volume_data_differenced) * 0.8)

    volume_data_train = volume_data_differenced[:train_split_volume]
    volume_data_val = volume_data_differenced[train_split_volume:val_split_volume]
    volume_data_test = volume_data_differenced[val_split_volume:]

    price_data_differenced.to_parquet('./data/price_data.parquet.gzip', compression='gzip')
    price_data_train.to_parquet('./data/price_data_train.parquet.gzip', compression='gzip')
    price_data_val.to_parquet('./data/price_data_val.parquet.gzip', compression='gzip')
    price_data_test.to_parquet('./data/price_data_test.parquet.gzip', compression='gzip')
    volume_data_differenced.to_parquet('./data/volume_data.parquet.gzip', compression='gzip')
    volume_data_train.to_parquet('./data/volume_data_train.parquet.gzip', compression='gzip')
    volume_data_val.to_parquet('./data/volume_data_val.parquet.gzip', compression='gzip')
    volume_data_test.to_parquet('./data/volume_data_test.parquet.gzip', compression='gzip')


if __name__=='__main__':

    main()
