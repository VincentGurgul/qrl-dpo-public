'''Main script for running all models with fixed hyperparameters.'''

import json
import time
import datetime
import logging
import numpy as np
import pandas as pd
import torch

from humanfriendly import format_timespan

from config import tickers
from predictors import NeuralNetwork, QuantumNeuralNetwork
from predictors.input_transformations import radial_to_linear

from models import (
    EqualWeights,
    MeanVarianceOptimization,
    DDPG,
    DeepQLearning,
)

from utilities.data_processing import TimeSeriesCrossValidation
from utilities.logger import ResultsLogger, activate_operations_logging


#-----------------------------------------------------------------------------
# Config
#-----------------------------------------------------------------------------

# Choose which pipelines to run.
RUN_MODELS = [
    'Equal Weights',
    'Mean Variance Optimization',
    'DDPG',
    'QDPG',
    'Deep Q-Learning',
    'Quantum Q-Learning',
]

GLOBAL_CONFIG = {
    'LOOKBACK_WINDOW': 30,  # lookback window for supervised learning models
    'FORECAST_WINDOW': 7,  # forecast window for RL models
    'DYNAMIC_PO': True,  # whether to run dynamic portfolio optimization
    'DPO_INTERVAL': 30,  # rebalancing interval for dynamic portfolio optimization
    'SHORT_SELLING': True,  # whether to allow negative asset weights
    'CLAMP_NEGATIVES': True,  # whether to clamp negative asset weights to -100%
    'CV_SPLITS': 7,  # number of crossvalidation splits
    'VERBOSE': 0,  # verbosity level for training process logging
    'SEED': 68,  # seed for reproducibility
    'DEBUG': False,  # whether to log all events to a file for debugging purposes
}


#-----------------------------------------------------------------------------
# Initialize Operations and Results Loggers
#-----------------------------------------------------------------------------

def print_results(cv_results, dpo: bool):
    '''Prints the average performance of the models.
    
    Args:
        avg_performance (arraylike): Average performance metrics.
        dpo (bool): Whether to print DPO results.
    '''

    average = np.mean(cv_results, axis=0)
    std = (np.nanpercentile(cv_results, 75, axis=0) - np.nanpercentile(cv_results, 25, axis=0))
    min = np.min(cv_results, axis=0)
    max = np.max(cv_results, axis=0)

    if dpo:
        return f'''Results:
    
Average Profit p.a. (SPO):\t{average[0][0]*100:.4f} % (std: {std[0][0]*100:.4f}, min: {min[0][0]*100:.4f}, max: {max[0][0]*100:.4f})
Average Sharpe Ratio (SPO):\t{average[0][1]:.4f} (std: {std[0][1]:.4f}, min: {min[0][1]:.4f}, max: {max[0][1]:.4f})

Average Profit p.a. (DPO):\t{average[1][0]*100:.4f} % (std: {std[1][0]*100:.4f}, min: {min[1][0]*100:.4f}, max: {max[1][0]*100:.4f})
Average Sharpe Ratio (DPO):\t{average[1][1]:.4f} (std: {std[1][1]:.4f}, min: {min[1][1]:.4f}, max: {max[1][1]:.4f})
'''
    else:
        return f'''Results:

Average Profit p.a. (SPO):\t{average[0]*100:.4f} % (std: {std[0]*100:.4f}, min: {min[0]*100:.4f}, max: {max[0]*100:.4f})
Average Sharpe Ratio (SPO):\t{average[1]:.4f} (std: {std[1]:.4f}, min: {min[1]:.4f}, max: {max[1]:.4f})
'''
    
TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
results_logger = ResultsLogger(f'./results_logs/{TIME}.log')
        
if GLOBAL_CONFIG['DEBUG']:
    activate_operations_logging(level=logging.DEBUG)
    
results_logger.log(f'''
-----------------------------------------------------------------------------

                     PORTFOLIO OPTIMIZATION RESULTS LOG                      

-----------------------------------------------------------------------------

Tickers: {tickers}

Config:
{(json.dumps(GLOBAL_CONFIG, indent=4)
    .replace('{\n', '')
    .replace('\n}', '')
    .replace('"', "'")
)}
''')


#-----------------------------------------------------------------------------
# Initialize Data and Crossvalidation
#-----------------------------------------------------------------------------

# load price data
price_data = pd.read_parquet('./data/price_data.parquet.gzip')

# initialize crossvalidation
crossvalidation = TimeSeriesCrossValidation(
    data=price_data,
    n_splits=GLOBAL_CONFIG['CV_SPLITS'])


#-----------------------------------------------------------------------------
# Run Portfolio Optimization Pipelines and Log Results
#-----------------------------------------------------------------------------

if 'Equal Weights' in RUN_MODELS:

    results_logger.log('''
-----------------------------------------------------------------------------
Naive Baseline: Equal Weights Model
-----------------------------------------------------------------------------
''')

    start_time = time.time()
    cv_results = []

    for train_index, test_index in crossvalidation():
        test_data = price_data.iloc[test_index]
        model = EqualWeights()
        results = model.evaluate(test_data=test_data)
        cv_results.append(results)
    
    results_logger.log('Execution time: ' + format_timespan(time.time() - start_time) + '\n',
        console=False)
    results_logger.log(print_results(cv_results, dpo=False))

if 'Mean Variance Optimization' in RUN_MODELS:

    results_logger.log('''
-----------------------------------------------------------------------------
Mean-Variance Optimization
-----------------------------------------------------------------------------
''')

    start_time = time.time()
    cv_results = []

    for train_index, test_index in crossvalidation():

        train_data = price_data.iloc[train_index]
        test_data = price_data.iloc[test_index]

        # Initialize and run evaluator
        model = MeanVarianceOptimization(
            train_data=train_data,
            test_data=test_data,
            short_selling=GLOBAL_CONFIG['SHORT_SELLING'],
            risk_aversion=10 if GLOBAL_CONFIG['SHORT_SELLING'] else 5,
        )
        spo_results = model.evaluate_spo(
            verbose=GLOBAL_CONFIG['VERBOSE'],
        )
        if GLOBAL_CONFIG['DYNAMIC_PO']:
            dpo_results = model.evaluate_dpo(
                interval=GLOBAL_CONFIG['DPO_INTERVAL'],
                verbose=GLOBAL_CONFIG['VERBOSE'],
            )
            cv_results.append((spo_results, dpo_results))
        else:
            cv_results.append(spo_results)

    results_logger.log('Execution time: ' + format_timespan(time.time() - start_time) + '\n',
        console=False)
    results_logger.log(print_results(cv_results, dpo=GLOBAL_CONFIG['DYNAMIC_PO']))


if 'DDPG' in RUN_MODELS:

    results_logger.log('''
-----------------------------------------------------------------------------
Deep Deterministic Policy Gradient
-----------------------------------------------------------------------------
''')
    
    start_time = time.time()
    cv_results = []

    for train_index, test_index in crossvalidation():

        val_split = int(len(price_data.iloc[train_index]) * 0.8)
        train_data = price_data.iloc[train_index][:val_split]
        val_data = price_data.iloc[train_index][val_split:]
        test_data = price_data.iloc[test_index]

        model = DDPG(
            lookback_window=GLOBAL_CONFIG['LOOKBACK_WINDOW'],
            forecast_window=GLOBAL_CONFIG['FORECAST_WINDOW'],
            batch_size=1,
            predictor=NeuralNetwork,
            hidden_sizes=(30,),
            short_selling=GLOBAL_CONFIG['SHORT_SELLING'],
            reduce_negatives=GLOBAL_CONFIG['CLAMP_NEGATIVES'],
            verbose=GLOBAL_CONFIG['VERBOSE'],
            seed=GLOBAL_CONFIG['SEED'],
        )
        model.train(
            train_data=train_data,
            val_data=val_data,
            actor_lr=0.020239765866555008,
            critic_lr=0.014249327834891122,
            optimizer=torch.optim.SGD,
            # l1_lambda=1e-7,
            l2_lambda=0.009585823379719707,
            # weight_decay=1e-6,
            soft_update=False,
            # tau=1e-3,
            risk_preference=-0.2832085400024138,
            gamma=0.028599514945159235,
            num_epochs=50,
            early_stopping=False,
            patience=10,
        )
        results = model.evaluate(
            test_data=test_data,
            dpo=GLOBAL_CONFIG['DYNAMIC_PO'],
        )
        cv_results.append(results)

    results_logger.log('Execution time: ' + format_timespan(time.time() - start_time) + '\n',
        console=False)
    results_logger.log(print_results(cv_results, dpo=GLOBAL_CONFIG['DYNAMIC_PO']))


if 'QDPG' in RUN_MODELS:

    results_logger.log('''
-----------------------------------------------------------------------------
Quantum Deterministic Policy Gradient
-----------------------------------------------------------------------------
''')

    start_time = time.time()
    cv_results = []

    for train_index, test_index in crossvalidation():

        val_split = int(len(price_data.iloc[train_index]) * 0.8)
        train_data = price_data.iloc[train_index][:val_split]
        val_data = price_data.iloc[train_index][val_split:]
        test_data = price_data.iloc[test_index]

        model = DDPG(
            lookback_window=GLOBAL_CONFIG['LOOKBACK_WINDOW'],
            forecast_window=GLOBAL_CONFIG['FORECAST_WINDOW'],
            batch_size=1,
            predictor=QuantumNeuralNetwork,
            num_weights=60,
            encoding='amplitude',
            input_transformation=radial_to_linear,
            rotation_axes='y',
            short_selling=GLOBAL_CONFIG['SHORT_SELLING'],
            reduce_negatives=GLOBAL_CONFIG['CLAMP_NEGATIVES'],
            verbose=GLOBAL_CONFIG['VERBOSE'],
            seed=GLOBAL_CONFIG['SEED'],
        )
        model.train(
            train_data=train_data,
            val_data=val_data,
            actor_lr=0.09935741130315447,
            critic_lr=0.0018039893844072358,
            optimizer=torch.optim.SGD,
            # l1_lambda=1e7,
            l2_lambda=3.2067524338595386e-06,
            # weight_decay=1e-6,
            soft_update=False,
            # tau=1e-3,
            risk_preference=-0.9286138365176491,
            gamma=0.009826640813865617,
            num_epochs=50,
            early_stopping=False,
            patience=10,
        )
        results = model.evaluate(
            test_data=test_data,
            dpo=GLOBAL_CONFIG['DYNAMIC_PO'],
        )
        cv_results.append(results)

    results_logger.log('Execution time: ' + format_timespan(time.time() - start_time) + '\n',
        console=False)
    results_logger.log(print_results(cv_results, dpo=GLOBAL_CONFIG['DYNAMIC_PO']))


if 'Deep Q-Learning' in RUN_MODELS:

    results_logger.log('''
-----------------------------------------------------------------------------
Deep Q-Learning
-----------------------------------------------------------------------------
''')

    start_time = time.time()
    cv_results = []

    for train_index, test_index in crossvalidation():

        val_split = int(len(price_data.iloc[train_index]) * 0.8)
        train_data = price_data.iloc[train_index][:val_split]
        val_data = price_data.iloc[train_index][val_split:]
        test_data = price_data.iloc[test_index]

        model = DeepQLearning(
            lookback_window=GLOBAL_CONFIG['LOOKBACK_WINDOW'],
            forecast_window=GLOBAL_CONFIG['FORECAST_WINDOW'],
            batch_size=1,
            predictor=NeuralNetwork,
            hidden_sizes=(30,),
            short_selling=GLOBAL_CONFIG['SHORT_SELLING'],
            reduce_negatives=GLOBAL_CONFIG['CLAMP_NEGATIVES'],
            verbose=GLOBAL_CONFIG['VERBOSE'],
            seed=GLOBAL_CONFIG['SEED'],
        )
        model.train(
            train_data=train_data,
            val_data=val_data,
            actor_lr=0.0011422800982086824,
            critic_lr=0.003990673146909851,
            optimizer=torch.optim.Adam,
            # l1_lambda=1e7,
            l2_lambda=0.005716467080685015,
            # weight_decay=1e-7,
            soft_update=True,
            # tau=1e-3,
            risk_preference=-0.8134627523331615,
            gamma=0.04711405953074143,
            num_epochs=50,
            early_stopping=False,
            patience=10,
            num_action_samples=10,
        )
        results = model.evaluate(
            test_data=test_data,
            dpo=GLOBAL_CONFIG['DYNAMIC_PO'],
        )
        cv_results.append(results)

    results_logger.log('Execution time: ' + format_timespan(time.time() - start_time) + '\n',
        console=False)
    results_logger.log(print_results(cv_results, dpo=GLOBAL_CONFIG['DYNAMIC_PO']))


if 'Quantum Q-Learning' in RUN_MODELS:

    results_logger.log('''
-----------------------------------------------------------------------------
Quantum Q-Learning
-----------------------------------------------------------------------------
''')

    start_time = time.time()
    cv_results = []

    for train_index, test_index in crossvalidation():

        val_split = int(len(price_data.iloc[train_index]) * 0.8)
        train_data = price_data.iloc[train_index][:val_split]
        val_data = price_data.iloc[train_index][val_split:]
        test_data = price_data.iloc[test_index]

        model = DeepQLearning(
            lookback_window=GLOBAL_CONFIG['LOOKBACK_WINDOW'],
            forecast_window=GLOBAL_CONFIG['FORECAST_WINDOW'],
            batch_size=1,
            predictor=QuantumNeuralNetwork,
            num_weights=60,
            encoding='amplitude',
            input_transformation=radial_to_linear,
            rotation_axes='y',
            short_selling=GLOBAL_CONFIG['SHORT_SELLING'],
            reduce_negatives=GLOBAL_CONFIG['CLAMP_NEGATIVES'],
            verbose=GLOBAL_CONFIG['VERBOSE'],
            seed=GLOBAL_CONFIG['SEED'],
        )
        model.train(
            train_data=train_data,
            val_data=val_data,
            actor_lr=0.09488160675184557,
            critic_lr=0.0011635537865649728,
            optimizer=torch.optim.SGD,
            # l1_lambda=1e7,
            l2_lambda=5.030375515009282e-05,
            # weight_decay=1e-7,
            soft_update=True,
            # tau=1e-3,
            risk_preference=-0.12009396389629173,
            gamma=0.0012179475752639956,
            num_epochs=50,
            early_stopping=False,
            patience=10,
        )
        results = model.evaluate(
            test_data=test_data,
            dpo=GLOBAL_CONFIG['DYNAMIC_PO'],
        )
        cv_results.append(results)

    results_logger.log('Execution time: ' + format_timespan(time.time() - start_time) + '\n',
        console=False)
    results_logger.log(print_results(cv_results, dpo=GLOBAL_CONFIG['DYNAMIC_PO']))
