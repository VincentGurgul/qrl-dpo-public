'''This module runs hyperparameter optimization for the reinforcement learning
portfolio optimization pipelines.
'''
import json
import datetime
import logging
import numpy as np
import pandas as pd
import torch
import wandb
import optuna

from config import tickers
from predictors import NeuralNetwork, QuantumNeuralNetwork
from predictors.input_transformations import radial_to_linear, normalized_arcsin

from models import DDPG, DeepQLearning

from utilities.logger import (
    ResultsLogger,
    activate_operations_logging,
    serialize_callable,
)
from utilities.data_processing import TimeSeriesCrossValidation
from utilities.wrappers import retry


#------------------------------------------------------------------------------
# Config
#------------------------------------------------------------------------------

# Choose which pipelines to run.
PIPELINES = [
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
    'SEED': 20,  # seed for reproducibility
    'DEBUG': False,  # whether to log all events to a file for debugging purposes
    'PROJECT_NAME': 'RLPO+ARIMA 100iter',  # Weights & Biases project name
    'N_TRIALS': 100,  # number of hyperparameter optimization trials
    'TARGET': 'val_dpo_sharpe',  # optimization target for hyperparameter tuning
}


#------------------------------------------------------------------------------
# Initialize Operations and Results Loggers
#------------------------------------------------------------------------------

TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
results_logger = ResultsLogger(f'./hyperparam_opt_logs/{TIME}.log')
        
if GLOBAL_CONFIG['DEBUG']:
    activate_operations_logging(level=logging.DEBUG)

results_logger.log(f'''
-------------------------------------------------------------------------------

                        HYPERPARAMETER OPTIMIZATION LOG

-------------------------------------------------------------------------------

Tickers: {tickers}

Config:
{(json.dumps(GLOBAL_CONFIG, default=serialize_callable, indent=4)
    .replace('{\n', '')
    .replace('\n}', '')
    .replace('"', "'")
)}
''')


#------------------------------------------------------------------------------
# Initialize Data and Crossvalidation
#------------------------------------------------------------------------------

# load price data
price_data = pd.read_parquet('./data/price_data.parquet.gzip')

# initialize crossvalidation
crossvalidation = TimeSeriesCrossValidation(
    data=price_data,
    n_splits=GLOBAL_CONFIG['CV_SPLITS'])


#------------------------------------------------------------------------------
# Run Portfolio Optimization Pipelines and Log Optimal Hyperparameters
#------------------------------------------------------------------------------

@retry(10, 100)
def wandb_init(group, config=None, reinit=True, **kwargs):
    '''Initialize Weights & Biases run.'''
    wandb.init(
        project=GLOBAL_CONFIG['PROJECT_NAME'],
        config=config,
        group=group,
        reinit=reinit,
        **kwargs)
    

def log_results(log, results):
    '''Log results to Weights & Biases and the results logger.'''
    log.append(results)
    wandb.log(data=results)
    wandb.finish()
    return log


for pipeline in PIPELINES:

    if pipeline == 'DDPG':
        approach = DDPG
        approach_name = f'Deep Deterministic Policy Gradient ({pipeline})'
    elif pipeline == 'QDPG':
        approach = DDPG
        approach_name = f'Quantum Deterministic Policy Gradient ({pipeline})'
    elif pipeline in ['Deep Q-Learning', 'Quantum Q-Learning']:
        approach = DeepQLearning
        approach_name = pipeline
    
    results_logger.log(f'''
-------------------------------------------------------------------------------
{approach_name}
-------------------------------------------------------------------------------
''')
    
    test_results_log = []
    
    def objective(trial):
        '''Objective function for hyperparameter optimization.'''

        model_config = {
            'actor_lr': trial.suggest_float('actor_lr', 1e-4, 1e-1, log=True),
            'critic_lr': trial.suggest_float('critic_lr', 1e-4, 1e-1, log=True),
            'l2_lambda': trial.suggest_float('l2_lambda', 1e-6, 1e-1, log=True),
            'soft_update': trial.suggest_categorical('soft_update', [True, False]),
            'risk_preference': trial.suggest_float('risk_preference', -1, -1e-2, log=False),
            'gamma': trial.suggest_float('gamma', 1e-3, 1e-1, log=True),
            'optimizer': trial.suggest_categorical('optimizer', [torch.optim.Adam, torch.optim.SGD]),
        }

        # Set up the predictor based on the pipeline.
        if pipeline in ['DDPG', 'Deep Q-Learning']:
            predictor_kwargs = {
                'predictor': NeuralNetwork,
                'hidden_sizes': (),
            }
        elif pipeline in ['QDPG', 'Quantum Q-Learning']:
            # model_config['encoding'] = trial.suggest_categorical(
            #     'encoding', ['angle', 'amplitude'])
            predictor_kwargs = {
                'predictor': QuantumNeuralNetwork,
                # 'rotation_axes': 'y',
                # 'encoding': model_config['encoding'],
                # 'input_transformation': radial_to_linear if model_config['encoding'] == 'amplitude' else None,
                'encoding': 'amplitude',
                'input_transformation': radial_to_linear,
            }

        log_config = {
            'trial_number': trial.number,
            **GLOBAL_CONFIG,
            **model_config,
        }
        wandb_init(
            group=pipeline,
            config=log_config,
        )

        cv_val_results = []
        cv_test_results = []
        for train_index, test_index in crossvalidation():

            # Split the data into training, validation, and test sets.
            val_split = int(len(price_data.iloc[train_index]) * 0.8)
            train_data = price_data.iloc[train_index][:val_split]
            val_data = price_data.iloc[train_index][val_split:]
            test_data = price_data.iloc[test_index]

            # Initialize and train the model.
            model = approach(
                lookback_window=GLOBAL_CONFIG['LOOKBACK_WINDOW'],
                forecast_window=GLOBAL_CONFIG['FORECAST_WINDOW'],
                short_selling=GLOBAL_CONFIG['SHORT_SELLING'],
                reduce_negatives=GLOBAL_CONFIG['CLAMP_NEGATIVES'],
                verbose=GLOBAL_CONFIG['VERBOSE'],
                seed=GLOBAL_CONFIG['SEED'],
                **predictor_kwargs,
            )
            model.train(
                train_data=train_data,
                val_data=val_data,
                actor_lr=model_config['actor_lr'],
                critic_lr=model_config['critic_lr'],
                optimizer=model_config['optimizer'],
                l2_lambda=model_config['l2_lambda'],
                soft_update=model_config['soft_update'],
                risk_preference=model_config['risk_preference'],
                gamma=model_config['gamma'],
                num_epochs=50,
                early_stopping=True,
                patience=10,
            )
            val_results = model.evaluate(
                test_data=val_data,
                dpo=True,
            )
            test_results = model.evaluate(
                test_data=test_data,
                dpo=True,
            )
            cv_val_results.append(val_results)
            cv_test_results.append(test_results)

        # Log the results for this trial.
        val_results = np.mean(cv_val_results, axis=0)
        test_results = np.mean(cv_test_results, axis=0)
        results = {
            'val_spo_profit': val_results[0][0],
            'val_spo_sharpe': val_results[0][1],
            'val_dpo_profit': val_results[1][0],
            'val_dpo_sharpe': val_results[1][1],
            'test_spo_profit': test_results[0][0],
            'test_spo_sharpe': test_results[0][1],
            'test_dpo_profit': test_results[1][0],
            'test_dpo_sharpe': test_results[1][1],
        }
        log_results(
            log=test_results_log,
            results=results,
        )

        # Return the target value for optimization.
        return results[GLOBAL_CONFIG['TARGET']]
    
    # Initialize Optuna study for hyperparameter optimization.
    study = optuna.create_study(
        direction='maximize',
        study_name=pipeline,
    )
    study.optimize(objective, n_trials=GLOBAL_CONFIG['N_TRIALS'])

    # Log the results of the study to W&B in the final iteration.
    try:
        param_importances = optuna.visualization.plot_param_importances(study)
        optimization_history = optuna.visualization.plot_optimization_history(study)
        wandb_init(group=pipeline)
        wandb.log({
            'param_importances': param_importances,
            'optimization_history': optimization_history,
        })
        wandb.finish()
    except:
        pass

    # Log the best hyperparameters and results to local log file.
    best_params = (json.dumps(study.best_params, default=serialize_callable, indent=4)
        .replace('{\n', '')
        .replace('\n}', '')
        .replace('"', "'")
    )

    test_results_log = pd.DataFrame(test_results_log).sort_values(
        by=GLOBAL_CONFIG['TARGET'], ascending=False)
    best_results = test_results_log.iloc[0]
    top_10_average_results = test_results_log.iloc[:10].mean()

    results_logger.log(f'''Optimal hyperparameters:
{best_params}

Optimal validation values:
Validation Sharpe Ratio (SPO) {best_results['val_spo_sharpe']:.4f} 
Validation Profit p.a. (SPO): {best_results['val_spo_profit']*100:.4f} %

Validation Sharpe Ratio (DPO): {best_results['val_dpo_sharpe']:.4f}
Validation Profit p.a. (DPO): {best_results['val_dpo_profit']*100:.4f} %

Test results of best configuration:
Test Sharpe Ratio (SPO): {best_results['test_spo_sharpe']:.4f}
Test Profit p.a. (SPO): {best_results['test_spo_profit']*100:.4f} %

Test Sharpe Ratio (DPO): {best_results['test_dpo_sharpe']:.4f}
Test Profit p.a. (DPO): {best_results['test_dpo_profit']*100:.4f} %

Average test results of top 10 configurations:
Test Sharpe Ratio (SPO): {top_10_average_results['test_spo_sharpe']:.4f}
Test Profit p.a. (SPO): {top_10_average_results['test_spo_profit']*100:.4f} %

Test Sharpe Ratio (DPO): {top_10_average_results['test_dpo_sharpe']:.4f}
Test Profit p.a. (DPO): {top_10_average_results['test_dpo_profit']*100:.4f} %
''')
