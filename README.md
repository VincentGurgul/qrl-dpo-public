# Quantum Reinforcement Learning for Portfolio Optimization

## Overview

This project implements **Quantum Reinforcement Learning (QRL)** for Portfolio Optimization using **Variational Quantum Circuits (VQCs)** for Quantum Neural Networks (QNNs) in Python 3.12.6. We leverage quantum computation to enhance the performance and speed up reinforcement learning tasks in dynamic financial applications.

The results of this project are published in the following paper: [Add link]. This publication details the methodologies, experiments, and findings achieved using this codebase.


## Features

- Implementation of **Variational Quantum Circuits** for Reinforcement Learning.
- Classical implementations: Q-Learning, DDPG, and simple baselines.
- Utilities for data processing, training, evaluation, and plotting results.
- Highly modular and extensible codebase for experimentation and research.


## Requirements

### Software
- Python 3.12.6

### Libraries
- The package is built around `pennylane` and `torch`.
- For a complete list of requirements, see `requirements.txt`.


## Installation

1. Clone the repository:
```bash
git clone https://github.com/VincentGurgul/qrl-dpo-public
```

2. Navigate to the project directory:
```bash
cd qrl-dpo-public
```

3. Set up a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick start

Run the main entrypoint to start training/evaluation pipelines:

```bash
python MAIN.py
```


## Repo layout (important files/folders)

- `MAIN.py` — project entrypoint to run all models with fixed hyperparameters.
- `config.py` — global configuration and data settings.
- `models.py` — high-level model/pipeline module.
- `requirements.txt` — dependency list.
- `ddpg/` — DDPG implementation.
- `q_learning/` — Q-Learning/DQN implementation.
- `mvo/` — mean-variance optimization.
- `predictors/` — classical and quantum predictor implementations.
- `utilities/` — data processing, training utilities, logging, metrics, and wrappers.
- `data/` — dataset download and preparation.
- `visualizations/` — notebooks for result plotting.


## Tips

- Adjust dataset paths and training hyperparameters in `MAIN.py` before running.
- If you plan to run quantum experiments, ensure any optional quantum backends (PennyLane plugins) are installed.


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
