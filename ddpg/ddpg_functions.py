'''Functions for training and evaluating Deep Deterministic Policy Gradient 
Reinforcement Learning for portfolio optimization.
'''
import pandas as pd
import torch
import torch.nn as nn
import predictors

from copy import deepcopy

from utilities.data_processing import RLDataLoader
from utilities.metrics import RLEvaluator
from utilities.model_training import EarlyStopper, ReplayBuffer, set_seeds


class DDPG:
    '''Meta-class for Deep Determinitic Policy Gradient Reinforcement Learning.

    Args:
        lookback_window (int): The size of the lookback window for input data.
        predictor (predictors): The predictor class to use for the model.
        batch_size (int, optional): The number of samples per batch. Defaults
            to 1.
        short_selling (bool, optional): Whether to allow short selling, i.e.
            negative portfolio weights in the model. Defaults to False.
        forecast_window (int, optional): The size of the forecast window for
            input data. Defaults to 0.
        reduce_negatives (bool, optional): Whether to clamp negative portfolio
            weights to -100 %. Defaults to False.
        verbose (int, optional): The verbosity level for logging and outputs.
            Defaults to 1.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        **kwargs: Keyword arguments to be passed to the predictor at init.
    '''
    def __init__(
        self,
        lookback_window: int,
        predictor: predictors,
        batch_size: int = 1,
        short_selling: bool = False,
        forecast_window: int = 0,
        reduce_negatives: bool = False,
        verbose: int = 1,
        seed: int = 42,
        **kwargs,
    ):
        self.lookback_window = lookback_window
        self.batch_size = batch_size
        self.predictor = predictor
        self.predictor_kwargs = kwargs
        self.short_selling = short_selling
        self.forecast_window = forecast_window
        self.reduce_negatives = reduce_negatives
        self.verbose = verbose
        self.seed = seed

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        actor_lr: float = 0.05,
        critic_lr: float = 0.01,
        optimizer: torch.optim = torch.optim.Adam,
        l1_lambda: float = 0,
        l2_lambda: float = 0,
        soft_update: bool = False,
        tau: float = 0.005,
        risk_preference: float = -0.5,
        weight_decay: float = 0,
        gamma: float = 1.0,
        num_epochs: int = 50,
        early_stopping: bool = True,
        patience: int = 2,
        min_delta: float = 0,
    ):
        '''Trains the DDPG model.

        Args:
            train_data (pd.DataFrame): Training dataset.
            val_data (pd.DataFrame): Validation dataset.
            actor_lr (float): Learning rate for the actor model.
            critic_lr (float): Learning rate for the critic model.
            optimizer (torch.optim): Optimizer class for updating model weights.
            l1_lambda (float): L1 regularization parameter for the actor model.
            l2_lambda (float): L2 regularization parameter for the actor model.
            weight_decay (float): Regularization parameter for weight decay.
            soft_update (bool): Whether to use soft updates for target networks.
            tau (float): Soft update factor for target networks.
            risk_preference (float): Risk preference factor for the reward function.
            gamma (float): Discount factor for future rewards.
            num_epochs (int): Number of training epochs.
            patience (int): Early stopping patience.
            min_delta (float, optional): Minimum change in loss for early stopping.

        Returns:
            Actor_Critic_RL_Model: The trained Actor-Critic RL model instance.
        '''
        self.val_data = val_data
        dataloader = RLDataLoader(train_data, val_data, shuffle=False)

        # set data-related hyperparameters
        self.number_of_assets = dataloader.number_of_assets
        number_of_datapoints = self.lookback_window + self.forecast_window
        self.input_size = self.number_of_assets * number_of_datapoints
        self.output_size = self.number_of_assets

        # build dataloaders
        train_loader, val_loader = dataloader(
            batch_size=self.batch_size,
            window_size=self.lookback_window,
            forecast_size=self.forecast_window,
        )

        # initialize models
        if self.short_selling:
            activation = lambda x: x / torch.sum(x, dim=-1, keepdim=True)
        else:
            activation = nn.Softmax(dim=-1)
        self.actor = self.predictor(
            input_size=self.input_size,
            output_size=self.output_size,
            output_activation=activation,
            **self.predictor_kwargs,
            seed=self.seed,
        )
        critic = self.predictor(
            input_size=self.input_size + self.output_size,
            output_size=1,
            **self.predictor_kwargs,
            seed=self.seed,
        )

        # run training loop
        trainer = DDPGTrainer(
            number_of_assets=self.number_of_assets,
            actor=self.actor,
            critic=critic,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            optimizer=optimizer,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
            weight_decay=weight_decay,
            soft_update=soft_update,
            tau=tau,
            risk_preference=risk_preference,
            gamma=gamma,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
        )
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            verbose=self.verbose,
            num_epochs=num_epochs,
        )

        return self

    def evaluate(self, test_data: pd.DataFrame, dpo: bool = True) -> tuple:
        '''Evaluates the DDPG model.

        Args:
            test_data (pd.DataFrame): Test dataset.
            dpo (bool, optional): Whether to evaluate the Dynamic Portfolio
                Optimization (DPO) strategy. Defaults to True.

        Returns:
            tuple: ((SPO profit, SPO sharpe ratio),
                    (DPO profit, DPO sharpe ratio))
                    if dpo is True, else (SPO profit, SPO sharpe ratio)
        '''
        evaluator = RLEvaluator(
            actor=self.actor,
            train_data=self.val_data,
            test_data=test_data,
            forecast_size=self.forecast_window,
            reduce_negatives=self.reduce_negatives,
        )
        spo_results = evaluator.evaluate_spo(verbose=self.verbose)
        if dpo:
            dpo_results = evaluator.evaluate_dpo(interval=self.lookback_window,
                                                verbose=self.verbose)
            return spo_results, dpo_results
        else:
            return spo_results


class DDPGTrainer:
    '''Facilitates the training of a DDPG pipeline for financial portfolio
    optimization, focusing on optimizing the actor network (decision-making)
    and the critic network (evaluation) using provided optimizers, loss
    functions, and hyperparameters. It supports early stopping to prevent
    overfitting based on validation performance.

    Args:
        number_of_assets (int): The number of assets in the portfolio, used for
            constructing the actor and critic models appropriately.
        actor (nn.Module): The actor network responsible for making portfolio
            allocation decisions.
        critic (nn.Module): The critic network responsible for evaluating the
            actor's decisions.
        optimizer (torch.optim, optional): Optimizer class (e.g., torch.optim.Adam
            or torch.optim.SGD) used to optimize both actor and critic networks.
            Defaults to torch.optim.Adam.
        weight_decay (float, optional): L2 regularization factor applied during
            optimization. Defaults to 0 (no weight decay).
        l1_lambda (float, optional): L1 regularization factor applied to the
            actor network. Defaults to 0 (no L1 regularization).
        l2_lambda (float, optional): L2 regularization factor applied to the
            actor network. Defaults to 0 (no L2 regularization).
        soft_update (bool, optional): Whether to use soft updates with target
            networks.
        tau (float, optional): Soft update factor for target networks. Defaults
            to 0.005.
        risk_preference (float, optional): Risk preference factor for the reward
            function. Negative value results in volatility lowering the reward.
            Defaults to -0.5.
        gamma (float, optional): Discount factor for future rewards in
            reinforcement learning. Defaults to 1.0.
        actor_lr (float, optional): Learning rate for the actor optimizer.
            Defaults to 0.05.
        critic_lr (float, optional): Learning rate for the critic optimizer.
            Defaults to 0.01.
        early_stopping (bool, optional): Whether to use early stopping based on
            validation performance. Defaults to True.
        patience (int, optional): Number of epochs to wait for improvement in
            validation performance before stopping training. Used for early
            stopping. Defaults to 2.
        min_delta (float, optional): Minimum change in validation performance to
            be considered as an improvement. Defaults to 0.
    '''
    def __init__(
        self,
        number_of_assets: int,
        actor: nn.Module,
        critic: nn.Module,
        actor_lr: float = 0.05,
        critic_lr: float = 0.01,
        optimizer: torch.optim = torch.optim.Adam,
        l1_lambda: float = 0,
        l2_lambda: float = 0,
        soft_update: bool = False,
        tau: float = 0.005,
        risk_preference: float = -0.5,
        weight_decay: float = 0,
        gamma: float = 1.0,
        early_stopping: bool = True,
        patience: int = 2,
        min_delta: float = 0,
    ):
        self.number_of_assets = number_of_assets
        self.actor = actor
        self.critic = critic
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.soft_update = soft_update
        self.tau = tau
        self.risk_preference = risk_preference
        self.gamma = gamma
        self.early_stopper = EarlyStopper(patience, min_delta) if early_stopping else None

        self.actor_optimizer = optimizer(
            actor.parameters(),
            lr=actor_lr,
            # weight_decay=weight_decay,  # weight decay optional for actor
        )
        self.critic_optimizer = optimizer(
            critic.parameters(),
            lr=critic_lr,
            weight_decay=weight_decay,
        )
    
        if soft_update:
            # Initialize update factor and target networks
            self.target_actor = deepcopy(actor)
            self.target_critic = deepcopy(critic)
            # Synchronize target networks with main networks
            self._soft_update(self.target_actor, self.actor, tau=1.0)
            self._soft_update(self.target_critic, self.critic, tau=1.0)

    def _soft_update(self, target, source, tau):
        '''Soft-update target network parameters.'''
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * source_param.data)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        noise: float = 0.2,
        verbose: int = 1,
    ):
        '''Training loop for the DDPG pipeline.
        
        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader providing
                training data in batches.
            val_loader (torch.utils.data.DataLoader, optional): DataLoader
                providing validation data in batches. Used for early stopping
                and performance monitoring. Defaults to None.
            num_epochs (int, optional): Number of training epochs to run.
                Defaults to 100.
            noise (float, optional): Standard deviation of Gaussian noise added
                to the actor's portfolio allocation. Defaults to 0.2.
            verbose (int, optional): Verbosity level for printing training
                details, can be 0, 1, or 2. Defaults to 1.
        '''
        replay_buffer = ReplayBuffer()
        
        for epoch in range(num_epochs):
            total_actor_loss = 0
            total_critic_loss = 0

            for state, next_state in train_loader:

                # Compute current portfolio allocation and Q-value
                portfolio_allocation = self.actor(state.flatten())
                exploration_noise = torch.normal(0, noise, portfolio_allocation.shape)
                noisy_portfolio_allocation = portfolio_allocation + exploration_noise

                # Set target value = average profit + risk preference * volatility
                avg_profit = torch.mean(
                    torch.sum(state.view(-1, self.number_of_assets) * noisy_portfolio_allocation,
                              dim=-1)
                ).detach().cpu()
                volatility = torch.std(
                    torch.sum(state.view(-1, self.number_of_assets) * noisy_portfolio_allocation,
                              dim=-1),
                    correction=0, # maximum likelihood estimation
                ).detach().cpu()
                reward = avg_profit + self.risk_preference * volatility

                # Store transition in replay buffer
                replay_buffer.push((
                    state.detach(),
                    noisy_portfolio_allocation.detach(),
                    reward.detach(),
                    next_state.detach()))

                # Sample transition from replay buffer
                transition = replay_buffer.sample(1)
                state = transition[0][0]
                noisy_portfolio_allocation = transition[0][1]
                reward = transition[0][2]
                next_state = transition[0][3]

                portfolio_allocation = self.actor(state.flatten())

                # Use target networks for next state action and Q-value if soft
                # updates are enabled, else use regular ones
                if self.soft_update:
                    next_portfolio_allocation = self.target_actor(next_state.flatten())
                    next_q_value = self.target_critic(
                        torch.cat((next_state.flatten(),
                                next_portfolio_allocation.flatten()))
                    )
                else:
                    next_portfolio_allocation = self.actor(next_state.flatten())
                    next_q_value = self.critic(
                        torch.cat((next_state.flatten(),
                                   next_portfolio_allocation.flatten()))
                    )

                # Calculate target Q-value according to update function
                target_q_value = reward + self.gamma * next_q_value

                # Critic loss and backpropagation
                q_value = self.critic(
                    torch.cat((state.flatten(),
                               noisy_portfolio_allocation.flatten()))
                )
                critic_loss = (target_q_value - q_value).pow(2)
                self.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optimizer.step()

                # Actor evaluation
                critic_input = torch.cat(
                    (state.flatten(), portfolio_allocation.flatten()))
                actor_loss = -self.critic(critic_input)

                # Add L1/L2 regularization to actor loss
                l1_actor = sum(weight.abs().sum() for weight in self.actor.parameters())
                l2_actor = sum(weight.pow(2).sum() for weight in self.actor.parameters())
                actor_loss += self.l1_lambda * l1_actor + self.l2_lambda * l2_actor

                # Actor backpropagation
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

            # Average losses
            avg_actor_loss = total_actor_loss / len(train_loader)
            avg_critic_loss = total_critic_loss / len(train_loader)

            # Early stopping
            if self.early_stopper:
                with torch.no_grad():
                    val_critic_loss = 0
                    for state, next_state in val_loader:
                        portfolio_allocation = self.actor(state.flatten())
                        q_value = self.critic(
                            torch.cat((state.flatten(), portfolio_allocation.flatten()))
                        )

                        if self.soft_update:
                            next_portfolio_allocation = self.target_actor(next_state.flatten())
                            next_q_value = self.target_critic(
                                torch.cat((next_state.flatten(), next_portfolio_allocation.flatten()))
                            )
                        else:
                            next_portfolio_allocation = self.actor(next_state.flatten())
                            next_q_value = self.critic(
                                torch.cat((next_state.flatten(), next_portfolio_allocation.flatten()))
                            )

                        avg_profit = torch.mean(
                            torch.sum(state.view(-1, self.number_of_assets) * portfolio_allocation,
                                    dim=-1)
                        ).detach().cpu()
                        volatility = torch.std(
                            torch.sum(state.view(-1, self.number_of_assets) * portfolio_allocation,
                                    dim=-1),
                            correction=0, # maximum likelihood estimation
                        ).detach().cpu()
                        reward = avg_profit + self.risk_preference * volatility

                        target_q_value = reward + self.gamma * next_q_value
                        val_critic_loss += (target_q_value - q_value).pow(2).item()

                    avg_val_critic_loss = val_critic_loss / len(val_loader)

                if verbose > 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Actor Loss: {avg_actor_loss:.10f}, Critic Loss: {avg_critic_loss:.10f}, Val Critic Loss: {avg_val_critic_loss:.10f}')

                if self.early_stopper.early_stop(avg_val_critic_loss, verbose=verbose):
                        break
            else:
                if verbose > 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Actor Loss: {avg_actor_loss:.10f}, Critic Loss: {avg_critic_loss:.10f}')

            # Synchronize target networks
            if self.soft_update:
                self._soft_update(self.target_actor, self.actor, self.tau)
                self._soft_update(self.target_critic, self.critic, self.tau)
