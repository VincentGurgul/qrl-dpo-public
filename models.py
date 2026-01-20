'''Meta-module containing all models used in the project.

Classes:
    EqualWeightsModel: Baseline model that assigns equal weights to all assets.
    MVO_Model: Model that uses Mean-Variance Optimization to allocate assets.
    ActorCriticModel: Model that uses an actor-critic neural network to allocate
        assets.
    Deep_SARSA_Model: Model that uses a deep SARSA neural network to allocate
        assets.
    DeepQLearningModel: Model that uses a deep Q-learning neural network to
        allocate assets.
'''
from naive_baseline import EqualWeights
from mvo.mvo_functions import MeanVarianceOptimization
from actor_critic.actor_critic_functions import DeepActorCritic
from ddpg.ddpg_functions import DDPG
from q_learning.q_learning_functions import DeepQLearning

__all__ = [
    'EqualWeights',
    'MeanVarianceOptimization',
    'DeepActorCritic',
    'DDPG',
    'DeepQLearning',
]
