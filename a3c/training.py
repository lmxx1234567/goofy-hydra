import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
from torch.utils.data import DataLoader
from torch.distributions import Categorical, Bernoulli
from a3c.models import TrafficScheduler, StateValueCritic
from a3c.dataset import A2CDataSet


def training():
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # define model
    traffic_scheduler = TrafficScheduler(19, 256, 4, 3, 3, 2048).to(dev)
    state_value_critic = StateValueCritic(19, 256, 4, 3, 3, 2048).to(dev)
    shared_state_feature_extractor = state_value_critic.state_feature_extractor

    # load data
    dataset = A2CDataSet("data/DL2-40M40ms-90M90ms_82_bbr_202310.csv")
    dataloader = DataLoader(dataset)

    # define loss function and hyperparameters
    alpha_scheduler = 0.1
    optimizer_scheduler = torch.optim.Adam(traffic_scheduler.parameters(), lr=1e-3)
    optimizer_critic = torch.optim.Adam(state_value_critic.parameters(), lr=1e-3)

    # train loop
    start_episode = 0
    num_episodes = 1000
    reward_expectation = 0

    for episode in range(start_episode, num_episodes):
        episode_loss_scheduler = 0
        episode_loss_critic = 0
        reward_expectation = 0

        for states, next_n_states, reward in dataloader:
            states, next_n_states, reward = (
                states.to(dev),  # (batch_size, seq_len, 19)
                next_n_states.to(dev),  # (batch_size, seq_len, 19)
                reward.to(dev),  # (batch_size, n_td, 1)
            )

            # get actions
            actions: torch.Tensor = traffic_scheduler(
                states
            )  # (batch_size, seq_len,  2)
            action_probs: torch.Tensor = choose_action(
                actions
            )  # (batch_size, seq_len, 2) contains choose result by 0 or 1
            action_probs = (
                actions * action_probs
            )  # (batch_size, seq_len, 2) contains choose result by probability

            # get state value
            values = state_value_critic(states)  # (batch_size, 1)
            next_n_values = state_value_critic(next_n_states)  # (batch_size, 1)

            # calculate loss
            dt = torch.sum(reward - reward_expectation) + next_n_values - values
            reward_expectation = reward_expectation + alpha_scheduler * dt

            episode_loss_scheduler -= action_probs.apply_(
                lambda x: torch.log(x) * dt**2
            ).sum()
            episode_loss_critic += dt**2

            # update loss
            # episode_loss_scheduler += loss_scheduler.item()
            # episode_loss_critic += loss_critic.item()

        # update model
        optimizer_scheduler.zero_grad()
        episode_loss_scheduler.backward()
        optimizer_scheduler.step()
        optimizer_critic.zero_grad()
        episode_loss_critic.backward()
        optimizer_critic.step()

        # save model


def choose_action_vectorized(actions: torch.Tensor) -> torch.Tensor:
    # Initialize a Bernoulli distribution with the actions tensor
    action_distri = Bernoulli(actions)
    action_probs = action_distri.sample()  # (batch_size, seq_len, 2)

    # Check if both elements in the last dimension are zero using torch.all
    zero_conditions = torch.all(action_probs == 0, dim=-1)

    # Apply Categorical distribution where the mask is True
    # Here, we sample and then mask the irrelevant samples
    categorical_samples = Categorical(actions).sample((actions.size(0), actions.size(1)))
    categorical_samples = torch.where(zero_conditions, categorical_samples, action_probs)

    return categorical_samples



if __name__ == "__main__":
    training()
