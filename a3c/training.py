import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
from torch.distributions import Categorical
from a3c.models import TrafficScheduler, StateValueCritic
from a3c.dataset import A2CDataSet


def training():
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # define model
    traffic_scheduler = TrafficScheduler(19, 256, 4, 3, 3, 2048).to(dev)
    state_value_critic = StateValueCritic(19, 256, 4, 3, 3, 2048).to(dev)
    shared_state_feature_extractor = state_value_critic.state_feature_extractor

    # load data
    batch_size = 32
    dataset = A2CDataSet("data/DL2-40M40ms-90M90ms_82_bbr_202310.csv")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    

    # define loss function
    criterion_critic = torch.nn.MSELoss()
    optimizer_scheduler = torch.optim.Adam(traffic_scheduler.parameters(), lr=1e-3)
    optimizer_critic = torch.optim.Adam(state_value_critic.parameters(), lr=1e-3)

    # train loop
    start_episode = 0
    num_episodes = 1000
    for episode in range(start_episode, num_episodes):
        episode_reward = 0
        episode_loss_scheduler = 0
        episode_loss_critic = 0
        num_steps = 60

        for step in range(num_steps):
             # read states. TODO: read from dataset
            states,next_states = torch.randn(batch_size, 100, 19).to(dev),torch.randn(batch_size, 100, 19).to(dev)

            # get actions
            actions:torch.Tensor = traffic_scheduler(states) # (batch_size, 100, 3)
            actions_dist = Categorical(actions)
            actions = actions_dist.sample() # (batch_size, 100)
            action_probs = actions_dist.log_prob(actions) # (batch_size, 100)

            # get state value
            values = state_value_critic(states) # (batch_size, 1)
            next_values = state_value_critic(next_states) # (batch_size, 1)

            # get reward. TODO: read from dataset
            reward = torch.randn(batch_size, 1).to(dev)
            episode_reward += reward.item()

            # calculate loss
            loss_critic = criterion_critic(values, reward + next_values)
            td = reward + next_values - values # (batch_size, 1) temporal difference
            td = td.detach()
            loss_scheduler = -torch.mean(action_probs * td.unsqueeze(-1))

            # update loss
            episode_loss_scheduler += loss_scheduler.item()
            episode_loss_critic += loss_critic.item()

        # update model
        optimizer_scheduler.zero_grad()
        episode_loss_scheduler.backward()
        optimizer_scheduler.step()
        optimizer_critic.zero_grad()
        episode_loss_critic.backward()
        optimizer_critic.step()

        # save model



if __name__ == "__main__":
    training()
