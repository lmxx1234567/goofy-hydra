import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
from torch.utils.data import DataLoader
from torch.distributions import Categorical, Bernoulli
from a3c.models import TrafficScheduler, StateValueCritic
from a3c.dataset import A2CDataSet
from a3c.load import load_checkpoint
from a3c.reload import reload_model


def training(epsisode_input, dataset_path="data/DL2-40M40ms-90M90ms_82_bbr_202310.csv"):
    print(epsisode_input, ",", dataset_path)
    epsisode_last= (str)((int)(epsisode_input)-1)
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # define model
    traffic_scheduler = TrafficScheduler(19, 256, 4, 3, 3, 2048).to(dev)
    state_value_critic = StateValueCritic(19, 256, 4, 3, 3, 2048).to(dev)
    shared_state_feature_extractor = state_value_critic.state_feature_extractor

    # load model
    # 模型文件的路径
    scheduler_model_path = f'/data/qwx/goofy-hydra/a3c/saved_models/trafficScheduler/traffic_scheduler_model_traced_{epsisode_last}.pt'
    critic_model_path = f'/data/qwx/goofy-hydra/a3c/saved_models/networkValue/state_value_critic_model_traced_{epsisode_last}.pt'

    # 检查 traffic_scheduler_model.pth 是否存在
    if os.path.exists(scheduler_model_path):
        # 如果文件存在，加载模型参数
        traffic_scheduler.load_state_dict(torch.load(scheduler_model_path))
        print("traffic_scheduler 模型已载入")
    else:
        print("traffic_scheduler 模型文件不存在，未载入模型")

    # 检查 state_value_critic_model.pth 是否存在
    if os.path.exists(critic_model_path):
        # 如果文件存在，加载模型参数
        state_value_critic.load_state_dict(torch.load(critic_model_path))
        print("state_value_critic 模型已载入")
    else:
        print("state_value_critic 模型文件不存在，未载入模型")

    # load data
    dataset = A2CDataSet(dataset_path)
    dataloader = DataLoader(dataset)

    # define loss function and hyperparameters
    alpha_scheduler = 0.1
    optimizer_scheduler = torch.optim.Adam(traffic_scheduler.parameters(), lr=1e-3)
    optimizer_critic = torch.optim.Adam(state_value_critic.parameters(), lr=1e-3)

    # train loop
    traffic_scheduler.train()
    state_value_critic.train()
    start_episode = (int)(epsisode_input)
    num_episodes = start_episode + 1
    reward_expectation = 0
    for episode in range(start_episode, num_episodes):
        for states, next_n_states, reward in dataloader:
            episode_loss_scheduler = 0
            episode_loss_critic = 0
            reward_expectation = 0
            states, next_n_states, reward = (
                states.to(dev),  # (batch_size, seq_len, 19)
                next_n_states.to(dev),  # (batch_size, seq_len, 19)
                reward.to(dev),  # (batch_size, n_td, 1)
            )

            # get actions
            actions: torch.Tensor = traffic_scheduler(
                states
            )  # (batch_size, seq_len,  2)
            action_probs: torch.Tensor = choose_action_vectorized(
                actions.squeeze(0)
            )  # (batch_size, seq_len, 2) contains choose result by 0 or 1
            action_probs = action_probs.unsqueeze(0)
            action_probs = (
                actions * action_probs
            )  # (batch_size, seq_len, 2) contains choose result by probability

            # get state value
            values = state_value_critic(states)  # (batch_size, 1)
            next_n_values = state_value_critic(next_n_states)  # (batch_size, 1)

            # calculate loss
            dt = torch.sum(reward - reward_expectation) + next_n_values - values
            reward_expectation = reward_expectation + alpha_scheduler * dt

            log_action_probs = torch.log(action_probs[action_probs != 0]).sum()
            episode_loss_scheduler = -log_action_probs * dt**2

            episode_loss_critic = dt**2
            print("episode_loss_scheduler:",episode_loss_scheduler.item(),"\t episode_loss_critic:", episode_loss_critic.item())

            # 检查是否满足保存模型的条件
            if episode_loss_scheduler < 0.01 and episode_loss_critic < 0.01:
                src = torch.rand(1, 100, 19).to(dev)
                traffic_scheduler.eval()
                traced_script_module = torch.jit.trace(traffic_scheduler, src)
                torch.save(
                    traffic_scheduler.state_dict(), "/data/qwx/goofy-hydra/a3c/saved_models/trafficScheduler/traffic_scheduler_model_convergence.pt"
                )
                # save state_value_critic model
                src = torch.rand(1, 100, 19).to(dev)
                state_value_critic.eval()
                traced_script_module = torch.jit.trace(state_value_critic, src)
                torch.save(
                    state_value_critic.state_dict(), "/data/qwx/goofy-hydra/a3c/saved_models/networkValue/state_value_critic_model_convergence.pt"
                )

                print("模型已保存，因为 episode_loss_scheduler and episode_loss_critic < 0.01")

                traffic_scheduler.train()
                state_value_critic.train()

                
            # update loss
            # episode_loss_scheduler += loss_scheduler.item()
            # episode_loss_critic += loss_critic.item()

            # update model
            optimizer_scheduler.zero_grad()
            episode_loss_scheduler.backward(retain_graph = True)
            optimizer_scheduler.step()

            optimizer_critic.zero_grad()
            episode_loss_critic.backward()
            optimizer_critic.step()

    # save model
    src = torch.rand(1, 100, 19).to(dev)
    traffic_scheduler.eval()
    traced_script_module = torch.jit.trace(traffic_scheduler, src)
    torch.save(
        traffic_scheduler.state_dict(), f"/data/qwx/goofy-hydra/a3c/saved_models/trafficScheduler/traffic_scheduler_model_traced_{epsisode_input}.pt"
    )
    # save state_value_critic model
    state_value_critic.eval()
    traced_script_module = torch.jit.trace(state_value_critic, src)
    torch.save(
        state_value_critic.state_dict(), f"/data/qwx/goofy-hydra/a3c/saved_models/networkValue/state_value_critic_model_traced_{epsisode_input}.pt"
    )

def choose_action_vectorized(actions: torch.Tensor) -> torch.Tensor:
    # Initialize a Bernoulli distribution with the actions tensor
    action_distri = Bernoulli(actions)
    action_probs = action_distri.sample()  # (batch_size, seq_len, 2)

    # Check if both elements in the last dimension are zero using torch.all
    zero_conditions = torch.all(action_probs == 0, dim=-1)

    # Apply Categorical distribution where the mask is True
    # Here, we sample and then mask the irrelevant samples
    # categorical_samples = torch.where(zero_conditions, categorical_samples, action_probs)
    
    # Find the indices where zero_conditions is True
    zero_indices = torch.nonzero(zero_conditions).squeeze()
    is_empty = zero_indices.numel() == 0
    if is_empty:
        return action_probs
    
    # Ensure the tensor is in a list-like format for iteration
    if zero_indices.ndim == 0:
        zero_indices = torch.tensor([zero_indices])
    
    for i in zero_indices:
        twice_action_probs = action_distri.sample()  # (batch_size, seq_len, 2)
        action_probs[i] = twice_action_probs[i]
    
    # Find the indices where zero_conditions is True
    zero_conditions = torch.all(action_probs == 0, dim=-1)
    zero_indices = torch.nonzero(zero_conditions).squeeze()
    is_empty = zero_indices.numel() == 0
    if is_empty:
        return action_probs
    
    # Ensure the tensor is in a list-like format for iteration
    if zero_indices.ndim == 0:
        zero_indices = torch.tensor([zero_indices])

    categorical_samples = Categorical(actions).sample()
    for i in zero_indices:
        x = categorical_samples[i].item()  # Get the value of tensor2 at index i
        action_probs[i][x] = torch.tensor(1)

    return action_probs

def test_action():
    action = torch.tensor([[1.0000, 1.0000],
                           [0.500, 0.0100],
                           [0.0100, 0.0100]])
    action_probs = choose_action_vectorized(action)
    print(action_probs)

if __name__ == "__main__":
    epsiode = sys.argv[1]
    dataset_path = sys.argv[2]

    training(epsiode, dataset_path)

    serv_dev = torch.device("cpu")
    rept_file_path = reload_model("/data/qwx/goofy-hydra/Croc_smr_20231013_all_forRL/trafficScheduler/", serv_dev, epsiode)
