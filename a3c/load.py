import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
from torch.utils.data import DataLoader
from torch.distributions import Categorical, Bernoulli
from a3c.models import TrafficScheduler, StateValueCritic
from a3c.dataset import A2CDataSet

def load_checkpoint(traffic_scheduler,state_value_critic,dev,epsiode):
    # 模型文件的路径
    scheduler_model_path = f'/data/qwx/goofy-hydra/a3c/saved_models/trafficScheduler/traffic_scheduler_model_traced_{epsiode}.pt'
    critic_model_path = f'/data/qwx/goofy-hydra/a3c/saved_models/networkValue/state_value_critic_model_traced_{epsiode}.pt'

    # 检查 traffic_scheduler_model.pth 是否存在
    if os.path.exists(scheduler_model_path):
        # 如果文件存在，加载模型参数
        traffic_scheduler.load_state_dict(torch.load(scheduler_model_path)).to(dev)
        print("traffic_scheduler 模型已载入")
    else:
        print("traffic_scheduler 模型文件不存在，未载入模型")

    # 检查 state_value_critic_model.pth 是否存在
    if os.path.exists(critic_model_path):
        # 如果文件存在，加载模型参数
        state_value_critic.load_state_dict(torch.load(critic_model_path)).to(dev)
        print("state_value_critic 模型已载入")
    else:
        print("state_value_critic 模型文件不存在，未载入模型")
