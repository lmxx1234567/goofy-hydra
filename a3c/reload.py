import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
from torch.utils.data import DataLoader
from torch.distributions import Categorical, Bernoulli
from a3c.models import TrafficScheduler, StateValueCritic
from a3c.dataset import A2CDataSet

# # m_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def reload_model(reload_output_dir, m_device, epsiode):
    # Test the model
    trafficScheduler = TrafficScheduler(19, 256, 4, 3, 3, 2048).to(m_device)
    src = torch.rand(1, 100, 19).to(m_device)

    pt_file = f"/data/qwx/goofy-hydra/a3c/saved_models/trafficScheduler/traffic_scheduler_model_traced_{epsiode}.pt"
    trafficScheduler.load_state_dict(torch.load(pt_file))

    trafficScheduler.eval()
    # trace
    x = torch.ones(1,100,19).to(m_device)
    traced_module = torch.jit.trace(trafficScheduler, x)
    # print(traced_module.code)
    rept_file_path = pt_file.split("/")[-1].split(".")[0] + "_rept.pt"
    traced_module.save(reload_output_dir+rept_file_path)
    print(rept_file_path)
    return rept_file_path
    