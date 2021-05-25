import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .models.model import get_model
from .data import get_dataset
from .utils import MODEL_PATH, model_class


EPSILON = 0.05
EPSILON_S = 0.03


class Tester():
    pass

    def __init__(self, model_name, test_params, use_cuda=True, net_type="lenet",):

        self.net_type = net_type
        self.dataset = get_dataset(test=True)[0]
        self.test_params = test_params
        self.data_loader = DataLoader(
            dataset=self.dataset,
            num_workers=self.test_params["threads"],
            batch_size=self.test_params["batch_size"],
            shuffle=False
        )
        self.net = get_model(net_type=self.net_type)
        self.epoch = epoch
        self.net_path = "{0}/{1}/{2}.pth".format(
            MODEL_PATH, model_class(net_type), model_name
        )
        self.net.load_state_dict(torch.load(self.net_path))
        self.use_cuda = use_cuda
        pass

    def test(self):
        total_count = 0
        total_hit_num = 0
        fault_count = 0
        fault_hit_num = 0

        total_hit_num_S = 0
        fault_hit_num_S = 0

        with torch.no_grad():
            for data in self.data_loader:
                x, y_label = data
                y_label = torch.unsqueeze(y_label, dim=1)
                if self.use_cuda:
                    x = x.cuda()
                y_pred = self.R(x)
                if self.use_cuda:
                    y_pred = y_pred.cpu()

                delta_y = torch.abs(y_label - y_pred)
                flag_hit = delta_y < EPSILON
                flag_hit_S = delta_y < EPSILON_S
                flag_fault = y_label < 0.6

                total_count += y_label.shape[0]
                fault_count += torch.sum(flag_fault).item()
                total_hit_num += torch.sum(flag_hit).item()
                fault_hit_num += torch.sum(flag_fault * flag_hit).item()
                total_hit_num_S += torch.sum(flag_hit_S).item()
                fault_hit_num_S += torch.sum(flag_fault * flag_hit_S).item()

        return[self.epoch,
               total_hit_num / total_count,
               total_hit_num_S / total_count,
               fault_hit_num / fault_count,
               fault_hit_num_S / fault_count]
