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

    def __init__(self, model_name, test_params, dataset="mnist", use_cuda=True, net_type="lenet",):

        self.net_type = net_type
        self.dataset = get_dataset(
            name=dataset, net_type=net_type, test=True)[0]
        self.test_params = test_params
        self.data_loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.test_params["batch_size"],
            shuffle=False
        )
        self.net = get_model(net_type=self.net_type, dataset=dataset)
        if use_cuda:
            self.net = self.net.cuda()
        self.net_path = "{0}{1}/{2}/{3}.pth".format(
            MODEL_PATH, dataset, model_class(net_type), model_name
        )
        print("Test model: ", self.net_path)
        self.net.load_state_dict(torch.load(self.net_path))
        self.use_cuda = use_cuda
        pass

    def test(self):
        batch_size = self.test_params["batch_size"]
        use_cuda = self.use_cuda

        total_num = 0
        correct_num = 0
        for i, data in enumerate(self.data_loader):
            with torch.no_grad():
                b_x, b_y = data
                if use_cuda:
                    b_x = b_x.cuda()
                b_predict_y = self.net(b_x)
                if use_cuda:
                    b_predict_y = b_predict_y.cpu()
                b_y = b_y.numpy()
                b_predict_y = b_predict_y.numpy()

                total_num += b_y.shape[0]
                correct_num += self.match_num(b_predict_y, b_y)
            pass
        acc = correct_num / total_num
        print("Test accuracy: ", acc)
        return acc

    def match_num(self, y_pred, y_label):
        y_pred_i = np.argmax(y_pred, axis=1)
        tmp = sum(y_pred_i == y_label)
        return tmp
