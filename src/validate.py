# validate
import torch
import numpy as np
import matplotlib.pyplot as plt


class Validator():
    pass

    def __init__(self, net,
                 hyper_params,
                 validate_params,
                 use_cuda,
                 data_loader,):
        self.net = net
        self.hyper_params = hyper_params
        self.validate_params = validate_params
        self.use_cuda = use_cuda
        self.data_loader = data_loader
        pass

    def validate(self):
        batch_size = self.validate_params["batch_size"]
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
        return correct_num / total_num

    def match_num(self, y_pred, y_label):
        y_pred_i = np.argmax(y_pred, axis=1)
        tmp = sum(y_pred_i == y_label)
        return tmp

    def RMSE(self, y_pred, y_label):
        len_ = y_pred.shape[0]
        return np.sqrt(np.sum((y_pred - y_label)**2) / len_)
