import cv2 as cv
import torch

def test():
    img = cv2.imread('./distortion/test_img.jpg')
    img = torch.tensor(img).cuda()
    print(img.shape)
    h, w, c = img.shape
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)