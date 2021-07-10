import torch
import local_att

# 非整倍数补齐: 在python代码里padding
# block_num大于65536情况: 不管，cuda报out of memory，靠减小batch_size或换卡

# TODO:
# 反向传播 

def local_attention_forward_cpu(X:torch.Tensor, w:torch.Tensor, kernel_size:int):
    wA = w[0]
    wB = w[1]
    wC = w[2]
    wD = w[3]
    wU = w[4]
    W = X.shape[2]
    H = X.shape[3]
    kernel_size = w.shape[2]
    assert(W % kernel_size == 0)
    assert(H % kernel_size == 0)
    att = torch.zeros_like(X)
    for i in range(0, W, kernel_size):
        for j in range(0, H, kernel_size):
            x_slice = X[:, :, i:i+kernel_size, j:j+kernel_size]
            x1 = torch.matmul(wA, x_slice)
            x2 = torch.matmul(x_slice, wB)
            x3 = torch.matmul(wC, x_slice.transpose(2,3))
            x4 = torch.matmul(x_slice.transpose(2,3), wD)
            att_slice = x1+x2+x3+x4
            att_slice = att_slice * wU
            att[:, :, i:i+kernel_size, j:j+kernel_size] = att_slice
    return att

def test_forward():
    batch_size = 8
    channels = 16
    x_size = 225
    y_size = 225
    k_zero = torch.zeros([3,3]).float()
    k_ones = torch.ones([3,3]).float()
    wA = torch.randint(-5, 5, [channels,3,3]).float()
    wB = torch.randint(-5, 5, [channels,3,3]).float()
    wC = torch.randint(-5, 5, [channels,3,3]).float()
    wD = torch.randint(-5, 5, [channels,3,3]).float()
    wU = torch.randint(-5, 5, [channels,3,3]).float()

    X = torch.randint(0, 10, [batch_size, channels, x_size, y_size]).float().cuda()
    w = torch.stack([wA, wB, wC, wD, wU], dim=0).cuda()
    print("x shape:", X.shape)
    print("w shape:", w.shape)
    cuda_output = local_att.local_attention_forward_cuda(X, w)
    cpu_output = local_attention_forward_cpu(X, w, 3).cuda()
    print("Equal: ", (cuda_output == cpu_output).all().item())


def local_attention_backward_cpu(grad_output, x, w):
    wA = w[0]
    wB = w[1]
    wC = w[2]
    wD = w[3]
    wU = w[4]
    width = x.shape[2]
    height = x.shape[3]
    kernel_size = w.shape[2]
    assert(width % kernel_size == 0)
    assert(height % kernel_size == 0)

    grad_x = torch.zeros_like(x)
    grad_w = torch.zeros_like(w)
    for i in range(0, width, kernel_size):
        for j in range(0, height, kernel_size):
            x_slice = x[:, :, i:i+kernel_size, j:j+kernel_size]
            grad_output_slice = grad_output[:, :, i:i+kernel_size, j:j+kernel_size]
            grad_x_slice = grad_output_slice * wU * (wA.t()+wB+wC+wD.t())
            grad_U_slice = grad_output_slice / (wU + 1e-6)
            grad_D_slice = grad_output_slice * wU * x_slice
            grad_C_slice = grad_output_slice * wU * x_slice.transpose(2,3)
            grad_B_slice = grad_output_slice * wU * x_slice.transpose(2,3)
            grad_A_slice = grad_output_slice * wU * x_slice
            grad_x[:, :, i:i+kernel_size, j:j+kernel_size] = grad_x_slice
            grad_w[0, :, :, :] += grad_A_slice
            grad_w[1, :, :, :] += grad_B_slice
            grad_w[2, :, :, :] += grad_C_slice
            grad_w[3, :, :, :] += grad_D_slice
            grad_w[4, :, :, :] += grad_U_slice

    return grad_x, grad_w 

def test_backward():
    batch_size = 1
    channels = 1
    x_size = 3
    y_size = 3
    X = torch.randint(0, 10, [batch_size, channels, x_size, y_size]).float().cuda()
    grad_output = torch.ones_like(X)

    k_zero = torch.zeros([3,3]).float()
    k_ones = torch.ones([3,3]).float()
    wA = torch.randint(-5, 5, [channels,3,3]).float()
    wB = torch.randint(-5, 5, [channels,3,3]).float()
    wC = torch.randint(-5, 5, [channels,3,3]).float()
    wD = torch.randint(-5, 5, [channels,3,3]).float()
    wU = torch.randint(-5, 5, [channels,3,3]).float()

    w = torch.stack([wA, wB, wC, wD, wU], dim=0).cuda()
    print("x shape:", X.shape)
    print("w shape:", w.shape)
    cuda_grad_x, cuda_grad_w = local_att.local_attention_backward_cuda(grad_output, X, w)
    cpu_grad_x, cpu_grad_w = local_attention_backward_cpu(grad_output, X, w).cuda()
    print("Equal: ", (cuda_grad_x == cuda_grad_x).all().item(), (cuda_grad_w == cuda_grad_w).all().item())

# test_forward()
test_backward()