import torch
import local_att

# 非整倍数补齐: 在python代码里padding
# block_num大于65536情况: 不管，cuda报out of memory，靠减小batch_size或换卡

# TODO:
# 1. cuda实现反向传播 √
# 2. 样例验证算法实现正确性 √
# 3. 搭建LANet √
# 4. 小数据集上训练，验证反传算法正确性 √
# 5. 和CNN网络比较模型大小、占用空间、推理速度等 √

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
    batch_size = 5
    channels = 7
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
    batch_size = x.shape[0]
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

            # dL/dX
            grad_T_slice = grad_output_slice / (wU+1e-6) # dL/dU
            grad_x_slice = torch.matmul(wA.transpose(1,2), grad_T_slice)
            grad_x_slice += torch.matmul(grad_T_slice, wB.transpose(1,2))
            grad_x_slice += torch.matmul(grad_T_slice.transpose(2,3), wC)
            grad_x_slice += torch.matmul(wD, grad_T_slice.transpose(2,3))
            grad_x[:, :, i:i+kernel_size, j:j+kernel_size] = grad_x_slice

            # A/B/C/D
            grad_w[0:1, :, :, :] += torch.sum(torch.matmul(grad_T_slice, x_slice.transpose(2,3)), dim=0) # dL/dA
            grad_w[1:2, :, :, :] += torch.sum(torch.matmul(x_slice.transpose(2,3), grad_T_slice), dim=0) # dL/dB
            grad_w[2:3, :, :, :] += torch.sum(torch.matmul(grad_T_slice, x_slice), dim=0) # dL/dC
            grad_w[3:4, :, :, :] += torch.sum(torch.matmul(x_slice, grad_T_slice), dim=0) # dL/dD
            
            # dL/dU
            x1 = torch.matmul(wA, x_slice)
            x2 = torch.matmul(x_slice, wB)
            x3 = torch.matmul(wC, x_slice.transpose(2,3))
            x4 = torch.matmul(x_slice.transpose(2,3), wD)
            grad_U_slice = x1+x2+x3+x4
            grad_w[4:5, :, :, :] += torch.sum(grad_U_slice, dim=0)
    
    grad_w = grad_w / batch_size
    grad_x = torch.clamp(grad_x, -1e4, 1e4)
    grad_w = torch.clamp(grad_w, -1e4, 1e4)
    return grad_x, grad_w

def test_backward(USE_RANDOM=True):
    batch_size = 7
    channels = 23
    x_size = 45
    y_size = 45

    if USE_RANDOM:
        X = torch.randint(0, 10, [batch_size, channels, x_size, y_size]).float()  / 10
        grad_output = torch.randint_like(X, 0, 10).float()  / 100
        wA = torch.randint(-5, 5, [channels,3,3]).float() / 10
        wB = torch.randint(-5, 5, [channels,3,3]).float() / 10
        wC = torch.randint(-5, 5, [channels,3,3]).float() / 10
        wD = torch.randint(-5, 5, [channels,3,3]).float() / 10
        wU = torch.randint(-5, 5, [channels,3,3]).float() / 10
    else:
        X = torch.ones([batch_size, channels, x_size, y_size]).float()
        X = torch.tensor([[[[-1,-5,-3]*3, [6,4,8]*3, [7,2,9]*3]*3]]).float()
        KPW = int(x_size / 3)
        grad_output = torch.tensor([[[[1,5,3]*KPW, [6,-4,8]*KPW, [7,2,9]*KPW]*KPW]*channels]*batch_size).float()
        grad_output = torch.ones_like(X).float()
        k_zero = torch.zeros([channels, 3,3]).float()
        k_ones = torch.ones([channels, 3,3]).float()
        wA = wB = wC = wD = wU = k_ones
        wA = torch.tensor([[[2,1,2], [-4,-5,1], [4,1,3]]]*channels).float()
        wB = torch.tensor([[[-2,2,-3], [3,1,1], [1,4,1]]]*channels).float()
        wC = torch.tensor([[[1,-2,-1], [-4,4,2], [-1,1,-1]]]*channels).float()
        wD = torch.tensor([[[2,4,4], [1,4,-4], [-1,1,2]]]*channels).float()
        wU = torch.tensor([[[1,-2,-3], [3,2,1], [-3,3,1]]]*channels).float()

    # print(wA, wB, wC, wD, wU)
    w = torch.stack([wA, wB, wC, wD, wU], dim=0)
    print("x shape:", X.shape)
    print("w shape:", w.shape)
    X = X.cuda()
    w = w.cuda()
    grad_output = grad_output.cuda()
    cuda_grad_x, cuda_grad_w = local_att.local_attention_backward_cuda(grad_output, X, w)
    cpu_grad_x, cpu_grad_w = local_attention_backward_cpu(grad_output, X, w)
    # print(cuda_grad_w[1,-1])
    # print(cpu_grad_w[1,-1])
    
    dx_equal = (torch.abs(cuda_grad_x - cpu_grad_x)/(1e-2+torch.abs(cuda_grad_x))< 1e-2).all().item()
    print("dx Equal: ", dx_equal)
    # if not dx_equal:
    #     print(cuda_grad_x)
    #     print(cpu_grad_x)
    #     print(cuda_grad_x - cpu_grad_x)

    print("dw Equal: ", 
        (torch.abs((cuda_grad_w[0] - cpu_grad_w[0])/cuda_grad_w[0])< 1e-4).all().item(),
        (torch.abs((cuda_grad_w[1] - cpu_grad_w[1])/cuda_grad_w[1])< 1e-4).all().item(),
        (torch.abs((cuda_grad_w[2] - cpu_grad_w[2])/cuda_grad_w[2])< 1e-4).all().item(),
        (torch.abs((cuda_grad_w[3] - cpu_grad_w[3])/cuda_grad_w[3])< 1e-4).all().item(),
        (torch.abs((cuda_grad_w[4] - cpu_grad_w[4])/cuda_grad_w[4])< 1e-4).all().item())
    print("diff:",         
        torch.abs(cuda_grad_x - cpu_grad_x).max().item(),
        torch.abs(cuda_grad_w[0] - cpu_grad_w[0]).max().item(),
        torch.abs(cuda_grad_w[1] - cpu_grad_w[1]).max().item(),
        torch.abs(cuda_grad_w[2] - cpu_grad_w[2]).max().item(),
        torch.abs(cuda_grad_w[3] - cpu_grad_w[3]).max().item(),
        torch.abs(cuda_grad_w[4] - cpu_grad_w[4]).max().item(),)

# test_forward()
test_backward(USE_RANDOM=True)