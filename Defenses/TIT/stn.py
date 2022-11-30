import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def stn(aff, x):
    grid = F.affine_grid(aff, x.size())
    x = F.grid_sample(x, grid)
    return x

def para2aff(theta, dx, dy):
    aff = torch.tensor([[np.cos(theta), np.sin(theta), dx], [-np.sin(theta), np.cos(theta), dy]]).float()
    return aff

def para2aff2(thetas, dxs, dys):
    aff = torch.zeros(len(thetas), 2, 3)
    aff[:, 0, 0] = torch.cos(thetas)
    aff[:, 0, 1] = torch.sin(thetas)
    aff[:, 0 ,2] = dxs
    aff[:, 1, 0] = -torch.sin(thetas)
    aff[:, 1, 1] = torch.cos(thetas)
    aff[:, 1, 2] = dys
    return aff

def gen_rand_aff(max_theta, max_d, n):
    thetas = 2*max_theta*(torch.rand(n)-0.5)
    dxs = 2*max_d*(torch.rand(n)-0.5)
    dys = 2*max_d*(torch.rand(n)-0.5)
    return para2aff2(thetas, dxs, dys)



def CWLoss(logits, target, kappa=0):
    # inputs to the softmax function are called logits.
    # https://arxiv.org/pdf/1608.04644.pdf
    target = torch.ones(logits.size(0)).type(logits.type()).fill_(target)
    target_one_hot = torch.eye(10).type(logits.type())[target.long()]

    # workaround here.
    # subtract large value from target class to find other max value
    # https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
    real = torch.sum(target_one_hot*logits, 1)
    other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)

    return torch.sum(torch.max(other-real, kappa))


class Loss_flow(nn.Module):
    def __init__(self, device, neighbours=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])):
        super(Loss_flow, self).__init__()

        filters = []
        for i in range(neighbours.shape[0]):
            for j in range(neighbours.shape[1]):
                if neighbours[i][j] == 1:
                    filter = np.zeros((1, neighbours.shape[0], neighbours.shape[1]))
                    filter[0][i][j] = -1
                    filter[0][neighbours.shape[0]//2][neighbours.shape[1]//2] = 1
                    filters.append(filter)

        filters = np.array(filters)
        self.filters = torch.from_numpy(filters).float().to(device)

    def forward(self, f):
        # TODO: padding
        '''
        f - f.size() =  [1, h, w, 2]
            f[0, :, :, 0] - u channel
            f[0, :, :, 1] - v channel
        '''
        f_u = f[:, :, :, 0].unsqueeze(1)
        f_v = f[:, :, :, 1].unsqueeze(1)

        diff_u = F.conv2d(f_u, self.filters)[0][0] # don't use squeeze
        diff_u_sq = torch.mul(diff_u, diff_u)

        diff_v = F.conv2d(f_v, self.filters)[0][0] # don't use squeeze
        diff_v_sq = torch.mul(diff_v, diff_v)

        dist = torch.sqrt(torch.sum(diff_u_sq, dim=0) + torch.sum(diff_v_sq, dim=0))
        return torch.sum(dist)


def attack_stadv(model, X, opt='sgd', lr=0.005, tau=10, iters=20):
    theta = torch.tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0).float() # identity transformation
    theta = theta.repeat(X.shape[0], 1, 1)
    grid = F.affine_grid(theta, X.size()).to(X.device) # flow = 0. This is base grid
    f = torch.zeros_like(grid, requires_grad=True)
    torch.nn.init.normal_(f, mean=0, std=0.01)
    grid_new = grid + f
    grid_new = grid_new.clamp(min=-1, max=1)
    X_new = F.grid_sample(X, grid_new, mode='bilinear')

    if opt == 'sgd':
        optimizer = torch.optim.SGD([f,], lr=lr)
    elif opt == 'lbfgs':
        optimizer = optimizer = torch.optim.LBFGS([f, ], lr=lr)

    loss_flow = Loss_flow(device=X.device)
    loss_adv = nn.CrossEntropyLoss()

    yp0 = model(X)
    y0 = yp0.max(dim=1)[1]
    
    for i in range(iters):
        optimizer.zero_grad()
        logits = model(X_new)
        print((logits.max(dim=1)[1]!=y0).sum().item())

        loss = -loss_adv(logits, y0) + tau*loss_flow(f)
        loss.backward()
        optimizer.step()
        print(nn.CrossEntropyLoss()(logits, y0).item())

        # update variables and predict on adversarial image
        grid_new = grid + f
        grid_new = grid_new.clamp(min=-1, max=1)
        X_new = F.grid_sample(X, grid_new, mode='bilinear')

    return X_new