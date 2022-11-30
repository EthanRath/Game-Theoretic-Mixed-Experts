import torch
import torch.nn as nn
import random
from Defenses.TIT import stn



def attack_stn(model, X, eps1, eps2, stepsize=0.01, num_iter=20, debug=False):
    aff = torch.zeros(X.shape[0], 2, 3, requires_grad=True).to(X.device)
    aff[:, 0, 0] = 1
    aff[:, 1, 1] = 1
    

    with torch.no_grad():
        yp0 = model(X)
        y0 = yp0.max(dim=1)[1]

    for t in range(num_iter):
        aff = aff.clone()
        X = X.clone()
        aff.retain_grad()
        affX = stn.stn(aff, X)
        loss = nn.CrossEntropyLoss()(model(affX), y0)
        loss.backward(retain_graph=True)
        # gd = torch.autograd.grad(loss, affX)[0]
        # print(gd)
        # print(aff.grad)
        if debug:
            print('iter', t, 'loss', loss.item())
            print('grad', aff.grad.shape)
            yp = model(X)
            y = yp.max(dim=1)[1]
            yp2 = model(affX)
            y2 = yp2.max(dim=1)[1]
            print('prediction', y)
            print('prediction adv', y2)
            print('err', (yp2.max(dim=1)[1] != y).sum().item())
            # print('aff0', aff[0,:,:])

        aff.data = aff + stepsize*aff.grad.detach()
        aff[:, 0, 0].clamp_(1-eps1, 1+eps1)
        aff[:, 0, 1].data.clamp_(-eps1, eps1)
        aff[:, 0, 2].data.clamp_(-eps2, eps2)
        aff[:, 1, 0].data.clamp_(-eps1, eps1)
        aff[:, 1, 1].data.clamp_(1-eps1, 1+eps1)
        aff[:, 1, 2].data.clamp_(-eps2, eps2)
        # aff = aff.detach()
        aff.grad.zero_()
    return stn.stn(aff, X).detach(), aff


def gen_rand_labels(model, X, num_classes):
    y = model(X).min(dim=1)[1]
    targets = torch.randint_like(y, low=0, high=num_classes)
    for i in range(len(targets)):
        while targets[i]==y[i]:
            targets[i] = torch.randint(low=0, high=num_classes, size=(1,))
    return targets


def gen_least_likely_labels(model, X):
    preds = model(X)
    return preds.min(dim=1)[1]

def pgd_linf_targ(model, X, epsilon=0.1, alpha=0.01,
         num_iter=20, y_targ='rand', num_classes=10, randomize=False):
    """ Construct targeted adversarial examples on the examples X"""

    if isinstance(y_targ, str):
        strlist = ['rand', 'leastlikely']
        assert(y_targ in strlist)
        if y_targ == 'rand':
            y_targ = gen_rand_labels(model, X, num_classes)
        elif y_targ == 'leastlikely':
            y_targ = gen_least_likely_labels(model, X)


    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)


    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        yp = model(X + delta)
        loss = nn.CrossEntropyLoss()(yp, y_targ)
        loss.backward()
        delta.data = (delta - alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def pgd_linf_untargeted_mostlikely(model, X, epsilon=0.1, stepsize=0.01, num_iter=20, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    with torch.no_grad():
        yp = model(X)
        y = yp.max(dim=1)[1]
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()

        delta.data = (delta + stepsize*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]


def pgd_l2_untargeted_mostlikely(model, X, epsilon=1.0, alpha=0.05, num_iter=20, randomize=False):
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
    else:
        delta = torch.zeros_like(X, requires_grad=True)
    delta = torch.zeros_like(X, requires_grad=True)

    with torch.no_grad():
        yp = model(X)
        y = yp.max(dim=1)[1]

    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.grad.zero_()
        
    return delta.detach()


class OptNetEnsemble(nn.Module):
    def __init__(self, basemodel, advmodels, epsilon, stepsize, iters, randomize, adv_mode='adv'):
        super(OptNetEnsemble, self).__init__()
        self.basemodel = basemodel
        self.advmodels = advmodels
        assert(len(self.advmodels)>0)
        self.epsilon = epsilon
        self.stepsize = stepsize
        self.iters = iters
        self.randomize = randomize
        self.adv_mode = adv_mode
        

    def train(self):
        self.basemodel.train()
        for model in self.advmodels:
            model.train()

    def eval(self):
        self.basemodel.eval()
        for model in self.advmodels:
            model.eval()


    def adversary(self, x):
        if self.adv_mode == 'adv':
            rd_model = random.choice(self.advmodels)
            return x+pgd_linf_untargeted_mostlikely(rd_model, x, self.epsilon, self.stepsize, self.iters, self.randomize)
        elif self.adv_mode == 'noise':
            delta = torch.rand_like(x, requires_grad=False)
            delta.data = delta.data * 2 * self.epsilon - self.epsilon
            return x+delta
        elif self.adv_mode == 'no':
            return x
        elif self.adv_mode == 'advl2':
            rd_model = random.choice(self.advmodels)
            return x+pgd_l2_untargeted_mostlikely(rd_model, x, self.epsilon, self.stepsize, self.iters, self.randomize)
        # elif self.adv_mode == 'll':
            # return x+pgd_linf_targ()

        else:
            print('no such mode:', self.adv_mode)
            raise NotImplementedError  

    def forward(self, x):
        
        x_prime = self.adversary(x)
        return self.basemodel(x_prime)

class OptNet(nn.Module):
    def __init__(self, basemodel, advmodel, epsilon, stepsize, iters, randomize, adv_mode='adv', resize = None):
        super(OptNet, self).__init__()
        self.basemodel = basemodel
        self.advmodel = advmodel
        self.epsilon = epsilon
        self.stepsize = stepsize
        self.iters = iters
        self.randomize = randomize
        self.adv_mode = adv_mode
        self.resize = resize
        

    def train(self):
        self.basemodel.train()
        self.advmodel.eval()

    def eval(self):
        self.basemodel.eval()
        self.advmodel.eval()

    def adversary_batch(self, x, bs = 16):
        model = self.advmodel.to("cuda")
        xt = torch.zeros(size = x.shape).to("cpu")
        for i in range(len(x) // bs):
            inx = x[i*bs : (i+1)*bs].to("cuda")
            xt[i*bs : (i+1)*bs] = self.adversary(inx, model).detach().cpu()
        
        if len(x) // bs < len(x) / bs:
            b = (len(x) // bs)
            inx = x[b*bs :].to("cuda")
            xt[b*bs :] = self.adversary(inx, model).detach().cpu()

        del(model)
        torch.cuda.empty_cache()
        return xt


    def adversary(self, x, model = None):
        if model == None:
            model = self.advmodel
        #xt = transforms.Resize((32,32))(x)
        if self.adv_mode == 'adv':
            temp = x+pgd_linf_untargeted_mostlikely(model, x, self.epsilon, self.stepsize, self.iters, self.randomize)
            return temp

        elif self.adv_mode == 'advl2':
            return x+pgd_l2_untargeted_mostlikely(self.advmodel, x, self.epsilon, self.stepsize, self.iters, self.randomize)
        elif self.adv_mode == 'noise':
            delta = torch.rand_like(x, requires_grad=False)
            delta.data = delta.data * 2 * self.epsilon - self.epsilon
            return x+delta
        elif self.adv_mode == 'stn':
            x, aff = attack_stn(self.advmodel, x, self.epsilon[0], self.epsilon[1], self.stepsize, self.iters)
            return x
        elif self.adv_mode == 'no':
            return x
        elif self.adv_mode == 'll':
            return x+pgd_linf_targ(self.advmodel, x, self.epsilon, self.stepsize, self.iters, y_targ='leastlikely', num_classes=10, randomize=self.randomize)
        elif self.adv_mode == 'randtarg':
            return x+pgd_linf_targ(self.advmodel, x, self.epsilon, self.stepsize, self.iters, y_targ='rand', num_classes=10, randomize=self.randomize)
        else:
            print('no such mode:', self.adv_mode)
            raise NotImplementedError 
        

    def forward(self, x):
        ##construct adversarial samples
        # delta = pgd_linf_untargeted_mostlikely(self.advmodel, x, self.epsilon, self.stepsize, self.iters, self.randomize)
        #x.requires_grad = True

        x_prime= self.adversary(x)
        if self.resize != None:
            x_prime = self.resize(x_prime)
        return self.basemodel(x_prime)

    def BPDA(self, x, stepsize, epsilon, samples=1, iters=20, randomize=True):

        if randomize:
            delta = torch.rand_like(x, requires_grad=False)
            delta.data = delta.data * 2 * epsilon - epsilon
        else:
            delta = torch.zeros_like(x, requires_grad=False)

        for t in range(iters):
            grad = None
            for m in range(samples):
                x_adv = self.adversary(x+delta) + x

                with torch.no_grad():
                    yp = self.basemodel(x)
                    y = yp.max(dim=1)[1]
                

                x_adv = x_adv.detach()
                x_adv.requires_grad = True
                # x_adv = torch.tensor(x_adv, requires_grad=True)
                loss = nn.CrossEntropyLoss()(self.basemodel(x_adv), y)
                loss.backward()
                if grad is None:
                    grad = x_adv.grad.detach().clone()
                else:
                    grad += x_adv.grad.detach().clone()
            grad = grad / m
            delta.data = (delta + stepsize*grad.detach().sign()).clamp(-epsilon,epsilon) 

        return delta.detach()

def expected_delta(X, model, exp, batch_size, epsilon, stepsize, num_iter, attack='untarg'):
    if exp<batch_size:
        X0 = X.repeat(exp, 1, 1, 1)
        iters = 1
    else:
        X0 = X.repeat(batch_size, 1, 1, 1)
        iters = int(exp/batch_size)
    
    adv_imgs = []
    for i in range(iters):
        if attack == 'untarg':
            delta = pgd_linf_untargeted_mostlikely(model, X0, epsilon=epsilon, stepsize=stepsize, num_iter=num_iter, randomize=True)
        elif attack == 'll':
            delta = pgd_linf_targ(model, X0, epsilon=epsilon, alpha=stepsize, num_iter=num_iter, y_targ='leastlikely', num_classes=10, randomize=True)
        adv_imgs.append(delta.detach().cpu().clone())
    adv_imgs = torch.cat(adv_imgs, dim=0)
    return adv_imgs