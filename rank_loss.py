import torch
import torch.nn as nn


class RankLoss(nn.Module):

    def __init__(self, device, mode = 'mean'):
        super(RankLoss, self).__init__()
        assert(mode in ['mean', 'sum']), 'mode need to be "mean" or "sum"'
        self.device = device
        self.mode = mode
    def forward1(self, input, target):
        input = torch.squeeze(input)

        pair_num = target.shape[0]

        pre_values = input[target]

        loss = torch.log(1 + torch.exp(pre_values[:,1] - pre_values[:,0]))
        loss = torch.sum(loss)

        if self.mode == 'sum':
            return loss
        else:
            return loss / pair_num

    def forward(self, input, target):

        self.target = target

        self.pair_num = input.shape[0]
        self.input = input

        self.pair_values = input[target]
        self.loss_lsep = torch.log(1 + torch.sum(torch.exp(self.pair_values[:,1] - self.pair_values[:,0])))
        return self.loss_lsep

    def backward(self, grad_output):

        tmp_sum = torch.zeros(self.pair_num, device = self.device)
        u = torch.zeros(self.pair_num, device = self.device)
        v = torch.zeros(self.pair_num, device = self.device)
        for i in range(self.pair_num-1):

            u[:] = 0
            v[:] = 0
            u[self.target[i,0]] = 1
            v[self.target[i,1]] = 1
            delta_y = u-v
            tmp_sum = tmp_sum + delta_y * torch.exp(delta_y * (-1 * self.input))


        der_input = (-1/self.loss_lsep*tmp_sum)
        grad_input = grad_output.float().to(self.device) * der_input

        return grad_input
