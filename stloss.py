import torch.nn as nn
import torch.nn.functional as F


class SoftTarget(nn.Module):


    def __init__(self, T):
        super(SoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s / self.T, dim=2),
                        F.softmax(out_t / self.T, dim=2),
                        reduction='batchmean') * self.T * self.T

        return loss
