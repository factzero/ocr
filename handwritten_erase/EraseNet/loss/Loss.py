import torch
from torch import nn


def bce_loss(input, target):
    input = torch.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)

    bce = nn.BCELoss()

    return bce(input, target)


class LossWithGAN_STE(nn.Module):
    def __init__(self):
        super(LossWithGAN_STE, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, mask, output, mm, gt):
        holeLoss = self.l1(mask*output, mask*gt)
        validAreaLoss = self.l1((1 -mask)*output, (1 - mask)*gt)
        mask_loss = bce_loss(mm, mask) + self.l1(mm, mask)
        image_loss = self.l1(output, gt)
        Gloss = 0.5*mask_loss + 0.5*holeLoss + 0.5*validAreaLoss + 1.5*image_loss

        return Gloss.sum()