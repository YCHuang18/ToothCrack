import torch.nn as nn
import torch

class CrackEnhance(nn.Module):
    def __init__(self, in_channels, out_channels, width=8, threshold=0.4):
        super(CrackEnhance, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.width = width
        self.threshold = threshold

    def forward(self, input):
        mask_crack = self._get_crack_mask(input)
        output = self.conv1(mask_crack)
        return output

    def _get_crack_mask(self, input):
        longitudinal_crack = self._l_crack(input)
        transversal_crack = self._t_crack(input)
        mask_crack = torch.max(longitudinal_crack, transversal_crack)
        mask_crack = (mask_crack + input) / 2

        return mask_crack

    def _t_crack(self, input):
        top_roll = torch.roll(input, shifts=-self.width, dims=2).clone()
        top_roll[:, :, -self.width:, :] = 0
        diff1 = torch.abs(input - top_roll) < self.threshold

        under_roll = torch.roll(input, shifts=self.width, dims=2).clone()
        under_roll[:, :, :self.width, :] = 0
        diff2 = torch.abs(input - under_roll) < self.threshold

        mask = torch.logical_or(diff1, diff2)
        masked_input = input.clone()
        masked_input[mask] = 0
        return masked_input

    def _l_crack(self, input):
        left_roll = torch.roll(input, shifts=-self.width, dims=3).clone()
        left_roll[:, :, :, -self.width:] = 0
        diff1 = torch.abs(input - left_roll) < self.threshold

        right_roll = torch.roll(input, shifts=self.width, dims=3).clone()
        right_roll[:, :, :, :self.width] = 0
        diff2 = torch.abs(input - right_roll) < self.threshold

        mask = torch.logical_or(diff1, diff2)
        masked_input = input.clone()
        masked_input[mask] = 0
        return masked_input
