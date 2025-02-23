# This model is modified from: 
# https://github.com/NVIDIA/CleanUNet
# Please respect the original license

import numpy
import torch


class CleanunetPositionalEncoding(torch.nn.Module):
    def __init__(self, d_hid, n_position=200):
        super().__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / numpy.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table: numpy.ndarray = numpy.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = numpy.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = numpy.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
