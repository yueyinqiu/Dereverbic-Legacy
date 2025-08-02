# This model is modified from (without modification of its behavior):
# https://github.com/NVIDIA/CleanUNet
# The original repository is licensed under MIT
# Please also respect the original author's rights

import numpy
from torch import Tensor
import torch

from models.cleanunet_models.submodules.cleanunet_transformer_encoder import CleanunetTransformerEncoder


class CleanunetNetwork(torch.nn.Module):
    @staticmethod
    def weight_scaling_init(layer: torch.nn.Conv1d | torch.nn.ConvTranspose1d):
        w: Tensor = layer.weight.detach()
        alpha: Tensor = 10.0 * w.std()
        layer.weight.data /= torch.sqrt(alpha)
        if layer.bias is not None:
            layer.bias.data /= torch.sqrt(alpha)

    @staticmethod
    def padding(x, D, K, S):
        L: int = x.shape[-1]
        for _ in range(D):
            if L < K:
                L = 1
            else:
                L = 1 + numpy.ceil((L - K) / S)

        for _ in range(D):
            L = (L - 1) * S + K
        
        L = int(L)
        x = torch.nn.functional.pad(x, (0, L - x.shape[-1]))
        return x

    def __init__(self, channels_input=1, channels_output=1,
                 channels_H=64, max_H=768,
                 encoder_n_layers=8, kernel_size=4, stride=2,
                 tsfm_n_layers=5, 
                 tsfm_n_head=8,
                 tsfm_d_model=512, 
                 tsfm_d_inner=2048):
        super().__init__()

        self.channels_input = channels_input
        self.channels_output = channels_output
        self.channels_H = channels_H
        self.max_H = max_H
        self.encoder_n_layers = encoder_n_layers
        self.kernel_size = kernel_size
        self.stride = stride

        self.tsfm_n_layers = tsfm_n_layers
        self.tsfm_n_head = tsfm_n_head
        self.tsfm_d_model = tsfm_d_model
        self.tsfm_d_inner = tsfm_d_inner

        # encoder and decoder
        self.encoder = torch.nn.ModuleList()
        self.decoder = torch.nn.ModuleList()

        i: int
        for i in range(encoder_n_layers):
            self.encoder.append(torch.nn.Sequential(
                torch.nn.Conv1d(channels_input, channels_H, kernel_size, stride),
                torch.nn.ReLU(),
                torch.nn.Conv1d(channels_H, channels_H * 2, 1), 
                torch.nn.GLU(dim=1)
            ))
            channels_input = channels_H

            if i == 0:
                # no relu at end
                self.decoder.append(torch.nn.Sequential(
                    torch.nn.Conv1d(channels_H, channels_H * 2, 1), 
                    torch.nn.GLU(dim=1),
                    torch.nn.ConvTranspose1d(channels_H, channels_output, kernel_size, stride)
                ))
            else:
                self.decoder.insert(0, torch.nn.Sequential(
                    torch.nn.Conv1d(channels_H, channels_H * 2, 1), 
                    torch.nn.GLU(dim=1),
                    torch.nn.ConvTranspose1d(channels_H, channels_output, kernel_size, stride),
                    torch.nn.ReLU()
                ))
            channels_output = channels_H
            
            # double H but keep below max_H
            channels_H *= 2
            channels_H = min(channels_H, max_H)
        
        # self attention block
        self.tsfm_conv1 = torch.nn.Conv1d(channels_output, tsfm_d_model, kernel_size=1)
        self.tsfm_encoder = CleanunetTransformerEncoder(d_word_vec=tsfm_d_model, 
                                               n_layers=tsfm_n_layers, 
                                               n_head=tsfm_n_head, 
                                               d_k=tsfm_d_model // tsfm_n_head, 
                                               d_v=tsfm_d_model // tsfm_n_head, 
                                               d_model=tsfm_d_model, 
                                               d_inner=tsfm_d_inner, 
                                               dropout=0.0, 
                                               n_position=0, 
                                               scale_emb=False)
        self.tsfm_conv2 = torch.nn.Conv1d(tsfm_d_model, channels_output, kernel_size=1)

        # weight scaling initialization
        layer: torch.nn.Module
        for layer in self.modules():
            if isinstance(layer, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                CleanunetNetwork.weight_scaling_init(layer)

    def forward(self, noisy_audio: Tensor):
        # (B, L) -> (B, C, L)
        if len(noisy_audio.shape) == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        B: int
        C: int
        L: int
        B, C, L = noisy_audio.shape
        # assert C == 1
        
        # normalization and padding
        std: Tensor = noisy_audio.std(dim=2, keepdim=True)
        std = std.mean(dim=1, keepdim=True) + 1e-3
        noisy_audio = noisy_audio / std
        x: Tensor = CleanunetNetwork.padding(noisy_audio, 
                                             self.encoder_n_layers, 
                                             self.kernel_size, 
                                             self.stride)
        
        # encoder
        skip_connections: list[Tensor] = []
        downsampling_block: torch.nn.Module
        for downsampling_block in self.encoder:
            x = downsampling_block(x)
            skip_connections.append(x)
        skip_connections = skip_connections[::-1]

        # attention mask for causal inference; for non-causal, set attn_mask to None
        len_s: int = x.shape[-1]  # length at bottleneck
        attn_mask: Tensor = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=x.device), 
            diagonal=1)).bool()

        x = self.tsfm_conv1(x)  # C 1024 -> 512
        x = x.permute(0, 2, 1)
        x = self.tsfm_encoder(x, src_mask=attn_mask)
        x = x.permute(0, 2, 1)
        x = self.tsfm_conv2(x)  # C 512 -> 1024

        i: int
        upsampling_block: torch.nn.Module
        for i, upsampling_block in enumerate(self.decoder):
            skip_i: Tensor = skip_connections[i]
            x = x + skip_i[:, :, :x.shape[-1]]
            x = upsampling_block(x)

        x = x[:, :, :L] * std
        return x
