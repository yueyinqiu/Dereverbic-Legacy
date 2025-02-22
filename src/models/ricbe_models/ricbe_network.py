# Inspired by https://github.com/ksasso1028/audio-reverb-removal/blob/main/dereverb/auto_verb.py

from statictorch import Tensor3d
import torch

from models.ricbe_models.ricbe_encoder_decoder_pair import RicbeEncoderDecoderPair
from models.ricbe_models.ricbe_postprocess import RicbePostprocess
from models.ricbe_models.ricbe_preprocess import RicbePreprocess


class RicbeNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        channels: int = 48
        self.preprocess_for_speech = RicbePreprocess(1, channels)
        self.pair_for_speech = RicbeEncoderDecoderPair(channels)
        self.postprocess_for_speech = RicbePostprocess(channels * 2, 1, 1, 1, 1)
        
        self.preprocess_for_rir = RicbePreprocess(2, channels)
        self.pair_for_rir = RicbeEncoderDecoderPair(channels)
        self.postprocess_for_rir = RicbePostprocess(channels * 2, 1, 1, 1, 5)


    def forward(self, reverb: Tensor3d):
        rev: Tensor3d = self.preprocess_for_speech(reverb)
        spe: Tensor3d = self.pair_for_speech(rev)
        spe = self.postprocess_for_speech(spe)

        rs: Tensor3d = Tensor3d(torch.cat([reverb, spe], dim=1))

        rs = self.preprocess_for_rir(rs)
        rir: Tensor3d = self.pair_for_rir(rs)
        rir = self.postprocess_for_rir(rir)
        return rir, spe
    