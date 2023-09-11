# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:40:59 2023

@author: repse
"""

import torch.nn as nn
import torch.nn.functional as F
from .long_sequence_model import LongSequenceModel
from common.epoch_encoder import MultipleEpochEncoder
from .classifier import Classifier


class LSeqSleepNet(nn.Module):
    class Config:
        def __init__(self, encoder_conf, lsm_conf, clf_conf):
            self.encoder_conf = encoder_conf
            self.lsm_conf = lsm_conf
            self.clf_conf = clf_conf

    def __init__(self, enc_conf, lsm_conf, clf_conf):
        super().__init__()
        self.epoch_encoder = MultipleEpochEncoder(enc_conf)
        self.sequence_model = LongSequenceModel(lsm_conf)
        self.classifier = Classifier(clf_conf)

    def forward(self, x):
        # x is (Batch, Epoch, Channels, Sequence, Feature)
        x = self.epoch_encoder(x)

        # x is (Batch, Epoch, Feature)
        x = self.sequence_model(x)

        # x is (Batch, Epoch, Feature)
        x = self.classifier(x)

        # x is (Batch, Epoch, Probabilities)
        return x
