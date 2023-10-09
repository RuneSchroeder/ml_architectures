"""
An implementation of SeqSleepNet using shared components from ml_architectures.
"""
import torch
import torch.nn as nn
from common.epoch_encoder import MultipleEpochEncoder
from short_sequence_model import ShortSequenceModel
from classifier import Classifier


class SeqSleepNet(nn.Module):
    """
    Implementation of the SeqSleepNet architecture.
    """

    class Config:
        def __init__(self, encoder_conf, ssm_conf, clf_conf):
            self.encoder_conf = encoder_conf
            self.lsm_conf = ssm_conf
            self.clf_conf = clf_conf

    def __init__(self, enc_conf, ssm_conf, clf_conf):
        super().__init__()
        self.epoch_encoder = MultipleEpochEncoder(enc_conf)
        self.sequence_model = ShortSequenceModel(ssm_conf)
        self.classifier = Classifier(clf_conf)

    def forward(self, x):
        # x is (Batch, Epoch, Channels, Sequence, Feature)
        x = self.epoch_encoder(x)

        # x is (Batch, Epoch, Feature)
        x = self.sequence_model(x)

        # x is (Batch, Epoch, Feature)
        x = self.classifier(x)

        # x is (Batch, Epoch, Logits)
        return x
