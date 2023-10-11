import torch
from ml_architectures.seqsleepnet.seqsleepnet import SeqSleepNet
from ml_architectures.seqsleepnet.utils import make_seqsleepnet_config


def test_forward_pass():
    config = make_seqsleepnet_config()
    model = SeqSleepNet(config)
    # 1 sample, 20 epochs, 1 channel, 29 time bins, 129 frequency bins
    x = torch.rand((1, 20, 1, 29, 129))
    y = model(x)

    assert y.shape == (1, 20, 5)
