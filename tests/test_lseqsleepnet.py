import torch
from ml_architectures.lseqsleepnet.lseqsleepnet import LSeqSleepNet
from ml_architectures.lseqsleepnet.utils import make_lseqsleepnet_config


def test_forward_pass():
    config = make_lseqsleepnet_config()
    model = LSeqSleepNet(config)
    # 1 sample, 20 epochs, 1 channel, 29 time bins, 129 frequency bins
    x = torch.rand((1, 200, 1, 29, 129))
    y = model(x)

    assert y.shape == (1, 200, 5)
