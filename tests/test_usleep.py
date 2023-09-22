
import torch

from ml_architectures.usleep.usleep import USleep

def test_forward_pass():
    net = USleep()
    x = torch.rand((1, 2, 35*30*128))
    y = net(x)

    assert y.shape == (1, 5, 35)