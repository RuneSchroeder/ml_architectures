
import torch

from ml_architectures.usleep.usleep import USleep

def test_forward_pass_2_channel():
    net = USleep()
    x = torch.rand((1, 2, 35*30*128))
    y = net(x)

    assert y.shape == (1, 5, 35)

def test_forward_pass_1_channel():
    net = USleep(num_channels=1)
    x = torch.rand((1, 1, 35*30*128))

    y = net(x)

    assert y.shape == (1, 5, 35)

def test_complexity_factor_change():
    net = USleep(num_channels=2,
                 initial_filters=5,
                 complexity_factor=2)
    
    x = torch.rand((1, 2, 35*30*128))

    y = net(x)

    assert y.shape == (1, 5, 35)

def test_progression_factor_change():
    net = USleep(num_channels=2,
                 initial_filters=5,
                 complexity_factor=1.67,
                 progression_factor=4)
    
    x = torch.rand((1, 2, 35*30*128))

    y = net(x)

    assert y.shape == (1, 5, 35)