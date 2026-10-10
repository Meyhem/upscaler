import torch

from upscaler.losses import PerceptualLoss, gan_loss


def test_perceptual_zero_for_identical_and_positive_otherwise():
    loss = PerceptualLoss(pretrained=False)
    x = torch.rand(1, 3, 32, 32)
    assert loss(x, x.clone()).item() == 0.0
    assert loss(x, torch.rand(1, 3, 32, 32)).item() > 0.0


def test_perceptual_grad_flows_to_input_only():
    loss = PerceptualLoss(pretrained=False)
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    loss(x, torch.rand(1, 3, 32, 32)).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in loss.parameters())


def test_perceptual_stays_in_eval_mode_and_relu_not_inplace():
    loss = PerceptualLoss(pretrained=False)
    loss.train()
    assert not loss.training and not loss.vgg.training
    assert not any(getattr(m, "inplace", False) for m in loss.vgg)


def test_gan_loss_direction():
    big = torch.full((2, 1, 4, 4), 10.0)
    assert gan_loss(big, True).item() < 1e-3
    assert gan_loss(big, False).item() > 5.0
    assert gan_loss(-big, False).item() < 1e-3
