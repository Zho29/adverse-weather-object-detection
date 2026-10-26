#!/usr/bin/env python3
"""Offline smoke check for the production DSNet detection loss."""

import torch
from detection import compute_detection_loss


def main():
    heads = [torch.randn(2, 21, height, height, requires_grad=True)
             for height in [16, 16, 8]]
    boxes = [torch.tensor([[100., 100., 200., 200.], [300., 300., 400., 400.]]),
             torch.tensor([[150., 150., 250., 250.]])]
    labels = [torch.tensor([0, 1]), torch.tensor([0])]
    loss = compute_detection_loss(heads, boxes, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(head.grad is not None and torch.isfinite(head.grad).all() for head in heads)
    print(f'DSNet loss and backward pass succeeded: {loss.item():.4f}')


if __name__ == '__main__':
    main()
