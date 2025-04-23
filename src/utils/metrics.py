# src/utils/metrics.py
import torch
import numpy as np

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        # Prevent division by zero if count is 0
        self.avg = self.sum / self.count if self.count > 0 else 0


def binary_accuracy(output, target):
    """
    Computes the accuracy for multiple binary predictions.
    Output should be sigmoid probabilities (0-1).
    Target should be integer labels (0 or 1).
    """
    with torch.no_grad(): # Ensure no gradients are calculated here
        target_int = target.int()
        # Threshold probabilities to get binary predictions
        pred = (output >= 0.5).int()

        # Compare predictions with ground truth
        correct = pred.eq(target_int)
        correct_sum = correct.sum().item() # .item() convert tensor -> number

        num_elements = target.numel() # Total number of predictions

        # Calculate accuracy
        accuracy = (correct_sum / num_elements) * 100.0 if num_elements > 0 else 0.0

    return accuracy

def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    output and target are Torch tensors
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res # Returns a list of accuracies for each k