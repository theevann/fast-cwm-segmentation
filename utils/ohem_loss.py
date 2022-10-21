import torch
import torch.nn as nn
import torch.nn.functional as F


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = thresh
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='mean')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        n_pixs = N * H * W
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.clone().detach().flatten()
        with torch.no_grad():
            scores = F.softmax(logits, dim=1)
            invalid_mask = labels == self.ignore_lb
            labels[invalid_mask] = 0
            picks = scores[torch.arange(n_pixs), labels]
            picks[invalid_mask] = 1
            sorteds, _ = torch.sort(picks)
            thresh = self.thresh if sorteds[self.n_min] < self.thresh else sorteds[self.n_min]
            labels[picks > thresh] = self.ignore_lb
        loss = self.criteria(logits, labels)
        return loss
