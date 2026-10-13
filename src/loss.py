import torch
import torch.nn as nn

class BCELoss(nn.Module):
    def __init__(self, is_logits=True):
        super().__init__()

        if is_logits:
            self.bce_loss = nn.BCEWithLogitsLoss()
        else:
            self.bce_loss = nn.BCELoss()

    def forward(self, preds, target):
        return self.bce_loss(preds, target)

class DiceLoss(nn.Module):
    def __init__(self, is_logits=True, tol=1e-6):
        super().__init__()
        self.is_logits = is_logits
        self.tol = tol

    def forward(self, preds, target):

        if self.is_logits:
            # Unbounded -> bounded [0, 1]
            preds = torch.sigmoid(preds)
        
        # Flatten to (Batch*H*W) for each image in the batch
        preds = preds.view(-1)
        target = target.view(-1)

        intersection = (preds * target).sum()

        cardinality = preds.sum() + target.sum()

        # Avoid division by zero
        dice_score = (2. * intersection + self.tol) / (cardinality + self.tol)

        return 1. - dice_score


class DiceBCELoss(nn.Module):
    def __init__(self, is_logits=True, tol=1e-6, weight_dice=0.5):
        super().__init__()
        self.is_logits = is_logits
        self.tol = tol
        self.weight_dice = weight_dice
        self.dice_loss = DiceLoss(is_logits=self.is_logits, tol=self.tol)
        self.bce_loss = BCELoss(is_logits=self.is_logits)

    def forward(self, preds, target):
        dice = self.dice_loss(preds, target)
        bce = self.bce_loss(preds, target)
        weighted = self.weight_dice * dice + bce

        return weighted, dice, bce