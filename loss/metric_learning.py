import torch
import torch.nn as nn
import torch.nn.functional as F


class Arcface(nn.Module):
    def __init__(self, in_feat, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.in_feat = in_feat
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_feat))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine * self.s


class Cosface(nn.Module):
    def __init__(self, in_feat, num_classes, s=30.0, m=0.35):
        super().__init__()
        self.in_feat = in_feat
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_feat))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine * self.s


class AMSoftmax(nn.Module):
    def __init__(self, in_feat, num_classes, s=30.0, m=0.35):
        super().__init__()
        self.in_feat = in_feat
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_feat))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine * self.s


class CircleLoss(nn.Module):
    def __init__(self, in_feat, num_classes, s=128, m=0.25):
        super().__init__()
        self.in_feat = in_feat
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_feat))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine * self.s
