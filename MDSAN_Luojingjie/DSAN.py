import torch
import torch.nn as nn
import lmmd
from  cnn_1d import cnn_features
class DSAN(nn.Module):

    def __init__(self, num_classes=10, bottle_neck=False):
        super(DSAN, self).__init__()
        self.feature_layers = cnn_features()
        self.lmmd_loss = lmmd.LMMD_loss(class_num=num_classes)
        self.bottle_neck = bottle_neck
        if bottle_neck:
            self.cls_fc = nn.Linear(256, num_classes)
        else:
            self.cls_fc = nn.Linear(256, num_classes)


    def forward(self, source, target, s_label):
        source = self.feature_layers(source)
        if self.bottle_neck:
            source = self.bottle(source)
        s_pred = self.cls_fc(source)
        target = self.feature_layers(target)
        if self.bottle_neck:
            target = self.bottle(target)
        t_label = self.cls_fc(target)
        loss_lmmd = self.lmmd_loss.get_loss(source, target, s_label, torch.nn.functional.softmax(t_label, dim=1))
        return s_pred, loss_lmmd

    def predict(self, x):
        x = self.feature_layers(x)
        if self.bottle_neck:
            x = self.bottle(x)
        return self.cls_fc(x)