import random
import torch
import torch.nn as nn

from utils import intersection_over_union
        
class YoloLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.lambda_class = 3
        self.lambda_noobj = 10
        self.lambda_obj = 10
        self.lambda_box = 5
        self.device = device  # Lưu device để tiện xử lý sau này

    def forward(self, predictions, target, anchors):
        # Đảm bảo mọi thứ trên cùng thiết bị
        predictions = predictions.to(self.device)
        target = target.to(self.device)
        anchors = anchors.to(self.device)

        obj = (target[..., 0] == 1).to(self.device)
        noobj = (target[..., 0] == 0).to(self.device)

        no_object_loss = self.bce(
            predictions[..., 0:1][noobj], target[..., 0:1][noobj]
        )

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat(
            [self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors],
            dim=-1
        )

        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(
            self.sigmoid(predictions[..., 0:1][obj]), 
            ious * target[..., 0:1][obj]
        )

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log(1e-16 + target[..., 3:5] / anchors)
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        class_loss = self.entropy(predictions[..., 5:][obj], target[..., 5][obj].long())
        print("train loss")
        print(f"box {self.lambda_box * box_loss}")
        print(f"obj {self.lambda_obj * object_loss}")
        print(f"noobj {self.lambda_noobj * no_object_loss}")
        print(f"class {self.lambda_class * class_loss}")
        print()
        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
