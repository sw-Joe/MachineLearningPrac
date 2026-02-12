import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, efficientnet_b4, vit_b_16

def freeze_bn(module):
    """사전 학습된 BatchNorm 통계치를 보호하기 위해 eval 모드로 고정"""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()

class CNNEnsemble(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        
        # 1. ResNet50: layer4와 fc만 학습
        self.resnet = resnet50(weights="IMAGENET1K_V1")
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        for p in self.resnet.parameters(): p.requires_grad = False
        for p in self.resnet.layer4.parameters(): p.requires_grad = True
        for p in self.resnet.fc.parameters(): p.requires_grad = True

        # 2. EfficientNet-B4: 마지막 블록과 classifier만 학습
        self.effnet = efficientnet_b4(weights="IMAGENET1K_V1")
        self.effnet.classifier[1] = nn.Linear(self.effnet.classifier[1].in_features, num_classes)
        for p in self.effnet.parameters(): p.requires_grad = False
        for p in self.effnet.features[-1].parameters(): p.requires_grad = True
        for p in self.effnet.classifier.parameters(): p.requires_grad = True

        # 3. ViT-B/16: 마지막 3개 인코더 레이어와 heads 학습
        self.vit = vit_b_16(weights="IMAGENET1K_V1")
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)
        for p in self.vit.parameters(): p.requires_grad = False
        for block in self.vit.encoder.layers[-3:]:
            for p in block.parameters(): p.requires_grad = True
        for p in self.vit.heads.parameters(): p.requires_grad = True

        # 모델별 가중치 등록
        self.register_buffer('model_weights', torch.FloatTensor([0.25, 0.45, 0.30]))

    def forward(self, x):
        # 학습 중에도 BN은 고정 (사전 학습 지식 유지)
        if self.training:
            freeze_bn(self.resnet)
            freeze_bn(self.effnet)

        x224 = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        out_r = self.resnet(x224)
        out_v = self.vit(x224)
        del x224

        x380 = F.interpolate(x, size=(380, 380), mode="bilinear", align_corners=False)
        out_e = self.effnet(x380)
        del x380

        # Soft Voting: 확률 평균 후 로그 변환 (NLLLoss용)
        p_r = F.softmax(out_r, dim=1)
        p_v = F.softmax(out_v, dim=1)
        p_e = F.softmax(out_e, dim=1)

        avg_probs = (p_r * self.model_weights[0] + 
                     p_e * self.model_weights[1] + 
                     p_v * self.model_weights[2])
        
        return torch.log(avg_probs + 1e-7)