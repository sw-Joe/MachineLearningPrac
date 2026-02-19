import timm    # 최신 프리트레인 모델과 Drop Path 지원을 위해 추가
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, efficientnet_b4, vit_b_16



class CNNEnsembleByTimm(nn.Module):
    def __init__(self, num_classes=7, drop_path_rate=0.2):
        """
        Args:
            num_classes (int): 분류할 클래스 개수 (HAM10k 기준 7)
            drop_path_rate (float): Stochastic Depth 확률. 과적합 억제를 위해 사용
        """
        super().__init__()

        # 1. ResNet-50 (timm 버전)
        # timm의 resnet50은 구조가 최적화되어 있으며 drop_path를 지원합니다.
        self.resnet = timm.create_model(
            'resnet50', 
            pretrained=True, 
            num_classes=num_classes,
            drop_path_rate=0.1 # ResNet은 상대적으로 얕으므로 낮게 설정
        )

        # 2. EfficientNet-B4 (timm 버전)
        # B4는 층이 깊으므로 drop_path_rate를 더 높게 주어 규제를 강화합니다.
        self.effnet = timm.create_model(
            'efficientnet_b4', 
            pretrained=True, 
            num_classes=num_classes,
            drop_path_rate=drop_path_rate # 외부 설정값(cfg) 적용
        )

        # 3. Vision Transformer (timm 버전)
        # ViT의 전역 문맥 파악 능력을 앙상블에 활용합니다.
        self.vit = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=True, 
            num_classes=num_classes,
            drop_path_rate=0.1
        )

        # [핵심] 학습 가능한 앙상블 가중치 (Learnable Weights)
        # 초기값은 기존 [0.25, 0.45, 0.30]의 비율과 유사하게 설정하되, 
        # 이제는 역전파(Backpropagation)를 통해 최적의 비율로 스스로 학습됩니다.
        self.ensemble_weights = nn.Parameter(torch.tensor([1.0, 1.8, 1.2]))

    def forward(self, x):
        """
        각 모델의 출력을 학습된 가중치로 결합합니다.
        """
        # 각 백본 모델의 Logits 계산
        res_logits = self.resnet(x)
        eff_logits = self.effnet(x)
        vit_logits = self.vit(x)

        # 가중치를 Softmax를 통해 합이 1이 되도록 정규화
        w = F.softmax(self.ensemble_weights, dim=0)

        # 수치적 안정성을 위해 확률 공간에서 가중 합산 후 다시 로그를 취함
        # 이는 NLLLoss(LabelSmoothingNLLLoss)와 호환되는 log-probability를 반환합니다.
        avg_probs = (
            torch.softmax(res_logits, dim=1) * w[0] +
            torch.softmax(eff_logits, dim=1) * w[1] +
            torch.softmax(vit_logits, dim=1) * w[2]
        )
        
        # log(0) 에러 방지를 위한 epsilon(1e-9) 추가
        return torch.log(avg_probs + 1e-9)

# [참고] 기존 고정 가중치 방식에서 사용하던 register_buffer 코드는 
# 유연한 파인튜닝을 위해 nn.Parameter로 대체되었습니다.


def freeze_bn(module):
    """사전 학습된 BatchNorm 통계치를 보호하기 위해 eval 모드로 고정"""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()


class CNNEnsembleByTorchVision(nn.Module):
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