import timm    # 최신 프리트레인 모델
import torch
import torch.nn as nn



class CNNRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # 마지막 fc 레이어를 제외하고 모델 로드
        self.backbone = timm.create_model(
            'resnet18', pretrained=True, num_classes=0, drop_path_rate=0.15)
        
        # resnet18의 최종 피처 맵 채널 수는 512입니다.
        self.reg_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1) # 최종 나이값
        )

    def forward(self, x):
        features = self.backbone(x) # [Batch, 512] 특성 추출
        age = self.reg_head(features)
        return age

    
class MultiLabelCNNRegression(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
