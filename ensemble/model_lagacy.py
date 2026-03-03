import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, efficientnet_b4, vit_b_16
# import timm  # 다양한 사전학습 모델을 위해 timm 라이브러리 사용 권장



def freeze_bn(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


''' conv1 → layer1 → layer2 → layer3 → layer4 → fc '''
resnet_50 = resnet50(weights="IMAGENET1K_V1")

# head 교체
resnet_50.fc = nn.Linear(resnet_50.fc.in_features, 7)

freeze_bn(resnet_50)

# 전체 freeze
for p in resnet_50.parameters():
    p.requires_grad = False

# fc + layer4 unfreeze
for p in resnet_50.fc.parameters():
    p.requires_grad = True

for p in resnet_50.layer4.parameters():
    p.requires_grad = True


''' stem → blocks[0…6] → head '''
eff_b4 = efficientnet_b4(weights="IMAGENET1K_V1")
eff_b4.classifier[1] = nn.Linear(eff_b4.classifier[1].in_features, 7)

freeze_bn(eff_b4)

# 전체 freeze
for p in eff_b4.parameters():
    p.requires_grad = False

# classifier unfreeze
for p in eff_b4.classifier.parameters():
    p.requires_grad = True

# 마지막 stage unfreeze
for p in eff_b4.features[-1].parameters():
    p.requires_grad = True


''' patch_embed → blocks[0..11] → head '''
vit_b16 = vit_b_16(weights="IMAGENET1K_V1")
vit_b16.heads.head = nn.Linear(vit_b16.heads.head.in_features, 7)

# 전체 freeze
for p in vit_b16.parameters():
    p.requires_grad = False

# head unfreeze
for p in vit_b16.heads.parameters():
    p.requires_grad = True

# 마지막 3 blocks unfreeze
for block in vit_b16.encoder.layers[-3:]:
    for p in block.parameters():
        p.requires_grad = True


class CNNEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.resnet = resnet_50
        self.effnet = eff_b4
        self.vit = vit_b16

        # 모델별 가중치 (Validation 성능에 따라 사후 조정 가능)
        self.register_buffer('weights', torch.FloatTensor([0.25, 0.45, 0.30]))

    def forward(self, x):
        # 1. 입력 크기 조정
        # 224 사이즈 모델들 먼저 처리
        x224 = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        out_r = self.resnet(x224)
        out_v = self.vit(x224)
        del x224 # 메모리 확보
        
        # 380 사이즈 모델 처리
        x380 = F.interpolate(x, size=(380, 380), mode="bilinear", align_corners=False)
        out_e = self.effnet(x380)
        del x380 # 메모리 확보
        
        # 3. Weighted Soft Voting (확률 기반 결합)
        # 학습 시 CrossEntropyLoss를 사용한다면 로짓 결합이 유리하지만, 
        # 안정성을 위해 아래와 같이 확률 합산 후 로그를 취하는 방식을 권장합니다.
        p_r = F.softmax(out_r, dim=1)
        p_v = F.softmax(out_v, dim=1)
        p_e = F.softmax(out_e, dim=1)

        avg_probs = (p_r * self.weights[0] + 
                     p_e * self.weights[1] + 
                     p_v * self.weights[2])
        
        # Loss 함수가 log_softmax를 기대하는 경우 (예: NLLLoss)를 대비
        return torch.log(avg_probs + 1e-7)

# 모델 선언
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CNNEnsemble(num_classes=7).to(device)