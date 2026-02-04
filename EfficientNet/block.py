import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Squeeze
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(), # Swish와 거의 동일
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid() # Excitation (0~1 사이 가중치)
        )

    def forward(self, x):
        return x * self.se(x) # 원본에 가중치 곱하기 (Scale)
    

class MBConv(nn.Module):
    """
    Docstring for MBConv
    in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(MBConv, self).__init__()
        
        # 1. 확장 채널 계산
        expanded_channels = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.use_expansion = (expand_ratio != 1)

        # (1) Expansion Phase: 1x1 Conv
        if self.use_expansion:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU()
            )

        # (2) Depthwise Conv Phase: 3x3 or 5x5
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, 
                      stride, padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        )

        # (3) Squeeze-and-Excitation Phase
        reduced_dim = max(1, int(in_channels * se_ratio))
        self.se_block = SEBlock(expanded_channels, reduced_dim)

        # (4) Pointwise Projection Phase: 1x1 Conv (Linear)
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
            # 마지막은 활성화 함수를 생략 (Linear Bottleneck)
        )

    def forward(self, x):
        identity = x
        
        if self.use_expansion:
            x = self.expand_conv(x)
            
        x = self.depthwise_conv(x)
        x = self.se_block(x)
        x = self.project_conv(x)
        
        # 입력과 출력 형태가 같을 때만 Residual Connection 적용
        if self.use_residual:
            return x + identity
        return x
    

class EfficientNet(nn.Module):
    def __init__(self, config, num_classes=1000):
        super(EfficientNet, self).__init__()
        
        # 1. Stem: Stage 1 (3x3 Conv)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        
        # 2. Stages: Stage 2 ~ 8 (MBConv Blocks)
        self.stages = nn.ModuleList()
        in_channels = 32
        
        for expand_ratio, out_channels, num_layers, stride, kernel_size in config:
            self.stages.append(self._make_stage(
                in_channels, out_channels, num_layers, stride, kernel_size, expand_ratio
            ))
            in_channels = out_channels # 다음 스테이지의 입력은 현재의 출력
            
        # 3. Head: Stage 9 (1x1 Conv & Pooling & FC)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_layers, stride, kernel_size, expand_ratio):
        layers = []
        # 스테이지의 첫 번째 블록만 stride를 적용하여 해상도를 줄임
        layers.append(MBConv(in_channels, out_channels, kernel_size, stride, expand_ratio))
        
        # 나머지 레이어들은 stride=1로 고정하여 반복 쌓기
        for _ in range(num_layers - 1):
            layers.append(MBConv(out_channels, out_channels, kernel_size, 1, expand_ratio))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights using proper initialization schemes"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for convolutional layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize batch norm parameters
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x

# 모델 생성 예시
# model = EfficientNet(base_config, num_classes=10) # CIFAR-10용