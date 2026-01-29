import math

def round_filters(filters, width_mult):
    if not width_mult:
        return filters
    # 8의 배수로 반올림하는 로직 (EfficientNet 공식 구현 방식)
    filters *= width_mult
    new_filters = max(8, int(filters + 8 / 2) // 8 * 8)
    if new_filters < 0.9 * filters:
        new_filters += 8
    return int(new_filters)

def round_repeats(repeats, depth_mult):
    if not depth_mult:
        return repeats
    # 레이어 수 올림 계산
    return int(math.ceil(depth_mult * repeats))

# B1 적용 예시 (B0 base_config 기준)
# base_config의 [expand_ratio, out_channels, num_layers, stride, kernel_size]
b1_width_mult = 1.0
b1_depth_mult = 1.1

# Stage 3의 경우: [6, 24, 2, 2, 3] -> B1에서는?
b1_channels = round_filters(24, b1_width_mult) # 24
b1_layers = round_repeats(2, b1_depth_mult)     # ceil(2 * 1.1) = 3