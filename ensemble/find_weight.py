import torch
import torch.nn.functional as F



if __name__ == "__main__":
    # 1. 최적 모델 파일 경로 설정
    model_path = "ensemble/result/26-02-13_02-24-31/best_model_ep096.pt" # 실제 파일 경로

    # 2. state_dict 불러오기
    state_dict = torch.load(model_path, map_location='cpu')

    # 3. ensemble_weights 키 확인 및 값 추출
    if 'ensemble_weights' in state_dict:
        raw_weights = state_dict['ensemble_weights']
        
        # 4. 소프트맥스를 적용하여 실제 반영 비율(0~1)로 변환
        final_ratios = F.softmax(raw_weights, dim=0)
        
        print("="*50)
        print("가중치 학습 결과 분석")
        print("="*50)
        print(f"Raw 파라미터 값: {raw_weights.numpy()}")
        print("-"*50)
        print(f"최종 앙상블 비율 (합계 1.0):")
        print(f" > ResNet-50    : {final_ratios[0]:.4f} ({final_ratios[0]*100:.1f}%)")
        print(f" > EfficientNet : {final_ratios[1]:.4f} ({final_ratios[1]*100:.1f}%)")
        print(f" > ViT-B/16     : {final_ratios[2]:.4f} ({final_ratios[2]*100:.1f}%)")
        print("="*50)

    else:
        print("가중치를 찾을 수 없습니다. 모델 클래스 정의를 확인하세요.")