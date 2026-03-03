# 멀티 GPU 사용 학습 commandline(torchrun)
# --master_port 옵션 추가 (예: 29505)
CUDA_VISIBLE_DEVICES=2,3 uv run torchrun --nproc_per_node=2 --master_port=29505 -m {project.runfile}


# single GPU 사용 학습 commandline(uv)
CUDA_VISIBLE_DEVICES=2 uv run python -m {project.runfile}


# uv에게 .env 파일을 명시적으로 알려주기
CUDA_VISIBLE_DEVICES=2 uv run --env-file .env python -m face.run