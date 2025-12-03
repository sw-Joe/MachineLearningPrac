from pathlib import Path



project_path = Path(__file__).parent.parent

SEED = 100


''' PATH '''
PATH_ORIGINAL_CAT = f"{project_path}/img/original/cat/"
PATH_ORIGINAL_DOG = f"{project_path}/img/original/dog/"
IMG_FORMAT = "*.jpg"


''' HYPER PARAMETERS '''
TOTAL_EPOCH = 50
BATCH_SIZE = 32




ROOT = Path(__file__).resolve().parents[2]  # → ul/
DATA_DIR = ROOT / "cat_n_dog"               # 원하는 경로
