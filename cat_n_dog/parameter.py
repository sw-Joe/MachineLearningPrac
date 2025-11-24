SEED = 100


''' PATH '''
PATH_ORIGINAL_CAT = "imgs/original/cat/"
PATH_ORIGINAL_DOG = "imgs/original/dog/"
PATH_TRAIN = "img/splited/train/"
PATH_SAVE = './cat_n_dog_CNN.pth'
IMG_FORMAT = "*.jpg"


''' HYPER PARAMETERS '''
TOTAL_EPOCH = 50
BATCH_SIZE = 128    # batch가 작으면 GPU 사용 효과(병렬 연산의 장점)를 살리기 어려움