from pathlib import Path



if __name__ != "__main__":
    project_path = Path(__file__).parent.parent

    SEED = 100


    ''' PATH '''
    PATH_ORIGINAL_CAT = f"{project_path}/img/original/cat/"
    PATH_ORIGINAL_DOG = f"{project_path}/img/original/dog/"
    IMG_FORMAT = "*.jpg"


    ''' HYPER PARAMETERS '''
    TOTAL_EPOCH = 50
    BATCH_SIZE = 32