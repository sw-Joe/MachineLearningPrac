import kagglehub



def data_download():
    # Download latest version
    path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")

    print("Path to dataset files:", path)


if __name__ == "__main__":
    data_download()
