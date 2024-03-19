import pandas as pd

# fetch dataset
def fetch_data_source(filepath):
    try:
        # load the dataset
        df = pd.read_csv(filepath)
        print("dataset loaded \n")
        return df
    except Exception as e:
        print(e, "error loading the dataset")
        return None
