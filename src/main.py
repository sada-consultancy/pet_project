from src.preprocessing.load_dataset import load_dataset
from src.preprocessing.preprocessing import preprocessing
from src.eda.eda import eda

def main():
    data=load_dataset()
    preprocessed_data=preprocessing(data)
    eda(preprocessed_data)

main()