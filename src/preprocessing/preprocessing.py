from src.preprocessing.load_dataset import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import os

data=load_dataset()    

def labeling(data):
    if data <=2:
        return 'Negative'
    elif data<4:
        return 'Neutral'
    else:
        return 'Positive'

def preprocessing(data):
    # Drop nan values
    data.dropna(inplace=True)

    data['sentiment']=data['star_rating'].apply(labeling)
    data[['star_rating','sentiment']]
    # Duplicates
    data.duplicated().sum()
    # 0 duplicates
    return data


