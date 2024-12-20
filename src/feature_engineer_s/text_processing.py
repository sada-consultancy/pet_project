import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.preprocessing.load_dataset import load_dataset
from src.preprocessing.preprocessing import preprocessing


data=load_dataset()
df=preprocessing(data)
nlp=spacy.load('en_core_web_sm')
#df=df.sample(frac=0.3)
def lemmatize_text(text):
    doc=nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct]) 

df['tokens']=df['review_body'].apply(lemmatize_text)

df.to_csv('data_final.csv',index=False)

train_df,test_df=train_test_split(df,test_size=0.2,random_state=42)

features=train_df['tokens']
target=train_df['sentiment']

features_train,features_valid,target_train,target_valid=train_test_split(features,target,test_size=0.2,random_state=42)

vectorizer=TfidfVectorizer(max_features=5000,ngram_range=(1,2))

features_train_tfidf = vectorizer.fit_transform(features_train)  # Transformamos el texto de entrenamiento
features_valid_tfidf = vectorizer.transform(features_valid)



