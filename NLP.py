import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

Friendcsv = 'data/friendMessages.csv'  #Messages from friend (label 1)
Othercsv = 'data/otherMessages.csv'    #Messages from others (label 0)
Combinedcsv = 'data/combinedMessages.csv'

Frienddf = pd.read_csv(Friendcsv)
Otherdf = pd.read_csv(Othercsv)

Frienddf.columns = Frienddf.columns.str.strip()
Otherdf.columns = Otherdf.columns.str.strip()

# Convert all entries in 'content' to strings
Frienddf['Content'] = Frienddf['Content'].astype(str)
Otherdf['Content'] = Otherdf['Content'].astype(str)

Frienddf['Label'] = 1
Otherdf['Label'] = 0

#Join both DataFrames
df = pd.concat([Frienddf, Otherdf], ignore_index=True)

df.to_csv(Combinedcsv, index=False)
print(f'Combined CSV saved to: {Combinedcsv}')

df.columns = df.columns.str.strip()
df['Content'] = df['Content'].astype(str) #turns to string

def preprocess(text):
    #Lowercasee
    text = text.lower()

    #Excludes embeds
    if text.startswith('https://'):
        return None  

    return text

df['ProcessedContent'] = df['Content'].apply(preprocess)

df = df.dropna(subset=['ProcessedContent'])

#splitting dataset
X = df['ProcessedContent']
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, 'model.pkl')
