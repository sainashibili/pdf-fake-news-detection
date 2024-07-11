import numpy as np
import pandas as pd
import itertools
from pypdf import PdfReader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

################################################################
#CHANGE FILENAME HERE
#MAKE SURE THE FILE IS IN THE SAME DIRECTORY AS THIS PYTHON FILE
################################################################
FILENAME = "example.pdf"

################################################################
#READS THE PDF IN AND SAVES IT AS A STRING
################################################################
reader = PdfReader(FILENAME)

#saving the text into a string
article = ""
for i in range(len(reader.pages)):
    page = reader.pages[i]
    text = page.extract_text()
    article += text

#replaces newline characters with spaces
article = ' '.join(article.splitlines())

################################################################
#TRAINING THE MODEL
################################################################
#read the data
df = pd.read_csv('news.csv')

#get the labels
labels = df.label

#split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    df['text'], labels, test_size=0.2, random_state=7)

#create a Tfidfvectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

#fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

#create a PassiveAggressiveClassifer
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

################################################################
#PREDICTING THE PDF 
################################################################
#creating a dataframe with the text
pdf = pd.DataFrame([ {"text" : article} ])
tfidf_pdf = tfidf_vectorizer.transform(pdf)

#predict the model
pdf["label"] = pac.predict(tfidf_pdf)

#print the outcome
print(pdf)
