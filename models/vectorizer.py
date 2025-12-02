# Import File
from src import preprocessing as prp

# Lib Import
from sklearn.feature_extraction.text import TfidfVectorizer

# Fillna Train Test Data
prp.X_train = prp.X_train.fillna("").astype(str)
prp.X_test = prp.X_test.fillna("").astype(str)

# Vectorizer 
Tfid= TfidfVectorizer(stop_words="english")

# Train Test Vectorizer
X_train = Tfid.fit_transform(prp.X_train)
X_test = Tfid.transform(prp.X_test)

print("Train Test Vectorizer Complete")

# print("NaN in X_train:", prp.X_train.isna().sum())
# print(type(prp.X_train))
# print(prp.X_train.head())
# print(prp.X_train.apply(type).value_counts())