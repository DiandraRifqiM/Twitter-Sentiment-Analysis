# Naive Bayes Model

# File Import
from models import vectorizer as vct
from src import preprocessing as prp

# Lib Import
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Model Training
MultiNB = MultinomialNB()
MultiNB.fit(vct.Tfid_X_train, prp.y_train)

print("Naive Bayes Model Successfull!")

# Model Evaluate
yPred = MultiNB.predict(vct.Tfid_X_test)
MultiNB_score = classification_report(prp.y_test, yPred)
print(f"Naive Bayes Accuracy Score:\n{MultiNB_score}")



