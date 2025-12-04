# Logistin Regression Model

# File import
from models import vectorizer as vct
from src import preprocessing as prp
from logs import logger as lg

# Lib import
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Model Training
logReg = LogisticRegression()
logReg.fit(vct.Tfid_X_train, prp.y_train)

print("Logistic Regression Model Successfull")

# Model Evaluate
y_pred = logReg.predict(vct.Tfid_X_test)
logReg_score = classification_report(prp.y_test, y_pred)
print(f"Logistic Regression Accuracy Score:\n{logReg_score}")

# Model Evaluate Saved
lg.log_report("Logistic Regerssion Accuracy Report", logReg_score)
