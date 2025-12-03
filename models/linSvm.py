# Linear SVM Model

# File import
from models import vectorizer as vct
from src import preprocessing as prp


# Lib import
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Model Training
linSvc = LinearSVC()
linSvc.fit(vct.Tfid_X_train, prp.y_train)

print("Linear SVM Model Successfull!")

# Model Evaluate
yPred = linSvc.predict(vct.Tfid_X_test)
linSvc_score = classification_report(prp.y_test, yPred)
print(f"Linear SVC Accuracy Score:\n{linSvc_score}")