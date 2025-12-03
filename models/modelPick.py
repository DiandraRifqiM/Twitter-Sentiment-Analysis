# Model Pick

# File Import
from models import logReg as lr
from models import vectorizer as vct

# Lib Import
import pickle

model = lr.logReg
vectorizer = vct.Tfid

# Model Saved
with open(r"D:\Python Course\Sentiment Analysis on Twitter Project\models\model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)
    
print("Model Saved Succesfull")
