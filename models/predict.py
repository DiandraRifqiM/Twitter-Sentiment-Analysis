# Predict

# Lib Import
import pickle
import sys

# Model Import
def loadModel(path=r"D:\Python Course\Sentiment Analysis on Twitter Project\models\model.pkl"):
    with open(path, 'rb') as f:
        return pickle.load(f)
        
def main():
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"your text here\"")
        sys.exit(1)
        
    text = " ".join(sys.argv[1:])
    
    # Load Vectorizer & Model
    model, vectorizer = loadModel()

    # Transform text
    X = vectorizer.transform([text])
    
    # Predict
    predict = model.predict(X)[0]

    # Probability (if supported)
    try:
        prob = model.predict_proba(X)[0]
        print("Prediction:", predict)
        print("Probability:", prob)
    except:
        print("Prediction:", predict)

if __name__ == "__main__":
    main()
    

