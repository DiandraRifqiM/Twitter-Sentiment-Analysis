
# Lib Import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Tweet Dataset
tweetDf = pd.read_csv(r"D:\Python Course\Sentiment Analysis on Twitter Project\data\processed\Cleaned_1.6M-Tweets.csv",
                      encoding='utf-8')
# print(tweetDf.columns)

# Train Test Split
X = tweetDf["text"]
y = tweetDf["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print("Train-test split completed!")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)