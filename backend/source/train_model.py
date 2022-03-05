import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from dataset import feature_creator
from dataset import extractor


df = feature_creator()

X = np.array(df["features"].tolist())
y = np.array(df["class"].tolist())
y = np.array(pd.get_dummies(y))
X_train, X_test, y_train, y_test = train_test_split(X,y)
model = KNeighborsClassifier()
model = model.fit(X_train,y_train)

audio = extractor("models/dashboard(0).wav")

print(model.predict(audio))






