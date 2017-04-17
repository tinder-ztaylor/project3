import numpy as np
import pandas as pd
import csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/intern_sample_dataset.csv', delimiter=',')

ids = []
vectors = []
genders = []
with open('vec_reps.csv', 'rb') as file:
    for line in csv.reader(file):
        curID = line[0]
        gender = int(data.loc[data["uid"] == curID]["gender"])
        ids.append(curID)
        vectors.append(line[1:])
        genders.append(gender)

with open('data/valid_ids.txt', 'wb') as f:
    for uid in ids:
        f.write(uid + '\n')


X_train, X_test, y_train, y_test = train_test_split(np.asarray(vectors), np.asarray(genders), test_size=0.25, random_state=5)
model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Percentage male:", 1 - float(sum(genders))/len(genders))
