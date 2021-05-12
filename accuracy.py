import pandas as pd
from sklearn.metrics import accuracy_score

pred = pd.read_csv("test_predictions.csv", header = None)
test = pd.read_csv("test_label.csv", header = None)

print(pred[0])
print(test[0])

print(accuracy_score(pred[0], test[0]))