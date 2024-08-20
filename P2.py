import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

file_path = "./nba_stats.csv"
nba_data = pd.read_csv(file_path)

columns_to_keep =['ORB', 'DRB', 'PF', 'TOV', 'AST', 'FGA', 'STL', '3PA', 'FTA', '2P', 'PTS', '3P%', 'FG', 'FT%']
X= nba_data[columns_to_keep]
target = nba_data['Pos'] 
train_feature, test_feature, train_class, test_class = train_test_split(X,target, test_size=0.2,stratify=target, random_state=0)

linearsvm = SVC(random_state=0 ,kernel= 'linear')
linearsvm.fit(train_feature, train_class)
print("SVM CLASSSIFICATION")
print("Training set score: {:.3f}".format(linearsvm.score(train_feature, train_class)))
print("Validation set score: {:.3f}".format(linearsvm.score(test_feature, test_class)))

train_prediction = linearsvm.predict(train_feature)
test_prediction = linearsvm.predict(test_feature)
print("CONFUSION MATRIX TRAINING SET")
print(pd.crosstab(train_class, train_prediction, rownames=['True'], colnames=['Predicted'], margins=True))
print("CONFUSION MATRIX VALIDATION SET")
print(pd.crosstab(test_class, test_prediction, rownames=['True'], colnames=['Predicted'], margins=True))

dummy_test_path = "dummy_test.csv"
dummy_test_data = pd.read_csv(dummy_test_path)
dummy_test_features = dummy_test_data[columns_to_keep]
dummy_test_predictions = linearsvm.predict(dummy_test_features)
accuracy = accuracy_score(dummy_test_data['Pos'], dummy_test_predictions)
print("DUMMY TEST SET ACCURACY")
print(f"Accuracy on Dummy test data: {accuracy:.3f}")

print("CONFUSION MATRIX DUMMY TEST SET")
print(pd.crosstab(dummy_test_data['Pos'], dummy_test_predictions, rownames=['True'], colnames=['Predicted'], margins=True))

scores = cross_val_score(linearsvm, X, target, cv=10)
print("CROSS VALIDATION")
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.4f}".format(scores.mean()))