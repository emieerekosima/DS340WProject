import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import os
import matplotlib.pyplot as plt

#data files
traindatadir = "340w/prediction_train"
testdata = "340w/prediction_test/prediction_test.txt"

# Load and concatenate all training data files
traindatalist = []
for file_name in os.listdir(traindatadir):
    if file_name.endswith('.txt'):
        file_path = os.path.join(traindatadir, file_name)
        df = pd.read_csv(file_path, sep='\s+', header=None)
        traindatalist.append(df)

# Combine all DataFrames into one
traindata = pd.concat(traindatalist, ignore_index=True)

# Assign column names based on the data structure
column_names = ['timestamp', 'object_id', 'object_type', 'position_x', 'position_y', 'position_z', 'object_length', 'object_width', 'object_height', 'heading']
traindata.columns = column_names
traindata['object_type'] = traindata['object_type'].astype(int)
print(traindata.columns)

testdata = pd.read_csv(testdata, sep='\s+', header=None)
testdata.columns = column_names
print(testdata.columns)
# Features and target
features = ['position_x', 'position_y', 'heading', 'object_length', 'object_width', 'object_height',]
target = 'object_type'

# Split the data into features (X) and target (y)
Xtrain = traindata[features]
ytrain = traindata[target]

Xtest = testdata[features]
ytest = testdata[target]
# Split the data into training and testing sets and validation
X_train, X_val, y_train, y_val = train_test_split(Xtrain, ytrain, test_size=0.2, random_state=42)

# Initialize Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Evaluate the model on the validation set
y_val_pred = clf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))

# Make predictions on the test set
y_pred = clf.predict(Xtest)

# Evaluate the model
print("Accuracy:", accuracy_score(ytest, y_pred))
print("Classification Report:\n", classification_report(ytest, y_pred))

#Confusion
confusion = confusion_matrix(ytest, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()