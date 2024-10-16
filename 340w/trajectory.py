import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv('sample_trajectory_1.txt', sep='\s+', header=None)
# Assign column names based on the data structure
column_names = ['timestamp', 'object_id', 'speed_x', 'speed_y', 'object_type', 'position_x', 'position_y', 'position_z', 'heading']
data.columns = column_names
print(data.columns)

# Features and target
features = ['speed_x', 'speed_y', 'position_x', 'position_y', 'heading']
target = 'object_type'

# Split the data into features (X) and target (y)
X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))