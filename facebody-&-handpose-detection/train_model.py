import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import glob

# Combining all CSV files
files = glob.glob("*.csv")
data = pd.concat([pd.read_csv(f, header=None) for f in files], ignore_index=True)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print("Model accuracy:", model.score(X_test, y_test))

# Saving model
with open("pose_model.pkl", "wb") as f:
    pickle.dump(model, f)


