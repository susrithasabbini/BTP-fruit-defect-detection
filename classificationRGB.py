import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

ripened_csv = "dataset_RGB.csv"

ripened_df = pd.read_csv(ripened_csv)

combined_df = pd.concat([ripened_df], ignore_index=True)

# Encode labels
label_encoder = LabelEncoder()
combined_df["Label"] = label_encoder.fit_transform(combined_df["Label"])

# Separate features and labels
X = combined_df.drop("Label", axis=1)
y = combined_df["Label"]

# Perform 80:20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize classifiers
classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=4),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "SVM": SVC(kernel="linear", C=1.0, random_state=42),
    "LDA": LinearDiscriminantAnalysis(),
}

# Train and evaluate classifiers
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy}")

# Save results to a DataFrame
results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"])
results_df.index.name = "Classifier"
results_csv = "resultsWithDataset(RGB).csv"
results_df.to_csv(results_csv)
print(f"Results saved to {results_csv}")
