import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score


# Load train, val, and test CSV files
train_csv = "histogramTrain.csv"
val_csv = "histogramVal.csv"
test_csv = "histogramTest.csv"

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

# Encode labels
label_encoder = LabelEncoder()
train_df["Label"] = label_encoder.fit_transform(train_df["Label"])
val_df["Label"] = label_encoder.transform(val_df["Label"])
test_df["Label"] = label_encoder.transform(test_df["Label"])

# Separate features and labels
X_train = train_df.drop("Label", axis=1)
y_train = train_df["Label"]

X_val = val_df.drop("Label", axis=1)
y_val = val_df["Label"]

X_test = test_df.drop("Label", axis=1)
y_test = test_df["Label"]

# Initialize classifiers
classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="linear", C=1.0, random_state=42),
    "LDA": LinearDiscriminantAnalysis(),
}

# Train and evaluate classifiers
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)

    # Now train on train + val and test on test
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    clf.fit(X_train_full, y_train_full)
    y_pred_test = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    results[name] = (val_accuracy, test_accuracy)
    print(
        f"{name} - Validation Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}"
    )

# Save results to a DataFrame
results_df = pd.DataFrame.from_dict(
    results, orient="index", columns=["Validation Accuracy", "Test Accuracy"]
)
results_df.index.name = "Classifier"
results_csv = "resultsWithHistogramTrainVal.csv"
results_df.to_csv(results_csv)
print(f"Results saved to {results_csv}")
