import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV files
train_csv = "train.csv"
test_csv = "test.csv"
val_csv = "val.csv"

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
val_df = pd.read_csv(val_csv)

# Combine all three datasets
combined_df = pd.concat([train_df, test_df, val_df], ignore_index=True)

# Encode labels
label_encoder = LabelEncoder()
combined_df["Label"] = label_encoder.fit_transform(combined_df["Label"])

# Separate features and labels
X = combined_df.drop("Label", axis=1)
y = combined_df["Label"]

# Impute missing values (NaN) with mean of the column
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Perform 80:20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=37
)

# Initialize classifiers
classifiers = {
    # "KNN": KNeighborsClassifier(n_neighbors=4),
    # "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "SVM": SVC(kernel="linear", C=1.0, random_state=42),
    # "LDA": LinearDiscriminantAnalysis(),
}

# Train and evaluate classifiers
results = {}
precision = {}
recall = {}
f1 = {}
confusion_matrices = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    precision[name] = precision_score(y_test, y_pred, average="weighted")
    recall[name] = recall_score(y_test, y_pred, average="weighted")
    f1[name] = f1_score(y_test, y_pred, average="weighted")
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")

# Save results to a DataFrame
results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"])
results_df.index.name = "Classifier"
results_csv = "results/thermal/resultsWithGLCMAndColor.csv"
results_df.to_csv(results_csv)
print(f"Results saved to {results_csv}")

# Function to plot a single confusion matrix with larger font size
def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.5)  # Increase the font size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 23})  # Adjust annotation size
    plt.title(title, fontsize=22)
    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    plt.show()

# Plot confusion matrix for the SVM classifier
plot_confusion_matrix(confusion_matrices["SVM"], title="SVM Confusion Matrix for thermal images")

# Print precision, recall, and F1-score for SVM
print("\nMetrics for SVM:")
print(f"  Precision: {precision['SVM']}")
print(f"  Recall: {recall['SVM']}")
print(f"  F1-score: {f1['SVM']}")

