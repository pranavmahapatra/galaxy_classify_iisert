import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from dataprep import data_r, data_c
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from analysis import latexify
import seaborn as sns

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False
})

def dataSplit(data):
    temp = data.iloc[:,1:-1]
    temp = temp.drop(columns=["Mg", "log_10(Mdyn)", "log_10(Mbary)","log_10(Mgas)","log_10(M*)"])
    X = temp

    y = data['class']
    feature_names = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=67)
    return  X_train, X_test, y_train, y_test, feature_names

X_train, X_test, y_train, y_test, feature_names = dataSplit(data_r)

# Random Forest Sweep
rf_results = []      # accuracies
rf_n_list = [50, 100, 150, 200, 300, 400, 500]

best_rf_acc = -1
best_rf_params = None

for n in rf_n_list:
    clf = RandomForestClassifier(
        n_estimators=n,
        max_depth=None,
        min_samples_leaf=1,
        random_state=67,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    rf_results.append(acc)

    print(f"RF | n_estimators={n} | Acc={acc:.4f}")

    if acc > best_rf_acc:
        best_rf_acc = acc
        best_rf_params = n


print("\nBest RF n_estimators =", best_rf_params, "Accuracy =", best_rf_acc)

# Plotting RF
plt.figure()
plt.plot(rf_n_list, rf_results)
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('Estimation of optimal n_estimators for RF Classification')
plt.show()

# Final RF classification
clf = RandomForestClassifier(
    n_estimators=best_rf_params,
    max_depth=None,
    min_samples_leaf=1,
    random_state=67,
    n_jobs=-1
)

featureNamesLatex = [latexify(f) for f in feature_names]

clf.fit(X_train, y_train)
y_pred_rf = clf.predict(X_test)
importances = clf.feature_importances_
plt.figure(figsize=(8, 4))
plt.barh(featureNamesLatex, importances, color='#71C0BB')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis
plt.show()

print("\n--- Final Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

plt.figure(figsize=(7,5))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

# SVM sweep
svm_results = []
C_list = [0.1, 1, 3, 5, 10]

best_svm_acc = -1
best_svm_C = None

for C in C_list:
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=C,
            kernel='rbf',
            gamma='scale',
            decision_function_shape='ovr'
        ))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    svm_results.append(acc)

    print(f"SVM | C={C} | Acc={acc:.4f}")

    if acc > best_svm_acc:
        best_svm_acc = acc
        best_svm_C = C


print("\nBest SVM C =", best_svm_C, "Accuracy =", best_svm_acc)

# Plotting SVM
plt.figure()
plt.plot(C_list, svm_results)
plt.xlabel('C values')
plt.ylabel('Accuracy')
plt.title('Estimation of optimal C for Support Vector Machine Classification')
plt.savefig(f"optimalcsvm", dpi=400, bbox_inches="tight")
plt.close()

# Final SVM classification
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        C=best_svm_C,
        kernel='rbf',
        gamma='scale',
        decision_function_shape='ovr'
    ))
])

clf.fit(X_train, y_train)
y_pred_svm = clf.predict(X_test)

print("\n--- Final SVM ---")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))

plt.figure(figsize=(7,5))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt="d", cmap="Blues")
plt.title("SVM (RBF Kernel) - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.show()
