# Email Spam Classification using SVM and KNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load dataset
df = pd.read_csv("emails.csv")
df.head()
# Features and labels
X = df.iloc[:, 1:3001]   # word frequency features
y = df.iloc[:, -1]       # target: 1 = spam, 0 = not spam
# Split dataset (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Feature scaling (important for KNN & SVM)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
# -------- Support Vector Machine --------
svm = SVC(kernel='rbf', C=1.0, gamma='auto')
svm.fit(X_train_s, y_train)
svm_pred = svm.predict(X_test_s)
print("=== Support Vector Machine (SVM) ===")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
print("Classification Report:\n", classification_report(y_test, svm_pred))
# -------- K-Nearest Neighbors --------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_s, y_train)
knn_pred = knn.predict(X_test_s)
print("\n=== K-Nearest Neighbors (KNN) ===")
print("Accuracy:", accuracy_score(y_test, knn_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print("Classification Report:\n", classification_report(y_test, knn_pred))