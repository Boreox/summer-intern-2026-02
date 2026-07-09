import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Step 1: Load MNIST Dataset
print("Loading MNIST Dataset...")

X, y = loadlocal_mnist(
    images_path="train-images.idx3-ubyte",
    labels_path="train-labels.idx1-ubyte"
)

print("Dataset Loaded Successfully!")

# Step 2: Convert Dataset to CSV
df = pd.DataFrame(X)
df["label"] = y
df.to_csv("mnist_train.csv", index=False)

print("CSV File Saved Successfully!")

# Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 4: Create Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Step 5: Train Model
print("Training Model...")
rf_model.fit(X_train, y_train)
print("Training Finished!")

# Step 6: Prediction
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Step 7: Evaluation
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average="weighted")
recall = recall_score(y_test, y_test_pred, average="weighted")

print("\n========== RESULTS ==========")
print("Training Accuracy :", train_acc)
print("Testing Accuracy  :", test_acc)
print("Precision         :", precision)
print("Recall            :", recall)

# Step 8: Predict One Image
sample = X_test[0].reshape(1, -1)
prediction = rf_model.predict(sample)

print("\nActual Label    :", y_test[0])
print("Predicted Label :", prediction[0])

# Step 9: Show Image
plt.imshow(X_test[0].reshape(28, 28), cmap="gray")
plt.title(f"Actual: {y_test[0]} | Predicted: {prediction[0]}")
plt.axis("off")
plt.show()