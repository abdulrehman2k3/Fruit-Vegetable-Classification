
import tensorflow as tf
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load trained model
model = tf.keras.models.load_model(
    "model/fruit_vegetable_model.keras"
)

# 2. Load class names
with open("model/class_names.json", "r") as file:
    class_names = json.load(file)

# 3. Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(128, 128),
    batch_size=32,
    label_mode="int",
    shuffle=False
)

# 4. Evaluate model
test_loss, test_accuracy = model.evaluate(test_ds)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")

# 5. Get actual and predicted labels
y_true = np.concatenate([
    labels.numpy() for images, labels in test_ds
])

predictions = model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)

# 6. Classification report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
))

# 7. Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(18, 15))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Fruit and Vegetable Classification")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig("model/confusion_matrix.png")
print("\nConfusion matrix saved successfully!")
