import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFile
import joblib

# Prevent crashing on corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Suppress performance info logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Dataset paths
data_path = r'Training and Validation'
test_path = r'Testing'

# Step 1: Data Generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    data_path, target_size=(64, 64), batch_size=32, class_mode='binary', subset='training'
)
val_generator = datagen.flow_from_directory(
    data_path, target_size=(64, 64), batch_size=32, class_mode='binary', subset='validation'
)
test_generator = test_datagen.flow_from_directory(
    test_path, target_size=(64, 64), batch_size=32, class_mode='binary'
)

class_names = list(train_generator.class_indices.keys())

# Step 2: CNN Model
model = Sequential([
    Input(shape=(64, 64, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train CNN
history = model.fit(train_generator, validation_data=val_generator, epochs=10)

# Evaluate CNN
loss, cnn_acc = model.evaluate(test_generator)
print(f"CNN Test Accuracy: {cnn_acc*100:.2f}%")

# Step 3: Random Forest Baseline
print("\nTraining Random Forest baseline...")

# Extract data for RF
X_train, y_train = [], []
for batch, labels in train_generator:
    X_train.extend(batch.reshape(batch.shape[0], -1))
    y_train.extend(labels)
    if len(X_train) >= train_generator.samples:
        break

X_test, y_test = [], []
for batch, labels in test_generator:
    X_test.extend(batch.reshape(batch.shape[0], -1))
    y_test.extend(labels)
    if len(X_test) >= test_generator.samples:
        break

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test   = np.array(X_test), np.array(y_test)

# Train RF
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate RF
y_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred)
print(f"RandomForest Test Accuracy: {rf_acc*100:.2f}%")

# Step 4: Comparative Analysis
print("\n=== Comparative Results ===")
print(f"CNN Accuracy        : {cnn_acc*100:.2f}%")
print(f"RandomForest Accuracy: {rf_acc*100:.2f}%")

# Step 5: Visualizations
# CNN Learning Curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("CNN Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("CNN Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# RandomForest Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("RandomForest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Classification Report
print("\nRandomForest Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

# Step 6: Save CNN Model
model.save('models/fire_nofire_model.h5')
joblib.dump(model, "fire_nofire_rf_model.pkl")

# Step 7: GUI for CNN Prediction
def load_image():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            img = Image.open(file_path).convert("RGB").resize((64, 64))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0][0]
            label = "✅ No Fire" if prediction > 0.5 else "🔥 Fire"
            result_label.config(text=f"Predicted: {label}")

            img_tk = ImageTk.PhotoImage(img)
            image_label.config(image=img_tk)
            image_label.image = img_tk
    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict.\n{str(e)}")

root = tk.Tk()
root.title("Fire Detection Classifier")
root.geometry("400x400")
root.configure(bg="white")

btn = tk.Button(root, text="Select Image", command=load_image, bg="#FF5722", fg="white", font=("Arial", 12))
btn.pack(pady=15)

image_label = tk.Label(root, bg="white")
image_label.pack()

result_label = tk.Label(root, text="", font=("Arial", 14), bg="white")
result_label.pack(pady=10)

root.mainloop()

