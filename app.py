import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from PIL import Image, ImageTk
import numpy as np

class ModelTester:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Tester GUI")

        # Load pre-trained model
        self.model = load_model('catvsdog_model.h5')

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Upload Image button
        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        # Display Image
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        # Predict button
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_image)
        self.predict_button.pack(pady=10)

        # Prediction result
        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            image = Image.open(file_path)
            image = image.resize((224, 224))
            self.image = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.image)

            # Save the image path
            self.image_path = file_path
            self.result_label.config(text="")

    def predict_image(self):
        if hasattr(self, 'image_path'):
            # Read the uploaded image
            image = Image.open(self.image_path)
            image = image.resize((224, 224))
            image = np.array(image)
            image = image / 255.0
            image = image.reshape((1, 224, 224, 3))

            # Make predictions
            prediction = self.model.predict(image)

            # Interpret the prediction
            confidence = prediction[0, 0] if prediction[0, 0] > 0.5 else 1 - prediction[0, 0]
            class_label = "Dog" if prediction[0, 0] > 0.5 else "Cat"

            # Display the result with confidence
            result_text = f"Prediction: {class_label}\nConfidence: {confidence:.2%}"
            self.result_label.config(text=result_text)
        else:
            self.result_label.config(text="Please upload an image first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelTester(root)
    root.mainloop()
