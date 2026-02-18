import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

app = Flask(__name__)

# Load model
model = load_model("cotton_disease_resnet50.h5")

# Class labels (MUST match training order)
class_names = [
    "Alternaria Leaf Spot",
    "Bacterial Blight",
    "Fusarium Wilt",
    "Healthy Leaf",
    "Verticillium Wilt"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    
    if request.method == "POST":
        file = request.files["file"]
        
        if file:
            img = Image.open(file).convert("RGB")
            img = img.resize((224, 224))
            
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            preds = model.predict(img_array)
            prediction = class_names[np.argmax(preds)]
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
