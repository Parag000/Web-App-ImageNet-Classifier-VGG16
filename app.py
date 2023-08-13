from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16

app = Flask(__name__)

UPLOAD_FOLDER = 'Images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = VGG16(weights='imagenet')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'imageUpload' not in request.files:
            return redirect(request.url)

        image_file = request.files['imageUpload']
        if image_file.filename == '':
            return redirect(request.url)

        if image_file:
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)

            image = Image.open(image_path)
            image = image.resize((224, 224))
            image = np.array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            yhat = model.predict(image)
            label = decode_predictions(yhat)
            label = label[0][0]
            classification = '%s (%.2f%%)' % (label[1], label[2] * 100)

            return render_template('index.html', prediction=classification, image_path = image_path)

    return render_template('index.html')

if __name__ == "__main__":
    app.run()
