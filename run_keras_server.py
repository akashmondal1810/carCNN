

# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np
import flask
import io

img_width, img_height = 50, 50

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model2():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	model = load_model('model.h5')
	model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			images = flask.request.files.get("image")

			# preprocess the image and prepare it for classification
			img = image.load_img(images, target_size=(img_width, img_height))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)

			# classify the input image and then initialize the list
			# of predictions to return to the client
			images = np.vstack([x])
			classes = model.predict_classes(images, batch_size=10)
			data["predictions"] = []

			# loop over the results and add them to the list of
			# returned predictions
			for imagenetID in classes:
				r = {"label": str(imagenetID)}
				data["predictions"].append(r)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model2()
	app.run()