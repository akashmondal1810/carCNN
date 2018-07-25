# Convolutional Neural Network for Car Recognition
A simple convolutional neural network for *car recognizer* using `keras` library that achieves over *95%* accuracy

## Setup:
- Simply `pip install -r dependencies.txt` or  
- Install `sklearn` using `sudo pip install sklearn`
- Install `keras` using `sudo pip install keras`
- Install `skimage` using `sudo pip install scikit-image`

## Architecture:
- Conv layer
- ReLU activation
- Pool layer
- Conv layer
- ReLU activation
- Pool layer
- Fully connected layer
- Softmax layer

## Execute:
`python predict.py` for prediction

## Keras REST API :
- run the server `run_keras_server.py`
- Submit a request via cURL `curl -X POST -F image=@imagename.jpg 'http://localhost:5000/predict'`
- Submita a request via Python `python simple_request.py`

## Result:
![alt text](https://github.com/akashmondal1810/carCNN/blob/master/Screenshot%20from%202018-07-20%2001-10-40.png)
