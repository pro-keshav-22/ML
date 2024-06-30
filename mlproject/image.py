import numpy as np
from PIL import Image,ImageOps
import io
_params = np.load("parameters.npz")
W1, b1, W2, b2, W3, b3 = _params["W1"], _params["b1"], _params["W2"], _params["b2"], _params["W3"], _params["b3"]
def relu(Z):
    return np.maximum(Z, 0)
def predict_classes(A3):
    return np.argmax(A3, axis=0)
def softmax(Z):
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=0)
def forward_propagation(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = relu(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3
def process_image(image_path,target_size=(28,28)):
    img = Image.open(image_path)
    
    # Resize the image
    img_resized = img.resize(target_size)
    
    # Convert the image to grayscale
    img_gray = img_resized.convert('L')
    img_gray= ImageOps.equalize(img_gray)
    # Convert the image to a numpy array
    img_array = np.array(img_gray)
    
    # Reshape the array to 1D array of size 784
    img_1d = img_array.reshape(-1)
  
    img_1d=np.where(img_1d>20,255,img_1d)
    img_1d=np.where(1,255-img_1d,0)
    column_vector = img_1d.reshape(-1, 1)
    arry=column_vector.astype(float)/255.
    _, _, _, _, _, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, arry)
    return predict_classes(A3)[0]
print(process_image("6t.jpg"))