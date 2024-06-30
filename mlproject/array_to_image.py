from PIL import Image
import numpy as np

def array_to_image(array, output_path):
    # Reshape the array to 2D
    array_2d = array.reshape((28, 28))
    
    # Convert the numpy array to an image
    img = Image.fromarray(array_2d)
   
    # Save the image to the output path
    img.show()
    return img

# Example usage:
array = np.random.rand(784) * 255  # Example numpy array of shape (784,)
output_path = "output_image.jpg"  # Output path for the image
array_to_image(array, output_path)
