import numpy as np
import PIL

from common import *

def check_file(filename):
	return path.exists(filename) and path.isfile(filename)

artifact_path = parse_arguments()

if not check_artifacts_directory(artifact_path) or not check_file(class_names_file_path(artifact_path)) or not check_file(tf_model_file_path(artifact_path)):
	raise Exception(f"artifact directory {artifact_path} is defect")

# read JSON file and parse contents
with open(class_names_file_path(artifact_path), 'r') as file:
    class_names = json.load(file)

print_label_note(class_names)  

# get a sunflower image and use it
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

# create interpreter instance
interpreter = tf.lite.Interpreter(model_path=tf_model_file_path(artifact_path))

classify_lite = interpreter.get_signature_runner('serving_default')

# run the actual inference
predictions_lite = classify_lite(sequential_input=img_array)['dense_1']
score_lite = tf.nn.softmax(predictions_lite)

print(f"The image belongs to class {class_names[np.argmax(score_lite)]} with {100 * np.max(score_lite):.2f}% probability")