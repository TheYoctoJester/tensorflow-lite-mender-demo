import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

import argparse
import json

from os import mkdir, path

IMG_HEIGHT = 180
IMG_WIDTH = 180

__ARTIFACTS_DIRECTORY_PATH = 'artifacts'

def tf_model_file_path(artifact_path=__ARTIFACTS_DIRECTORY_PATH):
	return artifact_path + '/model.tflite' # The default path to the saved TensorFlow Lite model

def class_names_file_path(artifact_path=__ARTIFACTS_DIRECTORY_PATH):
	return artifact_path + '/class_names.json' # The default path to the saved class names list

def check_artifacts_directory(artifact_path=__ARTIFACTS_DIRECTORY_PATH):
	return path.isdir(artifact_path)

def assert_artifacts_directory(artifact_path=__ARTIFACTS_DIRECTORY_PATH):
	if not path.exists(artifact_path):
		mkdir(artifact_path)
	if not path.isdir(artifact_path):
		raise Exception(f"artifact path {artifact_path} could not be created or is not a directory")

def print_label_note(class_names):
	print(f"classification will use the labels {class_names}")

def parse_arguments():
	argParser = argparse.ArgumentParser()
	argParser.add_argument(
		"-a",
		"--artifacts",
		help="the artifact directory providing class_names.json and model.tflite",
		default=[__ARTIFACTS_DIRECTORY_PATH],
		nargs=1
	)

	args = argParser.parse_args()
	artifact_path = args.artifacts[0]
	print(f"using artifacts directory {artifact_path}")
	return artifact_path
