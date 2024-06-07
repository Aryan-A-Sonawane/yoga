import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model

def load_data():
	is_init = False
	label = []
	dictionary = {}
	c = 0
	X, y = None, None

	# Loop through the files in the current directory
	for filename in os.listdir():
		if filename.endswith(".npy") and not filename.startswith("labels"):
			# Load the data from the .npy file
			data = np.load(filename)
			size = data.shape[0]

			# Initialize or concatenate the dataset
			if not is_init:
				is_init = True
				X = data
				y = np.array([filename.split('.')[0]] * size).reshape(-1, 1)
			else:
				X = np.concatenate((X, data))
				y = np.concatenate((y, np.array([filename.split('.')[0]] * size).reshape(-1, 1)))

			# Update labels and dictionary
			label.append(filename.split('.')[0])
			dictionary[filename.split('.')[0]] = c
			c += 1

	# Convert labels to numeric values using the dictionary
	for i in range(y.shape[0]):
		y[i, 0] = dictionary[y[i, 0]]
	y = np.array(y, dtype="int32")

	# One-hot encode the labels
	y = to_categorical(y)

	return X, y, label

def shuffle_data(X, y):
	# Shuffle the dataset
	indices = np.arange(X.shape[0])
	np.random.shuffle(indices)
	X_new = X[indices]
	y_new = y[indices]
	return X_new, y_new

def create_model(input_shape, output_shape):
	# Define the neural network model
	ip = Input(shape=(input_shape,))
	m = Dense(128, activation="tanh")(ip)
	m = Dense(64, activation="tanh")(m)
	op = Dense(output_shape, activation="softmax")(m)
	model = Model(inputs=ip, outputs=op)

	# Compile the model
	model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])
	return model

if __name__ == "__main__":
	# Load and shuffle the data
	X, y, label = load_data()
	X, y = shuffle_data(X, y)

	# Create and train the model
	model = create_model(X.shape[1], y.shape[1])
	model.fit(X, y, epochs=80)

	# Save the trained model and labels
	model.save("model.h5")
	np.save("labels.npy", np.array(label))
	print("Model and labels saved successfully.")
