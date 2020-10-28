from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras
from tensorflow.keras.optimizers import SGD
import os.path
import sklearn.metrics as metrics
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tkinter import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class MyThresholdCallback(tensorflow.keras.callbacks.Callback):
	def __init__(self, threshold):
		super(MyThresholdCallback, self).__init__()
		self.threshold = threshold

	def on_epoch_end(self, epoch, logs=None): 
		val_acc = logs["val_accuracy"]
		if val_acc >= self.threshold:
			self.model.stop_training = True

def model_train(model_add , x_train, y_train, x_test, y_test):

	# build a sequential model
	model = Sequential()
	model.add(InputLayer(input_shape=(512, 512, 1)))

	# 1st conv block
	model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
	model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
	# 2nd conv block
	model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
	model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
	model.add(BatchNormalization())
	# 3rd conv block
	model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
	model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
	model.add(BatchNormalization())
	# ANN block
	model.add(Flatten())
	model.add(Dense(units=100, activation='relu'))
	model.add(Dense(units=100, activation='relu'))
	model.add(Dropout(0.25))
	# output layer
	model.add(Dense(units=2, activation='softmax'))

	#optimizer = keras.optimizers.Adam(lr=0.01)
	#opt = SGD(lr=1)
	if os.path.isfile(model_add):
		print('Found saved model, loading now')
		user_choice = messagebox.askyesno('Found saved model' , 'You want to train again?')
		if(user_choice == True):
			model.summary()
			model.compile(loss='categorical_crossentropy', optimizer = 'adamax', metrics=['accuracy'])
			# fit on data for 30 epochs
			es = MyThresholdCallback(threshold=0.89)
			history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=100 , batch_size = 1, callbacks=[es])
			plot_acc(history)
			plot_loss(history)
			model.save(model_add) 
		else:
			model = tensorflow.keras.models.load_model(model_add)
			model.summary()

	else:
		print('No saved model found, fitting new model with the data')
		print('Saving to ' + model_add)
		model.summary()
		model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
		es = MyThresholdCallback(threshold=0.89)
		history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=100 , batch_size = 1, callbacks=[es])
		# fit on data for 30 epochs
		plot_acc(history)
		plot_loss(history)
		model.save(model_add) 
	return model


def plot_acc(history):

	print(history.history.keys())
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('epoch')
	plt.legend(['Train', 'Val'], loc='upper left')
	plt.show()
	return

def plot_loss(history):

	print(history.history.keys())
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(loss) + 1)
	plt.plot(epochs, loss, color='red', label='Training loss')
	plt.plot(epochs, val_loss, color='green', label='Validation loss')
	plt.title('Training and validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()
	return

def load_images(images_train , images_val):
	# create a new generator
	imagegen = ImageDataGenerator()
	# load train data
	train = imagegen.flow_from_directory(images_train, class_mode="categorical", color_mode="grayscale", batch_size = 40, shuffle = True, target_size=(512, 512))
	val   = imagegen.flow_from_directory(images_val, class_mode="categorical", color_mode="grayscale", batch_size = 40, shuffle = True, target_size=(512, 512))
	#train = imagegen.flow_from_directory("C:/MUSP_Local/MUSP_Data/New_data/Images", class_mode="categorical", color_mode="rgb" ,shuffle=False, batch_size=128, target_size=(224, 224))
	#val   = imagegen.flow_from_directory("C:/MUSP_Local/MUSP_Data/New_data/Images", class_mode="categorical", color_mode="rgb" ,shuffle=False, batch_size=128, target_size=(224, 224))

	x_train = train[0][0]
	y_train = train[0][1]

	x_test  = val[0][0]
	y_test  = val[0][1]

	print('High Feed images are hot encoded onto 0-Class')
	print('Low Feed images are hot encoded onto 1-Class')
	print('Shape of x_train' , x_train.shape )
	print('Shape of y_train' , y_train )
	print('Shape of x_test'  , x_test.shape )
	print('Shape of y_test'  , y_test )

	return x_train, y_train, x_test, y_test

def conf_mat(model,x_test,y_test):

	y_pred_ohe = model.predict(x_test)  # shape=(n_samples, 12)
	y_pred_labels = np.argmax(y_pred_ohe, axis=1)  # only necessary if output has one-hot-encoding, shape=(n_samples)
	print(y_pred_labels)

	y_true_labels = np.argmax(y_test, axis=1)  # only necessary if output has one-hot-encoding, shape=(n_samples)
	print(y_true_labels)

	cm = confusion_matrix(y_true_labels ,  y_pred_labels)  # shape=(12, 12)
	disp = ConfusionMatrixDisplay(cm , display_labels = ['.15 .2 [mm/tooth] \n High ' , '.1 [m/min]\n Low'])
	disp.plot(cmap = plt.cm.Blues)
	disp.ax_.set(title = 'Confusion matrix for High and Low \n Feed per tooth [fz]', xlabel='Predicted [fz] Class', ylabel='True [fz] Class' )
	plt.show()

	return


model_add    = 'C:/MUSP_Local/Keras_models/CNN_Fz.h5'
images_train = 'C:/MUSP_Local/MUSP_Data/New_data/Images/Hi_Lo_Fz/Train'
images_val   = 'C:/MUSP_Local/MUSP_Data/New_data/Images/Hi_Lo_Fz/Val'

x_train, y_train, x_test, y_test = load_images(images_train, images_val)
model = model_train(model_add , x_train, y_train, x_test, y_test)
#model.save_weights("C:/MUSP_Local/Keras_models/Hi_Lo_fz/CNN_Fz_weights")

conf_mat(model ,x_test,y_test)
