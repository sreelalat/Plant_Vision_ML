
# importing libraries 
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
from tensorflow.contrib import lite


img_width, img_height = 224, 224

train_data_dir = '/content/drive/My Drive/Pure_Dataset_Banana'
validation_data_dir = '/content/drive/My Drive/Pure_Dataset_Banana'
nb_train_samples = 500
nb_validation_samples = 200
epochs = 10
batch_size = 32
input_shape = (img_width, img_height, 3) 



if K.image_data_format() == 'channels_first': 
	input_shape = (3, img_width, img_height) 
else: 
	input_shape = (img_width, img_height, 3) 


#model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same",input_shape=input_shape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation("softmax"))

model.summary()




model.compile(loss ='categorical_crossentropy', 
					optimizer ='Adam', 
				metrics =['accuracy'])



train_datagen = ImageDataGenerator( 
				rescale = 1. / 255, 
				shear_range = 0.2, 
				zoom_range = 0.2, 
			horizontal_flip = True)


test_datagen = ImageDataGenerator(rescale = 1. / 255) 


datagen = ImageDataGenerator(
    data_format=K.image_data_format(),
    validation_split = 0.2)

train_generator = train_datagen.flow_from_directory(train_data_dir, 
							target_size =(img_width, img_height), 
					batch_size = batch_size, class_mode ='categorical')



validation_generator = test_datagen.flow_from_directory( 
									validation_data_dir, 
				target_size =(img_width, img_height), 
		batch_size = batch_size, class_mode ='categorical') 




history = model.fit_generator(train_generator, 
	steps_per_epoch = nb_train_samples, 
	epochs = epochs, validation_data = validation_generator, 
	validation_steps = nb_validation_samples) 



plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# scores = model.evaluate(x_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

#save model
keras_file = "model_leaf.h5"
model.save_weights("/content/drive/My Drive/model_leaf.h5")
print("Saved model to disk")

model.save("/content/drive/My Drive/model2_leaf.h5")
print("model 2 is saved")

model_json = model.to_json()
with open("/content/drive/My Drive/model_face.json", "w") as json_file:
    json_file.write(model_json)




converter = lite.TFLiteConverter.from_keras_model_file("/content/drive/My Drive/model2_leaf.h5")
tflite_model = converter.convert()
open("/content/drive/My Drive/model2_leaf.tflite", "wb").write(tflite_model)