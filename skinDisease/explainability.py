import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model


model = tf.keras.models.load_model('DWICNN.h5')


target_layer = model.get_layer('separable_conv2d_14')


grad_model = Model(inputs=model.inputs, outputs=[target_layer.output, model.output])


img_path = 'HAM10000/train/bcc/ISIC_0024360.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

with tf.GradientTape() as tape:
    conv_output, predictions = grad_model(x)
    loss = predictions[:, np.argmax(predictions[0])]

grads = tape.gradient(loss, conv_output)[0]


weights = tf.reduce_mean(grads, axis=(0, 1))


heatmap = tf.reduce_sum(tf.multiply(weights, conv_output), axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)


img_array = image.img_to_array(img)
img_shape = img_array.shape[0:2]


heatmap_resized = tf.image.resize(heatmap, img_shape)


plt.imshow(img)
plt.imshow(heatmap_resized[..., 0], alpha=0.5, cmap='jet')
plt.show()