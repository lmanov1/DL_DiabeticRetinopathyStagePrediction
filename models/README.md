# diabetic-retinopathy-stage-prediction
Diabetic Retinopathy Stage Prediction DL Project

# Load the saved model
from tensorflow.keras.models import load_model

model = load_model('v1.0_vgg16_model.keras')

#   Models Versioning
## ------------------- ##


## ** Version 1.0 vgg16 Model **

File Export - models/v1.0_vgg16_model.keras

# Training Output:

Epoch 1/10
92/92 [==============================] - 144s 2s/step - loss: 1.2455 - accuracy: 0.6197 - val_loss: 0.8122 - val_accuracy: 0.7077
Epoch 2/10
92/92 [==============================] - 143s 2s/step - loss: 0.9448 - accuracy: 0.6852 - val_loss: 0.8295 - val_accuracy: 0.7090
Epoch 3/10
92/92 [==============================] - 144s 2s/step - loss: 0.9659 - accuracy: 0.6828 - val_loss: 0.7849 - val_accuracy: 0.7145
Epoch 4/10
92/92 [==============================] - 143s 2s/step - loss: 0.9285 - accuracy: 0.6849 - val_loss: 0.7936 - val_accuracy: 0.7199
Epoch 5/10
92/92 [==============================] - 143s 2s/step - loss: 0.9330 - accuracy: 0.6746 - val_loss: 0.8282 - val_accuracy: 0.7213
Epoch 6/10
92/92 [==============================] - 144s 2s/step - loss: 0.8858 - accuracy: 0.6951 - val_loss: 0.8059 - val_accuracy: 0.7213
Epoch 7/10
92/92 [==============================] - 144s 2s/step - loss: 0.8744 - accuracy: 0.7060 - val_loss: 0.7728 - val_accuracy: 0.7227
Epoch 8/10
92/92 [==============================] - 143s 2s/step - loss: 0.8417 - accuracy: 0.7094 - val_loss: 0.7682 - val_accuracy: 0.7240
Epoch 9/10
92/92 [==============================] - 143s 2s/step - loss: 0.8437 - accuracy: 0.7070 - val_loss: 0.7866 - val_accuracy: 0.7158
Epoch 10/10
92/92 [==============================] - 143s 2s/step - loss: 0.8094 - accuracy: 0.7145 - val_loss: 0.7424 - val_accuracy: 0.7254


# Model Evaluation:

23/23 [==============================] - 29s 1s/step - loss: 0.7424 - accuracy: 0.7254
Validation Accuracy: 72.54%
