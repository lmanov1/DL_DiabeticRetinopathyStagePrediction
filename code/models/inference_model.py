# create a interface  to the model neural network model
import tensorflow as tf


def create_model_interface(model):
    """
    Creates a wrapper interface for a TensorFlow model for easy use in applications.
    """

    class ModelInterface:
        def __init__(self, model):
            self.model = model

        def predict(self, input_data):
            return self.model.predict(input_data)

        def evaluate(self, test_data):
            return self.model.evaluate(test_data)

        def fit(self, train_data, epochs, batch_size):
            self.model.fit(train_data, epochs=epochs, batch_size=batch_size)

    return ModelInterface(model)
