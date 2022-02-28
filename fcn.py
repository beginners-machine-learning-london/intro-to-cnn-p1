import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.sequence = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10),
            # Softmax activation function converts the raw output predictions to a probability distribution
            tf.keras.layers.Softmax()
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        y_prime = self.sequence(x)
        return y_prime


def main():
    image_size = 28
    num_train = 60000
    num_test = 10000
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = NeuralNetwork()
    model.build(input_shape=(1, image_size, image_size))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    print(model.summary())
    model.fit(train_images, train_labels, epochs=10)
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    output = model.predict(test_images)
    print(output)


if __name__ == '__main__':
    main()
