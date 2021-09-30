import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# The "model" object is an instance of the "Sequential" class. 
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()


# loss_fn will be used as a "Callback" function. I.e. a function as argument wrapped in another function. 
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()


# "Compile" means we translate the problem into a language - which works more closely to the hardware. 
# 62.0% of tensorflow is written in C++ ! 
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


# "model.fit" means: backpropagation algorithm. "epochs=5" means we train over the whole dataset 5 times. 
model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])
