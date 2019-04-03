import tensorflowjs as tfjs

def train(model):
    model = keras.models.Sequential([   
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(128, activation=tf.nn.relu),
      keras.layers.Dense(10, activation=tf.nn.softmax)
      ])
    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)
    tfjs.converters.save_keras_model(model, tfjs_target_dir)
