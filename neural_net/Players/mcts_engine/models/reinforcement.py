import tensorflow as tf

tf.get_logger().setLevel('ERROR')

def create_model():
    print("Creating model: reinforcement")
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu', input_shape=(8, 8, 12)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

    model.add(tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

    model.add(tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

<<<<<<< HEAD
    model.add(tf.keras.layers.Conv2D(1024, kernel_size=3, padding='same', strides=2, activation='relu'))
=======
    model.add(tf.keras.layers.Conv2D(1024, kernel_size=3, padding='same', activation='relu'))
>>>>>>> b2e210fce2ff2f7187dd8b0f9b943e1297a2ace9
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=2048, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
    model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
    
    optimiser = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.CategoricalCrossentropy()
    
    model.compile(optimizer=optimiser, loss=loss, metrics=['accuracy'])
    
    try:
        model.load_weights("neural_net/Players/mcts_engine/models/reinforcement.h5")
    except OSError:
        pass

    return model
