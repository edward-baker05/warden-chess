import tensorflow as tf

tf.get_logger().setLevel('ERROR')

def create_model() -> tf.keras.Sequential:
    print("Creating model: complex_large")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu',  input_shape=(8, 8, 12)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))

    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))

    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(units=1, activation='linear'))
    
    optimiser = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MeanSquaredError()
    
    model.compile(optimizer=optimiser, loss=loss, metrics=['mae'])
    try:
        model.load_weights("neural_net/Players/mcts_engine/models/complex_large.h5")
    except FileNotFoundError:
        pass
    
    return model