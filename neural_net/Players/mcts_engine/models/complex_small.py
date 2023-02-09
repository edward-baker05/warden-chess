import tensorflow as tf

def create():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu', input_shape=(8, 8, 12)))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(1024, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=2048, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='softmax'))
    
    return model