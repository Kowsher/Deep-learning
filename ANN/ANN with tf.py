import tensorflow as tf
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units = 128, activation = 'relu', input_shape = (len(X_train[1]),)))
model.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
model.add(tf.keras.layers.Dense(units = 524, activation = 'relu'))
model.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))

model.add(tf.keras.layers.Dense(units = 10, activation = 'softmax'))
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01), loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs =50)
