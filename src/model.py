import tensorflow as tf

class ParkingModel:
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
        
    def fit(self, train_ds, epochs=100, validation_data=None):
        return self.model.fit(
            train_ds, 
            epochs=epochs,
            validation_data=validation_data,
            class_weight={0: 1.5, 1: 1.0}  # Address class imbalance
        )
        
    def save_model(self):
        self.model.save(os.path.join(self.config.MODEL_DIR, 'parking_model.h5'))