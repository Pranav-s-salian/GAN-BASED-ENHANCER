from tensorflow import keras
from tensorflow.keras import layers



def create_weather_classifier_updated(input_shape=(256, 256, 3), num_classes=4):
    """
    Updated weather classifier that matches your original architecture
    """
    print("🔍 Creating Weather Classifier (The Weather Detective)...")
    
    model = keras.Sequential([
        
        layers.Input(shape=input_shape),
        
    
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        
       
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("✅ Weather Classifier created successfully!")
    
    
    return model, model.summary()


