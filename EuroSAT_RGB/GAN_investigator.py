from tensorflow import keras
from tensorflow.keras import layers

def create_discriminator():
    """
    Discriminator: Looks at image and says "This looks real" or "This looks fake"
    Also considers weather condition for better judgment
    """
    print("🔍 Creating Discriminator (The Reality Checker)...")
    
    
    image_input = layers.Input(shape=(256, 256, 3))
    
    
    weather_input = layers.Input(shape=(4,))
    
   
    weather_dense = layers.Dense(256 * 256)(weather_input)
    weather_reshaped = layers.Reshape((256, 256, 1))(weather_dense)
    
    
    combined = layers.concatenate([image_input, weather_reshaped])
    
    # investigator  network
    d1 = layers.Conv2D(64, (4, 4), strides=2, padding='same', activation='relu')(combined)
    d2 = layers.Conv2D(128, (4, 4), strides=2, padding='same', activation='relu')(d1)
    d3 = layers.Conv2D(256, (4, 4), strides=2, padding='same', activation='relu')(d2)
    d4 = layers.Conv2D(512, (4, 4), strides=2, padding='same', activation='relu')(d3)
    
    
    flattened = layers.Flatten()(d4)
    output = layers.Dense(1, activation='sigmoid')(flattened)
    
    model = keras.Model([image_input, weather_input], output)
    return model        

print("✅ Discriminator created successfully!") 