from tensorflow import keras
from tensorflow.keras import layers



def create_generator():
    """
    Generator: Takes weather image + weather type, outputs clear image
    This is the "cloud remover" that cleans up the image
    """
    print("🎨 Creating Generator (The Cloud Remover)...")
    
   
    image_input = layers.Input(shape=(256, 256, 3))
    
    
    weather_input = layers.Input(shape=(4,))
    
    
    weather_dense = layers.Dense(256 * 256)(weather_input)
    weather_reshaped = layers.Reshape((256, 256, 1))(weather_dense)
    
    
    combined = layers.concatenate([image_input, weather_reshaped])
    
    
    e1 = layers.Conv2D(64, (4, 4), strides=2, padding='same', activation='relu')(combined)  ##her we are actually upsampling the images, and then we are concatenating the weather condition, every encoder descrese the size of the image by half
    e2 = layers.Conv2D(128, (4, 4), strides=2, padding='same', activation='relu')(e1)
    e3 = layers.Conv2D(256, (4, 4), strides=2, padding='same', activation='relu')(e2)
    e4 = layers.Conv2D(512, (4, 4), strides=2, padding='same', activation='relu')(e3)
    
   
    bottleneck = layers.Conv2D(512, (4, 4), strides=2, padding='same', activation='relu')(e4)
    
    # Decoder (upsampling)  ## here w are upsampling the image, and then we are concatenating the weather condition, every decoder increase the size of the image by half
    d1 = layers.Conv2DTranspose(512, (4, 4), strides=2, padding='same', activation='relu')(bottleneck)
    d1 = layers.concatenate([d1, e4])  
    
    d2 = layers.Conv2DTranspose(256, (4, 4), strides=2, padding='same', activation='relu')(d1)
    d2 = layers.concatenate([d2, e3])  
    
    d3 = layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same', activation='relu')(d2) #filter size, strides, xize etc
    d3 = layers.concatenate([d3, e2])  
    
    d4 = layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='relu')(d3)
    d4 = layers.concatenate([d4, e1])  
    
    output = layers.Conv2DTranspose(3, (4, 4), strides=2, padding='same', activation='tanh')(d4)
    
    model = keras.Model([image_input, weather_input], output)
    return model
