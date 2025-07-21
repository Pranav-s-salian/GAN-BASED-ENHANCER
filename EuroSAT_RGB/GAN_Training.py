from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from loader_GAN import load_eurosat_dataset_Gan
from GAN_investigator import create_discriminator
from GAN_genrator import create_generator


def train_gan(generator, discriminator, weather_images, clear_images, weather_labels):
    """Train the GAN to remove weather interference"""
    print("Training Weather-Aware GAN...")
    
    
    discriminator.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    
    discriminator.trainable = False
    weather_input = layers.Input(shape=(256, 256, 3))
    condition_input = layers.Input(shape=(4,))
    
    generated_image = generator([weather_input, condition_input])
    validity = discriminator([generated_image, condition_input])
    
    combined = keras.Model([weather_input, condition_input], validity)
    combined.compile(optimizer='adam', loss='binary_crossentropy')
    
    
    epochs = 100
    batch_size = 32
    
    for epoch in range(epochs):
        
        idx = np.random.randint(0, len(weather_images), batch_size)
        real_images = clear_images[idx]
        weather_imgs = weather_images[idx]
        conditions = weather_labels[idx]
        
        # here we are applying one hot encoding to conevrt the labels into and hone hot encoded vectors
        conditions_onehot = keras.utils.to_categorical(conditions, 4)
        
         #fake images!!!
        fake_images = generator.predict([weather_imgs, conditions_onehot])
        
        # Our investigator
        d_loss_real = discriminator.train_on_batch([real_images, conditions_onehot], np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch([fake_images, conditions_onehot], np.zeros((batch_size, 1)))
        
        r
        g_loss = combined.train_on_batch([weather_imgs, conditions_onehot], np.ones((batch_size, 1)))
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: D_loss: {0.5 * np.add(d_loss_real[0], d_loss_fake[0]):.4f}, G_loss: {g_loss:.4f}")
    
    return generator, discriminator


def create_temporal_model():  # this is an temporay model, to ake the sure the process smoother and make sure it moves like butter, i mean like an vedio frame, NO Laggg
    
    
    model = keras.Sequential([
        layers.Input(shape=(5, 256, 256, 3)),  #
        layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu')),
        layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')),
        layers.TimeDistributed(layers.Flatten()),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(256 * 256 * 3, activation='tanh'),
        layers.Reshape((256, 256, 3))
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model


if __name__ == "__main__":
    print("getting all the iamges from the loader")
    
    weatger_images, clear_images, weather_labels = load_eurosat_dataset_Gan("give your path ex data/images")  
    print(f"Loaded {len(weatger_images)} weather images and {len(clear_images)} clear images.")
    
    genrator, investegator = train_gan(create_generator, create_discriminator, weatger_images, clear_images, weather_labels)
    print("GAN training completed!")
    
    tempory = create_temporal_model()  
    print("Temporal model created for video-like processing")
    tempory.save('temporal_model.h5')
    
    
    genrator.save('weather_aware_generator.h5')
    investegator.save('weather_aware_discriminator.h5')
    
    print("Models saved as 'weather_aware_generator.h5' and 'weather_aware_discriminator.h5'")
    