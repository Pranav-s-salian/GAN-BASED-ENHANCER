from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from folder_loader import load_eurosat_dataset
from cnn_model import create_weather_classifier_updated


def train_weather_classifier_eurosat(model, images, labels, validation_split=0.2, epochs=20, batch_size=32):
    
    print("Training Weather Classifier on EuroSAT data...")
    
    
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=validation_split, random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(x_train)} images")
    print(f"Test set: {len(x_test)} images")
    
    
    data_gen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
   
    history = model.fit(
        data_gen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"✅ Weather Classifier Test Accuracy: {test_acc:.4f}")
    
    model.save('weather_classifier_eurosat.h5')
    print("💾 Model saved as 'weather_classifier_eurosat.h5'")
    
    return test_acc
    
    
if __name__ == "__main__":
    
    base_path = "blah/blah.."  # Update this path to your dataset location
    images, labels, class_names, class_distribution = load_eurosat_dataset(base_path)
    
    print(f"Loaded {len(images)} images from EuroSAT dataset with classes: {class_names}")
    print(images.shape, labels.shape)
    
    model,model_info = create_weather_classifier_updated(input_shape=(256, 256, 3))
    print(model_info)
    
    accuracy = train_weather_classifier_eurosat(model, images, labels, validation_split=0.2, epochs=20, batch_size=32)
    print("Training complete! Yahoooooooo!!!!")    
    print(f"test accuracy: {accuracy}")
    print("model saved as weather_classifier_eurosat.h5")
    
    
    
    
