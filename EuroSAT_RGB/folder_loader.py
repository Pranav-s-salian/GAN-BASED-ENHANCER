import os
import numpy as np
from PIL import Image


def load_eurosat_dataset(base_path, target_size=(256, 256), max_images_per_class=200): #here we are taking only 500 images per folder, sinvce my pc i not that strong
    """
    Load EuroSAT dataset and adapt it for weather classification
    
    Args:
        base_path (str): Path to the folder containing all class folders
        target_size (tuple): Target size for resizing images (height, width)
        max_images_per_class (int): Maximum number of images to load per class (None for all)
    
    Returns:
        images (np.array): Array of preprocessed images
        labels (np.array): Array of weather condition labels
        class_names (list): List of weather condition names
        class_distribution (dict): Distribution of images per class
    """
    
   
    weather_mapping = {
        'SeaLake': 0,        # Clear (water bodies usually indicate clear visibility)
        'AnnualCrop': 0,     # Clear (agricultural areas are usually clear)
        'PermanentCrop': 0,  # Clear
        'Pasture': 1,        # Cloudy (grasslands can appear cloudy/overcast)
        'HerbaceousVegetation': 1,  # Cloudy
        'Forest': 2,         # Foggy (forests often have misty/foggy conditions)
        'Residential': 2,    # Foggy (urban areas can be hazy)
        'Industrial': 3,     # Hazy (industrial areas often have haze/pollution)
        'Highway': 3,        # Hazy (transportation corridors can be hazy)
        'River': 1           # Cloudy (rivers can indicate overcast conditions)
    }
    
    weather_names = ['Clear', 'Cloudy', 'Foggy', 'Hazy']
    
    print("Loading EuroSAT dataset...")
    print(f"Base path: {base_path}")
    print(f"Target size: {target_size}")
    
    images = []
    labels = []
    class_distribution = {name: 0 for name in weather_names}
    
    
    class_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    print(f"📂 Found {len(class_folders)} class folders: {class_folders}")
    
    for class_folder in class_folders:
        if class_folder not in weather_mapping:
            print(f"Skipping unknown class: {class_folder}")
            continue
            
        weather_label = weather_mapping[class_folder]
        weather_name = weather_names[weather_label]
        
        class_path = os.path.join(base_path, class_folder)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
    
        if max_images_per_class:
            image_files = image_files[:max_images_per_class]
        
        print(f"Loading {len(image_files)} images from {class_folder} -> {weather_name}")
        
        for i, image_file in enumerate(image_files):
            try:
                image_path = os.path.join(class_path, image_file)
                
               
                image = Image.open(image_path)
                
                # Convert to RGB if not already
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                
                image = image.resize(target_size, Image.Resampling.LANCZOS)
                
                
                image_array = np.array(image, dtype=np.float32) / 255.0
                
                images.append(image_array)
                labels.append(weather_label)
                class_distribution[weather_name] += 1
                
                # Progress indicator
                if (i + 1) % 100 == 0:
                    print(f"   Processed {i + 1}/{len(image_files)} images from {class_folder}")
                    
            except Exception as e:
                print(f"Error loading {image_path}: {str(e)}")
                continue
    
    #numpy
    images = np.array(images)
    labels = np.array(labels)
    
    print("\nDataset loading completed!")
    print(f"Total images loaded: {len(images)}")
    print(f"Image shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print("\n Class distribution:")
    for weather_name, count in class_distribution.items():
        percentage = (count / len(images)) * 100
        print(f"   {weather_name}: {count} images ({percentage:.1f}%)")
    
    return images, labels, weather_names, class_distribution


