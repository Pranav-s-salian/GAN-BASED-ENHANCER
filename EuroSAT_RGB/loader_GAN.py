import os
import numpy as np
from PIL import Image


def load_eurosat_dataset_Gan(base_path, target_size=(256, 256), max_images_per_class=200):
   
    
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
    
    
    clear_classes = ['SeaLake', 'AnnualCrop', 'PermanentCrop']
    
    weather_names = ['Clear', 'Cloudy', 'Foggy', 'Hazy']
    
    
    
    weather_images = []
    clear_images = []
    weather_labels = []
    class_distribution = {name: 0 for name in weather_names}
    
    
    clear_image_pool = []
    
    
    class_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    print(f"Found {len(class_folders)} class folders: {class_folders}")
    
    
    print(" Loading clear images...")
    for class_folder in class_folders:
        if class_folder not in weather_mapping or class_folder not in clear_classes:
            continue
            
        class_path = os.path.join(base_path, class_folder)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if max_images_per_class:
            image_files = image_files[:max_images_per_class]
        
        print(f"Loading {len(image_files)} clear images from {class_folder}")
        
        for image_file in image_files:
            try:
                image_path = os.path.join(class_path, image_file)
                image = Image.open(image_path)
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image = image.resize(target_size, Image.Resampling.LANCZOS)
                image_array = np.array(image, dtype=np.float32) / 255.0
                
                clear_image_pool.append(image_array)
                    
            except Exception as e:
                print(f"Error loading {image_path}: {str(e)}")
                continue
    
    print(f"✅ Loaded {len(clear_image_pool)} clear images")
    
    
    print("🌦️ Loading all images and creating pairs...")
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
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image = image.resize(target_size, Image.Resampling.LANCZOS)
                image_array = np.array(image, dtype=np.float32) / 255.0
                
                
                weather_images.append(image_array)
                weather_labels.append(weather_label)
                class_distribution[weather_name] += 1
                
                
                if class_folder in clear_classes:
                    clear_images.append(image_array)
                else:
                    
                    if clear_image_pool:
                        random_clear_idx = np.random.randint(0, len(clear_image_pool))
                        clear_images.append(clear_image_pool[random_clear_idx])
                    else:
                        
                        clear_images.append(image_array)
                
                
                if (i + 1) % 100 == 0:
                    print(f"   Processed {i + 1}/{len(image_files)} images from {class_folder}")
                    
            except Exception as e:
                print(f"Error loading {image_path}: {str(e)}")
                continue
    
    # Convert to numpy arrays
    weather_images = np.array(weather_images)
    clear_images = np.array(clear_images)
    weather_labels = np.array(weather_labels)
    
    print("\n Dataset loading completed!")
    print(f" Total weather images loaded: {len(weather_images)}")
    print(f"Total clear images loaded: {len(clear_images)}")
    print(f"Weather image shape: {weather_images.shape}")
    print(f"Clear image shape: {clear_images.shape}")
    print(f" Labels shape: {weather_labels.shape}")
    print("\n Class distribution:")
    for weather_name, count in class_distribution.items():
        percentage = (count / len(weather_images)) * 100
        print(f"   {weather_name}: {count} images ({percentage:.1f}%)")
    
    # just to verif y thease things
    assert len(weather_images) == len(clear_images) == len(weather_labels), \
        f"Array lengths don't match: weather={len(weather_images)}, clear={len(clear_images)}, labels={len(weather_labels)}"
    
    return weather_images, clear_images, weather_labels, weather_names, class_distribution

