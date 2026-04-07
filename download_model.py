import urllib.request
import os

# Model trained on PlantVillage by the community
url = "https://github.com/AbdullahTabassam/Tomato-Leaf-Disease-Detection/raw/main/tomato_disease_model.h5"
save_path = "tomato_disease_model.h5"

if os.path.exists(save_path):
    os.remove(save_path)

print("Downloading trained model... please wait")

try:
    urllib.request.urlretrieve(url, save_path)
    size = os.path.getsize(save_path) / (1024*1024)
    print(f"✅ Downloaded! Size: {size:.1f} MB")
except Exception as e:
    print(f"❌ Failed: {e}")