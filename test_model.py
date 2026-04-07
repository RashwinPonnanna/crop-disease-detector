import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('tomato_disease_model.h5', compile=False)

# Convert to RGB to handle PNG transparency (4 channels → 3 channels)
img = Image.open('tomato-leaf.jpg').convert('RGB').resize((224, 224))
arr = np.expand_dims(np.array(img).astype('float32') / 255.0, 0)
preds = model.predict(arr, verbose=0)[0]

classes = [
    'Bacterial_spot', 'Early_blight', 'Late_blight',
    'Leaf_Mold', 'Septoria', 'Spider_mites',
    'Target_Spot', 'YellowLeaf_Curl', 'Mosaic_virus', 'healthy'
]

print("\nAll predictions:")
for i, (c, p) in enumerate(zip(classes, preds)):
    print(f"  {i}: {c} = {p*100:.1f}%")

print(f"\nWinner: {classes[np.argmax(preds)]}")