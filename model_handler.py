"""
model_handler.py — STABLE VERSION
───────────────────────────────────
Exact class names from class_order.json:
  0 Tomato_Bacterial_spot
  1 Tomato_Early_blight
  2 Tomato_Late_blight
  3 Tomato_Leaf_Mold
  4 Tomato_Septoria_leaf_spot
  5 Tomato_Spider_mites_Two_spotted_spider_mite
  6 Tomato__Target_Spot
  7 Tomato__Tomato_YellowLeaf__Curl_Virus
  8 Tomato__Tomato_mosaic_virus
  9 Tomato_healthy
"""

import numpy as np
import os
import streamlit as st
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# ─────────────────────────────────────────────────────────────
# EXACT CLASS NAMES from class_order.json
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

DISPLAY_NAMES = {
    "Tomato_Bacterial_spot":                       "Tomato Bacterial Spot",
    "Tomato_Early_blight":                         "Tomato Early Blight",
    "Tomato_Late_blight":                          "Tomato Late Blight",
    "Tomato_Leaf_Mold":                            "Tomato Leaf Mold",
    "Tomato_Septoria_leaf_spot":                   "Tomato Septoria Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato Spider Mites",
    "Tomato__Target_Spot":                         "Tomato Target Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus":       "Tomato Yellow Leaf Curl Virus",
    "Tomato__Tomato_mosaic_virus":                 "Tomato Mosaic Virus",
    "Tomato_healthy":                              "Healthy Tomato Leaf",
}

DISEASE_INFO = {
    "Tomato Bacterial Spot": {
        "severity": "Medium",
        "scientific_name": "Xanthomonas campestris pv. vesicatoria",
        "symptoms": "Small dark brown water-soaked spots (1-3mm) on leaves, stems and fruits. Spots surrounded by yellow halos. Leaves turn yellow and drop prematurely.",
        "causes": "Caused by bacterium Xanthomonas campestris. Spreads through infected seeds, rain splash, and contaminated tools. Favoured by warm (24-30°C) wet weather.",
        "remedies": ["Apply copper-based bactericide (copper hydroxide)", "Spray fixed copper + mancozeb every 7 days", "Remove and destroy all infected plant parts", "Use disease-free certified seeds", "Avoid overhead irrigation"],
        "prevention": ["Use bacterial-spot resistant varieties", "Practice 3-year crop rotation", "Sanitise tools with 10% bleach solution", "Avoid working with wet plants", "Plant in well-drained soil"],
    },
    "Tomato Early Blight": {
        "severity": "Medium",
        "scientific_name": "Alternaria solani",
        "symptoms": "Dark brown circular lesions with concentric rings (bulls-eye pattern). Yellow area surrounds each lesion. Starts on older lower leaves and moves upward.",
        "causes": "Caused by fungus Alternaria solani. Favoured by warm temperatures (24-29°C), high humidity, and prolonged leaf wetness.",
        "remedies": ["Apply chlorothalonil or mancozeb fungicide", "Spray copper-based fungicide every 7-10 days", "Remove infected lower leaves immediately", "Apply neem oil (5ml/litre) as organic option", "Use azoxystrobin for severe cases"],
        "prevention": ["Mulch around base of plants", "Water at soil level using drip irrigation", "Use certified disease-free transplants", "Rotate crops annually", "Space plants 45-60cm apart"],
    },
    "Tomato Late Blight": {
        "severity": "High",
        "scientific_name": "Phytophthora infestans",
        "symptoms": "Large irregular water-soaked greyish-green to dark brown lesions. White cottony mould on leaf undersides. Rapid browning and plant collapse.",
        "causes": "Caused by oomycete Phytophthora infestans. Spreads in cool (10-24°C), wet foggy conditions. Same pathogen as the 1840s Irish Potato Famine.",
        "remedies": ["Apply metalaxyl + mancozeb (Ridomil Gold) immediately", "Remove and BURN all infected plant material", "Spray chlorothalonil every 5-7 days", "Apply cymoxanil + famoxadone as systemic fungicide", "Improve field drainage"],
        "prevention": ["Plant resistant varieties (Mountain Magic, Defiant)", "Monitor disease forecast for blight-risk alerts", "Avoid overhead watering completely", "Ensure excellent field drainage", "Never save seed from infected plants"],
    },
    "Tomato Leaf Mold": {
        "severity": "Low",
        "scientific_name": "Passalora fulva",
        "symptoms": "Pale greenish-yellow spots on upper leaf surface. Olive-green to greyish-purple velvety mould on lower surface. Older leaves affected first.",
        "causes": "Caused by fungus Passalora fulva. Thrives in high humidity (above 85%) and moderate temperatures (22-25°C). Common in greenhouses.",
        "remedies": ["Apply mancozeb or copper oxychloride fungicide", "Reduce humidity by improving ventilation", "Remove affected leaves carefully", "Apply potassium bicarbonate spray", "Use Bacillus subtilis biofungicide"],
        "prevention": ["Maintain humidity below 85%", "Ensure 60cm+ plant spacing", "Avoid excess nitrogen fertilisation", "Use leaf-mold resistant varieties", "Ventilate greenhouses morning and evening"],
    },
    "Tomato Septoria Leaf Spot": {
        "severity": "Medium",
        "scientific_name": "Septoria lycopersici",
        "symptoms": "Small circular spots (3-6mm) with dark brown borders and light grey centres. Tiny black dots visible inside spots. Yellowing and defoliation from bottom up.",
        "causes": "Caused by fungus Septoria lycopersici. Spreads by rain splash and tools. Favoured by warm (20-25°C) wet weather.",
        "remedies": ["Apply chlorothalonil at first sign", "Spray mancozeb every 7-10 days", "Remove infected lower leaves", "Apply copper-based fungicide preventively", "Use azoxystrobin for systemic protection"],
        "prevention": ["Mulch soil surface to prevent spore splash", "Practice 2-3 year crop rotation", "Stake plants to keep foliage off ground", "Water in the morning", "Remove plant debris after harvest"],
    },
    "Tomato Spider Mites": {
        "severity": "Medium",
        "scientific_name": "Tetranychus urticae",
        "symptoms": "Tiny yellow or white speckles on upper leaf surface. Fine silk webbing on leaf undersides. Leaves turn bronze then yellow and drop.",
        "causes": "Caused by two-spotted spider mite Tetranychus urticae. Populations explode in hot (above 30°C), dry, dusty conditions.",
        "remedies": ["Spray with insecticidal soap (2% solution)", "Apply neem oil (5ml/litre) every 5 days", "Use abamectin miticide for severe infestations", "Introduce predatory mites", "Forceful water spray to dislodge mites"],
        "prevention": ["Maintain adequate soil moisture", "Avoid excessive nitrogen fertilisation", "Control dust on plants", "Avoid broad-spectrum insecticides", "Monitor with magnifying glass weekly"],
    },
    "Tomato Target Spot": {
        "severity": "Medium",
        "scientific_name": "Corynespora cassiicola",
        "symptoms": "Brown circular lesions with concentric target-ring pattern and yellow halo. Spots on leaves, stems, and fruit. Premature leaf drop in severe cases.",
        "causes": "Caused by fungus Corynespora cassiicola. Favoured by high humidity and temperatures of 20-30°C.",
        "remedies": ["Apply azoxystrobin or chlorothalonil fungicide", "Spray tebuconazole for systemic action", "Remove infected plant debris", "Apply copper-based fungicide preventively", "Reduce leaf wetness duration"],
        "prevention": ["Use certified disease-free transplants", "Avoid dense planting — maintain 50cm+ spacing", "Ensure proper drip irrigation", "Rotate crops with cereals or legumes", "Remove all crop debris after harvest"],
    },
    "Tomato Yellow Leaf Curl Virus": {
        "severity": "High",
        "scientific_name": "Tomato yellow leaf curl virus (TYLCV)",
        "symptoms": "Upward curling and yellowing of leaf margins. Leaves become small and crinkled. Severely stunted plant growth. Flowers drop without setting fruit.",
        "causes": "Caused by TYLCV, transmitted exclusively by silverleaf whitefly (Bemisia tabaci). Cannot spread by contact or tools.",
        "remedies": ["Control whitefly with imidacloprid or thiamethoxam drench", "Remove and destroy infected plants immediately", "Apply reflective silver mulch to repel whiteflies", "Use yellow sticky traps to monitor and trap", "Apply spirotetramat for long-lasting whitefly control"],
        "prevention": ["Use TYLCV-resistant varieties", "Install 50-mesh insect-proof nets in nursery", "Control whitefly before and after transplanting", "Remove infected plants immediately", "Avoid planting near older infected crops"],
    },
    "Tomato Mosaic Virus": {
        "severity": "High",
        "scientific_name": "Tomato mosaic virus (ToMV)",
        "symptoms": "Mottled light and dark green mosaic pattern on leaves. Leaf distortion, curling, and stunting. Fruit may show internal browning and uneven ripening.",
        "causes": "Caused by Tomato mosaic virus (ToMV). Spread by contact — contaminated hands, tools, and clothing. No insect vector.",
        "remedies": ["No cure — remove and destroy infected plants immediately", "Disinfect all tools with 10% bleach solution", "Wash hands thoroughly before handling plants", "Do not smoke near tomato plants", "Control aphids to prevent secondary spread"],
        "prevention": ["Use certified virus-free seed", "Plant TMV/ToMV resistant varieties", "Disinfect all tools before use", "Remove solanaceous weeds near crop", "Do not handle plants unnecessarily"],
    },
    "Healthy Tomato Leaf": {
        "severity": "None",
        "scientific_name": "N/A",
        "symptoms": "No disease symptoms detected. Leaf appears healthy with uniform vibrant green colouration. No spots, lesions, mould, or discolouration visible.",
        "causes": "No pathogen detected. Plant appears to be in good health.",
        "remedies": ["No treatment required", "Continue current care routine", "Monitor weekly for any early signs of disease"],
        "prevention": ["Continue regular watering schedule", "Apply balanced NPK fertiliser monthly", "Inspect plants weekly for early detection", "Maintain good air circulation between plants", "Practice preventive crop hygiene"],
    },
}

MODEL_PATH = "tomato_disease_model.h5"

@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            return model, "cnn"
        except Exception as e:
            st.warning(f"Model load error: {e}")
    return None, "heuristic"


def heuristic_classify(features: dict) -> tuple:
    green_ratio = features.get("green_ratio", 50)
    brown_ratio = features.get("brown_ratio", 0)
    brightness  = features.get("brightness", 128)
    saturation  = features.get("saturation", 50)
    r = features.get("channel_r", 100)
    g = features.get("channel_g", 100)

    if green_ratio > 40 and brown_ratio < 5 and saturation > 30:
        return 9, 82.0
    if brown_ratio > 15 and green_ratio > 20:
        return 1, 74.0
    if brightness < 80 and brown_ratio > 20:
        return 2, 71.0
    if r > g + 30 and saturation > 20:
        return 7, 68.0 if brightness > 140 else 0
    if brown_ratio > 8 and brightness < 120:
        return 4, 65.0
    return 1, 55.0


def predict_disease(image: Image.Image, features: dict) -> dict:
    model, mode = load_model()

    if mode == "cnn" and model is not None:
        img_array = np.array(image.resize((224, 224))).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array, verbose=0)[0]
        pred_idx    = int(np.argmax(predictions))
        confidence  = float(predictions[pred_idx]) * 100
        method      = "MobileNetV2 CNN (PlantVillage — 86.8% accuracy)"
    else:
        pred_idx, confidence = heuristic_classify(features)
        method = "Colour-Texture Heuristic (offline fallback)"

    raw_name     = CLASS_NAMES[pred_idx]
    disease_name = DISPLAY_NAMES.get(raw_name, raw_name)
    info         = DISEASE_INFO.get(disease_name, DISEASE_INFO["Healthy Tomato Leaf"])

    return {
        "disease_name":    disease_name,
        "confidence":      round(confidence, 1),
        "severity":        info["severity"],
        "symptoms":        info["symptoms"],
        "causes":          info["causes"],
        "remedies":        info["remedies"],
        "prevention":      info["prevention"],
        "scientific_name": info["scientific_name"],
        "method":          method,
    }
