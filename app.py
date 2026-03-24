import os

# WICHTIG: Diese Zeilen MÜSSEN ganz oben stehen, vor allen anderen Imports!
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import streamlit as st
from PIL import Image
import numpy as np

# Erst jetzt die KI-Library laden

from ultralytics import YOLO

st.set_page_config(page_title="Wahlzettel-Prüfer", layout="centered")

@st.cache_resource
def load_model():
    # Lädt dein Modell yolov8n-2.pt
    return YOLO('yolov8n-2.pt')

st.title("🎓 Profilwahl KI-Check")

try:
    model = load_model()
    
    uploaded_file = st.file_uploader("Wahlzettel Bild hochladen", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        
        # Inferenz: Wir schalten alles aus, was Plotting oder GUI erfordert
        results = model.predict(source=img, save=False, show=False, imgsz=640)
        
        # Statt results[0].plot() (was cv2 nutzt), extrahieren wir nur die Daten
        boxes = results[0].boxes
        
        st.subheader("Analyse-Ergebnis")
        
        # Wir zeichnen die Boxen selbst mit PIL, um cv2 komplett zu umgehen
        if len(boxes) > 0:
            st.success(f"{len(boxes)} Kreuze erkannt!")
            
            # Zeige das Originalbild an
            st.image(img, caption="Hochgeladener Wahlzettel", use_container_width=True)
            
            # Hier kannst du die Logik für die Fächer einfügen
            for box in boxes:
                # Koordinaten: box.xyxy
                st.write(f"Kreuz gefunden bei: {box.xyxy[0].tolist()}")
        else:
            st.warning("Keine Kreuze gefunden. Bitte schärferes Foto machen.")
            st.image(img, use_container_width=True)

except Exception as e:
    st.error(f"Ein Fehler ist aufgetreten: {e}")
