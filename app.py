import streamlit as st
import os

# TRICK: Wir simulieren eine Umgebung ohne GUI, bevor CV2 geladen wird
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Seite konfigurieren
st.set_page_config(page_title="Profilwahl Check", layout="centered")

# Modell laden mit Cache
@st.cache_resource
def load_model():
    model_path = 'yolov8n-2.pt'
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_model()

st.title("🎓 Profilwahl-Prüfer")

if model is None:
    st.error("Fehler: 'yolov8n-2.pt' nicht im Repository gefunden!")
else:
    uploaded_file = st.file_uploader("Lade deinen Wahlzettel hoch...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Bild einlesen
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        with st.spinner('KI analysiert...'):
            # Inferenz (Wir nutzen .predict statt direktem Call für mehr Kontrolle)
            results = model.predict(source=img_array, conf=0.25, save=False)
            
            # Ergebnis visualisieren
            # result.plot() erzeugt ein numpy array (BGR), das wir für Streamlit umwandeln
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            st.image(res_rgb, caption="Ergebnis der Erkennung", use_container_width=True)

            # Analyse der gefundenen Kästchen
            boxes = results[0].boxes
            st.success(f"Analyse abgeschlossen: {len(boxes)} Markierungen gefunden.")

            if len(boxes) > 0:
                with st.expander("Details der erkannten Bereiche"):
                    for i, box in enumerate(boxes):
                        st.write(f"Markierung {i+1}: Konfidenz {float(box.conf):.2f}")
