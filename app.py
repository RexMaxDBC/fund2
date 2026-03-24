import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# Konfiguration der Seite
st.set_page_config(page_title="Profilwahl Check", layout="wide")

# Lade-Funktion mit Cache, damit das Modell nicht bei jedem Klick neu geladen wird
@st.cache_resource
def get_model():
    model_path = 'yolov8n-2.pt'
    if not os.path.exists(model_path):
        st.error(f"Datei {model_path} nicht gefunden! Bitte ins GitHub-Repo hochladen.")
        return None
    return YOLO(model_path)

st.title("🚀 KI-Profilwahl Assistent")
st.info("Lade deinen Wahlzettel hoch. Die KI erkennt die Kreuze und prüft die Belegung.")

# Modell initialisieren
model = get_model()

# Datei Upload
file = st.file_uploader("Wahlzettel Bild (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if file and model:
    # 1. Bild einlesen und standardisieren
    pil_image = Image.open(file).convert("RGB")
    img_array = np.array(pil_image)
    
    # 2. KI Vorhersage
    # confidence=0.25 (Standard), kannst du anpassen falls er zu wenig erkennt
    results = model.predict(img_array, conf=0.25)
    
    # 3. Layout für Ergebnisse
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Scan-Ergebnis")
        # Das Ergebnis-Bild von YOLO zeichnen lassen
        res_plotted = results[0].plot(line_width=2)
        st.image(res_plotted, caption="Erkannte Markierungen", use_container_width=True)
        
    with col2:
        st.subheader("Daten-Analyse")
        boxes = results[0].boxes
        st.metric("Erkannte Kreuze", len(boxes))
        
        if len(boxes) > 0:
            st.success("Kreuze wurden erfolgreich lokalisiert.")
            # Hier kannst du später die Logik einbauen:
            # z.B. "Wenn Box-X Koordinate in Bereich Y liegt -> Fach ist Biologie"
        else:
            st.warning("Keine Markierungen gefunden. Ist das Bild hell genug?")

    # Optional: Debug-Ansicht der Koordinaten
    with st.expander("Technische Details (Koordinaten)"):
        for i, box in enumerate(boxes):
            coords = box.xyxy[0].tolist()
            st.write(f"Box {i+1}: {coords}")
