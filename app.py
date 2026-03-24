import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Seite konfigurieren
st.set_page_config(page_title="KI Profil-Check", layout="centered")

# Modell laden
@st.cache_resource
def load_custom_model():
    # Hier wird dein hochgeladenes Modell verwendet
    # Stelle sicher, dass die Datei 'yolov8n-2.pt' im selben Ordner auf GitHub liegt
    return YOLO('yolov8n-2.pt')

try:
    model = load_custom_model()
    st.success("KI-Modell 'yolov8n-2.pt' erfolgreich geladen!")
except Exception as e:
    st.error(f"Fehler beim Laden des Modells: {e}")
    st.info("Checke, ob die Datei 'yolov8n-2.pt' im Hauptverzeichnis deines GitHub-Repos liegt.")

st.title("📝 Wahlzettel-Scanner")
st.write("Lade ein Foto deines ausgefüllten Wahlzettels hoch.")

# File Uploader
uploaded_file = st.file_uploader("Bild hochladen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild vorbereiten
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert("RGB")) # Sicherstellen, dass es RGB ist
    
    # Button zum Starten der Analyse
    if st.button("Wahlzettel analysieren"):
        with st.spinner('KI analysiert das Dokument...'):
            # Inferenz durchführen
            results = model(img_array)
            
            # Ergebnisse anzeigen
            st.subheader("Analyse-Ergebnis")
            
            # Gezeichnetes Bild generieren
            res_plotted = results[0].plot() 
            st.image(res_plotted, caption="Erkannte Bereiche", use_container_width=True)
            
            # Details zu den Boxen
            boxes = results[0].boxes
            st.write(f"Anzahl erkannter Markierungen: **{len(boxes)}**")
            
            if len(boxes) > 0:
                st.write("### Einzelansicht der Kästchen:")
                cols = st.columns(4) # Erstellt 4 Spalten für kleine Vorschauen
                
                for i, box in enumerate(boxes):
                    # Koordinaten extrahieren
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Ausschnitt erstellen (Crop)
                    crop = img_array[y1:y2, x1:x2]
                    
                    # In der passenden Spalte anzeigen
                    cols[i % 4].image(crop, caption=f"Box {i+1}")
            else:
                st.warning("Keine Markierungen gefunden. Versuche es mit einem schärferen Bild.")
