import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.title("🏢 Digitales Fundbüro")
st.write("Lade ein Bild hoch, um Gegenstände zu erkennen")

# Modell laden (wird automatisch heruntergeladen)
@st.cache_resource  # Wichtig: Caching, damit das Modell nicht bei jedem Neuladen neu geladen wird
def load_model():
    return YOLO("yolov8n.pt")  # Automatischer Download!

model = load_model()

# Datei-Upload
uploaded_file = st.file_uploader("Wähle ein Bild...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild öffnen
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
    
    # Temporäre Datei für YOLO
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name
    
    # Objekterkennung
    with st.spinner("Analysiere Bild..."):
        results = model(tmp_path)
    
    # Ergebnisse anzeigen
    result_image = results[0].plot()  # Bild mit Bounding Boxes
    st.image(result_image, caption="Erkannte Gegenstände", use_column_width=True)
    
    # Liste der gefundenen Objekte
    st.subheader("📋 Gefundene Gegenstände:")
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        confidence = float(box.conf[0])
        st.write(f"- **{class_name}** (Sicherheit: {confidence:.2%})")
    
    # Aufräumen
    os.unlink(tmp_path)
