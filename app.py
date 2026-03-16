import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import pandas as pd
from datetime import datetime

# --- Seitenkonfiguration ---
st.set_page_config(
    page_title="Digitales Fundbüro",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Digitales Fundbüro")
st.write("Lade ein Bild hoch, um Gegenstände zu erkennen und im Fundbuch zu erfassen.")

# --- Initialisierung des Session States ---
if 'fundstuecke' not in st.session_state:
    st.session_state.fundstuecke = pd.DataFrame(
        columns=['Datum', 'Gegenstand', 'Sicherheit', 'Bildname']
    )

# --- Modell laden ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --- Layout in zwei Spalten ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Bild hochladen & analysieren")
    uploaded_file = st.file_uploader(
        "Wähle ein Bild...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Original-Bild anzeigen
        image = Image.open(uploaded_file)
        st.image(image, caption="Original-Bild", use_column_width=True)

        # Analysieren-Button
        if st.button("🔍 Gegenstände erkennen", type="primary"):
            with st.spinner("Analysiere Bild..."):
                # RGBA zu RGB konvertieren falls nötig
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                
                # Temporäre Datei
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    image.save(tmp.name)
                    tmp_path = tmp.name

                # YOLO Analyse
                results = model(tmp_path)
                
                # **WICHTIG: Hier das Bild MIT Boxen anzeigen**
                result_image = results[0].plot()
                st.image(result_image, caption="Erkannte Gegenstände (mit Markierungen)", use_column_width=True)

                # Ergebnisse als Text
                st.subheader("📋 Gefundene Gegenstände:")
                neue_funde = []
                
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = results[0].names[class_id]
                    confidence = float(box.conf[0])

                    # Textausgabe
                    st.write(f"- **{class_name}** (Sicherheit: {confidence:.2%})")

                    # Für Tabelle speichern
                    neue_funde.append({
                        'Datum': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'Gegenstand': class_name,
                        'Sicherheit': f"{confidence:.2%}",
                        'Bildname': uploaded_file.name
                    })

                # In Session State speichern
                if neue_funde:
                    neue_df = pd.DataFrame(neue_funde)
                    st.session_state.fundstuecke = pd.concat(
                        [st.session_state.fundstuecke, neue_df],
                        ignore_index=True
                    )
                    st.success(f"{len(neue_funde)} Gegenstände wurden dem Fundbuch hinzugefügt!")

                # Aufräumen
                os.unlink(tmp_path)

with col2:
    st.subheader("📚 Fundbuch")
    
    # Fundbuch anzeigen
    if not st.session_state.fundstuecke.empty:
        st.dataframe(
            st.session_state.fundstuecke,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Datum": "Datum/Uhrzeit",
                "Gegenstand": "Gefundener Gegenstand",
                "Sicherheit": "Erkennungs-Sicherheit",
                "Bildname": "Quelldatei"
            }
        )

        # CSV Download
        csv = st.session_state.fundstuecke.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Als CSV herunterladen",
            data=csv,
            file_name=f'fundbucheintraege_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )

        # Tabelle leeren
        if st.button("🗑️ Tabelle leeren"):
            st.session_state.fundstuecke = pd.DataFrame(
                columns=['Datum', 'Gegenstand', 'Sicherheit', 'Bildname']
            )
            st.rerun()
    else:
        st.info("Noch keine Fundstücke erfasst.")

# --- Sidebar ---
with st.sidebar:
    st.header("ℹ️ Info")
    st.markdown("""
    **Digitales Fundbüro**
    
    1. Bild hochladen
    2. "Gegenstände erkennen" klicken
    3. Ergebnisse werden im Bild markiert
    4. Automatische Speicherung im Fundbuch
    """)
