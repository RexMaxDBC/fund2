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

# --- Initialisierung des Session States (einfache "Datenbank") ---
if 'fundstuecke' not in st.session_state:
    # Leeres DataFrame mit den Spalten für unsere Tabelle
    st.session_state.fundstuecke = pd.DataFrame(
        columns=['Datum', 'Gegenstand', 'Sicherheit', 'Bildname']
    )

# --- Modell laden (mit Caching) ---
@st.cache_resource
def load_model():
    """Lädt das YOLO-Modell. Wird nur einmal ausgeführt."""
    return YOLO("yolov8n.pt")  # Automatischer Download beim ersten Start!

model = load_model()

# --- Layout in zwei Spalten aufteilen ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Bild hochladen & analysieren")
    # Datei-Upload
    uploaded_file = st.file_uploader(
        "Wähle ein Bild...",
        type=["jpg", "jpeg", "png"],
        help="Lade ein Bild mit verlorenen Gegenständen hoch."
    )

    if uploaded_file is not None:
        # Bild anzeigen
        image = Image.open(uploaded_file)
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

        # --- Objekterkennung starten ---
        if st.button("🔍 Gegenstände erkennen", type="primary"):
            with st.spinner("Analysiere Bild... Bitte warten."):
                # Temporäre Datei für YOLO
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    image.save(tmp.name)
                    tmp_path = tmp.name

                # Objekterkennung durchführen
                results = model(tmp_path)

                # Bild mit Bounding Boxes anzeigen
                result_image = results[0].plot()
                st.image(result_image, caption="Erkannte Gegenstände", use_column_width=True)

                # --- Ergebnisse auslesen und in Session State speichern ---
                st.subheader("📋 Gefundene Gegenstände:")
                neue_funde = []
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = results[0].names[class_id]
                    confidence = float(box.conf[0])

                    # Ausgabe in der App
                    st.write(f"- **{class_name}** (Sicherheit: {confidence:.2%})")

                    # Für die Tabelle vorbereiten
                    neue_funde.append({
                        'Datum': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'Gegenstand': class_name,
                        'Sicherheit': f"{confidence:.2%}",
                        'Bildname': uploaded_file.name
                    })

                # Neue Funde zum DataFrame hinzufügen
                if neue_funde:
                    neue_df = pd.DataFrame(neue_funde)
                    st.session_state.fundstuecke = pd.concat(
                        [st.session_state.fundstuecke, neue_df],
                        ignore_index=True
                    )
                    st.success(f"{len(neue_funde)} Gegenstand/Gegenstände wurden dem Fundbuch hinzugefügt!")

                # Aufräumen
                os.unlink(tmp_path)

with col2:
    st.subheader("📚 Fundbuch")
    if st.button("🔄 Tabelle aktualisieren"):
        st.rerun()

    # Anzeige der gespeicherten Fundstücke
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

        # Download-Button für die Tabelle
        csv = st.session_state.fundstuecke.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tabelle als CSV herunterladen",
            data=csv,
            file_name=f'fundbucheintraege_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )

        # Button zum Löschen der Tabelle
        if st.button("🗑️ Tabelle leeren"):
            st.session_state.fundstuecke = pd.DataFrame(
                columns=['Datum', 'Gegenstand', 'Sicherheit', 'Bildname']
            )
            st.rerun()
    else:
        st.info("Noch keine Fundstücke erfasst. Lade ein Bild hoch und analysiere es.")

# --- Sidebar mit Informationen ---
with st.sidebar:
    st.header("ℹ️ Info")
    st.markdown("""
    **Willkommen im digitalen Fundbüro!**

    1.  Lade ein Bild mit verlorenen Gegenständen hoch.
    2.  Klicke auf "Gegenstände erkennen".
    3.  Die KI markiert die gefundenen Objekte.
    4.  Die Ergebnisse werden automatisch im Fundbuch (rechts) gespeichert.

    **Hinweis:** Die Erkennung basiert auf dem YOLOv8n-Modell.
    """)

    st.divider()
    st.caption("Entwickelt mit Streamlit und YOLOv8")
