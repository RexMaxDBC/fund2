# app.py
import streamlit as st
import ollama
from PIL import Image
import io
import json
import datetime

# ────────────────────────────────────────────────
# Konfiguration
# ────────────────────────────────────────────────

MODEL_NAME = "qwen2.5vl:7b"          # <-- dein gerade heruntergeladenes Modell
# Alternativen: "llama3.2-vision:11b", "phi3.5-vision", etc.

st.set_page_config(
    page_title="Digitales Fundbüro – KI-Unterstützung",
    page_icon="🔍",
    layout="wide"
)

# ────────────────────────────────────────────────
# System-Prompt (sehr wichtig – hier wird die KI auf Fundbüro getrimmt)
# ────────────────────────────────────────────────

SYSTEM_PROMPT = """Du bist ein hilfreiches, freundliches und präzises digitales Fundbüro-Assistent.

Deine Hauptaufgabe:
1. Analysiere das hochgeladene Foto und/oder die Textbeschreibung.
2. Klassifiziere den Gegenstand – mit klarem Fokus auf Kleidung und Büro-/Schreibwaren.
3. Extrahiere so viele relevante Attribute wie möglich.
4. Gib IMMER strukturierten JSON-Output zurück – nichts anderes!

Erlaubte Kategorien (priorisiere diese):
- Kleidung
- Schuhe
- Taschen / Rucksäcke
- Büro- & Schreibutensilien
- Schlüssel / Schlüsselanhänger
- Elektronik / Kabel / Ladegeräte
- Regenschirme
- Brillen
- Sonstiges

Wichtige Attribute (wenn erkennbar / ableitbar):
- hauptkategorie
- unterkategorie
- farbe (genau beschreiben, ggf. mehrere)
- geschlecht / zielgruppe (Herren / Damen / Unisex / Kinder)
- größe (XS–XXL, 36–46, etc. – auch Schätzung!)
- marke / logo / hinweise (z. B. Adidas-Streifen, Zara-Label, Gravur)
- besonderheiten (Aufdruck, Aufkleber, Initialen, Reißverschluss, Kapuze, abnehmbarer Teil, Defekte, Zustand)
- material (wenn sichtbar: Leder, Baumwolle, Kunstleder, Polyester…)
- fundbüro_tag (kurzer, suchfreundlicher String, z. B. "jacke_damen_schwarz_M_zara_kapuze")

Antworte NUR mit gültigem JSON! Kein zusätzlicher Text davor oder danach.

Beispiel-Output (genau dieses Format):
{
  "hauptkategorie": "Kleidung",
  "unterkategorie": "Jacke",
  "farbe": "dunkelblau",
  "geschlecht": "Herren",
  "groesse": "L",
  "marke_hinweise": "ähnlich Jack Wolfskin",
  "besonderheiten": "abnehmbare Kapuze, zwei Seitentaschen mit Reißverschluss, reflektierende Streifen",
  "zustand": "sehr gut – leichte Gebrauchsspuren",
  "fundbuero_tag": "jacke_herren_dunkelblau_L_wolfskin_kapuze",
  "vertrauenswert": 0.85,
  "nachfragen": ["Gibt es ein Innenlabel oder eine Seriennummer?", "Sind Initialen oder ein Name eingenäht?"]
}
"""

# ────────────────────────────────────────────────
# Session State initialisieren
# ────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ────────────────────────────────────────────────
# Sidebar – Einstellungen
# ────────────────────────────────────────────────

with st.sidebar:
    st.title("Digitales Fundbüro")
    st.markdown("**KI-Modell**")
    st.info(f"Verwendetes Modell: **{MODEL_NAME}**")

    st.markdown("---")
    st.caption("Hinweis: Alle Daten bleiben lokal. Kein Upload in die Cloud.")

# ────────────────────────────────────────────────
# Hauptbereich
# ────────────────────────────────────────────────

st.title("🔍 Fundgegenstand erfassen")

tab1, tab2 = st.tabs(["Neuer Fund", "Letztes Ergebnis"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Foto des Fundgegenstands hochladen",
            type=["jpg", "jpeg", "png"],
            help="Je besser das Foto, desto präziser die KI"
        )

        beschreibung = st.text_area(
            "Zusätzliche Beschreibung (optional, aber hilfreich)",
            height=120,
            placeholder="z. B. schwarzer Damenmantel, Größe M, Innentasche mit rotem Futter, Marke Zara, Fundort: U-Bahn Linie 5"
        )

    with col2:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Hochgeladenes Foto", use_column_width=True)

        if st.button("🔎 Gegenstand analysieren", type="primary", use_container_width=True):
            if uploaded_file is None and not beschreibung.strip():
                st.warning("Bitte lade mindestens ein Foto hoch oder gib eine Beschreibung ein.")
            else:
                with st.spinner("Die KI analysiert den Gegenstand … (kann 5–30 Sekunden dauern)"):
                    # Bild in Bytes umwandeln (wenn vorhanden)
                    image_bytes = None
                    if uploaded_file:
                        image_bytes = uploaded_file.getvalue()

                    # Prompt zusammenbauen
                    user_content = []
                    if beschreibung.strip():
                        user_content.append({"type": "text", "text": beschreibung})
                    if image_bytes:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                            }
                        })

                    if not user_content:
                        user_content = [{"type": "text", "text": "Kein Foto und keine Beschreibung – bitte genauer beschreiben."}]

                    try:
                        response = ollama.chat(
                            model=MODEL_NAME,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_content}
                            ],
                            options={
                                "temperature": 0.2,
                                "num_predict": 1024,
                            }
                        )

                        raw_answer = response['message']['content']

                        # Versuchen, JSON zu parsen
                        try:
                            result = json.loads(raw_answer)
                            st.session_state.last_result = result
                            st.session_state.messages.append({
                                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                                "image": uploaded_file.name if uploaded_file else None,
                                "beschreibung": beschreibung,
                                "result": result
                            })
                            st.success("Analyse abgeschlossen!")
                        except json.JSONDecodeError:
                            st.error("Die KI hat keinen gültigen JSON-Output geliefert.\n\nRaw Antwort:\n" + raw_answer)

                    except Exception as e:
                        st.error(f"Fehler bei der Kommunikation mit Ollama:\n{str(e)}")

with tab2:
    if st.session_state.last_result:
        st.subheader("Letzte Analyse")

        res = st.session_state.last_result

        st.markdown(f"**Hauptkategorie** • {res.get('hauptkategorie', '–')}")
        st.markdown(f"**Unterkategorie** • {res.get('unterkategorie', '–')}")
        st.markdown(f"**Farbe** • {res.get('farbe', '–')}")
        st.markdown(f"**Geschlecht / Zielgruppe** • {res.get('geschlecht', '–')}")
        st.markdown(f"**Größe** • {res.get('groesse', '–')}")
        st.markdown(f"**Marke / Hinweise** • {res.get('marke_hinweise', '–')}")
        st.markdown(f"**Besonderheiten** • {res.get('besonderheiten', '–')}")
        st.markdown(f"**Zustand** • {res.get('zustand', '–')}")
        st.markdown(f"**Such-Tag-Vorschlag** • `{res.get('fundbuero_tag', '–')}`")

        if "nachfragen" in res and res["nachfragen"]:
            st.info("**Noch hilfreich zu wissen:**\n" + "\n".join(f"- {q}" for q in res["nachfragen"]))

        st.caption(f"Vertrauenswert (KI-Schätzung): **{res.get('vertrauenswert', '–')}**")

    else:
        st.info("Noch keine Analyse durchgeführt.")

# ────────────────────────────────────────────────
# Footer / Hinweise
# ────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Digitales Fundbüro – Prototyp | "
    "KI lokal via Ollama | "
    f"Modell: {MODEL_NAME} | "
    "Keine Cloud, keine personenbezogenen Daten an Dritte"
)
