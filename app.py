# FALSCH: Wir zeigen das originale Bild an, nicht das mit Boxen
st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

# RICHTIG: Wir müssen das Bild MIT Boxen anzeigen
result_image = results[0].plot()  # Hier sind die Boxen drauf
st.image(result_image, caption="Erkannte Gegenstände", use_column_width=True)  # Das kommt zu spät!
