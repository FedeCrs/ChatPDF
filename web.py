import streamlit as st
import requests
import os
import uuid

# Configuración de la página de Streamlit
st.title("Chat para consultas en Documentos PDF 📄🤖")

# Variable global para guardar el contexto extraído del PDF
context_chunks = None

# Subidad de archivo PDF
subida = st.file_uploader("Sube un archivo PDF", type=["pdf"])

if subida is not None:
    # Crear una carpeta "temporal" si no existe para guardar los PDFs
    pdf_dir = "temp_dir"
    os.makedirs(pdf_dir, exist_ok=True)

    # Guardar el PDF en el servidor temporalmente
    pdf_path = os.path.join(pdf_dir, subida.name)
    with open(pdf_path, 'wb') as f:
        f.write(subida.getbuffer())

    # Mostrar un mensaje de éxito
    st.success(f"PDF '{pdf_path}' cargado correctamente ✅")

    # Enviar el PDF a la API de FastAPI
    try:
        with open(pdf_path, "rb") as f:
            response = requests.post("http://127.0.0.1:8000/subir_pdf/", files={"file": f}, timeout=60)

        if response.status_code == 200:
            data = response.json()
            print("Respuesta JSON de FastAPI (subir_pdf):", data)   # Imprimir la respuesta JSON para depuración
            context_chunks = data.get("context_chunks", [])  # Obtener los fragmentos de contexto del PDF
            if context_chunks:
                st.session_state["context_chunks"] = context_chunks  # Guardar los fragmentos en la sesión de Streamlit
                # Mostrar un mensaje de éxito
                st.write("PDF procesado correctamente. Contexto extraído ✅")
                st.write(f"Se extrajeron {len(context_chunks)} fragmentos de contexto.")
            else:
                st.error("⚠️ No se encontraron fragmentos de contexto en el PDF.")
        else:
            st.error(f"Error al subir el PDF: {response.status_code} - {response.text}")    
    except requests.exceptions.RequestException as e:
        st.error(f"Error al subir el PDF: {e}")

    # Eliminar el PDF del servidor después de procesarlo    
    os.remove(pdf_path)
    
# Entrada de texto para preguntas
pregunta = st.text_input("Escribe tu pregunta:")

# Enviar la pregunta a la API para obtener una respuesta basada en el contenido del PDF
if pregunta.strip():
    try:
        with st.spinner("Procesando tu pregunta..."):
            # Asegura que context_chunks seguarde antes de enviar la pregunta para que no sea 'None'
            context_chunks = st.session_state.get("context_chunks", [])
            # Crear un payload con la pregunta y el contexto
            payload = {"pregunta": pregunta, "context_chunks": context_chunks} # Enviar la pregunta y el contexto
            print("Payload enviado al servidor FastAPI:", payload)  # Imprimir el payload para depuración   
            response = requests.post("http://localhost:8000/preguntar/", headers={"Content-Type": "application/json"}, json=payload)
        # Verificar si la solicitud fue exitosa
        if response.status_code == 200:
            data = response.json()
            # Imprimir la respuesta JSON recibida para depuración
            print("🔍 Respuesta JSON recibida:", data)
            # Verificar si la clave 'respuesta' está presente en la respuesta JSON
            if "respuesta" in data:
                st.write(f"**Respuesta:** {data['respuesta']}")  # Mostrar la respuesta generada
            else:
                st.error("⚠️ No se encontró respuesta en la respuesta JSON del servidor.") 
        else:
            st.error(f"Error en la solicitud: {response.status_code} - {response.text}") # Mostrar un mensaje de error     
    except requests.exceptions.RequestException as e:
        st.error(f"Error al conectar con el servidor: {e}")
else:
    st.warning("Por favor, sube un archivo PDF y haz una pregunta para continuar.")
      

        
        
    
    
    
