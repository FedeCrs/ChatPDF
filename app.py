import tiktoken
import numpy as np
import pdfplumber
import openai
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import json
from contextlib import asynccontextmanager


# Cargar variables de entorno
load_dotenv()

# Aquí se obtiene la clave de OpenAI desde el archivo .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Verificar que la clave se haya cargado
if not OPENAI_API_KEY:
    raise ValueError("Error: La clave de OpenAI no está configurada correctamente en .env")
else:
    print("Clave de OpenAI cargada correctamente")

# Aquí se inicializa el cliente de OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ API iniciada correctamente (startup)")
    yield
    print("🛑 API cerrada (shutdown)")

app = FastAPI(lifespan=lifespan)


# Aquí se crea un directorio para guardar los PDFs
TEMP_DIR = os.path.abspath("temp_dir")
if not os.path.exists(TEMP_DIR):
    os.makedirs("temp_dir", exist_ok=True)

# Aquí se definen las funciones para procesar el PDF y generar respuestas
def extract_text_from_pdf(file_path: str) -> str:
    """Extrae el texto de un archivo PDF."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text.replace("\n", " ") + " "  # Elimina saltos de línea
        # Imprimir el texto extraído para depuración            
        if not text.strip():
            print("El PDF no tiene texto extraíble o está escaneado.")
            return ""
        
        print("Texto extraído del PDF:", text[:500])  # Muestra los primeros 500 caracteres

    # Manejar errores al leer el PDF
    except Exception as e:
        print(f"Error al leer el PDF: {e}")
        return ""
    return text.strip()

# Aquí se definen las funciones para dividir el contexto en fragmentos
def chunk_context(context_text: str, max_tokens: int = 1000) -> list:
    """Divide el contexto en fragmentos de tamaño máximo max_tokens."""
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = encoding.encode(context_text)
    # Dividir el texto en fragmentos de tamaño máximo max_tokens
    chunks = [encoding.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
    # Imprimir el número de fragmentos generados
    print(f"Se generaron {len(chunks)} fragmentos de contexto.")
    # Imprimir los primeros 2 fragmentos para depuración
    for i, chunk in enumerate(chunks[:2]):
        print(f"Fragmento {i + 1}:", chunk[:300])
    return chunks

# Aquí se definen las funciones para generar embeddings y obtener respuestas
def get_embeddings(text: str) -> list:
    """Genera embeddings para un texto dado."""
    try:
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
        # Imprimir la respuesta de la API OpenAI para depuración
        print("Respuesta API OpenAI (embeddins):", response)
        
        # Extraer el embedding del texto
        #embedding = response["data"][0]["embedding"]
        
        return response.data[0].embedding
    except Exception as e:
        print(f"Error al generar embeddings: {e}")
        return []
       
# Aquí se definen las funciones para obtener el contexto más relevante
def get_relevant_context(query: str, context_chunks: list) -> str:
    """Encuentra el fragmento de contexto más relevante para una consulta."""
    query_embedding = get_embeddings(query)
    if not query_embedding:
        return "No se pudo generar el embedding para la consulta."
    
    # Imprimir el embedding de la consulta para depuración
    print("Embedding de la consulta:", query_embedding) 

    # Generar embeddings para cada fragmento de contexto
    context_embeddings = [get_embeddings(chunk) for chunk in context_chunks]
    context_embeddings = [emb for emb in context_embeddings if emb]  # Filtrar embeddings vacíos
    
    # Si no hay embeddings válidos, devolver cadena vacía
    if not context_embeddings:
        return "No se generaron embeddings válidos para los fragmentos de contexto."
    
    # Calcular similitud coseno entre la consulta y los fragmentos de contexto
    similarities = cosine_similarity([query_embedding], context_embeddings)[0]
    
    # Imprimir las similitudes para depuración
    print("Similitudes entre consulta y fragmentos:", similarities)
    
    # Imprimir el índice del fragmento más relevante
    if max(similarities) < 0.2: # Si la similitud es demasiado baja, baja el umbral de 0.3 a 0.2
        # Imprimir advertencia si la similitud es demasiado baja
        print("⚠️ No se encontró contexto relevante con una similitud suficiente para la consulta.")
        return ""
    top_index = np.argmax(similarities)
    # Imprimir el fragmento más relevante
    print("Fragmento más relevante (similitud: {similarities[top_index]:.2f})", context_chunks[top_index][:300])
    # Devolver el fragmento de contexto más relevante
    return context_chunks[top_index]
    
# Aquí se define la función para obtener una respuesta
def get_response(query: str, conversation_history: list, context_chunks: list) -> str:
    """Genera una respuesta basada en la consulta y contexto relevante."""
    relevant_context = get_relevant_context(query, context_chunks)
    
    # Imprimir el contexto relevante para depuración
    print("Contexto relevante:", relevant_context)
    
    if not relevant_context:
        return "Lo siento, no pude encontrar información relevante para responder a tu pregunta."
    messages = conversation_history + [{"role": "system", "content": relevant_context}, {"role": "user", "content": query}]
    
    try:
        response = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=messages
        )
        # Imprimir la respuesta generada por OpenAI para depuración
        print("Respuesta generada por OpenAI:", response)
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error al obtener respuesta: {e}")
        return "Error al generar respuesta."

# Endpoint para cargar un archivo PDF
@app.post("/subir_pdf/")
async def subir_pdf(file: UploadFile):
    # Verificar que el archivo sea un PDF
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF.")

    # Guarda el archivo cargado en el servidor
    file_path = os.path.join(TEMP_DIR, file.filename)
    
    try:
        # Guardar el archivo en el sistema de archivos
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        
        # Extraer el texto del PDF
        pdf_text = extract_text_from_pdf(file_path)
        if not pdf_text:
            print("El PDF no tiene texto extraíble o está escaneado.")
            raise HTTPException(status_code=400, detail="El PDF no tiene texto extraíble o está escaneado.")
    
        # Dividir el texto en fragmentos para la búsqueda semántica
        context_chunks = chunk_context(pdf_text)


        # Imprimir los primeros 2 fragmentos para depuración
        print("Fragmentos generados:", context_chunks [:2])

        return JSONResponse(content={"message": "PDF cargado y procesado correctamente", "context_chunks": context_chunks, "file_id": file.filename})   
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al cargar el archivo: {str(e)}")

# Aquí se define el modelo de solicitud para hacer preguntas
class PreguntaRequest(BaseModel):
    pregunta: str
    context_chunks: List[str]

# Endpoint para hacer preguntas sobre el contenido del PDF
@app.post("/preguntar/")
async def preguntar(request: PreguntaRequest):
    print("Query recibida en FastAPI:", request.pregunta) # Verificar la consulta recibida
    print("Contexto recibido en FastAPI:", len(request.context_chunks), "fragmentos")   # Depurar el contexto recibido
    """Realiza una consulta sobre el texto extraído del PDF."""
    ###pregunta = query.get("pregunta", "")

    # Obtener el contexto relevante
    ###context_chunks = query.get("context_chunks", [])
    if not request.context_chunks:
       # Imprimir advertencia si no se encontraron fragmentos de contexto
       print("⚠️ No se encontraron fragmentos de contexto")
       return JSONResponse(content={"respuesta": "No se encontró contexto relevante para la consulta."})
    
    # Generar una respuesta basada en la consulta y el contexto
    respuesta = get_response(request.pregunta, [], request.context_chunks)
    print("Respuesta generada:", respuesta) # Verificar la respuesta generada                         
    return JSONResponse(content={"respuesta": respuesta})


# Ruta de inicio
@app.get("/")
async def root():
    return {"message": "Bienvenido a la API de preguntas sobre PDFs 🚀"}


if __name__ == "__main__":
    print("Iniciando la API de preguntas sobre PDFs...")
