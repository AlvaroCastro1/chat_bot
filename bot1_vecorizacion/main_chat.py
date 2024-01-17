import os
import openai
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext
from langchain_openai import ChatOpenAI



import textwrap
from dotenv import load_dotenv

# Carga las variables de entorno desde el archivo .env
load_dotenv()

# Accede a la clave API
API_KEY = os.getenv("OPENAI_API_KEY")

#Cámbiala por tu API de OpenAI
os.environ["OPENAI_API_KEY"] = API_KEY

#Leer los PDFs
pdf = SimpleDirectoryReader('datos').load_data()

#Definir e instanciar el modelo

modelo = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'))

#Indexar el contenido de los PDFs
service_context = ServiceContext.from_defaults(llm_predictor=modelo)
index = GPTVectorStoreIndex.from_documents(pdf, service_context = service_context)

#Guardar el índice a disco para no tener que repetir cada vez
#Recordar que necesistaríamos persistir el drive para que lo mantenga
#index.save_to_disk('index.json')

#Cargar el índice del disco
#index = GPTVectorStoreIndex.load_from_disk('index.json')

while True:
  pregunta = input('Escribe tu pregunta   \n') + "Responde en español"
  respuesta = index.as_query_engine().query(pregunta)
  for frase in textwrap.wrap(respuesta.response, width=100):
    print(frase)