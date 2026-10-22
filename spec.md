Pre-entrega 3: Sistema de recuperación semántica local (RAG)

¿Qué debes construir?

Debes entregar un script o notebook de Python que implemente un flujo End-to-End de RAG. El sistema debe ser capaz de recibir una consulta del usuario, buscar la información relevante en una base vectorial previamente poblada y generar una respuesta que utilice exclusivamente esa información.

Componentes a entregar:

Módulo de Ingesta (Setup): Una función que tome un conjunto de documentos (pueden ser archivos .txt o .md simples), los fragmente (chunking) y los persista en una colección de ChromaDB.
Capa de Recuperación (Retriever): Una lógica que convierta la pregunta del usuario en un embedding y recupere los fragmentos más relevantes.
Generación Grounded: Una cadena de LangChain (LCEL) que reciba los documentos recuperados y la pregunta, generando una respuesta. El prompt debe instruir al modelo a decir "No lo sé" si la respuesta no está en el contexto.
Pasos sugeridos
Puebla tu "Cerebro": Elige 3 o 4 archivos de texto sobre un tema específico (ej. manuales técnicos, apuntes de clase o normativas). Usa las técnicas de RecursiveCharacterTextSplitter que vimos en la Unidad 2 para procesarlos.
Configura ChromaDB: Inicializa el cliente persistente en una carpeta local (ej. ./vectorstore). Asegúrate de usar el mismo modelo de embeddings para indexar y para consultar.
Construye el Prompt de Sistema: Diseña un prompt que actúe como un "filtro de veracidad". Ejemplo: "Eres un asistente técnico. Responde solo basándote en el CONTEXTO proporcionado. Si la respuesta no está allí, di que no tienes acceso a esa información."
Crea la Cadena LCEL: Une el retriever con el transformador de documentos y el modelo de lenguaje. Recuerda que el output debe pasar por un PydanticOutputParser.
Errores comunes a evitar
El "Contexto Infinito": No intentes pasar 50 fragmentos al LLM. Esto causará errores de límite de tokens o degradación de la atención del modelo (Lost in the Middle). Mantén un top_k de entre 3 y 5.
Embeddings no coincidentes: Es el error #1. Si indexas tus documentos con OpenAIEmbeddings y consultas con HuggingFaceEmbeddings, la distancia vectorial no tendrá sentido y los resultados serán aleatorios.
Falta de persistencia: Asegúrate de que el script verifique si la base de datos ya existe antes de volver a indexar todo, optimizando así el tiempo y costo de ejecución.


Entregable

Crea un entorno virtual e instala las dependencias necesarias: langchain, chromadb, openai (o el provider seleccionado), y pydantic.
Implementa un script de carga que lea archivos de una carpeta /data, aplique un chunking estratégico (mínimo 500 tokens con 50 de overlap) y los guarde en una instancia local de ChromaDB.
Define una clase o función asíncrona get_rag_response(query: str) que: a. Realice una búsqueda de similitud en ChromaDB. b. Construya un prompt incluyendo los fragmentos recuperados. c. Llame al LLM de forma asíncrona. d. Parsee la respuesta a un modelo Pydantic que incluya el texto y las referencias.
Realiza dos pruebas: una pregunta cuya respuesta esté en los documentos y una "pregunta trampa" cuya respuesta no esté, verificando que el modelo no alucine.
Sube tu código asegurándote de no incluir llaves de API (usa variables de entorno .env).