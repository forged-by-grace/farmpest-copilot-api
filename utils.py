# Import dependencies
import numpy as np
from io import BytesIO
import tensorflow as tf
from PIL import Image
import keras

from langchain.retrievers.web_research import WebResearchRetriever
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain

# Load the pest classes
class_names = ['aphids',
 'armyworm',
 'beetle',
 'bollworm',
 'grasshopper',
 'mites',
 'mosquito',
 'sawfly',
 'stem_borer']

# Set image size
IMAGE_SIZE = (150, 150)

# Load model
model = keras.models.load_model("./ml_models/1")

async def read_file_as_image(data) -> np.ndarray:
    # Read file and convert to np array
    image = np.array(Image.open(BytesIO(data)))
    return image


async def preprocess_image(img):
    # Resize image
    resized_img = tf.image.resize(images=img, size=IMAGE_SIZE) 
    
    # Convert img to np array
    input_arr = keras.utils.img_to_array(resized_img)
    
    # Batch image
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    
    # Return img as array
    return input_arr


async def predict(input_arr):
    # Predict image    
    predictions = model.predict(input_arr)

    # Identify the class
    predicted_class = class_names[np.argmax(predictions[0])]
    
    # Compute the confidence
    confidence = round(100 * (np.max(predictions[0])), 2)
    
    # Return result
    return predicted_class, confidence


async def generate_pest_control(pest, language, location):
    # Vectorstore
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai"
    )

    # LLM
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", max_tokens=500)

    # Search
    search = GoogleSearchAPIWrapper()

    # Initialize
    web_research_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore, llm=llm, search=search
    )

    # Retriever
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm, retriever=web_research_retriever
    )

    # Query
    query = f"In multiple paragraphs, professionally state the economic importance  of {pest} and the different methods I can use in controlling it on my farm in {location}."
    
    # Get the result
    result = qa_chain({"question": query})
    
    return result