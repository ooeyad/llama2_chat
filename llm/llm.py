# check gpu
from torch import cuda

# used to log into huggingface hub
from huggingface_hub import login
import pandas as pd
# used to load text
from langchain.document_loaders import WebBaseLoader, UnstructuredFileLoader

# used to create the retriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings

# used to setup language generation pipeline
from transformers import AutoTokenizer
import transformers
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from azure.storage.blob import BlobServiceClient, ContainerClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
# used to agent
from langchain import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain

HUGGINGFACE_TOKEN = "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"
MODEL = "meta-llama/Llama-2-7b-chat-hf"
account_name = 'st05559d002'
account_key = 'lRa7z5WLpANIH2HudGel4u2Kmp5R7iCFX+LxQmb8VtNRBivxQnG18ZgoGYpxSveCtRE0Gwd9jhpV+AStBNq4uA=='
container_name = 'fs05559d002'

file_name = ''
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
login(token=HUGGINGFACE_TOKEN)

connect_str = 'DefaultEndpointsProtocol=https;AccountName=' + account_name + ';AccountKey=' + account_key + ';EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(container_name)
for blob_i in container_client.list_blobs():
    f_name = str(blob_i.name)
    if f_name.find('LLM') != -1 and f_name.find('pdf') != -1:
        if f_name.find('somatosensory') !=-1:
            file_name = blob_i.name

sas_i = generate_blob_sas(account_name=account_name,
                              container_name=container_name,
                              blob_name=file_name,
                              account_key=account_key,
                              permission=BlobSasPermissions(read=True),
                              expiry=datetime.utcnow() + timedelta(hours=1))
sas_url = 'https://' + account_name + '.blob.core.windows.net/' + container_name + '/' + file_name + '?' + sas_i

# file_url = "https://st05559d002.blob.core.windows.net/fs05559d002/azure-ml-Pipelines/LLM/file1.pdf?sp=r&st=2023-12-06T14:59:12Z&se=2023-12-06T22:59:12Z&spr=https&sv=2022-11-02&sr=b&sig=Hd%2Fsy6TPGPGsHcIgvhSqVGpemAyLr%2F2rIsNdomz9Rag%3D"
print(sas_url)

# df = pd.read_csv(sas_url)
# print(df.head())
# loader = UnstructuredFileLoader(file_url)
# documents = loader.load()

import requests
import PyPDF2

# Replace with your Blob Storage URL
# blob_url = "https://st05559d002.blob.core.windows.net/fs05559d002/azure-ml-Pipelines/LLM/file1.pdf?sp=r&st=2023-12-06T14:59:12Z&se=2023-12-06T22:59:12Z&spr=https&sv=2022-11-02&sr=b&sig=Hd%2Fsy6TPGPGsHcIgvhSqVGpemAyLr%2F2rIsNdomz9Rag%3D"

# Fetch the PDF content from the URL
response = requests.get(sas_url)

pdf_url = 'downloaded_pdf.pdf'
# Save the PDF content to a local file (optional)
with open(pdf_url, 'wb') as f:
    f.write(response.content)

loader = UnstructuredFileLoader(pdf_url)
documents = loader.load()

text_splitter=CharacterTextSplitter(separator='\n',
                                    chunk_size=1000,
                                    chunk_overlap=50)
texts=text_splitter.split_documents(documents)

print(texts)
embeddings = GPT4AllEmbeddings()
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
stop_token_ids

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])


pipeline = transformers.pipeline(
    task="text-generation", #task
    model=MODEL,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    repetition_penalty=1.1,  # without this output begins repeating
    torch_dtype=torch.bfloat16,
    device="auto",
    trust_remote_code=True,
    max_length=4000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

print("Done loaded the model!")
