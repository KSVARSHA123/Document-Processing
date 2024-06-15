from langchain.document_loaders import TextLoader
loader=TextLoader("data.txt")
document=loader.load()
print(document)
#Preprocessing
import textwrap
def wrap_text_preserve_newlines(text,width=110):
  lines=text.split('\n')
  wrapped_lines=[textwrap.fill(line,width=width) for line in lines]
  wrapped_text='\n'.join(wrapped_lines)
  return wrapped_text
print(wrap_text_preserve_newlines(str(document[0])))
#Text Splitting
from langchain_text_splitters import CharacterTextSplitter
text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
docs=text_splitter.split_documents(document)
print(docs)
print(len(docs))
#Embedding
from langchain.embeddings import HuggingFaceEmbeddings
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"
from langchain.vectorstores import FAISS
embeddings=HuggingFaceEmbeddings()
db=FAISS.from_documents(docs,embeddings)
query="ai"
doc=db.similarity_search(query)
print(wrap_text_preserve_newlines(str(doc[0].page_content)))
