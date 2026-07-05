from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint, HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough , RunnableParallel, RunnableSequence , RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

#LLM setup
llm  = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature = 0.2
)
model = ChatHuggingFace(llm = llm)


#1- Build prompt template
prompt = PromptTemplate(
    template = """ 
    You are helpul assistant. 
    Answer ONLY from the provided transcript context,
    If the context is not sufficient , just say you don't know.
     
    {context}
    Question : {question}

    Answer:
    """,
    input_variables = ["context" , "question"]

)


#2- get yt tanscript and create FAISS vector store
#2a-Indexing- use yt api to get video transcript
video_id = "5KmopXwjXik"
transcript = ""
try: 
    #sentance by sentance 
    ytt = YouTubeTranscriptApi()
    fetched = ytt.fetch(video_id , languages =["en"])
    transcript_list = fetched.snippets

    #Flatten it to plain txt
    transcript = " ".join(chunk.text for chunk in transcript_list)
    #print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

    #Gaurd before continuing
    if not transcript: 
        exit("No transcript available. Exiting the program.")

#2b-Text Splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
chunks = splitter.create_documents([transcript])

#2c-Initialize Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#2d. Create the FAISS vector store
vector_store = FAISS.from_documents(chunks , embeddings)

#print number of vectors in FAISS
print(f"Vector store created with {vector_store.index.ntotal} vectors")

#2e-Retrieval 
retriever = vector_store.as_retriever(search_type="similarity" , search_kwargs={"k":4})



#2f-convert the list of docs into a single string for the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



parser = StrOutputParser()

#3- Create a runnable chain
parallel_chain = RunnableParallel ({
    'question' : RunnablePassthrough(),
    'context' : retriever | RunnableLambda(format_docs)

})

main_chain = parallel_chain | prompt | model | parser

result = main_chain.invoke("What is the main topic of the video?")
print(result)
