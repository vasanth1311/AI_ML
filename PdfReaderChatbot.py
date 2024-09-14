import pdftotext
from ollama import chat
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer  # Import SentenceTransformer

# Function to read PDF using PyPDF2 and extract text
def read_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf = PdfReader(f)
        text = ""
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            text += page.extract_text()
    return text

# Function to chunk text using langchain's CharacterTextSplitter
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to get word embeddings using SentenceTransformer
def get_word_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a pre-trained SentenceTransformer model
    embeddings = [model.encode(chunk) for chunk in chunks]
    return embeddings

# Function to process chunks (for demonstration purposes, you can customize this)
def process_chunks(chunks):
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk[:200]}...")  # Print the first 200 characters of each chunk

# Main function to run the process
def main(pdf_path):
    # Read and chunk the PDF text
    text = read_pdf(pdf_path)
    chunks = chunk_text(text)

    # Get embeddings for each chunk
    embeddings = get_word_embeddings(chunks)
    for i, embedding in enumerate(embeddings):
        print(f"Chunk {i+1} Embeddings: {embedding[:10]}...")  # Print the first 10 elements of the embeddings

    # Process chunks (optional step depending on your use case)
    process_chunks(chunks)

    # Define your chatbot's personality (LLaMA model)
    model = 'tinyllama'
    print("Welcome to the LLaMA-based PDF reader! Type 'quit' to stop.")

    while True:
        # Store the user's query in a variable
        choice = input("Enter your question: ")

        if choice.lower() == 'quit':
            print("Goodbye!")
            break

        # Combine the user's query with the PDF text
        combined_input = f"PDF content:\n{text}\n\nUser question: {choice}"

        # Construct the message payload
        messages = [
            {
                'role': 'user',
                'content': combined_input,
            },
        ]

        # Generate a response using LLaMA
        response = chat(model, messages=messages)

        # Print the chatbot's response
        print(response['message']['content'])

# Example usage
pdf_path = r"C:\Users\harih\Downloads\4.MULTIPLE LINEAR REGRESSION.pdf"
main(pdf_path)
