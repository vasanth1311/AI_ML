import pdftotext
from ollama import chat

# Load your PDF
with open(r"C:\Users\harih\Desktop\vas.sample pdf.pdf","rb") as f:
    pdf = pdftotext.PDF(f)

# Combine all PDF pages into one string
pdf_text = "\n\n".join(pdf)

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
    combined_input = f"PDF content:\n{pdf_text}\n\nUser question: {choice}"

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
