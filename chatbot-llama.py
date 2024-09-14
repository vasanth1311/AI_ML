from ollama import chat

# Define your chatbot's personality  (LLaMA model)
model = 'llama3'
print("Welcome to the LLaMA-based chatbot! Type 'quit' to stop.")

while True:
    # Store the user's query in a variable
    choice = input("Enter your question: ")

    if choice.lower() == 'quit':
        print("Goodbye!")
        break

    # Construct the message payload
    messages = [
        {
            'role': 'user',
            'content': choice,
        },
    ]

    # Generate a response using LLaMA
    response = chat(model, messages=messages)

    # Print the chatbot's response
    print(response['message']['content'])
