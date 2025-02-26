from llama_stack_client import LlamaStackClient

# Initialize the client
client = LlamaStackClient(base_url="http://localhost:8321")

import gradio as gr

def chat_with_llama(message, history):
    # Prepare the messages including history
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    # Call the Llama Stack API for chat completion
    response = client.inference.chat_completion(
        model="Llama3.2-8B",
        messages=messages,
        stream=False
    )

    # Extract the assistant's response
    assistant_response = response.choices[0].message.content

    # Run safety check
    safety_response = client.safety.run_shield(
        messages=[{"role": "assistant", "content": assistant_response}],
        shield_type="llama_guard",
        params={}
    )

    if safety_response.violation:
        return "I apologize, but I can't provide a response to that request."

    return assistant_response

# Create Gradio interface
iface = gr.ChatInterface(
    chat_with_llama,
    title="Llama Stack Chat Demo",
    description="Chat with an AI assistant powered by Llama Stack"
)

# Launch the interface
iface.launch()
