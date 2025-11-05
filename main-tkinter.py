import gradio as gr
import torch
from transformers import pipeline
from huggingface_hub import login

# Login to huggingface (replace with your actual token or use Colab secrets)
login(token="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") # Uncomment and replace with your token or use secrets

# Inisialisasi pipeline dengan model LLM
model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Pesan sistem tetap untuk mengarahkan chatbot
system_message = "You are a pirate chatbot who always responds in pirate speak!"

def pirate_chatbot(user_prompt):
    """
    Generates a pirate-style response to the user prompt using the LLM.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]

    try:
        outputs = pipe(messages, max_new_tokens=256)
        # Extract the generated text (assuming the structure remains consistent)
        generated_text = outputs[0]["generated_text"]

        # Find the last message with 'role': 'assistant' in the generated text
        # This is a simple heuristic, may need adjustment based on actual model output
        assistant_response_start = generated_text.rfind("assistant\n")
        if assistant_response_start != -1:
            assistant_response = generated_text[assistant_response_start + len("assistant\n"):].strip()
            # Remove the trailing "user" role if it exists
            user_role_start = assistant_response.rfind("user\n")
            if user_role_start != -1:
                assistant_response = assistant_response[:user_role_start].strip()
            return assistant_response
        else:
            # If "assistant" role is not found, return the whole generated text after the user prompt
            user_prompt_index = generated_text.rfind(user_prompt)
            if user_prompt_index != -1:
                 return generated_text[user_prompt_index + len(user_prompt):].strip()
            else:
                 return generated_text.strip()

    except Exception as e:
        return f"Error: {e}"

# Create the Gradio interface
iface = gr.Interface(
    fn=pirate_chatbot,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs=gr.Textbox(label="Pirate Chatbot Response"),
    title="Pirate Chatbot",
    description="Ask me anything, but beware, I only speak in pirate tongue!"
)

# Launch the interface
iface.launch(debug=True)