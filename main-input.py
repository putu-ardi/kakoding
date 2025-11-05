import torch
from transformers import pipeline

#login to huggingface, please copy your token
from huggingface_hub import login
login(token="") #paste token anda di antara ""

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Pesan sistem yang tetap untuk mengarahkan chatbot agar selalu menggunakan bahasa bajak laut
system_message = "You are a pirate chatbot who always responds in pirate speak!"

while True:
    user_prompt = input("Enter your prompt (or type 'exit' to quit): ").strip()
    if user_prompt.lower() in ['exit', 'quit']:
        print("Exiting the chatbot. Farewell, matey!")
        break

    # Membuat pesan dengan menggabungkan pesan sistem dan input pengguna
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]

    # Menghasilkan teks berdasarkan pesan yang diberikan
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )

    # Menampilkan hasil teks yang dihasilkan
    print("Generated Text:")

    print(outputs[0]["generated_text"])
