import tensorflow as tf
import tensorflow_hub as hub
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import h5py
import json

model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
model = hub.load(model_url)

instructions = []
directories = os.listdir("data")

for directory in directories:
    if not os.path.isdir(f"data/{directory}"):
        continue
    file = os.listdir(f"data/{directory}")[0]
    file = f"data/{directory}/{file}"
    file = h5py.File(file, "r")
    instructions.append(file["instruction"][()].decode("utf-8"))
    
embeddings = model(instructions)
embeddings = {
    f'{instructions[i]}': embeddings[i].numpy().tolist() for i in range(len(instructions))
}

with open("text_embeddings.json", "w") as f:
    json.dump(embeddings, f, indent=4)

    