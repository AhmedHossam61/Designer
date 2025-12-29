import json
with open("config.json", "r") as f:
    config_data = json.load(f) 

img_gen_config = config_data["configuration"]["generate_images"]
print(img_gen_config)

from llama_cpp import Llama
llm = Llama(model_path="/teamspace/studios/this_studio/Designer/models/gemma-3-4b-it-Q8_0.gguf")
print("Loaded OK")
print(llm("Hello"))
