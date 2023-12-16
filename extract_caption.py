import torch
from PIL import Image
import os
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
import json

from transformers import pipeline

# loads InstructBLIP model
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

#processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
#model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

#model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

data_root = "data/images_test"
output_dir = "data/instruct_captions.json"

all_images = os.listdir(data_root)
all_images.sort(key=lambda x:int(x))

name_map = {}

for image in tqdm(all_images):
    if os.path.exists(os.path.join(data_root, image, "image.png")):
        curr_dir = os.path.join(data_root, image, "image.png")
    else:
        curr_dir = os.path.join(data_root, image, "choice_0.png")
    raw_image = Image.open(curr_dir).convert("RGB")
    # prepare the image
    image_features = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    output = model.generate({"image": image_features, "prompt": "Write a detailed description."})
    
    #inputs = processor(raw_image, return_tensors="pt")
    #out = model.generate(**inputs)
    #output = processor.decode(out[0], skip_special_tokens=True)
    
    #output = image_to_text(curr_dir)
    #output = image_caption[0]['generated_text']
    
    name_map[str(image)] = output

with open(output_dir, 'w') as outfile:
    json.dump(name_map, outfile, indent=2)