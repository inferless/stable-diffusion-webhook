from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import base64
import os
import requests

WEBHOOK_URL = "http://your-webhook-url.com" # You can also use os.env.get("WEBHOOkURL")

class InferlessPythonModel:
    def initialize(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            device_map='auto'
        )
        
    def infer(self, inputs):
        prompt = inputs["prompt"]
        image = self.pipe(prompt).images[0]
        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.get value()).decode()
        data = { "generated_image_base64" : img_str }
        // Call the Webhook 
        response = requests.post(WEBHOOK_URL, json=data)
        return {"response": "success"}

    def finalize(self):
        self.pipe = None
