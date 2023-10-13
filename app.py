from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import base64
from huggingface_hub import snapshot_download
import os

class InferlessPythonModel:
    def initialize(self):
        
        local_path = "/var/nfs-mount/LLama2-finetune"
        if os.path.exists(local_path + "model_index.json") == False :
            os.makedirs(local_path)
            snapshot_download(
                "runwayml/stable-diffusion-v1-5",
                local_dir=local_path,
            )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            local_path,
            torch_dtype=torch.float16,
            device_map='auto'
        )
    def infer(self, inputs):
        prompt = inputs["prompt"]
        image = self.pipe(prompt).images[0]
        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode()
        return { "generated_image_base64" : img_str }

    def finalize(self):
        self.pipe = None
