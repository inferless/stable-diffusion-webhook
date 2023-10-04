import json
import numpy as np
import PIL
import requests as req
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import base64


class InferlessPythonModel:
    def initialize(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16,
            
        )
        self.pipe = self.pipe.to("cuda:0")

    def infer(self, prompt):
        image = self.pipe(prompt).images[0]
        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue())
        return img_str

    def finalize(self):
        self.pipe = None
