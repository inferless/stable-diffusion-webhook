import app
import json
import triton_python_backend_utils as pb_utils
import numpy as np

inferless_model = app.InferlessPythonModel()


class TritonPythonModel:
    def initialize(self, args):
        inferless_model.initialize()

    def execute(self, requests):
        responses = []
        for request in requests:
            prompt = pb_utils.get_input_tensor_by_name(request, "prompt")
            prompt_string = prompt.as_numpy()[0].decode()
            output = inferless_model.infer(prompt_string)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "generated_image_base64",
                        np.array([output]),
                    )
                ]
            )
            responses.append(inference_response)
        return responses

    def finalize(self, args):
        inferless_model.finalize()
