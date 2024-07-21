
# Whisper: https://github.com/openai/whisper
# note: If you want to use GPU, You need to install CUDA and PyTorch witch is compatible with CUDA.
# Check CUDA version: https://pytorch.org/get-started/locally/
# CUDA download: https://developer.nvidia.com/cuda-toolkit-archive
import whisper
from model import Model

from pprint import pprint
import json

MODEL = Model.BASE
INPUT_FILE = "sample.mp3"

model = whisper.load_model(MODEL.value)
result = model.transcribe(INPUT_FILE)
print(result["text"])

# save to file
with open("result.json", "w") as f:
    json.dump(result, f)
