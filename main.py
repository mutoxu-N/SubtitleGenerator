
# Faster-Whisper: https://github.com/SYSTRAN/faster-whisper
# note: If you want to use GPU, You need to install CUDA and PyTorch witch is compatible with CUDA.
# Check CUDA version: https://pytorch.org/get-started/locally/

import os
from faster_whisper import WhisperModel
from model import ModelSize

MODEL = ModelSize.BASE
INPUT_FILE = "sample.mp4"

#  Initializing *.dll, but found *.dll already initialized. が出たら必要な環境変数
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = WhisperModel(MODEL.value, device="cuda", compute_type="float16")
# model = WhisperModel(MODEL.value, device="cuda", compute_type="int8_float16")

segments, _ = model.transcribe(INPUT_FILE, beam_size=5)

# save to file
with open("result.out", "w") as f:
    text = ""
    for segment in segments:
        f.write("[%.2fs -> %.2fs] %s\n" %
                (segment.start, segment.end, segment.text))
        text += segment.text
    print(text)
