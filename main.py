
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
text = ""
with open("result.srt", "w") as f:
    for i, segment in enumerate(segments):
        start_h = int(segment.start) // 3600
        start_m = int(segment.start) % 3600 // 60
        start_s = int(segment.start) % 60
        start_ms = int(segment.start * 1000) % 1000

        end_h = int(segment.end) // 3600
        end_m = int(segment.end) % 3600 // 60
        end_s = int(segment.end) % 60
        end_ms = int(segment.end * 1000) % 1000

        f.write(f"{i}\n")
        f.write("%2d:%02d:%02d,%03d --> %2d:%02d:%02d,%03d\n" %
                (start_h, start_m, start_s, start_ms, end_h, end_m, end_s, end_ms))
        f.write(f"{segment.text}\n")
        f.write("\n")
        text += segment.text

with open("result.out", "w") as f:
    f.write(text + "\n\n")

    for segment in segments:
        f.write(f"[{segment.start}s -> {segment.end}s] {segment.text}\n")
print(text)
print("Done!")
