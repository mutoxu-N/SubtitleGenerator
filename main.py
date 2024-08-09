
# Faster-Whisper: https://github.com/SYSTRAN/faster-whisper
# note: If you want to use GPU, You need to install CUDA and PyTorch witch is compatible with CUDA.
# Check CUDA version: https://pytorch.org/get-started/locally/

import os
import time
import tkinter.filedialog as fd
from faster_whisper import WhisperModel
from model import ModelSize

MODEL = ModelSize.LARGE_V3

# config 読み込み
config = os.path.dirname(os.path.abspath(__file__)) + "/config"
dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(config):
    tmp = ""
    with open(config) as f:
        tmp = f.read()

    if os.path.exists(tmp):
        dir = tmp

# ファイル読み込み
input_file = fd.askopenfile(initialdir=dir).name
dir = os.path.dirname(os.path.abspath(input_file))
fname = os.path.splitext(os.path.basename(input_file))[0]

# dir を config に保存
with open(config, "w") as f:
    f.write(dir)

#  Initializing *.dll, but found *.dll already initialized. が出ても処理が中断しないようにする環境変数
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently... を隠すための環境変数
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

start_time = time.perf_counter()
model = WhisperModel(MODEL.value, device="cuda", compute_type="float16")
# model = WhisperModel(MODEL.value, device="cuda", compute_type="int8_float16")

segments, _ = model.transcribe(input_file, beam_size=5)
print("Whisper is Done!")
end_time = time.perf_counter()
elapsed = end_time - start_time
elapsed_h = int(elapsed // 3600)
elapsed_m = int(elapsed % 3600 // 60)
elapsed_s = int(elapsed % 60)
elapsed_ms = int(elapsed * 1000) % 1000
print(" -> Elapsed time: "
      + f"{elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}.{elapsed_ms:03d}\n")

# save to file
text, out_result = "", ""
with open(f"{dir}/{fname}.srt", "w", encoding="utf-8") as f:
    for i, segment in enumerate(segments):
        write = ""
        start_sec = int(segment.start)
        start_h = start_sec // 3600
        start_m = start_sec % 3600 // 60
        start_s = start_sec % 60
        start_ms = int(segment.start * 1000) % 1000

        end_sec = int(segment.end)
        end_h = end_sec // 3600
        end_m = end_sec % 3600 // 60
        end_s = end_sec % 60
        end_ms = int(segment.end * 1000) % 1000

        write += f"{i}\n"
        write += f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d} --> "
        write += f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}\n"
        write += f"{segment.text}\n"
        write += "\n"
        text += segment.text
        out_result += f"[{segment.start}s -> {segment.end}s] {segment.text}\n"
        f.write(write)

with open(f"{dir}/{fname}.out", "w", encoding="utf-8") as f:
    f.write(text + "\n\n")
    f.write(out_result)

print("All is Done!")
end_time = time.perf_counter()
elapsed = end_time - start_time
elapsed_h = int(elapsed // 3600)
elapsed_m = int(elapsed % 3600 // 60)
elapsed_s = int(elapsed % 60)
elapsed_ms = int(elapsed * 1000) % 1000
print(" -> Elapsed time: "
      + f"{elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}.{elapsed_ms:03d}")
