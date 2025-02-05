# แก้ปัญหา SIGKILL ก่อน import อื่นๆ
import signal
if not hasattr(signal, 'SIGKILL'):
    signal.SIGKILL = signal.SIGTERM

# นำเข้าไลบรารีที่จำเป็น
import os
import torch
import IPython.display as ipd
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder

# แสดงรายชื่อโมเดล Spectrogram ที่มีใน NeMo
print("Spectrogram Models:")
print(SpectrogramGenerator.list_available_models())

print()
print("Vocoder Models:")
print(Vocoder.list_available_models())

# เลือกอุปกรณ์ประมวลผล (GPU หรือ CPU)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# โหลดโมเดล Spectrogram และ Vocoder เข้า GPU/CPU
spectrogram_model = SpectrogramGenerator.from_pretrained("tts_en_tacotron2").eval().to(device)
vocoder = Vocoder.from_pretrained("tts_en_hifigan").eval().to(device)

# ข้อความที่ต้องการสังเคราะห์เป็นเสียง
text = "This audio was generated with a text-to-speech model."

# แปลงข้อความเป็น tokens และทำ normalization
tokens = spectrogram_model.parse(text, normalize=True)

# สร้าง spectrogram จากข้อความ
spectrogram = spectrogram_model.generate_spectrogram(tokens=tokens)

# แปลง spectrogram เป็นเสียง
audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

# แปลงข้อมูลจาก PyTorch tensor เป็น NumPy array
spectrogram = spectrogram.cpu().detach().numpy()[0]
audio = audio.cpu().detach().numpy()[0]

# แสดงข้อความและเล่นเสียง
print(f'"{text}"\n')
ipd.Audio(audio, rate=22050)

# สร้างและบันทึกรูป spectrogram
output_dir = r"C:\NeMo_TTS_Primer\output"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 5))
imshow(spectrogram, origin="lower")
plt.xlabel("Audio Frame")
plt.ylabel("Frequency Band")
plt.title("Spectrogram Visualization")
# บันทึกรูป spectrogram
plt.savefig(os.path.join(output_dir, "spectrogram.png"), dpi=300, bbox_inches='tight')
plt.show()

# บันทึกผลลัพธ์ไปยังไฟล์ txt
output_file = os.path.join(output_dir, "7.4 End to end example.txt")

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("End-to-end Text-to-Speech Example\n")
    f.write("================================\n\n")
    f.write(f"Input text: {text}\n\n")
    f.write("Available Models:\n")
    f.write("----------------\n")
    f.write("Spectrogram Models:\n")
    f.write(str(SpectrogramGenerator.list_available_models()) + "\n\n")
    f.write("Vocoder Models:\n")
    f.write(str(Vocoder.list_available_models()) + "\n\n")
    f.write("Processing Device: " + device + "\n\n")
    f.write("Output Files:\n")
    f.write("1. spectrogram.png - Visualization of the generated spectrogram\n")
    f.write("Note: Audio and spectrogram visualization were generated successfully.")