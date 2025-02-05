# แก้ปัญหา SIGKILL ก่อน import อื่นๆ
import signal
if not hasattr(signal, 'SIGKILL'):
    signal.SIGKILL = signal.SIGTERM

# นำเข้าไลบรารีที่จำเป็น
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models.base import Vocoder
import scipy.io.wavfile as wav

# กำหนดค่าพื้นฐาน
sample_rate = 22050

# โหลดโมเดลและตั้งค่า
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading FastPitch model...")
fastpitch_model = FastPitchModel.from_pretrained("tts_en_fastpitch").eval().to(device)
print("Loading HiFi-GAN vocoder...")
vocoder = Vocoder.from_pretrained("tts_en_hifigan").eval().to(device)

# สร้างข้อความทดสอบ 2 ประโยค
texts = [
    "That is not only my accusation.",
    "This audio was generated with the FastPitch text-to-speech model."
]

# สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
output_dir = r"C:\NeMo_TTS_Primer\output"
os.makedirs(output_dir, exist_ok=True)

# สร้างและบันทึกเสียงสำหรับแต่ละข้อความ
for i, text in enumerate(texts, 1):
    print(f"\nProcessing text {i}: {text}")
    
    # แปลงข้อความเป็น tokens และทำ normalization
    tokens = fastpitch_model.parse(text, normalize=True)
    
    # สร้าง spectrogram
    spectrogram = fastpitch_model.generate_spectrogram(tokens=tokens)
    
    # บันทึก spectrogram plot
    plt.figure(figsize=(10, 5))
    plt.imshow(spectrogram.cpu().detach().numpy()[0], origin="lower")
    plt.colorbar(label='Magnitude')
    plt.title(f"Spectrogram for Text {i}")
    plt.xlabel("Time Frame")
    plt.ylabel("Mel Frequency Bin")
    plt.savefig(os.path.join(output_dir, f"spectrogram_{i}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # แปลง spectrogram เป็นเสียง
    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
    audio = audio.cpu().detach().numpy()[0]
    
    # บันทึกไฟล์เสียง
    wav.write(os.path.join(output_dir, f"audio_{i}.wav"), sample_rate, audio.astype(np.float32))

# บันทึกรายละเอียดการทำงาน
with open(os.path.join(output_dir, "7.8.1 and 7.8.2 FastPitch.txt"), 'w', encoding='utf-8') as f:
    f.write("FastPitch Text-to-Speech Synthesis Report\n")
    f.write("=====================================\n\n")
    
    for i, text in enumerate(texts, 1):
        f.write(f"Text {i}:\n")
        f.write(f'"{text}"\n\n')
        f.write(f"Generated Files:\n")
        f.write(f"1. spectrogram_{i}.png - Mel spectrogram visualization\n")
        f.write(f"2. audio_{i}.wav - Synthesized speech\n\n")
    
    f.write("\nSystem Information:\n")
    f.write(f"- Sample rate: {sample_rate} Hz\n")
    f.write(f"- Processing device: {device}\n")
    f.write(f"- Models used: FastPitch (text-to-spectrogram) and HiFi-GAN (vocoder)\n")

print("\nProcess completed successfully!")
print(f"Output files are saved in: {output_dir}")