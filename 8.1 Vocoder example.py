# แก้ปัญหา SIGKILL ก่อน import อื่นๆ
import signal
if not hasattr(signal, 'SIGKILL'):
    signal.SIGKILL = signal.SIGTERM

# นำเข้าไลบรารีที่จำเป็น
import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from nemo.collections.tts.models.base import Vocoder
from nemo.collections.tts.models import FastPitchModel
import scipy.io.wavfile as wav

# กำหนดค่าพื้นฐาน
sample_rate = 22050
output_dir = r"C:\NeMo_TTS_Primer\output"
os.makedirs(output_dir, exist_ok=True)

# โหลดโมเดลและตั้งค่า
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading models...")
fastpitch_model = FastPitchModel.from_pretrained("tts_en_fastpitch").eval().to(device)
vocoder = Vocoder.from_pretrained("tts_en_hifigan").eval().to(device)

# สร้างเสียงต้นฉบับด้วย FastPitch
print("Generating original audio...")
text = "This is a test audio for vocoder analysis."
tokens = fastpitch_model.parse(text, normalize=True)
spectrogram = fastpitch_model.generate_spectrogram(tokens=tokens)
original_audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
original_audio = original_audio.cpu().detach().numpy()[0]

# แปลงเสียงเป็น tensor
print("Processing audio through vocoder...")
audio_tensor = torch.from_numpy(original_audio).unsqueeze(0).to(device)
audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.long, device=device)

# สร้าง spectrogram จากเสียงที่สร้าง
print("Extracting spectrogram...")
input_spec, spec_len = fastpitch_model.preprocessor(
    input_signal=audio_tensor,
    length=audio_len
)

# สร้างเสียงใหม่จาก spectrogram
print("Reconstructing audio...")
reconstructed_audio = vocoder.convert_spectrogram_to_audio(spec=input_spec)
reconstructed_audio = reconstructed_audio.cpu().detach().numpy()[0]

# แสดงและบันทึก spectrograms เปรียบเทียบ
plt.figure(figsize=(15, 5))

# Original spectrogram
plt.subplot(1, 2, 1)
plt.imshow(input_spec.cpu().detach().numpy()[0], origin='lower')
plt.colorbar(label='Magnitude')
plt.title('Original Spectrogram')
plt.xlabel('Time Frame')
plt.ylabel('Mel Frequency Bin')

# Reconstructed spectrogram
reconstructed_spec, _ = fastpitch_model.preprocessor(
    input_signal=torch.from_numpy(reconstructed_audio).unsqueeze(0).to(device),
    length=torch.tensor([reconstructed_audio.shape[0]], dtype=torch.long, device=device)
)

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_spec.cpu().detach().numpy()[0], origin='lower')
plt.colorbar(label='Magnitude')
plt.title('Reconstructed Spectrogram')
plt.xlabel('Time Frame')
plt.ylabel('Mel Frequency Bin')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spectrogram_comparison.png"), dpi=300, bbox_inches='tight')

# บันทึกไฟล์เสียง
print("\nSaving audio files...")
wav.write(os.path.join(output_dir, "original_audio.wav"), sample_rate, original_audio.astype(np.float32))
wav.write(os.path.join(output_dir, "reconstructed_audio.wav"), sample_rate, reconstructed_audio.astype(np.float32))

# บันทึกรายงาน
with open(os.path.join(output_dir, "8.1 Vocoder example.txt"), 'w', encoding='utf-8') as f:
    f.write("Vocoder Analysis Report\n")
    f.write("=====================\n\n")
    f.write(f"Input Text: {text}\n\n")
    f.write("Process Steps:\n")
    f.write("1. Generate original audio using FastPitch\n")
    f.write("2. Extract mel spectrogram from generated audio\n")
    f.write("3. Reconstruct audio using HiFi-GAN vocoder\n")
    f.write("4. Compare original and reconstructed spectrograms\n\n")
    f.write(f"Audio Settings:\n")
    f.write(f"- Sample rate: {sample_rate} Hz\n")
    f.write(f"- Processing device: {device}\n\n")
    f.write("Output Files:\n")
    f.write("1. original_audio.wav - Original synthesized audio\n")
    f.write("2. reconstructed_audio.wav - Reconstructed audio using vocoder\n")
    f.write("3. spectrogram_comparison.png - Visual comparison of spectrograms\n")

print("\nProcess completed successfully!")
print("Output files are saved in:", output_dir)