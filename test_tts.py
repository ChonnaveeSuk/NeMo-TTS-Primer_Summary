import signal
if not hasattr(signal, 'SIGKILL'):
    signal.SIGKILL = signal.SIGTERM

from nemo.collections.tts.models import FastPitchModel, HifiGanModel
import soundfile as sf
import os
import torch

# ลองพิมพ์รายชื่อโมเดลที่มีให้เลือกใช้
print("Available HifiGan models:", HifiGanModel.list_available_models())

# กำหนด path สำหรับบันทึกไฟล์
output_path = os.path.join(os.getcwd(), "output.wav")

# Load models โดยใช้ชื่อโมเดลที่ถูกต้อง
spec_generator = FastPitchModel.from_pretrained("tts_en_fastpitch")
vocoder = HifiGanModel.from_pretrained("tts_en_hifigan")  # แก้เป็น tts_en_hifigan

# ตั้งค่าโมเดลให้อยู่ในโหมด eval
spec_generator.eval()
vocoder.eval()

# Text to speech
with torch.no_grad():
    text = "Hello, this is a test of text to speech using NeMo!"
    parsed = spec_generator.parse(text)
    spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

# Save audio with Windows path
sf.write(output_path, audio.detach().cpu().numpy()[0], 22050)
print(f"Audio saved to: {output_path}")