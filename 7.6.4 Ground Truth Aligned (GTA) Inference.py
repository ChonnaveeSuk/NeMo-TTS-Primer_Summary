# แก้ปัญหา SIGKILL ก่อน import อื่นๆ
import signal
if not hasattr(signal, 'SIGKILL'):
    signal.SIGKILL = signal.SIGTERM

# นำเข้าไลบรารีที่จำเป็น
import os
import numpy as np
import torch
import IPython.display as ipd
from nemo.collections.tts.models import Tacotron2Model
from nemo.collections.tts.models.base import Vocoder
import scipy.io.wavfile as wav

# เตรียมข้อความสำหรับทดสอบ
text = "That is not only my accusation."

# โหลดโมเดลและตั้งค่า
device = "cuda" if torch.cuda.is_available() else "cpu"
tacotron2_model = Tacotron2Model.from_pretrained("tts_en_tacotron2").eval().to(device)
vocoder = Vocoder.from_pretrained("tts_en_hifigan").eval().to(device)

# แปลงข้อความเป็น tokens
tokens = tacotron2_model.parse(text, normalize=True)
token_len = torch.Tensor([tokens.shape[1]]).type(torch.long).to(device=tacotron2_model.device)

# ตั้งค่าเริ่มต้นให้ปิด training และ calculate_loss
tacotron2_model.calculate_loss = False
tacotron2_model.training = False
tacotron2_model.decoder.training = False

# ทำ regular inference
print("Generating regular synthesis...")
outputs = tacotron2_model.forward(tokens=tokens, token_len=token_len)
predicted_spectogram = outputs[0] if isinstance(outputs, tuple) else outputs
predicted_audio = vocoder.convert_spectrogram_to_audio(spec=predicted_spectogram)
predicted_audio = predicted_audio.cpu().detach().numpy()[0]

# สร้างเสียงจำลองสำหรับ ground truth
print("Creating simulated ground truth...")
sample_rate = 22050
audio = predicted_audio.copy()
audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(tacotron2_model.device)
audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.long).to(device=tacotron2_model.device)

# เปิดใช้งาน training และ calculate_loss สำหรับ GTA inference
print("Performing GTA inference...")
tacotron2_model.calculate_loss = True
tacotron2_model.training = True
tacotron2_model.decoder.training = True

# ทำ GTA inference
outputs = tacotron2_model.forward(
    tokens=tokens,
    token_len=token_len,
    audio=audio_tensor,
    audio_len=audio_len
)
gta_spectogram = outputs[0] if isinstance(outputs, tuple) else outputs

# สร้างเสียงจาก GTA spectrogram
gta_audio = vocoder.convert_spectrogram_to_audio(spec=gta_spectogram)
gta_audio = gta_audio.cpu().detach().numpy()[0]

# บันทึกผลลัพธ์
output_dir = r"C:\NeMo_TTS_Primer\output"
os.makedirs(output_dir, exist_ok=True)

# บันทึกเสียงเป็นไฟล์ .wav
print("\nSaving audio files...")
wav.write(os.path.join(output_dir, "simulated_original.wav"), sample_rate, audio.astype(np.float32))
wav.write(os.path.join(output_dir, "gta_synthesis.wav"), sample_rate, gta_audio.astype(np.float32))
wav.write(os.path.join(output_dir, "regular_synthesis.wav"), sample_rate, predicted_audio.astype(np.float32))

# บันทึกข้อมูลลงไฟล์
print("Saving analysis report...")
with open(os.path.join(output_dir, "7.6.4 GTA Inference.txt"), 'w', encoding='utf-8') as f:
    f.write("Ground Truth Aligned (GTA) Inference Analysis\n")
    f.write("=========================================\n\n")
    f.write(f"Input text: {text}\n\n")
    f.write("Processing Steps:\n")
    f.write("1. Regular Inference: Generate audio directly from text\n")
    f.write("2. GTA Inference: Use ground truth audio to guide generation\n")
    f.write("3. Compare outputs to understand teacher forcing effects\n\n")
    f.write(f"Audio sample rate: {sample_rate} Hz\n")
    f.write(f"Audio duration: {len(audio)/sample_rate:.2f} seconds\n")
    f.write(f"Processing device: {device}\n\n")
    f.write("Output Files:\n")
    f.write("1. simulated_original.wav - Simulated ground truth audio\n")
    f.write("2. gta_synthesis.wav - Audio generated using GTA inference\n")
    f.write("3. regular_synthesis.wav - Audio generated using regular inference\n")

print("\nProcess completed successfully!")
print("Output files are saved in:", output_dir)