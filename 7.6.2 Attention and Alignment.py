# แก้ปัญหา SIGKILL ก่อน import อื่นๆ
import signal
if not hasattr(signal, 'SIGKILL'):
    signal.SIGKILL = signal.SIGTERM

# นำเข้าไลบรารีที่จำเป็น
import os
import torch
from matplotlib import pyplot as plt
from nemo.collections.tts.models import Tacotron2Model
from nemo.collections.tts.g2p.models.en_us_arpabet import EnglishG2p

# ตรวจสอบโมเดล Tacotron 2 ที่มีให้ใช้งาน
print("Available Tacotron2 Models:")
print(Tacotron2Model.list_available_models())

# เลือกอุปกรณ์ประมวลผล (GPU หรือ CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# โหลดโมเดล Tacotron 2 และตั้งค่า
tacotron2_model = Tacotron2Model.from_pretrained("tts_en_tacotron2").eval().to(device)
tacotron2_model.calculate_loss = False

# สร้าง G2P object สำหรับแปลงข้อความ
g2p = EnglishG2p(ignore_ambiguous_words=False)

# ข้อความสำหรับสร้าง attention map
text = "This is an attention map."

# แปลงข้อความเป็น tokens
tokens = tacotron2_model.parse(text, normalize=True)
token_len = torch.Tensor([tokens.shape[1]]).type(torch.long).to(device=tacotron2_model.device)

# สร้าง alignments (attention weights)
_, alignments, _, _ = tacotron2_model.forward(tokens=tokens, token_len=token_len)
alignment = alignments[0].cpu().detach().numpy()

# เตรียมข้อมูลสำหรับแกน x (ตัวอักษร)
# เพิ่ม <BOS> (beginning-of-speech) และ <EOS> (end-of-speech) tokens
characters = ["<BOS>"] + [char for char in text] + ["<EOS>"]

# สร้างและบันทึกรูป attention map
output_dir = r"C:\NeMo_TTS_Primer\output"
os.makedirs(output_dir, exist_ok=True)

# สร้าง subplot
fig, ax = plt.subplots(figsize=(10, 8))

# แสดง attention map
ax.imshow(alignment.transpose(), origin='upper', aspect='auto')
ax.set_xlabel("Audio Frame")
ax.set_yticks(range(len(characters)))
ax.set_yticklabels(characters)
plt.title("Tacotron 2 Attention Map")

# บันทึกรูป
plt.savefig(os.path.join(output_dir, "attention_map.png"), dpi=300, bbox_inches='tight')
plt.show()

# บันทึกข้อมูลลงไฟล์
output_file = os.path.join(output_dir, "7.6.2 Attention and Alignment.txt")

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("Tacotron 2 Attention Analysis\n")
    f.write("===========================\n\n")
    f.write(f"Input text: {text}\n\n")
    f.write("Model Information:\n")
    f.write("-----------------\n")
    f.write("Available Tacotron2 Models:\n")
    f.write(str(Tacotron2Model.list_available_models()) + "\n\n")
    f.write(f"Processing Device: {device}\n\n")
    f.write("Attention Map Analysis:\n")
    f.write("-------------------\n")
    f.write("- Vertical axis: Input text characters (including <BOS> and <EOS> tokens)\n")
    f.write("- Horizontal axis: Output audio frames\n")
    f.write("- Brighter colors indicate stronger attention weights\n\n")
    f.write("Output Files:\n")
    f.write("1. attention_map.png - Visualization of the Tacotron 2 attention mechanism\n")