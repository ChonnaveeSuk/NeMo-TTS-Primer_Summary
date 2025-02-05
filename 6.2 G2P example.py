# แก้ปัญหา SIGKILL ก่อน import อื่นๆ
import signal
if not hasattr(signal, 'SIGKILL'):
    signal.SIGKILL = signal.SIGTERM

import os
from nemo.collections.tts.g2p.models.en_us_arpabet import EnglishG2p
# เพิ่ม import สำหรับ Tokenizer
from nemo.collections.tts.torch.tts_tokenizers import EnglishPhonemesTokenizer, IPATokenizer

# สร้าง object สำหรับแปลงข้อความเป็น phonemes
arpabet_g2p = EnglishG2p()

# ทดสอบการแปลงด้วยข้อความ "hello world"
text = "Hello world"
arpabet_phonemes = arpabet_g2p(text)  # แปลงเป็น ARPABET phonemes

# แสดงผลลัพธ์สำหรับ G2P
print("\nG2P Results:")
print(f"Input text: {text}")
print(f"ARPABET phonemes: {arpabet_phonemes}")

# ดูว่า "hello" มี phonemes อะไรบ้างในพจนานุกรม
print(f"\nPronunciations for 'hello' in dictionary: {arpabet_g2p.phoneme_dict['hello']}")

# การจัดการกับคำที่มีการออกเสียงกำกวม
print("\nG2P with ignore_ambiguous_words=False:")
arpabet_g2p = EnglishG2p(ignore_ambiguous_words=False)
arpabet_phonemes = arpabet_g2p(text)
print(f"Input text: {text}")
print(f"ARPABET phonemes: {arpabet_phonemes}")

# ส่วนของ Tokenization
print("\nTokenization Results:")
# สร้าง tokenizer objects
arpabet_tokenizer = EnglishPhonemesTokenizer(arpabet_g2p)
arpabet_tokens = arpabet_tokenizer(text)
print(f"Input text: {text}")
print(f"ARPABET tokens: {arpabet_tokens}")

# บันทึกผลลัพธ์
output_dir = r"C:\NeMo_TTS_Primer\output"
output_file = os.path.join(output_dir, "6.2 G2P example.txt")

os.makedirs(output_dir, exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("G2P (Grapheme-to-Phoneme) and Tokenization Results\n")
    f.write("=============================================\n\n")
    
    f.write("G2P Results:\n")
    f.write(f"Input text: {text}\n")
    f.write(f"ARPABET phonemes: {arpabet_phonemes}\n\n")
    
    f.write(f"Pronunciations for 'hello' in dictionary:\n")
    f.write(f"{arpabet_g2p.phoneme_dict['hello']}\n\n")
    
    f.write("G2P with ignore_ambiguous_words=False:\n")
    f.write(f"Input text: {text}\n")
    f.write(f"ARPABET phonemes: {arpabet_phonemes}\n\n")
    
    f.write("Tokenization Results:\n")
    f.write(f"Input text: {text}\n")
    f.write(f"ARPABET tokens: {arpabet_tokens}\n")