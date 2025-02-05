# นำเข้าโมดูล Normalizer จาก NeMo สำหรับการทำ text normalization
try:
    from nemo_text_processing.text_normalization.normalize import Normalizer
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "ไม่พบแพ็คเกจ 'nemo_text_processing' ในระบบ กรุณาติดตั้งแพ็คเกจก่อนใช้งานที่: "
        "https://github.com/NVIDIA/NeMo-text-processing"
    )

# นำเข้าโมดูล os สำหรับจัดการ path และสร้างโฟลเดอร์
import os

# สร้าง object text_normalizer โดยกำหนด:
# - input_case="cased" คือให้คงรูปแบบตัวพิมพ์เล็ก-ใหญ่ตามต้นฉบับ
# - lang="en" คือกำหนดให้ใช้ภาษาอังกฤษ
text_normalizer = Normalizer(input_case="cased", lang="en")

# ข้อความที่ต้องการทำ normalization
text = "Mr. Johnson is turning 35 years old on 04-15-2023."

# ทำ text normalization โดยใช้ฟังก์ชัน normalize()
normalized_text = text_normalizer.normalize(text)

# แสดงผลข้อความต้นฉบับและข้อความที่ผ่านการ normalize แล้ว
print(text)  # แสดงข้อความต้นฉบับ
print(normalized_text)  # แสดงข้อความที่ผ่านการ normalize

# กำหนด path สำหรับโฟลเดอร์ output และไฟล์ผลลัพธ์
output_dir = r"C:\NeMo_TTS_Primer\output"
output_file = os.path.join(output_dir, "5.1 TN example.txt")

# สร้างโฟลเดอร์ output ถ้ายังไม่มี
os.makedirs(output_dir, exist_ok=True)

# บันทึกผลลัพธ์ลงไฟล์ txt
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("Original text:\n")
    f.write(text + "\n\n")
    f.write("Normalized text:\n")
    f.write(normalized_text + "\n")