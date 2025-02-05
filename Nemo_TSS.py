# Nemo_TSS.py

# แก้ปัญหา SIGKILL ก่อน import อื่นๆ
import signal
if not hasattr(signal, 'SIGKILL'):
    signal.SIGKILL = signal.SIGTERM

import torch
from nemo.collections.tts.models import FastPitchModel, HifiGanModel 
import soundfile as sf
import os
from datetime import datetime

class SimpleTTS:
    def __init__(self):
        # ตั้งค่า device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            # โหลดโมเดล
            print("Loading models...")
            self.spec_generator = FastPitchModel.from_pretrained("tts_en_fastpitch").eval().to(self.device)
            self.vocoder = HifiGanModel.from_pretrained("tts_en_hifigan").eval().to(self.device)
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

    def generate_speech(self, text, output_path=None):
        """
        แปลงข้อความเป็นเสียงพูด
        """
        try:
            # สร้าง output path ถ้าไม่ได้ระบุ
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
                output_path = f"output_{timestamp}.wav"

            print(f"Converting text to speech: {text}")
            
            # แปลงข้อความเป็น tokens
            with torch.no_grad():
                tokens = self.spec_generator.parse(text, normalize=True)
                spec = self.spec_generator.generate_spectrogram(tokens=tokens) 
                audio = self.vocoder.convert_spectrogram_to_audio(spec=spec)

            # แปลงเป็น numpy array และบันทึกไฟล์
            audio_numpy = audio.to('cpu').detach().numpy()[0]
            sf.write(output_path, audio_numpy, 22050)
            
            print(f"Audio saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            return None

# ทดสอบการทำงาน
if __name__ == "__main__":
    try:
        # สร้าง TTS object
        tts = SimpleTTS()
        
        # ทดสอบแปลงข้อความ
        test_text = "This is a test of the text to speech system."
        tts.generate_speech(test_text)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")