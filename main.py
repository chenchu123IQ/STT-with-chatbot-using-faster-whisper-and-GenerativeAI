import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import time
import threading
from tqdm import tqdm
import google.generativeai as genai
from faster_whisper import WhisperModel

# CONFIGURATION

GEMINI_API_KEY = "your API key"  # Replace with your real Gemini API key
SAMPLE_RATE = 16000
OUTPUT_FILE = "recorded.wav"
DEVICE = None  # Use default microphone

# INIT GEMINI

genai.configure(api_key=GEMINI_API_KEY)
chat_model = genai.GenerativeModel("models/gemini-1.5-flash")
chat_session = chat_model.start_chat(history=[])

# ENERGY-BASED VAD RECORDING

def record_with_energy_vad(output_file, silence_timeout=1.5):
    buffer = []
    silence_start = None
    energy_threshold = 0.01  # Adjust if needed

    def callback(indata, frames, time_info, status):
        nonlocal silence_start
        chunk = indata[:, 0]
        buffer.extend(chunk.tolist())

        energy = np.sqrt(np.mean(chunk**2))
        if energy < energy_threshold:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > silence_timeout:
                raise sd.CallbackStop()
        else:
            silence_start = None

    print("\n🎤 Speak now. It will stop after 1.5s of silence...")

    with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=1024, dtype='float32', device=DEVICE):
        try:
            sd.sleep(15000)  # Max wait = 15 seconds
        except sd.CallbackStop:
            pass

    if not buffer:
        print("⏳ No speech detected. Timing out...")
        return False

    audio_np = np.array(buffer, dtype='float32')
    audio_int16 = np.int16(audio_np * 32767)
    wav.write(output_file, SAMPLE_RATE, audio_int16)
    print("✅ Finished recording.")
    return True

# TRANSCRIBE AUDIO WITH FASTER-WHISPER + TQDM PROGRESS BAR

def transcribe_audio(filename):
    result_text = ""
    progress_bar = tqdm(total=100, desc="🔍 Transcribing with Whisper", bar_format="{l_bar}{bar} {elapsed}", ncols=80)
    progress_done = False

    def show_progress():
        for _ in range(100):
            if progress_done:
                break
            progress_bar.update(1)
            time.sleep(0.08)
        progress_bar.close()

    thread = threading.Thread(target=show_progress)
    thread.start()

    try:
        # Load the faster-whisper model
        model = WhisperModel("medium", compute_type="int8")  # Options: int8, float16, float32
        

        segments, info = model.transcribe(filename, task="translate")

        print(f"\n🌐 Language detected: {info.language}")
        for segment in segments:
            result_text += segment.text

    finally:
        progress_done = True
        thread.join()

    return result_text.strip()

# GEMINI CHAT

def get_gemini_response(prompt):
    print("🤖 Generating response...")
    try:
        response = chat_session.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini Error: {e}"

# MAIN LOOP

def main():
    print("\n🎙️ Voice Chatbot")
    print("------------------------------------------")
    print("Speak in any language. Type 'q' to exit.\n")

    while True:
        try:
            recorded = record_with_energy_vad(OUTPUT_FILE)
            if not recorded:
                continue

            user_input = transcribe_audio(OUTPUT_FILE)
            print(f"\n🗣️ Transcribed: {user_input}")

            if not user_input.strip():
                print("⚠️ Not recognized. Try again.")
                continue

            response = get_gemini_response(user_input)
            print(f"\n💬 Gemini: {response}")
            print("------------------------------------------")

            user_choice = input("Press Enter to continue or type 'q' to quit: ").strip().lower()
            if user_choice == "q":
                print("\n👋 See you soon!")
                break

        except Exception as e:
            print(f"⚠️ Error: {e}")
            continue

# RUN
if __name__ == "__main__":
    main()
