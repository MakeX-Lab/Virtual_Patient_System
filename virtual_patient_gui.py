import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import threading
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
import os
import whisper
import openai
import requests
from datetime import datetime
from dotenv import load_dotenv
import pygame
import subprocess
import re

# Load environment variables
load_dotenv()

# API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
assistant_id = os.getenv("VP_ASSIST_KEY")

class VirtualPatientApp:
    def __init__(self, root, thread_id):
        self.root = root
        self.root.title("Virtual Patient Assistant")

        self.thread_id = thread_id

        # Set fonts
        self.font = ("Mona Sans", 10)
        self.bold_font = ("Georgia", 10, "bold")  

        # Camera frame
        self.camera_label = ttk.Label(root)
        self.camera_label.grid(row=0, column=0, padx=10, pady=10)

        # Text input and output frame
        self.text_frame = ttk.Frame(root)
        self.text_frame.grid(row=0, column=1, padx=10, pady=10)

        self.student_input_label = ttk.Label(self.text_frame, text="Student Input:", font=self.bold_font)
        self.student_input_label.grid(row=0, column=0, padx=5, pady=5, sticky='W')
        self.student_input_text = tk.Text(self.text_frame, height=10, width=50, font=self.font, wrap=tk.WORD, padx=5, pady=5, bd=2, relief="groove")
        self.student_input_text.grid(row=1, column=0, padx=5, pady=5)

        self.virtual_patient_output_label = ttk.Label(self.text_frame, text="Virtual Patient Output:", font=self.bold_font)
        self.virtual_patient_output_label.grid(row=2, column=0, padx=5, pady=5, sticky='W')
        self.virtual_patient_output_text = tk.Text(self.text_frame, height=10, width=50, font=self.font, wrap=tk.WORD, padx=5, pady=5, bd=2, relief="groove")
        self.virtual_patient_output_text.grid(row=3, column=0, padx=5, pady=5)

        # self.student_input_label = ttk.Label(self.text_frame, text="Student Input:", font=self.bold_font)
        # self.student_input_label.grid(row=0, column=0, padx=5, pady=5)
        # self.student_input_text = tk.Text(self.text_frame, height=10, width=50,font=self.font)
        # self.student_input_text.grid(row=1, column=0, padx=5, pady=5)

        # self.virtual_patient_output_label = ttk.Label(self.text_frame, text="Virtual Patient Output:", font=self.bold_font)
        # self.virtual_patient_output_label.grid(row=2, column=0, padx=5, pady=5)
        # self.virtual_patient_output_text = tk.Text(self.text_frame, height=20, width=50, font=self.font)
        # self.virtual_patient_output_text.grid(row=3, column=0, padx=5, pady=5)

        # Buttons for recording
        self.record_button = ttk.Button(root, text="Start Recording", command=self.start_recording)
        self.record_button.grid(row=1, column=0, padx=10, pady=10)
        self.stop_button = ttk.Button(root, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.grid(row=1, column=1, padx=10, pady=10)

        # Variables for recording
        self.is_recording = False
        self.recording = []
        self.fs = 44100  # Sample rate
        self.channels = 1
        self.start_time = None
        self.end_time = None

        

        # Start visual emotion detection script
        visual_emotion_script_path = "visual_emotion_model/visual_emotion_2.py"
        subprocess.Popen(["python", visual_emotion_script_path])
        print("Starting up the Visual_Emotion_Detection Model!!!")
        time.sleep(10)  # Wait for the model to start

        # Update the camera feed
        self.update_camera_feed()

    def update_camera_feed(self):
        try:
            # Read the latest frame
            img = Image.open('cache/current_frame.jpg')
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
        except Exception as e:
            print(f"Error reading image: {e}")
        self.root.after(100, self.update_camera_feed)

    # def update_camera_feed(self):
    #     ret, frame = self.cap.read()
    #     if ret:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         img = Image.fromarray(frame)
    #         imgtk = ImageTk.PhotoImage(image=img)
    #         self.camera_label.imgtk = imgtk
    #         self.camera_label.configure(image=imgtk)
    #     self.root.after(10, self.update_camera_feed)

    def start_recording(self):
        self.is_recording = True
        self.recording = []
        self.start_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
        self.record_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        threading.Thread(target=self.record_audio).start()

    def stop_recording(self):
        self.is_recording = False
        self.end_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        threading.Thread(target=self.process_recording).start()

    def record_audio(self):
        def callback(indata, frames, time, status):
            if self.is_recording:
                self.recording.append(indata.copy())
            else:
                raise sd.CallbackStop

        with sd.InputStream(samplerate=self.fs, channels=self.channels, callback=callback):
            while self.is_recording:
                sd.sleep(100)

    def process_recording(self):
        recorded_audio = np.concatenate(self.recording, axis=0)
        input_audio_file_path = os.path.join('cache', 'audio_input.wav')
        write(input_audio_file_path, self.fs, recorded_audio)
        transcribed_text = self.transcribe_audio(input_audio_file_path)
        # self.student_input_text.insert(tk.END, transcribed_text + "\n")

        # Get current timestamp
        timestamp = datetime.now().strftime('%H:%M:%S')

        # Insert the transcribed text with timestamp
        self.student_input_text.insert(tk.END, f"{timestamp} : {transcribed_text}\n")


        # Update the UI to indicate we are getting a response
        self.virtual_patient_output_text.insert(tk.END, "\nGetting response...\n\n")
        threading.Thread(target=self.get_and_process_response, args=(transcribed_text,)).start()

    def transcribe_audio(self, file_path):
        model = whisper.load_model("tiny.en")
        result = model.transcribe(file_path)
        return result["text"]

    def get_and_process_response(self, user_input):
        response = self.get_virtual_patient_response(user_input)
        patient_response = re.split(r'\[', response, maxsplit=1)[0].rstrip()
        # self.virtual_patient_output_text.delete("1.0", tk.END)

        timestamp = datetime.now().strftime('%H:%M:%S')

        # Insert the response with timestamp
        self.virtual_patient_output_text.insert(tk.END, f"{timestamp} : {response}\n")

        # Play the response
        self.text_to_speech(patient_response)

        # self.virtual_patient_output_text.insert(tk.END, response + "\n")
        
        # self.text_to_speech(patient_response)

    def get_virtual_patient_response(self, user_input):
        # thread_id = create_thread()
        response = send_message_and_get_response(self.thread_id, user_input)
        return response

    def text_to_speech(self, text):
        headers = {"Authorization": f"Bearer {openai.api_key}"}
        data = {"model": "tts-1", "input": text, "voice": "echo", "response_format": "mp3"}
        response = requests.post("https://api.openai.com/v1/audio/speech", headers=headers, json=data, stream=True)
        output_file = os.path.join('cache', 'patient_response.mp3')
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            self.play_mp3(output_file)
        else:
            print(f"Error in text-to-speech conversion: {response.status_code} {response.text}")

    def play_mp3(self, path):
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)

def create_thread():
    thread = openai.beta.threads.create()
    return thread.id

def send_message_and_get_response(thread_id, message_text):
    openai.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message_text
    )
    run = openai.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    run_id = run.id
    while True:
        status = check_status(run_id, thread_id)
        if status == "completed":
            break
        time.sleep(2)
    response = openai.beta.threads.messages.list(thread_id=thread_id)
    return response.data[0].content[0].text.value

def check_status(run_id, thread_id):
    run = openai.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id,
    )
    return run.status

if __name__ == "__main__":
    root = tk.Tk()
    thread_id = create_thread()
    app = VirtualPatientApp(root, thread_id)
    root.mainloop()
