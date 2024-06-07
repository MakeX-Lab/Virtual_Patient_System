import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
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
import random
import visual_emotion_model.visual_emotion_new as ve

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

        # Set a prettier font
        self.font = ("Helvetica", 12)
        self.bold_font = ("Helvetica", 12, "bold")

        # Main frame for layout
        main_frame = ttk.Frame(root, padding="10 10 10 10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create frames for different sections
        self.avatar_frame = ttk.LabelFrame(main_frame, text="Virtual Patient Avatar", padding="10 10 10 10")
        self.avatar_frame.grid(row=0, column=0, rowspan=3, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.responses_frame = ttk.LabelFrame(main_frame, text="Virtual Patient Responses", padding="10 10 10 10")
        self.responses_frame.grid(row=0, column=1, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.student_frame = ttk.LabelFrame(main_frame, text="Student", padding="10 10 10 10")
        self.student_frame.grid(row=1, column=1, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.camera_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10 10 10 10")
        self.camera_frame.grid(row=2, column=1, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Avatar area
        self.avatar_label = ttk.Label(self.avatar_frame)
        self.avatar_label.grid(row=0, column=0)
        self.update_avatar_image()

        # Responses area
        self.virtual_patient_output_text = tk.Text(self.responses_frame, height=15, width=58, font=self.font, wrap=tk.WORD, padx=5, pady=5, bd=2, relief="groove")
        self.virtual_patient_output_text.grid(row=0, column=0, padx=5, pady=5)

        # Student input area
        self.student_input_text = tk.Text(self.student_frame, height=5, width=58, font=self.font, wrap=tk.WORD, padx=5, pady=5, bd=2, relief="groove")
        self.student_input_text.grid(row=0, column=0, padx=5, pady=5)

        # Camera feed area
        self.camera_label = ttk.Label(self.camera_frame, text="Camera feed will appear here", font=self.font)
        self.camera_label.grid(row=0, column=0)

        # Record button
        self.record_button = ttk.Button(main_frame, text="Start Recording", command=self.toggle_recording, style='TButton')
        self.record_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Customize button font
        self.style = ttk.Style()
        self.style.configure('TButton', font=self.font)

        # Variables for recording
        self.is_recording = False
        self.recording = []
        self.fs = 44100  # Sample rate
        self.channels = 1
        self.start_time = None
        self.end_time = None

        # Start visual emotion detection script
        threading.Thread(target=self.start_visual_emotion_detection).start()
        
        # visual_emotion_script_path = "visual_emotion_model/visual_emotion_2.py"
        # subprocess.Popen(["python", visual_emotion_script_path])
        # print("Starting up the Visual_Emotion_Detection Model!!!")
        # time.sleep(10)  # Wait for the model to start

        # Update the camera feed
        self.update_camera_feed()
        
    def start_visual_emotion_detection(self):
        # ve.initialize_visual_emotion()
        ve.initialize_visual_emotion(self.update_camera_frame)
        
    def update_camera_feed(self):
        self.root.after(100, self.update_camera_feed)
    
    def update_camera_frame(self, frame):
        img = Image.fromarray(frame)
        width, height = img.size
        aspect_ratio = width / height
        # print(f"aspect ratio: {aspect_ratio}")
        new_width = 500
        new_height = int(new_width / aspect_ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)
        
        
        

    # def update_camera_feed(self):
    #     try:
    #         # Read the latest frame
    #         img = Image.open('cache/current_frame.jpg')

            # width, height = img.size
            # aspect_ratio = width / height
            # # print(f"aspect ratio: {aspect_ratio}")
            # new_width = 500
            # new_height = int(new_width / aspect_ratio)
    #         img = img.resize((new_width, new_height), Image.LANCZOS)

    #         imgtk = ImageTk.PhotoImage(image=img)
    #         self.camera_label.imgtk = imgtk
    #         self.camera_label.configure(image=imgtk)
    #     except Exception as e:
    #         print(f"Error reading image: {e}")
    #     self.root.after(100, self.update_camera_feed)
        
    def update_avatar_image(self):
        try:
            # Load the avatar image
            avatar_img_path = 'cache/virtual_patient_avatar.jpg'
            img = Image.open(avatar_img_path)

            width, height = img.size
            aspect_ratio = width / height
            new_width = 900
            new_height = int(new_width / aspect_ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)
            self.avatar_label.imgtk = imgtk
            self.avatar_label.configure(image=imgtk)
        except Exception as e:
            print(f"Error reading avatar image: {e}")


    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        # import visual_emotion_model.visual_emotion_2 as ve
        ve.reset_nods()
        
        self.is_recording = True
        self.recording = []
        self.start_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
        self.record_button.config(text="Stop Recording")
        threading.Thread(target=self.record_audio).start()

    def stop_recording(self):
        self.is_recording = False
        self.end_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
        self.record_button.config(text="Start Recording")
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

        # Get current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Insert the transcribed text with timestamp
        self.student_input_text.insert(tk.END, f"{timestamp} - Student: {transcribed_text}\n")

        # Update the UI to indicate we are getting a response
        self.virtual_patient_output_text.insert(tk.END, f"{timestamp} - Getting response...\n")
        threading.Thread(target=self.get_and_process_response, args=(transcribed_text, timestamp)).start()

    def transcribe_audio(self, file_path):
        model = whisper.load_model("tiny.en")
        result = model.transcribe(file_path)
        return result["text"]

    def get_and_process_response(self, user_input, timestamp):
        response = self.get_virtual_patient_response(user_input)

        # Insert the response with timestamp
        self.virtual_patient_output_text.insert(tk.END, f"{timestamp} - Virtual Patient: {response}\n")

        # Play the response
        self.text_to_speech(response)

    def get_virtual_patient_response(self, user_input):
        # thread_id = create_thread()
        response = send_message_and_get_response(thread_id, user_input)
        return response

    def text_to_speech(self, text):
        headers = {"Authorization": f"Bearer {openai.api_key}"}
        voices = ["alloy","echo","fable","onyx","nova","shimmer"]
        selected_voice = random.choice(voices)
        data = {"model": "tts-1", "input": text, "voice": selected_voice, "response_format": "mp3"}
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
