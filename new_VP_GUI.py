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
from collections import Counter
import re
import audonnx
import audinterface
import visual_emotion_model.visual_emotion_new as ve

tone_emotion_model = audonnx.load('tone_emotion_model')

interface = audinterface.Feature(
    tone_emotion_model.labels('logits'),
    process_func=tone_emotion_model,
    process_func_args={
        'outputs': 'logits',
    }
)

# Load environment variables
load_dotenv()

# API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
assistant_id = os.getenv("VP_ASSIST_KEY")

class VirtualPatientApp:
    def __init__(self, root, thread_id, voice):
        self.root = root
        self.root.title("Virtual Patient Assistant")

        self.thread_id = thread_id
        self.voice = voice

        # Load and resize microphone images
        mic_off = Image.open('cache/mic_off.png')
        mic_on = Image.open('cache/mic_on.png')
        # Resize images to a suitable button size, e.g., 50x50 pixels
        mic_off_resized = mic_off.resize((50, 50), Image.Resampling.LANCZOS)
        mic_on_resized = mic_on.resize((50, 50), Image.Resampling.LANCZOS)
        
        # Convert resized images to PhotoImage
        self.mic_off_image = ImageTk.PhotoImage(mic_off_resized)
        self.mic_on_image = ImageTk.PhotoImage(mic_on_resized)

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
        self.virtual_patient_output_text = tk.Text(self.responses_frame, height=15, width=65, font=self.font, wrap=tk.WORD, padx=5, pady=5, bd=2, relief="groove")
        self.virtual_patient_output_text.grid(row=0, column=0, padx=5, pady=5)

        # Student input area
        self.student_input_text = tk.Text(self.student_frame, height=5, width=65, font=self.font, wrap=tk.WORD, padx=5, pady=5, bd=2, relief="groove")
        self.student_input_text.grid(row=0, column=0, padx=5, pady=5)

        # Camera feed area
        self.camera_label = ttk.Label(self.camera_frame, text="Camera feed will appear here", font=self.font)
        self.camera_label.grid(row=0, column=0)

        # Record button with circular image
        self.create_circular_button(main_frame, 3, 0, 2)

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

        # Update the camera feed
        self.update_camera_feed()
        
        
    def create_circular_button(self, parent, row, column, columnspan):
        self.canvas = tk.Canvas(parent, width=60, height=60, highlightthickness=0)
        self.canvas.grid(row=row, column=column, columnspan=columnspan, pady=10)

        # Draw a circle and place the image inside it
        self.circle = self.canvas.create_oval(5, 5, 55, 55, outline='black', width=2)
        self.image_id = self.canvas.create_image(30, 30, image=self.mic_off_image)

        # Bind events to the canvas
        self.canvas.bind("<Enter>", self.on_enter)
        self.canvas.bind("<Leave>", self.on_leave)
        self.canvas.bind("<Button-1>", self.toggle_recording)

    def on_enter(self, event):
        self.canvas.itemconfig(self.circle, outline='blue')

    def on_leave(self, event):
        self.canvas.itemconfig(self.circle, outline='black')

    def start_visual_emotion_detection(self):
        ve.initialize_visual_emotion(self.update_camera_frame)
        
    def update_camera_feed(self):
        self.root.after(100, self.update_camera_feed)
    
    def update_camera_frame(self, frame):
        img = Image.fromarray(frame)
        width, height = img.size
        aspect_ratio = width / height
        new_width = 500
        new_height = int(new_width / aspect_ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)
        
    def update_avatar_image(self):
        try:
            # Load the avatar image
            avatar_img_path = 'cache/patient-2.jpg'
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

    def toggle_recording(self, event=None):
        if self.is_recording:
            self.stop_recording()
            self.canvas.itemconfig(self.image_id, image=self.mic_off_image)
        else:
            self.start_recording()
            self.canvas.itemconfig(self.image_id, image=self.mic_on_image)

    def start_recording(self):
        self.is_recording = True
        self.recording = []
        self.start_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
        threading.Thread(target=self.record_audio).start()

    def stop_recording(self):
        ve.reset_nods()   # reset the nods count
        self.is_recording = False
        self.end_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
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
        
        visual_emotion_summary = get_visual_emo_data(self.start_time, self.end_time)
        speech_emotion_summary = get_speech_emo_data()
        
        user_input = transcribed_text + visual_emotion_summary + speech_emotion_summary
        
        print(visual_emotion_summary)
        print(speech_emotion_summary)

        # Get current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Insert the transcribed text with timestamp
        self.student_input_text.insert(tk.END, f"{timestamp} - Student: {transcribed_text}\n")

        # Update the UI to indicate we are getting a response
        self.virtual_patient_output_text.insert(tk.END, f"{timestamp} - Getting response...\n\n")
        threading.Thread(target=self.get_and_process_response, args=(user_input, timestamp)).start()

    def transcribe_audio(self, file_path):
        model = whisper.load_model("tiny.en")
        result = model.transcribe(file_path)
        return result["text"]

    def get_and_process_response(self, user_input, timestamp):
        response = self.get_virtual_patient_response(user_input)
        parts = re.split(r'\[|\]', response)
        patient_response = parts[0].rstrip()
        behavioral_cues = parts[1] if len(parts) > 1 else 'Behavioral Cues: Not available'
        disease_name = re.search(r'\{(.+?)\}', response)
        disease_name = f"Disease name: {{{disease_name.group(1)}}}" if disease_name else "Disease name: Not specified"

        # Formatting the output with the new structure
        new_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_response = f"{new_timestamp}\nVirtual Patient Response: {patient_response}\n\nBehavioral Cues: [{behavioral_cues}]\n\n{disease_name}\n\n"

        self.virtual_patient_output_text.insert(tk.END, formatted_response)

        # Play the response
        self.text_to_speech(patient_response)

    def get_virtual_patient_response(self, user_input):
        response = send_message_and_get_response(self.thread_id, user_input)
        return response

    def text_to_speech(self, text):
        headers = {"Authorization": f"Bearer {openai.api_key}"}
        data = {"model": "tts-1", "input": text, "voice": self.voice, "response_format": "mp3"}
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
            
def summarize_emotion_data(data):
    leaning = Counter()
    eye_contact = Counter()
    smiling = Counter()
    posture = Counter()
    expression = Counter()
    nods = 0

    for record in data:
        parts = record.strip().split(", ")
        if len(parts) < 2:
            continue

        behaviors = parts[1:5]
        current_expression = parts[5].split(' is ')[-1]
        nods = int(parts[-1].split(' ')[-2])

        if len(behaviors) == 4:
            leaning[behaviors[0]] += 1
            eye_contact[behaviors[1]] += 1
            smiling[behaviors[2]] += 1
            posture[behaviors[3]] += 1
            expression[current_expression] += 1

    total_entries = len(data)
    if total_entries == 0:
        return "No data to summarize."

    summary = [
        f"[Visual emotion: ",
        f"The student is {'not leaning forward' if leaning['The student is not leaning forward'] > leaning['The student is leaning forward'] else 'leaning forward'} ({max(leaning['The student is not leaning forward'], leaning['The student is leaning forward']) / total_entries:.1%}) more often than {'leaning forward' if leaning['The student is not leaning forward'] > leaning['The student is leaning forward'] else 'not leaning forward'} ({min(leaning['The student is not leaning forward'], leaning['The student is leaning forward']) / total_entries:.1%})",
        f"He/she is {'not making eye contact' if eye_contact['not making eye contact'] > eye_contact['making eye contact'] else 'making eye contact'} ({max(eye_contact['not making eye contact'], eye_contact['making eye contact']) / total_entries:.1%}) more often than {'making eye contact' if eye_contact['not making eye contact'] > eye_contact['making eye contact'] else 'not making eye contact'} ({min(eye_contact['not making eye contact'], eye_contact['making eye contact']) / total_entries:.1%})",
        f"He/she is {'not smiling' if smiling['not smiling'] > smiling['smiling'] else 'smiling'} ({max(smiling['not smiling'], smiling['smiling']) / total_entries:.1%}) more often than {'smiling' if smiling['not smiling'] > smiling['smiling'] else 'not smiling'} ({min(smiling['not smiling'], smiling['smiling']) / total_entries:.1%})",
        f"He/she is {'not displaying open posture' if posture['not displaying open posture'] > posture['displaying open posture'] else 'displaying open posture'} ({max(posture['not displaying open posture'], posture['displaying open posture']) / total_entries:.1%}) more often than {'displaying open posture' if posture['not displaying open posture'] > posture['displaying open posture'] else 'not displaying open posture'} ({min(posture['not displaying open posture'], posture['displaying open posture']) / total_entries:.1%})",
        f"He/she has nodded {nods} times in acknowledgment since the last interaction,",
    ]

    expressions_summary = ", ".join(f"{expr} ({count / total_entries:.1%})" for expr, count in expression.items())
    summary.append(f"Expressions: {expressions_summary}]")
    
    return ". ".join(summary)
        
def get_visual_emo_data(start_time, end_time):
    
    start_time = datetime.strptime(start_time, '%Y-%m-%d_%H:%M:%S.%f')
    end_time = datetime.strptime(end_time, '%Y-%m-%d_%H:%M:%S.%f')
    
    time.sleep(2)
    
    directory = '/Users/saran/CODE/GITHUB/Virtual_Patient_System/data'
    files = os.listdir(directory)
    text_files = [file for file in files if file.endswith('.txt')]

    if not text_files:
        print("No text files found in the directory.")
    else:
        text_files.sort()
        latest_file = text_files[-1]
        latest_file_path = os.path.join(directory, latest_file)
        with open(latest_file_path, 'r') as file:
            lines = file.readlines()
            filtered_lines = []
            for line in lines:
                timestamp_str = line.split(',')[0].strip()
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d_%H:%M:%S.%f')
                if start_time <= timestamp <= end_time:
                    print(line)
                    filtered_lines.append(line)

        summary = summarize_emotion_data(filtered_lines)
        return(summary)
    
def get_speech_emo_data():
    audio_files = []
    audio_files.append('./cache/audio_input.wav')
    result = interface.process_files(audio_files)

    row = result.iloc[0]
    arousal = round(row['arousal'], 2)
    dominance = round(row['dominance'], 2)
    valence = round(row['valence'], 2)

    summary = f"[Speech emotion: Arousal {arousal}, Dominance {dominance}, Valence {valence}]"
    return summary

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
    voices = ["alloy","echo","fable","onyx"]
    selected_voice = random.choice(voices)
    print("Selected Voice: ", selected_voice)
    app = VirtualPatientApp(root, thread_id, selected_voice)
    root.mainloop()
