import openai
import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import requests
import os
from dotenv import load_dotenv
from pydub import AudioSegment
import pygame
import time
import re
import random
import socket
import audeer
import audonnx
import audinterface
import subprocess
from datetime import datetime
from collections import Counter

tone_emotion_model = audonnx.load('tone_emotion_model')

interface = audinterface.Feature(
    tone_emotion_model.labels('logits'),
    process_func=tone_emotion_model,
    process_func_args={
        'outputs': 'logits',
    }
)

def cache_path(file):
    return os.path.join('cache', file)

def send_trigger_to_server(trigger_message, host='127.0.0.2', port=8080):
    # Create a socket (SOCK_STREAM means a TCP socket)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to server and send data
        sock.connect((host, port))
        sock.sendall(bytes(trigger_message, "utf-8"))
        # Optionally, you can receive a response here if your server sends any
        # response = sock.recv(1024)
        # print("Received: {}".format(response))

# Load environment variables
load_dotenv()

# API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
assistant_id = os.getenv("VP_ASSIST_KEY")


def create_thread():

    #create a thread
    thread = openai.beta.threads.create()
    my_thread_id = thread.id
    print(my_thread_id)

    return my_thread_id

def send_message_and_get_response(thread_id, message_text):
    # Create a message in the thread
    openai.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message_text
    )

    #run
    run = openai.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    ) 

    run_id = run.id

    # Wait for the run to complete
    while True:
        status = check_status(run_id, thread_id)
        print("...",status)
        if status == "completed":
            break
        time.sleep(2)

    # Retrieve and return the latest message from the thread
    response = openai.beta.threads.messages.list(thread_id=thread_id)

    return response.data[0].content[0].text.value

def check_status(run_id,thread_id):
    run = openai.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id,
    )
    return run.status

#########################################################################

# Recording parameters
fs = 44100  # Sample rate
channels = 1


# Function to record audio
def record_audio():
    global is_recording, recording
    print("Press Enter to start recording...")
    input()  # Wait for Enter key to start recording
    is_recording = True
    start_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
    recording = []
    print("Recording started. Press Enter again to stop.")

    def callback(indata, frames, time, status):
        recording.append(indata.copy())

    with sd.InputStream(samplerate=fs, channels=channels, callback=callback):
        input()  # Wait for Enter key to stop recording
        is_recording = False
        end_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
    
    print("Recording stopped.")
    # Concatenate all recorded audio fragments and save as a WAV file
    recorded_audio = np.concatenate(recording, axis=0)
    input_audio_file_path = cache_path('audio_input.wav')
    write(input_audio_file_path, fs, recorded_audio)
    return(start_time, end_time)

# Function to transcribe audio to text

def transcribe_audio(file_path="./cache/audio_input.wav"):
    # Load the Whisper model
    model = whisper.load_model("tiny.en")  
    # "tiny", "base", "small", "medium", "large"

    # Load and transcribe the audio file
    result = model.transcribe(file_path)
    return result["text"]


def text_to_speech(input_text, voice="echo", model="tts-1", output_file="cache/patient_response.mp3"):
    headers = {"Authorization": f"Bearer {openai.api_key}"}
    data = {"model": model, "input": input_text, "voice": voice, "response_format": "mp3"}
    response = requests.post("https://api.openai.com/v1/audio/speech", headers=headers, json=data, stream=True)
    
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        # print(f"File successfully written to {output_file}")
    else:
        print(f"Error in text-to-speech conversion: {response.status_code} {response.text}")

def play_mp3(path="cache/patient_response.mp3"):
    if os.path.exists(path):
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)
    else:
        print(f"Error: File {path} does not exist.")

def get_speech_emo_data():
    audio_files = []
    audio_files.append('./cache/audio_input.wav')
    result = interface.process_files(audio_files)

    row = result.iloc[0]

    # Extract specific values
    arousal = round(row['arousal'], 2)
    dominance = round(row['dominance'], 2)
    valence = round(row['valence'], 2)

    # Print the values
    summary = f"[Speech emotion: Arousal {arousal}, Dominance {dominance}, Valence {valence}]"
    # print(summary)

    return summary

def get_visual_emo_data(start_time, end_time):
    # Directory containing the files
    directory = '/Users/saran/CODE/GITHUB/Virtual_Patient_System/data'

    # Get all files in the directory
    files = os.listdir(directory)

    # Filter for text files, assuming they end with '.txt'
    text_files = [file for file in files if file.endswith('.txt')]

    # Check if there are any text files
    if not text_files:
        print("No text files found in the directory.")
    else:
        # Sort files by name (which includes the timestamp if formatted correctly)
        text_files.sort()

        # The latest file will be the last one in the sorted list
        latest_file = text_files[-1]

        # Full path to the latest file
        latest_file_path = os.path.join(directory, latest_file)

        # Read the contents of the latest file
        with open(latest_file_path, 'r') as file:
            # Read the lines from the file
            lines = file.readlines()

            # Filter lines
            filtered_lines = []
            for line in lines:
                # Extract the timestamp from the line
                timestamp_str = line.split(',')[0].strip()
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d_%H:%M:%S.%f')
                # print(timestamp)

                # Check if the timestamp is within the range
                if start_time <= timestamp <= end_time:
                    filtered_lines.append(line)

        # for line in filtered_lines:
        #     print(line)
        summary = summarize_emotion_data(filtered_lines)
        return(summary)

def summarize_emotion_data(data):
    # Initializing counters for each behavior and expression
    leaning = Counter()
    eye_contact = Counter()
    smiling = Counter()
    posture = Counter()
    expression = Counter()

    # Parse each record in the data
    for record in data:
        # Clean up any newline characters and split the record
        parts = record.strip().split(", ")
        if len(parts) < 2:
            continue  # Skip if the record is not in expected format

        # The first part is timestamp, ignore it, the rest are behaviors and expression
        behaviors = parts[1:5]  # Assumes exactly four behavior phrases before the expression
        current_expression = parts[5].split(' is ')[-1]

        # Tallying each behavior safely
        if len(behaviors) == 4:  # Check if there are exactly four behavior components
            leaning[behaviors[0]] += 1
            eye_contact[behaviors[1]] += 1
            smiling[behaviors[2]] += 1
            posture[behaviors[3]] += 1
            expression[current_expression] += 1

    # Total number of records processed
    total_entries = len(data)
    if total_entries == 0:
        return "No data to summarize."

    # Building the summary
    
    summary = [
        f"[Visual emotion: ",
        f"The student is {'not leaning forward' if leaning['The student is not leaning forward'] > leaning['The student is leaning forward'] else 'leaning forward'} ({max(leaning['The student is not leaning forward'], leaning['The student is leaning forward']) / total_entries:.1%}) more often than {'leaning forward' if leaning['The student is not leaning forward'] > leaning['The student is leaning forward'] else 'not leaning forward'} ({min(leaning['The student is not leaning forward'], leaning['The student is leaning forward']) / total_entries:.1%})",
        f"The student is {'not making eye contact' if eye_contact['not making eye contact'] > eye_contact['making eye contact'] else 'making eye contact'} ({max(eye_contact['not making eye contact'], eye_contact['making eye contact']) / total_entries:.1%}) more often than {'making eye contact' if eye_contact['not making eye contact'] > eye_contact['making eye contact'] else 'not making eye contact'} ({min(eye_contact['not making eye contact'], eye_contact['making eye contact']) / total_entries:.1%})",
        f"The student is {'not smiling' if smiling['not smiling'] > smiling['smiling'] else 'smiling'} ({max(smiling['not smiling'], smiling['smiling']) / total_entries:.1%}) more often than {'smiling' if smiling['not smiling'] > smiling['smiling'] else 'not smiling'} ({min(smiling['not smiling'], smiling['smiling']) / total_entries:.1%})",
        f"The student is {'not displaying open posture' if posture['not displaying open posture'] > posture['displaying open posture'] else 'displaying open posture'} ({max(posture['not displaying open posture'], posture['displaying open posture']) / total_entries:.1%}) more often than {'displaying open posture' if posture['not displaying open posture'] > posture['displaying open posture'] else 'not displaying open posture'} ({min(posture['not displaying open posture'], posture['displaying open posture']) / total_entries:.1%})",
    ]

    # Adding all expression percentages
    expressions_summary = ", ".join(f"{expr} ({count / total_entries:.1%})" for expr, count in expression.items())
    summary.append(f"Expressions: {expressions_summary}]")
    
    # print(". ".join(summary))
    return ". ".join(summary)

#########################################################################

def main():

    visual_emotion_script_path = "visual_emotion_model/visual_emotion.py"
    subprocess.Popen(["python", visual_emotion_script_path])

    print("Starting up the Visual_Emotion_Detection Model!!!")
    time.sleep(10)

    thread_id = create_thread()
    voices = ["alloy","echo","fable","onyx","nova","shimmer"]
    selected_voice = random.choice(voices)
    print(selected_voice)

    while True:
        start_time, end_time = record_audio()
        
        start_time = datetime.strptime(start_time, '%Y-%m-%d_%H:%M:%S.%f')
        end_time = datetime.strptime(end_time, '%Y-%m-%d_%H:%M:%S.%f')

        time.sleep(2)

        visual_emotion_summary = get_visual_emo_data(start_time, end_time)
        speech_emotion_summary = get_speech_emo_data()

        transcribed_text = transcribe_audio()
        print(visual_emotion_summary)
        print(speech_emotion_summary)
        print(transcribed_text)

        print(">>> Medical Student:", transcribed_text)
        user_input = transcribed_text + visual_emotion_summary + speech_emotion_summary
        check = user_input.rstrip()
        
        if check.lower() == 'exit' or check.lower() == 'exit.' or check.lower() == ' exit' or check.lower() == ' exit.':
            break
        whole_response = send_message_and_get_response(thread_id, user_input)
        
        patient_response = re.split(r'\[', whole_response, maxsplit=1)[0].rstrip()
        behavioral_cues = re.search(r'\[(.*?)\]', whole_response)
        disease_name = re.search(r'\{(.*?)\}', whole_response)

        # print(behavioral_cues)
        # print(disease_name)
        
        text_to_speech(patient_response, voice=selected_voice)
        print(">>> Virtual Patient:", whole_response)
        # send_trigger_to_server('trigger animator!!')
        play_mp3()
        print("")

if __name__ == "__main__":
    main()

