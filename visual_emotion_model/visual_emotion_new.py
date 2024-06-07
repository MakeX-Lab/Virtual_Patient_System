import cv2
import mediapipe as mp
import numpy as np
import math
from collections import Counter
from deepface import DeepFace
from datetime import datetime

# Constants for Face Landmarks
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
            249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# MediaPipe Initializations
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Global Variables
profile = -1
age = np.zeros((10, 1))
gender, race = [], []
emotion_probs = np.zeros((7, 1))

# initialize hard-coded positions
furrowFrames = 0
browFrames = 0
frameCounter = 0
rFrames = 0
furrow_sum = 0.0
furrow_average = 0.0
brow_sum = 0.0
brow_average = 0.0
nodFrames = 0
nods = 0
oldhead_height = 0
head_height = 0
recording = False
smiling = False
browFurrow = False
browRaised = False
browSkew = False
beginNod = False
forward = False
leanFlag = False
openFlag = True
prev_left_shoulder = None
prev_right_shoulder = None

# initialize text outputs
outLeaning = "not "
outOpen = ""
outEyeContact = "not "
outSmiling = "not "
outAge = ""
outGender = ""
outRace = ""
pronoun1 = ""
pronoun2 = ""
dominant_emotion = "not detected"

recording = True

filename = "./data/visual_data.txt"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filenameTime = f"{filename[:-4]}_{timestamp}.txt"
recording_control = "recording_control.txt"


def reset_nods():
    global nods
    nods = 0
    print("Nods value has been reset to 0")


def initialize_visual_emotion(callback):
    global cap, holistic, face_mesh, image, profile, furrowFrames, furrow_average, outAge
    global furrow_frames, browFrames, frameCounter, rFrames, furrow_sum, furrow_average, brow_sum, brow_average, nodFrames, nods, oldhead_height, head_height
    global recording, smiling, browFurrow, browRaised, browSkew, beginNod, forward, leanFlag, openFlag, prev_left_shoulder, prev_right_shoulder
    global outLeaning, outOpen, outEyeContact, outSmiling, outGender, outRace, pronoun1, pronoun2, dominant_emotion

    cap = cv2.VideoCapture(0)
    
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
            mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:


        with open(filenameTime, "w") as file:

            while cap.isOpened():
                ret, frame = cap.read()

                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                img_h, img_w, img_c = frame.shape
                face_2d = []
                face_3d = []

                # Make Detections
                results = holistic.process(image)
                face_results = face_mesh.process(image)

                # Recolor image back to BGR for rendering
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 1. Draw face landmarks
                # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))

                # 2. Right hand
                # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))

                # 3. Left Hand
                # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))

                # 4. Pose Detections
                # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                # Export coordinates

                if face_results.multi_face_landmarks:
                    face_landmarks = face_results.multi_face_landmarks[0]

                    # EYE TRACKING

                    # Calculate left eye midpoint
                    left_eye_points = np.array([[int(p.x * img_w), int(p.y * img_h)]
                                            for idx, p in enumerate(face_landmarks.landmark) if idx in LEFT_EYE])
                    left_midpoint = np.mean(left_eye_points, axis=0)

                    # Calculate right eye midpoint
                    right_eye_points = np.array([[int(p.x * img_w), int(p.y * img_h)]
                                                for idx, p in enumerate(face_landmarks.landmark) if idx in RIGHT_EYE])
                    right_midpoint = np.mean(right_eye_points, axis=0)

                    # Calculate iris centers
                    left_iris_center = np.mean(np.array([[int(p.x * img_w), int(p.y * img_h)]
                                            for idx, p in enumerate(face_landmarks.landmark) if idx in LEFT_IRIS]), axis=0)
                    right_iris_center = np.mean(np.array(
                        [[int(p.x * img_w), int(p.y * img_h)] for idx, p in enumerate(face_landmarks.landmark) if idx in RIGHT_IRIS]), axis=0)

                    # Calculate deviation of iris centers from eye midpoints
                    left_deviation = left_iris_center[0] - left_midpoint[0]
                    right_deviation = right_iris_center[0] - right_midpoint[0]

                    # cv2.putText(image, f"Left Deviation: {int(left_deviation)}", (320, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    # cv2.putText(image, f"Right Deviation: {int(right_deviation)}", (320, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # SMILE DETECTION

                    lip_width = math.sqrt((abs(face_landmarks.landmark[287].y - face_landmarks.landmark[57].y))**2 + (
                        abs(face_landmarks.landmark[287].x - face_landmarks.landmark[57].x))**2)
                    jaw_width = math.sqrt((abs(face_landmarks.landmark[401].y - face_landmarks.landmark[177].y))**2 + (
                        abs(face_landmarks.landmark[401].x - face_landmarks.landmark[177].x))**2)

                    smile_ratio = lip_width/jaw_width
                    # print(smile_ratio)

                    lip_left = math.sqrt((abs(face_landmarks.landmark[287].y - face_landmarks.landmark[0].y))**2 + (
                        abs(face_landmarks.landmark[287].x - face_landmarks.landmark[0].x))**2)
                    lip_right = math.sqrt((abs(face_landmarks.landmark[0].y - face_landmarks.landmark[57].y))**2 + (
                        abs(face_landmarks.landmark[0].x - face_landmarks.landmark[57].x))**2)
                    lip_semi = (lip_right + lip_left + lip_width)/2
                    lip_area = math.sqrt(
                        lip_semi*(lip_semi - lip_right)*(lip_semi - lip_left)*(lip_semi - lip_width))
                    lip_height = 2*lip_area/lip_width
                    # print(lip_height)

                    if lip_height < 0.01 and smile_ratio > 0.49:
                        smiling = True
                        outSmiling = ""
                        cv2.putText(image, 'Smiling', (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        smiling = False
                        outSmiling = "not "

                    # EYEBROW TRACKING

                    brow_width = math.sqrt((abs(face_landmarks.landmark[285].y - face_landmarks.landmark[55].y))**2 + (
                        abs(face_landmarks.landmark[285].x - face_landmarks.landmark[55].x))**2)
                    eye_width = math.sqrt((abs(face_landmarks.landmark[362].y - face_landmarks.landmark[133].y))**2 + (
                        abs(face_landmarks.landmark[362].x - face_landmarks.landmark[133].x))**2)

                    furrow_ratio = brow_width/eye_width
                    if furrowFrames > 0:
                        furrow_sum = furrow_sum + furrow_ratio
                        furrow_average = furrow_sum/(furrowFrames + 1)
                    else:
                        furrow_sum = furrow_ratio

                    if furrow_average - furrow_ratio >= 0.03:
                        browFurrow = True
                        furrow_sum = furrow_sum - furrow_ratio
                        furrowFrames -= 1
                    else:
                        browFurrow = False

                    # print(f"furrow? {browFurrow} average: {furrow_average} ratio: {furrow_ratio}")

                    left_eyelid = math.sqrt((abs(face_landmarks.landmark[442].y - face_landmarks.landmark[295].y))**2 + (
                        abs(face_landmarks.landmark[442].x - face_landmarks.landmark[295].x))**2)
                    left_brow = math.sqrt((abs(face_landmarks.landmark[295].y - face_landmarks.landmark[296].y))**2 + (
                        abs(face_landmarks.landmark[295].x - face_landmarks.landmark[296].x))**2)
                    right_eyelid = math.sqrt((abs(face_landmarks.landmark[222].y - face_landmarks.landmark[65].y))**2 + (
                        abs(face_landmarks.landmark[222].x - face_landmarks.landmark[65].x))**2)
                    right_brow = math.sqrt((abs(face_landmarks.landmark[65].y - face_landmarks.landmark[66].y))**2 + (
                        abs(face_landmarks.landmark[65].x - face_landmarks.landmark[66].x))**2)

                    lbrow_ratio = left_brow/left_eyelid
                    rbrow_ratio = right_brow/right_eyelid
                    brow_ratio = (lbrow_ratio + rbrow_ratio)/2
                    # print(f"left: {lbrow_ratio} right: {rbrow_ratio}")
                    # print(brow_ratio)

                    if browFrames > 0:
                        brow_sum = brow_sum + brow_ratio
                        brow_average = brow_sum/(browFrames + 1)
                    else:
                        brow_sum = brow_ratio
                    if brow_average - brow_ratio >= 0.085:
                        browRaised = True
                        brow_sum = brow_sum - brow_ratio
                        browFrames -= 1
                    else:
                        browRaised = False
                    # print(f"raised? {browRaised} average: {brow_average} ratio: {brow_ratio}")

                    if ((brow_average - lbrow_ratio >= 0.08) or (brow_average - lbrow_ratio >= 0.08)) and not browRaised:
                        browSkew = True
                    else:
                        browSkew = False
                    # print(f"skewed? {browSkew} left: {brow_average - lbrow_ratio} right: {brow_average - rbrow_ratio}")

                    # HEAD TRACKING

                    for face_landmarks in face_results.multi_face_landmarks:
                        for idx, lm in enumerate(face_landmarks.landmark):
                            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                                if idx == 1:
                                    nose_2d = (lm.x * img_w, lm.y * img_h)
                                    nose_3d = (lm.x * img_w, lm.y *
                                            img_h, lm.z * 3000)
                                x, y = int(lm.x * img_w), int(lm.y * img_h)

                                face_2d.append([x, y])
                                face_3d.append(([x, y, lm.z]))

                        # Get 2d Coord
                        face_2d = np.array(face_2d, dtype=np.float64)

                        face_3d = np.array(face_3d, dtype=np.float64)

                        focal_length = 1 * img_w

                        cam_matrix = np.array([[focal_length, 0, img_h/2],
                                            [0, focal_length, img_w/2],
                                            [0, 0, 1]])
                        distortion_matrix = np.zeros((4, 1), dtype=np.float64)

                        success, rotation_vec, translation_vec = cv2.solvePnP(
                            face_3d, face_2d, cam_matrix, distortion_matrix)

                        # getting rotation of face
                        rmat, jac = cv2.Rodrigues(rotation_vec)

                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                        x = angles[0] * 360
                        y = angles[1] * 360
                        z = angles[2] * 360

                        # NODDING
                        if profile == 11 or profile == -1:
                            # if not recording:
                            # print(f"beginNod: {beginNod} head_height: {head_height} x: {x}")
                            if beginNod:
                                if (x - head_height) >= 5 or (x - oldhead_height) >= 2:
                                    nods += 1

                                    nodFrames = 0
                                    beginNod = False
                                    break
                                else:
                                    nodFrames += 1
                                    if nodFrames > 3:
                                        nodFrames = 0
                                        beginNod = False
                                        break
                            else:
                                if (head_height - x) >= 3 or (oldhead_height - x) >= 3:
                                    beginNod = True
                            oldhead_height = head_height
                            head_height = x
                            rFrames += 1
                            # else:
                            #     oldhead_height = x
                            #     head_height = x
                            print("HERE>>>>>>>>>>0")
                            cv2.putText(image, "Nods:", (440, 240),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                            cv2.putText(image, str(nods), (500, 290),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

                        # here based on axis rot angle is calculated
                        if y < -10:
                            text = "Looking Right"
                            forward = False
                            outEyeContact = "not "
                        elif y > 10:
                            text = "Looking Left"
                            forward = False
                            outEyeContact = "not "
                        elif x < -3:
                            text = "Looking Down"
                            forward = False
                            outEyeContact = "not "
                        elif x > 15:
                            text = "Looking Up"
                            forward = False
                            outEyeContact = "not "
                        else:
                            text = "Forward"
                            forward = True
                            outEyeContact = "not "
                            if (left_deviation < 2 and left_deviation > -2) and (right_deviation < 2 and right_deviation > -2):
                                text = "Eye Contact"
                                outEyeContact = ""

                        nose_3d_projection, jacobian = cv2.projectPoints(
                            nose_3d, rotation_vec, translation_vec, cam_matrix, distortion_matrix)

                        p1 = (int(nose_2d[0]), int(nose_2d[1]))
                        p2 = (int(nose_2d[0] + y*10), int(nose_2d[1] - x * 10))

                        # cv2.line(image,p1,p2,(255,0,0),3)

                        cv2.putText(image, text, (20, 400),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        # cv2.putText(image,"x: " + str(np.round(x,2)),(500,350),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                        # cv2.putText(image,"y: "+ str(np.round(y,2)),(500,400),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                        # cv2.putText(image,"z: "+ str(np.round(z, 2)), (500, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                try:
                    if profile < 10 and profile > -1:
                        demographics = DeepFace.analyze(
                            image, actions=['age', 'gender', 'race'], silent=True)
                        for result in demographics:
                            age[profile] = result["age"]
                            gender.append(result["dominant_gender"])
                            race.append(result["dominant_race"])
                        profile += 1
                    elif profile == 10:
                        # print(np.mean(age))
                        genders = Counter(gender)
                        # print(genders.most_common(1)[0][0])
                        races = Counter(race)
                        # print(races.most_common(1)[0][0])
                        profile += 1
                    elif profile == -1:
                        pass
                    else:
                        # Draw text on the frame
                        outAge = np.mean(age)
                        text = f"Age: {np.mean(age)}"
                        cv2.putText(image, text, (300, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        outGender = genders.most_common(1)[0][0]
                        if (outGender == "Woman"):
                            pronoun1 = "She"
                            pronoun2 = "Her"
                            outGender = "woman"
                        elif (outGender == "Man"):
                            pronoun1 = "He"
                            pronoun2 = "His"
                            outGender = "man"
                        text = f"Gender: {genders.most_common(1)[0][0]}"
                        cv2.putText(image, text, (300, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        outRace = races.most_common(1)[0][0]
                        text = f"Race: {races.most_common(1)[0][0]}"
                        cv2.putText(image, text, (300, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    emotion_results = DeepFace.analyze(
                        image, actions=['emotion'], silent=True)

                    # Extract all emotions with their probabilities for each face
                    for idx, emotion in enumerate(emotion_results):
                        for i, (emo, prob) in enumerate(emotion['emotion'].items()):
                            emotion_probs[i] = prob
                            # print(emotion_probs[i])
                    # print()

                    # order is 0 angry, 1 disgust, 2 fear, 3 happy, 4 sad, 5 surprise, 6 neutral

                    if emotion_probs[3] > 50.0:
                        dominant_emotion = "happy"
                    elif emotion_probs[1] > 2.0:
                        dominant_emotion = "disgust"
                    elif (smiling and browFurrow) or (not smiling and browSkew):
                        dominant_emotion = "doubt"
                    elif smiling:
                        dominant_emotion = "sympathetic"
                    elif browFurrow and emotion_probs[0] > 10.0:
                        dominant_emotion = "angry"
                    elif emotion_probs[4] > 60.0:
                        dominant_emotion = "sad"
                    elif emotion_probs[2] > 90.0 and (browFurrow or browRaised):
                        dominant_emotion = "fear"
                    elif browFurrow:
                        dominant_emotion = "confused"
                    elif browRaised:
                        dominant_emotion = "surprise"
                    else:
                        dominant_emotion = "neutral"
                        # Extract dominant emotion for each face
                        # for emotion in emotion_results:
                        # dominant_emotion = emotion['dominant_emotion']
                        # print("Dominant Emotion:", dominant_emotion)

                    text = f"Dominant Emotion: {dominant_emotion}"
                    cv2.putText(image, text, (20, 350),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                except ValueError:
                    pass

                try:

                    # if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    

                    # SHOULDERS / LEAN
                    print("HERE>>>>>>>>>>1")
                    left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                    print("HERE>>>>>>>>>>10000")
                    right_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                    print("HERE>>>>>>>>>>20000")
                    
                    print(frameCounter)
                    if frameCounter == 10:
                        frameCounter = 0
                    print(frameCounter)

                    print("HERE>>>>>>>>>>2")
                    
                    shoulder_width = math.sqrt(
                        (abs(left_shoulder.y - right_shoulder.y))**2 + (abs(left_shoulder.x - right_shoulder.x))**2)
                    # print(shoulder_width)

                    if shoulder_width > 0.60:
                        leanFlag = True
                        outLeaning = ""
                    elif shoulder_width < 0.50:
                        leanFlag = False
                        outLeaning = "not "
                        
                    print("HERE>>>>>>>>>>3")

                    if frameCounter == 0:

                        if prev_left_shoulder != None and prev_right_shoulder != None:

                            oldshoulder_width = prev_left_shoulder - prev_right_shoulder
                            shoulder_growth = (
                                shoulder_width - oldshoulder_width)/oldshoulder_width

                            if shoulder_growth > 0.06 and leanFlag == False:
                                leanFlag = True
                                outLeaning = ""
                            elif shoulder_growth < -0.08 and leanFlag == True:
                                leanFlag = False
                                outLeaning = "not "

                        prev_left_shoulder = left_shoulder.x
                        prev_right_shoulder = right_shoulder.x

                        # print(shoulderGrowth)

                    # Display Lean Status
                    cv2.putText(image, 'LEANING: {}'.format(leanFlag),
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # OPEN POSE

                    left_wrist = landmarks[mp_holistic.PoseLandmark.LEFT_WRIST]
                    right_wrist = landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST]

                    left_knee = landmarks[mp_holistic.PoseLandmark.LEFT_KNEE]
                    right_knee = landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE]

                    left_ankle = landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE]
                    right_ankle = landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE]

                    if left_wrist.visibility > 0.8 and right_wrist.visibility > 0.8:

                        # print(left_wrist.x,right_wrist.x)
                        # print(left_wrist.visibility,right_wrist.visibility)

                        # wrist cross check
                        if left_wrist.x < right_wrist.x:
                            openFlag = False
                            outOpen = "not "
                        else:
                            openFlag = True
                            outOpen = ""

                        handThreshold = 0.70

                        # hand distance check
                        if results.left_hand_landmarks and results.right_hand_landmarks:

                            left_hand_landmarks = results.left_hand_landmarks
                            right_hand_landmarks = results.right_hand_landmarks

                            for left_landmark in left_hand_landmarks.landmark:
                                # Get the coordinates of the left landmark
                                left_x, left_y, left_z = left_landmark.x, left_landmark.y, left_landmark.z

                                # Iterate over all landmarks of the other hand
                                for right_landmark in right_hand_landmarks.landmark:
                                    # Get the coordinates of the right landmark
                                    right_x, right_y, right_z = right_landmark.x, right_landmark.y, right_landmark.z

                                    # Calculate the distance between the two landmarks
                                    distance = (
                                        (left_x - right_x)**2 + (left_y - right_y)**2 + (left_z - right_z)**2)**0.5

                                    # Check if the distance is less than the threshold
                                    if distance < handThreshold:
                                        # Hands are intersecting
                                        openFlag = False
                                        outOpen = "not "
                                        break  # Exit the inner loop if intersection is found
                                else:
                                    continue  # Continue to the next left landmark if no intersection found
                                break  # Exit the outer loop if intersection is found

                        if (left_knee.visibility > 0.8 and right_knee.visibility > 0.8) and abs(left_knee.y - right_knee.y) == 0.5:
                            openFlag = False
                            outOpen = "not "

                        if (left_ankle.visibility > 0.8 and right_ankle.visibility > 0.8) and left_ankle.x < right_ankle.x:
                            openFlag = False
                            outOpen = "not "

                    cv2.putText(image, 'OPEN POSE: {}'.format(openFlag),
                                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    frameCounter += 1

                except:
                    pass

                furrowFrames += 1
                browFrames += 1

                if (outAge != ""):
                    if recording:
                        print(f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')}, The student is a {outAge} year old {outRace} {outGender}. {pronoun1} is {outLeaning}leaning forward, {outEyeContact}making eye contact, {outSmiling}smiling, and {outOpen}displaying open posture. {pronoun2} current expression is {dominant_emotion}.", file=file)
                    else:
                        print(f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')}, The student is a {outAge} year old {outRace} {outGender}. {pronoun1} is {outLeaning}leaning forward, {outEyeContact}making eye contact, {outSmiling}smiling, and {outOpen}displaying open posture. {pronoun2} current expression is {dominant_emotion}. {pronoun1} has nodded in acknowledgement {nods} times.", file=file)
                elif profile == -1:
                    if recording:
                        print(f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')}, The student is {outLeaning}leaning forward, {outEyeContact}making eye contact, {outSmiling}smiling, and {outOpen}displaying open posture, current expression is {dominant_emotion}.", file=file)
                    else:
                        print(f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')}, The student is {outLeaning}leaning forward, {outEyeContact}making eye contact, {outSmiling}smiling, and {outOpen}displaying open posture, current expression is {dominant_emotion}, nodded {nods} times.", file=file)

                # Save the frame to a file
                cv2.imwrite('cache/current_frame.jpg', image)

                # Call the callback function with the frame
                callback(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # If you want to break the loop on 'q' key press (not applicable here)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
                # cv2.imshow('Raw Webcam Feed', image)
                # if cv2.waitKey(10) & 0xFF == ord('q'):
                #     break
    cap.release()
    cv2.destroyAllWindows()

# Main Execution
# if __name__ == "__main__":
#     initialize_visual_emotion(None)  
#     cap.release()
#     cv2.destroyAllWindows()

