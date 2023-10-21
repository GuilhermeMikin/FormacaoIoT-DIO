from ultralytics import YOLO
import cv2
import math 
from gtts import gTTS
from pydub import AudioSegment
import simpleaudio as sa
from time import sleep
from threading import Thread

object_list = list()

def text2speech():
    global object_list
    while True:
        audio_path = "text.mp3"
        try:
            text = f"I saw a {object_list[-1]}"
        except Exception:
            text = f"Hi"
        gtts_object = gTTS(text=text, lang="en", slow=False)
        gtts_object.save(audio_path)
        sleep(.5)
        audio = AudioSegment.from_file(audio_path)
        # Play the audio
        playback_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)
        # Wait for the audio to finish playing
        playback_obj.wait_done()
        try:
            print(f"thread text2speech saying {object_list[-1]}")
        except Exception:
            pass
        sleep(1)

def process_video_frames():
    count = 0
    global object_list
    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values
                
                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100

                # class name
                cls = int(box.cls[0])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                label = f"{classNames[cls]}: {confidence:.2f}"

                cv2.putText(img, label, org, font, fontScale, color, thickness)
        try:
            print(f"classname: {label} ")
            if count > 15 and float(confidence) >= 0.7 and object_list[-1] != classNames[cls]:
                object_list.append(classNames[cls])
                print(f"listaa:{object_list}")
                count = 0
            elif count > 15 and float(confidence) >= 0.4 and object_list[-1] != classNames[cls] and object_list[-2] != classNames[cls]:
                object_list.append(classNames[cls])
                print(f"listaa2:{object_list}")
                count = 0
        except Exception:
            print("Exception")
            object_list.append(classNames[cls])
        count += 1
        print(f"count: {count}")
                
        sleep(.1)
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # start webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # model
    model = YOLO("yolo-Weights/yolov8n.pt")

    # object classes
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

    thread = Thread(target=process_video_frames)
    # Run text2speech in a separate thread
    text_thread = Thread(target=text2speech)
    thread.start()
    text_thread.start()
    # thread.join()
    # text_thread.join()

    

