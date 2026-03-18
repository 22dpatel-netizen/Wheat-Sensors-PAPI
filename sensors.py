import numpy as np
import tflite as tf
from tflite_runtime.interpreter import Interpreter
import cv2
import time
import adafruit_dht
import board
import RPi.GPIO as GPIO
import tkinter
import Adafruit_ADS1x15
from picamera2 import Picamera2

import os
import subprocess
import threading   
 
last_photo = None
running = True
            
def find_atmosphere_conditions():
    dht_device = adafruit_dht.DHT11(board.D4)

    while True:
        try:
            temp = dht_device.temperature
            humidity = dht_device.humidity
            print ("Temperature:"+str(temp)+"C\nHumidity: "+str(humidity))
        except RuntimeError as error:
            print(error.args[0])

        time.sleep(3)

        
def check_soil():
    # Create an ADS1115 ADC object
    adc = Adafruit_ADS1x15.ADS1115(busnum=1)

    # Set the gain to ±4.096V (adjust if needed)
    GAIN = 1
    max_m = 21750
    min_m = 7700
    dry_threshold = 10
    wet_threshold = 60
    # Main loop to read the analog value from the soil moisture sensor and print the raw ADC value
    try:
        while True:
            # Read the raw analog value from channel A3
            raw_value = adc.read_adc(3, gain=GAIN)
            moisture = (max_m-raw_value)*100/(max_m-min_m)
            
            # Print the raw ADC value
            print("Raw Value: {}".format(raw_value))
            print("Moisture: {}%".format(moisture))
            if moisture < dry_threshold:
                print('soil needs WATER')
            elif moisture > wet_threshold:
                print('too much WATER')
            else:
                print('adaquate water level')


            # Add a delay between readings (adjust as needed)
            time.sleep(3)

    except KeyboardInterrupt:
        print("\nExiting the program.")
        

def predict_image():
    subprocess.run(['sudo', 'pkill', '-f', 'libcamera'], capture_output=True)
    subprocess.run(['sudo', 'pkill', '-f', 'picamera'], capture_output=True)
    os.environ['DISPLAY'] = ':0'

    os.environ['QT_QPA_PLATFORM'] = 'xcb'

    camera = Picamera2()
    camera.resolution = (704, 704)
    camera.start()
    time.sleep(2)



    def keyboard_input():
        global last_photo, running
        print("Press ENTER to take photo, type 'delete' to delete, type 'quit' to exit")
    
        while running == True:
            key = input()
            if key.lower() == 'quit':
                running = False

            else:
                last_photo = r'/home/pi5-alpha/Pictures/wheat.jpg'
                camera.capture_file(last_photo)
                print(f"Saved photo!")
                running = False
# Run keyboard input in background thread
    thread = threading.Thread(target=keyboard_input, daemon=True)
    thread.start()

    try:
        while running:
            frame = camera.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Camera Preview", frame)
            cv2.waitKey(1)

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Camera released.")
    
    # 1. Load and prepare interpreter (same as before)
    interpreter = Interpreter(model_path="/home/pi5-alpha/Downloads/model1.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. Preprocess Image

    start_img = cv2.imread(r'/home/pi5-alpha/Pictures/wheat.jpg')
    img = cv2.resize(start_img, (224, 224))

    img_array = np.expand_dims(img, axis=0)
    img_array = img_array.astype('float32')
    
    # 3. Run Inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # 4. Get Output and Calculate Confidence
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # If your model ends in a Softmax, output_data is already probabilities.
    # If not, we apply it manually for clarity:
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    probabilities = output_data[0]

    # 5. Extract results
    predicted_class = np.argmax(probabilities)
    match predicted_class:
        case 0:
            a = 'healthy'
        case 1:
            a = 'mildew'
        case 2:
            a = 'not_wheat'
        case 3:
            a = 'insect'
        case 4:
            a = 'rust'
            
    confidence = probabilities[predicted_class] * 100 # Convert to percentage

    print(f"Prediction: Class: {a}")
    print(f"Confidence: {confidence:.2f}%")

    
def main_screen():
    print("AgriRover 1.0\nby the Moloch team\nOptions:\n1: Check Atmosphere\n2: Check Soil Conditions\n3: Wheat-Check-Up")
    while True:    
        x = input()
        if x == '1' or x == '2' or x == '3':
            break
        print('valid input pls:)')
 
    print('loading...')
    time.sleep(1.0)
    if x == '1':
        find_atmosphere_conditions()
    elif x == '2':
        check_soil()
    elif x == '3':
        predict_image()
    else:
        print('ERROR')


main_screen()


