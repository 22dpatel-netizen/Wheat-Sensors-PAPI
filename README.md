# Raspberry Pi Integration

This part of the project runs on a Raspberry Pi and connects physical sensors and a camera to the AI model.

## Overview

The Raspberry Pi acts as the interface between hardware (sensors and camera) and the trained machine learning model. It allows real-time data collection and image-based disease detection.

## Features

* Terminal-based menu system for user input
* Reads environmental and soil data
* Captures images using a connected camera
* Runs AI model on captured images
* Converts trained model for lightweight deployment

## Files

* sensors.py

  * collects atmosphere and soil moisture readings
  * prints sensor data to the terminal

* convert.py

  * converts the trained .keras model into TensorFlow Lite format (.tflite)
  * optimised for running on low-power devices like Raspberry Pi

## How It Works

* user selects option from terminal menu
* sensors.py reads and displays environmental data
* camera captures image when selected
* image is resized to 224x224
* converted AI model (.tflite) processes the image
* output classification is displayed (e.g. healthy, rust, pest)

## Notes

* TensorFlow Lite is used for faster inference on the Raspberry Pi
* model conversion is required before deployment
* system is designed for real-time use with minimal hardware

## Future Improvements

* integrate all components into a single interface
* display results on a GUI instead of terminal
* connect to cloud for data storage and analysis
