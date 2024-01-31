
# COMPILE Whisper as a shared lib:
    - cmake -Wdev .
    - sudo make install 

DEPENDENCIES:
    - sudo apt-get install libsdl2-dev
    - sudo apt install ffmpeg libavcodec-dev libavformat-dev libavfilter-dev libswscale-dev libavdevice-dev
    - https://www.geeksforgeeks.org/how-to-install-opencv-in-c-on-linux/
WHOLE APPLICATION:
    - cmake .
    - make 

#TODO: Bring start_whisper functions back!