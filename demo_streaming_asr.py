from frame_asr import FrameASR
import numpy as np
import pyaudio as pa
import time

CHANNELS = 1
RATE = 16000
DURATION = 2.0
CHUNK_SIZE = int(DURATION*RATE)

p = pa.PyAudio()

print('Available audio input devices:')
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev.get('maxInputChannels'):
        print(i, dev.get('name'))
print('Please type input device ID:')
dev_idx = int(input())


asr = FrameASR()
print('Initialization was successful')


def callback(in_data, frame_count, time_info, status):
    signal = np.frombuffer(in_data, dtype=np.int16)
    pred = asr.transcribe(signal)
    if len(pred.strip()):
        print('"{}"'.format(pred))
    return (in_data, pa.paContinue)


stream = p.open(format=pa.paInt16,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=dev_idx,
                stream_callback=callback,
                frames_per_buffer=CHUNK_SIZE)

stream.start_stream()

while stream.is_active():
    time.sleep(0.1)

stream.stop_stream()
stream.close()
p.terminate()

