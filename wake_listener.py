#!/usr/bin/env python3
"""
Wake-word listener for WaveShare stereo audio card:
- Captures stereo 16kHz S16_LE audio using arecord (-c 2)
- Converts to mono inside Python
- Runs Vosk STT to detect wake-phrase "hey visionaid"
- On detection, runs record_and_run.sh then returns to listening
- Shows ALL logs from record_and_run.sh and the main pipeline
"""

import subprocess
import sys
import os
import time
import numpy as np
from vosk import Model, KaldiRecognizer

MODEL_PATH = "/opt/vosk-model/model_small_en"   # <-- EDIT
ARECORD_DEVICE = "hw:0,0"
WAKE_PHRASE = "okay vision"

SAMPLE_RATE = 16000
CHANNELS = 2
SAMPLE_WIDTH = 2
FRAME_SIZE = SAMPLE_WIDTH * CHANNELS
CHUNK_FRAMES = 8000
CHUNK_BYTES = CHUNK_FRAMES * FRAME_SIZE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_SCRIPT = os.path.join(SCRIPT_DIR, "run.sh")


def start_arecord_stream():
    return subprocess.Popen(
        [
            "arecord",
            "-D", ARECORD_DEVICE,
            "-f", "S16_LE",
            "-r", str(SAMPLE_RATE),
            "-c", str(CHANNELS),
            "-t", "raw"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )


def run_with_live_output(cmd_list):
    """
    Run subprocess and print stdout+stderr in real time.
    """
    print("\n================ PIPELINE STARTED ================\n")
    process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in process.stdout:
        print(line, end="")  # live streaming output

    process.wait()
    print("\n================ PIPELINE FINISHED ===============\n")
    return process.returncode


def stereo_to_mono(raw_bytes):
    data = np.frombuffer(raw_bytes, dtype=np.int16)
    if len(data) % 2 != 0:
        return None
    stereo = data.reshape(-1, 2)
    mono = stereo.mean(axis=1).astype(np.int16)
    return mono.tobytes()


def main():
    print("Loading Vosk model:", MODEL_PATH)
    model = Model(MODEL_PATH)

    print("Starting stereo wake listener on device:", ARECORD_DEVICE)

    while True:
        arec = start_arecord_stream()
        rec = KaldiRecognizer(model, SAMPLE_RATE)

        try:
            while True:
                buf = arec.stdout.read(CHUNK_BYTES)
                if not buf:
                    break

                mono = stereo_to_mono(buf)
                if mono is None:
                    continue

                if rec.AcceptWaveform(mono):
                    res = rec.Result().lower()
                    # print("[FULL]:", res)   # <---- add this
                    if WAKE_PHRASE in res:
                        print("\n*** Wake phrase detected (full):", res)
                        arec.terminate()
                        arec.wait()   # <-- this is required
                        time.sleep(0.3)   # <-- give ALSA time to release device
                        run_with_live_output(["stdbuf", "-oL", RECORD_SCRIPT, ARECORD_DEVICE])
                        time.sleep(0.5)
                        break
                else:
                    partial = rec.PartialResult().lower()
                    # print("[PARTIAL]:", partial)   # <---- add this
                    if WAKE_PHRASE in partial:
                        print("\n*** Wake phrase detected :", partial)
                        arec.terminate()
                        arec.wait()   # <-- this is required
                        time.sleep(0.3)   # <-- give ALSA time to release device

                        run_with_live_output(["stdbuf", "-oL", RECORD_SCRIPT, ARECORD_DEVICE])
                        time.sleep(0.5)
                        break


        except KeyboardInterrupt:
            print("Stopped by user.")
            arec.terminate()
            sys.exit(0)

        except Exception as e:
            print("Error:", e)
            try:
                arec.terminate()
            except:
                pass
            time.sleep(1)
            continue


if __name__ == "__main__":
    main()
