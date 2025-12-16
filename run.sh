rm -rf output
rm test.wav
export PERPLEXITY_API_KEY=""
arecord -D hw:0,0 -f S16_LE -r 16000 -c 2 -t wav -d 10 test_stereo.wav
ffmpeg -i test_stereo.wav -ac 1 -ar 16000 test.wav
python vision_hw_lite_rt.py
cd output
aplay -D plughw:0,0 result.wav
cd ..
