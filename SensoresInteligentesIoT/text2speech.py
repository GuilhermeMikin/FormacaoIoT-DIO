from gtts import gTTS
from pydub import AudioSegment
import simpleaudio as sa
from time import sleep

text2speech = "fala memo coxa velox"
audio_path = "gtts.mp3"
language = "pt"

gtts_object = gTTS(text= text2speech, lang=language, tld="com.br", slow=False)

gtts_object.save(audio_path)

sleep(1)
audio = AudioSegment.from_file(audio_path)
# Play the audio
playback_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)

# Wait for the audio to finish playing
playback_obj.wait_done()

print(audio_path)
