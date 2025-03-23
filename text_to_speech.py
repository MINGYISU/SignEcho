import os
from google.cloud import texttospeech
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def text_to_speech(text='No audio Available!', output_file="audio/output.mp3"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'vivid-plateau-442021-f3-9b91f2ec315c.json'

    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="cmn-CN",
        name="cmn-CN-Wavenet-D",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.15,
        pitch=2.0
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    with open(output_file, "wb") as out:
        out.write(response.audio_content)
    print(f"Audio File saved to: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    text_to_speech("Hello!")