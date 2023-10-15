import speech_recognition as sr
import os
from gtts import gTTS
from pydub import AudioSegment
import simpleaudio as sa
from time import sleep
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

#Função para ouvir e reconhecer a fala
def listen():
    """Func to listen and recognize speak"""
    #Enable User's microphone
    mic = sr.Recognizer()

    #Using the mic
    with sr.Microphone() as source:
        #Calls a noise reduction algorithm to the sound
        mic.adjust_for_ambient_noise(source)

        text2peech("text.mp3", "Oi, diga algo")
        #Stores what was said in a variable
        audio = mic.listen(source)

    try:
        #Pass the variable to the recognition function
        frase = mic.recognize_google(audio, language='pt-BR')

        if "navegador" in frase:
            text2peech("text.mp3", "Abrindo navegador")
            os.system("start Chrome.exe")

        elif "e-mail" in frase:
            now = datetime.now()
            corpo_email = f"""#Mensagem a ser enviada
            \nDatetime: {now.strftime('%d/%m/%Y %H:%M:%S')}"""
            if send_email(corpo_email, 'email@gmail.com', 'email_2@gmail.com'):
                text2peech("text.mp3", 'Email enviado!')
            else:
                text2peech("text.mp3", 'Falha ao enviar email')

        #Print what was said 
        print(f"You said: {frase}")

    except sr.UnknownValueError:
        text2peech("msg.mp3", "Não entendi")
    return frase


def text2peech(audio_path, text):
    gtts_object = gTTS(text=text, lang="pt", tld="com.br", slow=False)
    gtts_object.save(audio_path)
    sleep(.5)
    audio = AudioSegment.from_file(audio_path)
    # Play the audio
    playback_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)
    # Wait for the audio to finish playing
    playback_obj.wait_done()


def send_email(corpo_email, email_recipt1, email_recipt2 = None):
    msg = MIMEMultipart()
    password = 'senha'
    msg['Subject'] = f'Assistente Python'
    msg['From'] = 'emailexemplo@gmail.com'
    recipts = [email_recipt1, email_recipt2]
    msg['To'] = ', '.join(recipts)
    msg.attach(MIMEText(corpo_email, "plain"))

    try:
        s = smtplib.SMTP('smtp.gmail.com: 587')
        s.starttls()
        s.login(msg['From'], password)
        s.sendmail(msg['From'], msg['To'], msg.as_string().encode('utf-8'))
        s.quit()
        return 1
    except Exception as e: 
        return 0

listen()