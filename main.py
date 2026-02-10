import os
import requests
import base64
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from googletrans import Translator
from gtts import gTTS

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL")

CHAT_API_URL = "https://router.huggingface.co/v1/chat/completions"

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request schema
# -----------------------------
class UserInput(BaseModel):
    text: str
    category: str
    language: str = "en"

# -----------------------------
# Translator
# -----------------------------
translator = Translator()

# -----------------------------
# Malayalam TTS using Google
# -----------------------------
def google_malayalam_tts(text: str) -> str:
    mp3_fp = BytesIO()
    tts = gTTS(text=text, lang="ml")
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    audio_base64 = base64.b64encode(mp3_fp.read()).decode("utf-8")
    return audio_base64

# -----------------------------
# Core response generator
# -----------------------------
def generate_response(user_text: str, category: str, language: str = "en") -> dict:
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }

    # ---------- Malayalam ----------
    if language == "ml":
        try:
            translated = translator.translate(user_text, src="auto", dest="ml")
            user_text_to_use = translated.text
        except Exception:
            user_text_to_use = user_text

        system_prompt = f"""
നിങ്ങൾ ഒരു സ്നേഹനിറഞ്ഞ വൈകാരിക സഹായിയാണ്.
എപ്പോഴും സ്വാഭാവിക മലയാളത്തിൽ മാത്രം സംസാരിക്കുക.
ഒരേ വാക്യങ്ങൾ ആവർത്തിക്കരുത്.
വൈദ്യപരമായ ഉപദേശങ്ങൾ നൽകരുത്.

വിഷയം: {category}
"""
    else:
        user_text_to_use = user_text
        system_prompt = f"""
You are a kind emotional assistant.
Do not repeat phrases.
No medical advice.

Focus: {category}
"""

    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text_to_use},
        ],
        "temperature": 0.9,
        "max_tokens": 200,
    }

    r = requests.post(CHAT_API_URL, headers=headers, json=payload, timeout=60)
    data = r.json()

    reply = data["choices"][0]["message"]["content"]

    # ---------- Generate voice if Malayalam ----------
    if language == "ml":
        audio = google_malayalam_tts(reply)
        return {
            "response": reply,
            "audio": audio,
            "audio_format": "mp3"
        }

    return {"response": reply}

# -----------------------------
# API endpoint
# -----------------------------
@app.post("/analyze")
async def analyze(data: UserInput):
    return generate_response(data.text, data.category, data.language)
