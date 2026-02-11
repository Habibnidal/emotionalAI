import os
import requests
import base64
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from gtts import gTTS

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL")

CHAT_API_URL = "https://router.huggingface.co/v1/chat/completions"

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://emotionalai-ecru.vercel.app/"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Request schema
# --------------------------------------------------
class UserInput(BaseModel):
    text: str
    category: str
    language: str = "en"

# --------------------------------------------------
# gTTS helpers
# --------------------------------------------------
def malayalam_tts(text: str) -> str:
    mp3_fp = BytesIO()
    tts = gTTS(text=text, lang="ml", slow=False)
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return base64.b64encode(mp3_fp.read()).decode("utf-8")


def english_tts(text: str) -> str:
    mp3_fp = BytesIO()
    tts = gTTS(text=text, lang="en", slow=False)
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return base64.b64encode(mp3_fp.read()).decode("utf-8")

# --------------------------------------------------
# Core response generator
# --------------------------------------------------
def generate_response(user_text: str, category: str, language: str) -> dict:
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }

    # -----------------------------
    # Prompts
    # -----------------------------
    if language == "ml":
        system_prompt = system_prompt =f"""
നീ ഉപയോക്താവിന്റെ ഏറ്റവും അടുത്ത ആത്മാർത്ഥ സുഹൃത്താണ്.

മലയാളത്തിൽ മാത്രം സംസാരിക്കുക.
കണ്ണൂർ സ്ലാങ് ഉപയോഗിച്ച് വളരെ ലളിതമായ സംസാര ഭാഷയിൽ മറുപടി നൽകുക.
കഠിനമായ മലയാളം വാക്കുകളും പുസ്തക ശൈലിയുമൊന്നും ഉപയോഗിക്കരുത്.

സ്നേഹത്തോടും കരുതലോടും സംസാരിക്കുക.
ഉപയോക്താവിനെ കുറ്റപ്പെടുത്തരുത്.
ധൈര്യവും ആശ്വാസവും നൽകുന്ന രീതിയിൽ മറുപടി നൽകുക.
ഉപയോക്താവിന് ഇപ്പോൾ സഹായം വേണ്ട വിഷയം: {category}
"""


    else:
        system_prompt = f"""
You are the user’s close friend.

Reply in a warm, friendly, and natural tone,
like talking to someone you genuinely care about.
Do not sound formal, robotic, or like a book.

Use simple, clear language.
Talk the way a close friend would speak in real life.

You may answer briefly or at length,
but only say as much as you can say clearly and completely.

Do NOT stop in the middle of a sentence.
Do NOT stop in the middle of an idea.
If you feel you are nearing the end,
finish the current thought naturally and then stop.

The final sentence must feel complete and natural,
not cut off or unfinished.

Do not repeat phrases.
Do not give medical advice.

Focus: {category}
"""

    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.9,
        "max_tokens": 500,
    }

    # -----------------------------
    # HuggingFace request
    # -----------------------------
    r = requests.post(
        CHAT_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    data = r.json()
    reply = data["choices"][0]["message"]["content"]

    # -----------------------------
    # Audio generation
    # -----------------------------
    if language == "ml":
        audio = malayalam_tts(reply)
    else:
        audio = english_tts(reply)

    return {
        "response": reply,
        "audio": audio,
        "audio_format": "mp3"
    }

# --------------------------------------------------
# API endpoint
# --------------------------------------------------
@app.post("/analyze")
async def analyze(data: UserInput):
    return generate_response(
        data.text,
        data.category,
        data.language
    )
