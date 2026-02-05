import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("❌ GOOGLE_API_KEY not found in .env")
    exit()

# Configure Gemini
genai.configure(api_key=API_KEY)

print("✅ API Key Loaded Successfully")
print("🔍 Fetching available models...\n")

try:
    models = genai.list_models()

    print("📌 Available Gemini Models:\n")
    for m in models:
        print(f"- {m.name}")

except Exception as e:
    print(f"❌ Error fetching models: {e}")
