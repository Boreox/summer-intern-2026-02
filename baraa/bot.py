import sqlite3, uuid, re, os
from ollama import Client
from duckduckgo_search import DDGS
import yt_dlp
import whisper

DB_NAME = "chat_history.db"

# Load the local Whisper AI model once at start
print("🎧 Loading Whisper Audio Model...")
whisper_model = whisper.load_model("base")

def db_query(query, params=(), fetch=False, fetchall=False):
    """Handles all database interactions in a single helper."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.execute(query, params)
        conn.commit()
        if fetchall: return cursor.fetchall()
        if fetch: return cursor.fetchone()

def init_db():
    db_query("CREATE TABLE IF NOT EXISTS chat_messages (session_id TEXT, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
    db_query("CREATE TABLE IF NOT EXISTS user_profile (key TEXT PRIMARY KEY, value TEXT)")

def ai_detect_name(client, model, user_input):
    """Uses Gemma behind the scenes to extract your name from natural sentences."""
    prompt = f"Analyze: '{user_input}'. If the user states their name, reply with ONLY the name (no punctuation). Otherwise reply: None"
    try:
        res = client.generate(model=model, prompt=prompt).get('response', '').strip()
        if res and res.lower() != "none" and len(res) < 30:
            db_query("INSERT OR REPLACE INTO user_profile VALUES ('user_name', ?)", (res,))
            return res
    except: pass
    return None

def web_search_tool(query):
    """Searches DuckDuckGo and returns top results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results: return "[No results found]"
            return "\n".join([f"Source {i+1}: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}\n" for i, r in enumerate(results)])
    except Exception as e:
        return f"[Search error: {e}]"

def whisper_youtube_tool(url):
    """Downloads YouTube audio and uses local Whisper AI to transcribe speech."""
    temp_audio_file = f"temp_{uuid.uuid4().hex[:6]}.mp3"
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': temp_audio_file.replace('.mp3', ''),
        'quiet': True,
        'no_warnings': True
    }
    
    try:
        print("⏬ Downloading video audio track...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        print("🧠 Listening to voice audio with Whisper AI...")
        # fp16=False prevents CPU precision warnings on non-GPU setups
        result = whisper_model.transcribe(temp_audio_file, fp16=False)
        transcript_text = result.get("text", "").strip()
        
        # Clean up temporary audio file after processing
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
            
        if not transcript_text:
            return "[Error: No spoken speech could be detected in this video.]"
            
        return transcript_text[:30000] # Safe context limit
    except Exception as e:
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
        return f"[Error processing video audio with Whisper: {e}]"

def main():
    init_db()
    
    ollama_host = "http://127.0.0.1:11434/"
    client = Client(host=ollama_host)
    model_to_use = "gemma4:31b-cloud"
    
    print("\n🤖 OLLAMA TERMINAL CHATBOT\n[1] New Chat\n[2] Resume Chat")
    choice = input("\nChoice: ").strip()
    session_id = str(uuid.uuid4())[:8]

    if choice == "2":
        sessions = db_query("SELECT session_id, content FROM chat_messages WHERE role='user' GROUP BY session_id ORDER BY timestamp DESC LIMIT 5", fetchall=True)
        if sessions:
            for i, (s_id, text) in enumerate(sessions, 1):
                print(f"  {i}. [{s_id}] {text[:30]}...")
            try:
                session_id = sessions[int(input("\nSelect number: ").strip()) - 1][0]
                print(f"\n🔄 Resumed session: {session_id}")
                for r, c in db_query("SELECT role, content FROM chat_messages WHERE session_id=? ORDER BY timestamp ASC", (session_id,), fetchall=True)[-4:]:
                    print(f"{'You' if r=='user' else 'Bot'}: {c}")
            except: print("\nInvalid choice. Starting new chat...")
        else: print("\nNo past sessions found. Starting new chat...")

    print(f"\n🤖 Active: {model_to_use} @ {ollama_host} (Session: {session_id})")
    print("💡 TOOL COMMAND CHEAT SHEET:")
    print("  1. WEB SEARCH             -> search: <your query>")
    print("  2. WHISPER AUDIO SUMMARY  -> youtube: <video URL>")
    print("Type 'exit' to quit.\n" + "-"*40)

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ['quit', 'exit']: break
        if not user_input.strip(): continue

        db_query("INSERT INTO chat_messages (session_id, role, content) VALUES (?, 'user', ?)", (session_id, user_input))
        
        saved_name = db_query("SELECT value FROM user_profile WHERE key='user_name'", fetch=True)
        saved_name = saved_name[0] if saved_name else ai_detect_name(client, model_to_use, user_input)

        processed_input = user_input
        is_tool_call = False

        if user_input.lower().startswith("search:"):
            search_query = user_input[7:].strip()
            print(f"🔍 Searching the web for: '{search_query}'...")
            search_results = web_search_tool(search_query)
            processed_input = f"[Live Web Search Results for '{search_query}']:\n{search_results}\n\nAnswer the query using this live data."
            is_tool_call = True

        elif user_input.lower().startswith("youtube:"):
            yt_url = user_input[8:].strip()
            transcript = whisper_youtube_tool(yt_url)
            
            # Format requested: Dialogue script format (User 1, User 2, etc.)
            processed_input = (
                f"[Speech Transcript Extracted via Whisper AI]:\n{transcript}\n\n"
                f"Task:\n"
                f"1. Convert and structure this audio transcript into a natural conversational script.\n"
                f"2. Label the participants clearly as 'User 1:', 'User 2:' (or by their specific names if stated in the audio).\n"
                f"3. Below the conversation script, include a 1-2 sentence core takeaway summary."
            )
            is_tool_call = True

        messages = []
        if saved_name:
            messages.append({"role": "system", "content": f"The user's name is {saved_name}."})
        
        history = db_query("SELECT role, content FROM chat_messages WHERE session_id=? ORDER BY timestamp ASC", (session_id,), fetchall=True)
        formatted_history = [{"role": r, "content": c} for r, c in history]
        
        if is_tool_call and formatted_history:
            formatted_history[-1]["content"] = processed_input

        messages.extend(formatted_history)
        
        print("Thinking...")
        try:
            bot_res = client.chat(model=model_to_use, messages=messages).get('message', {}).get('content', '')
            print(f"\nBot: {bot_res}\n" + "-"*40)
            db_query("INSERT INTO chat_messages (session_id, role, content) VALUES (?, 'assistant', ?)", (session_id, bot_res))
        except Exception as e:
            print(f"\n❌ Connection or API Error: {e}\n")

if __name__ == "__main__":
    main()