# Mega Streamlit Project – One-File Monster App
# Author: ChatGPT (for Wizzy)
# Run:  streamlit run mega_app.py
# Python: 3.8+
"""
Feature Map
===========
Dashboard / Home – welcome screen with app descriptions.

🧮 Mega Calculator – math, BMI, age, temp converter, currency (fake/live*).
🎮 Fun Zone – Guess the Number, Quiz Game, Rock-Paper-Scissors.
📚 Productivity – Notes, To-Do, Text Summarizer, Dictionary/Translator.
📊 Data Playground – Upload CSV → auto charts + stats.
🎤 Voice / AI Zone – Voice-powered calculator*, Sentiment analyzer.
⚽ Football Match Predictor – simple rule-based predictor.

* Optional / Best-effort (works if extra libs available). The app gracefully falls back if not installed.

Recommended (optional) packages to improve features:
    pip install streamlit speechrecognition streamlit-mic-recorder pydub numpy pandas matplotlib nltk

Safe to run with just: pip install streamlit pandas numpy matplotlib nltk
"""

import math
import random
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Tuple

import streamlit as st
import numpy as np
import pandas as pd

# Optional imports guarded

try:
    import speech_recognition as sr  # For voice calc
    HAS_SR = True
except Exception:
    HAS_SR = False

try:
    # Community component that records mic audio in-browser
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC = True
except Exception:
    HAS_MIC = False

# NLTK Sentiment (VADER)
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        _ = nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    HAS_NLTK = True
except Exception:
    HAS_NLTK = False

# ---------- Utilities ----------

APP_NAME = "Mega Matrix"
st.set_page_config(page_title=APP_NAME, page_icon="⚡", layout="wide")

# Initialize session state containers
if 'notes' not in st.session_state:
    st.session_state.notes: List[Dict] = []
if 'todos' not in st.session_state:
    st.session_state.todos: List[Dict] = []
if 'guess_target' not in st.session_state:
    st.session_state.guess_target = random.randint(1, 100)
if 'guess_attempts' not in st.session_state:
    st.session_state.guess_attempts = 0
if 'quiz_idx' not in st.session_state:
    st.session_state.quiz_idx = 0
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0


# Safe expression evaluator
ALLOWED_FUNCS = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
ALLOWED_NAMES = {**ALLOWED_FUNCS, 'abs': abs, 'round': round}

def safe_eval(expr: str) -> float:
    expr = expr.replace('^', '**')
    # Security guard: allow only digits, operators, parentheses, dots, commas, spaces, and letters for math funcs
    import re
    if not re.fullmatch(r"[0-9\-+*/().,^% eE\sA-Za-z_]+", expr):
        raise ValueError("Unsupported characters in expression.")
    return eval(expr, {"__builtins__": {}}, ALLOWED_NAMES)


# Simple frequency-based text summarizer
def summarize_text(text: str, max_sentences: int = 3) -> str:
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= max_sentences:
        return text.strip()
    words = re.findall(r"[a-zA-Z']+", text.lower())
    stop = set("""a an the and or if in on at to for of with from by this that is are was were be been being as it's it'sn't it'sn t i you he she we they them us our your my me his her their it's its than then so but not no yes do did done have has had can could should would will just about into over under more most less many much very really""".split())
    freq = {}
    for w in words:
        if w not in stop:
            freq[w] = freq.get(w, 0) + 1
    scores = []
    for s in sentences:
        score = sum(freq.get(w.lower(), 0) for w in re.findall(r"[a-zA-Z']+", s))
        scores.append((score, s))
    top = [s for _, s in sorted(scores, key=lambda x: x[0], reverse=True)[:max_sentences]]
    return ' '.join(top)


# Fake/Static currency rates (base USD)
STATIC_RATES = {
    'USD': 1.0,
    'EUR': 0.92,
    'GBP': 0.78,
    'NGN': 1550.0,
    'JPY': 145.0,
    'CAD': 1.31,
    'INR': 83.0,
}


def convert_currency(amount: float, from_ccy: str, to_ccy: str, live: bool = False) -> float:
    # Live flag only simulates tiny random drift to mimic movement; no external APIs used.
    base_from = STATIC_RATES.get(from_ccy)
    base_to = STATIC_RATES.get(to_ccy)
    if base_from is None or base_to is None:
        raise ValueError("Unsupported currency")
    # Simulated live wobble
    wobble_from = 1 + (random.random() - 0.5) * 0.01 if live else 1
    wobble_to = 1 + (random.random() - 0.5) * 0.01 if live else 1
    return amount * (base_to * wobble_to) / (base_from * wobble_from)


# ------------- UI Sections -------------

def section_home():
    st.title("⚡ Mega Matrix")
    st.write("Welcome, legend. This is your all‑in‑one playground: calculators, games, tools, data, AI vibes, and football predictions. Dive in using the sidebar.")

    cols = st.columns(3)
    with cols[0]:
        st.header("🧮 Mega Calculator")
        st.write("Math, BMI, age, temperature & currency (fake/live-sim) conversions.")
    with cols[1]:
        st.header("🎮 Fun Zone")
        st.write("Guess the Number, Quiz Game, Rock-Paper-Scissors. Zero boredom allowed.")
    with cols[2]:
        st.header("📚 Productivity")
        st.write("Notes, To‑Do, Summarizer, Dictionary/Translator.")

    cols2 = st.columns(3)
    with cols2[0]:
        st.header("📊 Data Playground")
        st.write("Upload CSV → quick stats and auto charts.")
    with cols2[1]:
        st.header("🎤 Voice / AI Zone")
        st.write("Voice calculator (optional) + Sentiment analyzer.")
    with cols2[2]:
        st.header("⚽ Football Predictor")
        st.write("Simple model with vibes and sliders. No VAR drama here.")


def section_calculator():
    st.header("🧮 Mega Calculator")
    tab_math, tab_bmi, tab_age, tab_temp, tab_ccy = st.tabs([
        "Math", "BMI", "Age", "Temp Converter", "Currency"
    ])

    with tab_math:
        st.subheader("Math Expression Evaluator")
        expr = st.text_input("Enter expression (use math funcs like sin, cos, sqrt). Use ^ for power.", "2^8 + sqrt(144) - sin(0)")
        if st.button("Calculate", key="calc_btn"):
            try:
                result = safe_eval(expr)
                st.success(f"Result: {result}")
            except Exception as e:
                st.error(f"Error: {e}")
        st.caption("Allowed: + - * / % ^ ( ) and math functions like sin, cos, tan, log, sqrt…")

    with tab_bmi:
        st.subheader("BMI Calculator")
        c1, c2 = st.columns(2)
        with c1:
            w = st.number_input("Weight (kg)", 1.0, 400.0, 70.0)
        with c2:
            h = st.number_input("Height (cm)", 50.0, 250.0, 175.0)
        if st.button("Compute BMI"):
            bmi = w / ((h/100)**2)
            st.info(f"BMI: {bmi:.2f}")
            if bmi < 18.5:
                st.warning("Underweight")
            elif bmi < 25:
                st.success("Normal")
            elif bmi < 30:
                st.warning("Overweight")
            else:
                st.error("Obese")

    with tab_age:
        st.subheader("Age Calculator")
        dob = st.date_input("Date of Birth", dt.date(2005,1,1))
        today = dt.date.today()
        if st.button("Calculate Age"):
            years = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            days = (today - dob).days
            st.success(f"Age: {years} years (~{days} days)")

    with tab_temp:
        st.subheader("Temperature Converter")
        c1, c2, c3 = st.columns(3)
        unit_from = c1.selectbox("From", ["C", "F", "K"], index=0)
        unit_to = c2.selectbox("To", ["C", "F", "K"], index=1)
        val = c3.number_input("Value", value=25.0)

        def to_kelvin(u, x):
            return x + 273.15 if u == 'C' else (x - 32)*5/9 + 273.15 if u == 'F' else x
        def from_kelvin(u, k):
            return k - 273.15 if u == 'C' else (k - 273.15)*9/5 + 32 if u == 'F' else k
        if st.button("Convert"):
            k = to_kelvin(unit_from, val)
            out = from_kelvin(unit_to, k)
            st.success(f"{val}°{unit_from} = {out:.2f}°{unit_to}")

    with tab_ccy:
        st.subheader("Currency Converter (Fake/\"Live\" Sim)")
        c1, c2, c3, c4 = st.columns(4)
        amount = c1.number_input("Amount", value=100.0, min_value=0.0)
        from_ccy = c2.selectbox("From", list(STATIC_RATES.keys()), index=0)
        to_ccy = c3.selectbox("To", list(STATIC_RATES.keys()), index=3)
        live = c4.toggle("Simulate live movement", value=False)
        if st.button("Convert 💱"):
            try:
                res = convert_currency(amount, from_ccy, to_ccy, live)
                st.success(f"≈ {res:,.2f} {to_ccy}")
                st.caption("Rates are static or gently wobbled locally. No external API.")
            except Exception as e:
                st.error(str(e))


def section_fun_zone():
    st.header("🎮 Fun Zone")
    tab_guess, tab_quiz, tab_rps = st.tabs(["Guess the Number", "Quiz Game", "Rock‑Paper‑Scissors"])

    with tab_guess:
        st.subheader("Guess the Number (1–100)")
        guess = st.number_input("Your guess", 1, 100, 50)
        if st.button("Try"):
            st.session_state.guess_attempts += 1
            tgt = st.session_state.guess_target
            if guess == tgt:
                st.success(f"Correct! Number was {tgt}. Attempts: {st.session_state.guess_attempts}")
                if st.button("Play again"):
                    st.session_state.guess_target = random.randint(1, 100)
                    st.session_state.guess_attempts = 0
            elif guess < tgt:
                st.info("Too low. Go higher!")
            else:
                st.warning("Too high. Chill, go lower!")

    with tab_quiz:
        st.subheader("Quick Quiz")
        questions = [
            ("What is the capital of France?", ["Paris", "Lyon", "Marseille"], 0),
            ("2^5 = ?", ["16", "32", "64"], 1),
            ("Who invented Python?", ["Linus Torvalds", "Guido van Rossum", "James Gosling"], 1),
        ]
        idx = st.session_state.quiz_idx
        score = st.session_state.quiz_score
        if idx < len(questions):
            q, opts, ans = questions[idx]
            st.write(f"Q{idx+1}. {q}")
            choice = st.radio("Select", list(range(len(opts))), format_func=lambda i: opts[i])
            if st.button("Submit Answer"):
                if choice == ans:
                    st.success("Correct!")
                    st.session_state.quiz_score += 1
                else:
                    st.error(f"Wrong. Correct: {opts[ans]}")
                st.session_state.quiz_idx += 1
        else:
            st.info(f"Done! Score: {score}/{len(questions)}")
            if st.button("Restart Quiz"):
                st.session_state.quiz_idx = 0
                st.session_state.quiz_score = 0

    with tab_rps:
        st.subheader("Rock‑Paper‑Scissors")
        user = st.selectbox("You", ["Rock", "Paper", "Scissors"])
        if st.button("Shoot!"):
            comp = random.choice(["Rock", "Paper", "Scissors"])
            st.write(f"Computer chose **{comp}**")
            if user == comp:
                st.info("Draw. Rematch?")
            elif (user, comp) in [("Rock","Scissors"),("Paper","Rock"),("Scissors","Paper")]:
                st.success("You win. Light work ✨")
            else:
                st.error("L 😭 – try again")


def section_productivity():
    st.header("📚 Study / Productivity Tools")
    tab_notes, tab_todo, tab_sum, tab_dict = st.tabs(["Notes", "To‑Do", "Summarizer", "Dictionary/Translator"])

    with tab_notes:
        st.subheader("Notes (session only)")
        note_title = st.text_input("Title")
        note_body = st.text_area("Note")
        if st.button("Save Note"):
            if note_title or note_body:
                st.session_state.notes.append({"title": note_title, "body": note_body, "ts": dt.datetime.now()})
        if st.session_state.notes:
            for i, n in enumerate(reversed(st.session_state.notes)):
                st.markdown(f"**{n['title'] or 'Untitled'}** — _{n['ts'].strftime('%Y-%m-%d %H:%M')}_")
                st.write(n['body'])
                st.divider()
        else:
            st.caption("No notes yet. Type something genius.")

    with tab_todo:
        st.subheader("To‑Do List (session only)")
        new_todo = st.text_input("Add a task")
        if st.button("Add") and new_todo:
            st.session_state.todos.append({"task": new_todo, "done": False})
        for i, t in enumerate(st.session_state.todos):
            cols = st.columns([0.1, 0.7, 0.2])
            with cols[0]:
                st.session_state.todos[i]['done'] = st.checkbox("", value=t['done'], key=f"todo_{i}")
            with cols[1]:
                st.write("~~"+t['task']+"~~" if st.session_state.todos[i]['done'] else t['task'])
            with cols[2]:
                if st.button("Delete", key=f"del_{i}"):
                    st.session_state.todos.pop(i)
                    st.rerun()

    with tab_sum:
        st.subheader("Text Summarizer")
        text = st.text_area("Paste text to summarize")
        max_sents = st.slider("Summary sentences", 1, 7, 3)
        if st.button("Summarize"):
            if text.strip():
                st.success(summarize_text(text, max_sents))
            else:
                st.warning("Drop some text first.")

    with tab_dict:
        st.subheader("Dictionary / Translator")
        word = st.text_input("Word (EN)")
        text = st.text_area("Text to translate (optional)")
        lang = st.selectbox("Translate to", ["en", "fr", "es", "de", "it", "ha", "ig", "yo"])

        # Tiny built-in dictionary fallback
        MINI_DICT = {
            'python': 'a high-level programming language',
            'streamlit': 'a Python framework for data apps',
            'algorithm': 'a step-by-step procedure for calculations',
            'football': 'a sport played by two teams of eleven players',
        }
        if st.button("Lookup / Translate"):
            meaning = MINI_DICT.get(word.lower(), "No local definition. Try Google later.") if word else None
            if meaning:
                st.info(f"Definition of **{word}**: {meaning}")
            if text:
                # Mock translator: very small phrasebook + fallback pseudo-translation (word reversal per word)
                phrasebook = {
                    ('hello','fr'):'bonjour', ('thank you','fr'):'merci', ('good morning','fr'):'bonjour',
                    ('hello','es'):'hola', ('thank you','es'):'gracias', ('good morning','es'):'buenos días',
                    ('hello','de'):'hallo', ('thank you','de'):'danke',
                    ('hello','it'):'ciao', ('thank you','it'):'grazie',
                    ('hello','ha'):'sannu', ('thank you','ha'):'na gode',
                    ('hello','ig'):'ndeewo', ('thank you','ig'):'daalụ',
                    ('hello','yo'):'ẹ n lẹ', ('thank you','yo'):'ẹ ṣé',
                }
                key = (text.strip().lower(), lang)
                if key in phrasebook:
                    out = phrasebook[key]
                elif lang == 'en':
                    out = text
                else:
                    # playful reversible obfuscation to avoid external APIs
                    out = ' '.join(w[::-1] for w in text.split())
                st.success(f"Translation ({lang}): {out}")
            if not word and not text:
                st.warning("Type a word or some text, champ.")


def section_data_playground():
    st.header("📊 Data Playground")
    st.write("Upload a CSV and we’ll give you quick stats and easy charts.")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, encoding='latin1')
        st.subheader("Preview")
        st.dataframe(df.head())
        st.subheader("Describe")
        st.dataframe(df.describe(include='all').transpose())
        st.subheader("Quick Chart")
        cols = df.columns.tolist()
        x = st.selectbox("X", cols)
        y = st.selectbox("Y (numeric preferred)", cols, index=min(1, len(cols)-1))
        chart_type = st.radio("Chart", ["line", "bar", "area"], horizontal=True)
        if pd.api.types.is_numeric_dtype(df[y]):
            chart_df = df[[x, y]].dropna()
            chart_df = chart_df.set_index(x)
            if chart_type == 'line':
                st.line_chart(chart_df)
            elif chart_type == 'bar':
                st.bar_chart(chart_df)
            else:
                st.area_chart(chart_df)
        else:
            st.warning("Select a numeric Y column for charts.")
    else:
        st.caption("No file yet. Toss a CSV in here.")


def parse_spoken_math(text: str) -> str:
    """Very light parser mapping words to symbols."""
    text = text.lower()
    replacements = {
        'plus': '+', 'add': '+', 'minus': '-', 'subtract': '-', 'times': '*', 'multiply': '*',
        'multiplied by': '*', 'divide': '/', 'divided by': '/', 'over': '/', 'into': '*',
        'power of': '^', 'to the power of': '^',
        'sine': 'sin', 'cosine': 'cos', 'tangent': 'tan', 'logarithm': 'log', 'log': 'log', 'square root': 'sqrt',
        'pi': '3.141592653589793',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


import io
from pydub import AudioSegment
import speech_recognition as sr
import streamlit as st

def section_voice_ai():
    st.header("🎤 Voice AI")

    # Let user upload mic/audio file
    audio = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "webm"])

    if audio:
        st.audio(audio)

        try:
            # Convert to WAV (so SpeechRecognition accepts it)
            sound = AudioSegment.from_file(audio, format=audio.type.split("/")[-1])
            wav_bytes = io.BytesIO()
            sound.export(wav_bytes, format="wav")
            wav_bytes.seek(0)

            # Recognizer
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_bytes) as source:
                data = recognizer.record(source)
                text = recognizer.recognize_google(data)
                st.success(f"✅ Transcribed: {text}")

        except Exception as e:
            st.error(f"❌ Speech recognition failed: {e}")


# Football predictor – simple logistic based on ratings/goals/home adv

def section_football():
    st.header("⚽ Football Match Predictor (Simple)")
    c1, c2 = st.columns(2)
    with c1:
        team_a = st.text_input("Home Team", "Team A")
        rating_a = st.slider("Team A Strength", 0, 100, 70)
        form_a = st.slider("Team A Recent Form (last 5 pts)", 0, 15, 10)
        goals_for_a = st.number_input("Team A Avg Goals For", 0.0, 5.0, 1.6)
        goals_against_a = st.number_input("Team A Avg Goals Against", 0.0, 5.0, 1.1)
    with c2:
        team_b = st.text_input("Away Team", "Team B")
        rating_b = st.slider("Team B Strength", 0, 100, 65)
        form_b = st.slider("Team B Recent Form (last 5 pts)", 0, 15, 8)
        goals_for_b = st.number_input("Team B Avg Goals For", 0.0, 5.0, 1.4)
        goals_against_b = st.number_input("Team B Avg Goals Against", 0.0, 5.0, 1.3)

    home_adv = st.slider("Home Advantage (goals)", 0.0, 1.0, 0.25, 0.05)

    if st.button("Predict ⚽"):
        # Expected goals proxy
        xg_a = goals_for_a * 0.7 + (100 - goals_against_b) * 0.003 + (rating_a - rating_b) * 0.004 + home_adv
        xg_b = goals_for_b * 0.7 + (100 - goals_against_a) * 0.003 + (rating_b - rating_a) * 0.004
        # Poisson-based simple outcome probability (approx)
        lam_a = max(0.1, xg_a)
        lam_b = max(0.1, xg_b)
        # Compute probabilities for scores up to 6-6
        def pois(k, lam):
            return math.exp(-lam) * lam**k / math.factorial(k)
        PA, PB, PD = 0.0, 0.0, 0.0
        for i in range(7):
            for j in range(7):
                p = pois(i, lam_a) * pois(j, lam_b)
                if i > j:
                    PA += p
                elif i < j:
                    PB += p
                else:
                    PD += p
        # Normalize (should already be ~1 but just in case)
        s = PA + PB + PD
        if s > 0:
            PA, PB, PD = PA/s, PB/s, PD/s
        st.metric(f"{team_a} win %", f"{PA*100:.1f}%")
        st.metric("Draw %", f"{PD*100:.1f}%")
        st.metric(f"{team_b} win %", f"{PB*100:.1f}%")
        st.caption("Toy model. For fun, not for betting.")


# --------- Sidebar Navigation ---------

st.sidebar.title("⚡ Mega Menu")
page = st.sidebar.radio("Go to", [
    "Home", "Mega Calculator", "Fun Zone", "Productivity", "Data Playground", "Voice / AI", "Football Predictor"
])

if page == "Home":
    section_home()
elif page == "Mega Calculator":
    section_calculator()
elif page == "Fun Zone":
    section_fun_zone()
elif page == "Productivity":
    section_productivity()
elif page == "Data Playground":
    section_data_playground()
elif page == "Voice / AI":
    section_voice_ai()
elif page == "Football Predictor":
    section_football()

st.sidebar.divider()
st.sidebar.write("Made with ❤️ in Streamlit. Optional deps: speechrecognition, streamlit-mic-recorder, nltk.")