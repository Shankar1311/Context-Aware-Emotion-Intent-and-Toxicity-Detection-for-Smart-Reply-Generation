"""
=============================================================================
Context-Aware Emotion, Intent and Toxicity Detection
for Smart Reply Generation using NLP
=============================================================================
Author     : [Your Name]
Framework  : Streamlit
NLP Libs   : TextBlob, scikit-learn, re, collections, pandas, json
Description: A modular NLP pipeline that detects emotion, classifies intent
             using TF-IDF + Logistic Regression, performs toxicity analysis
             with weighted keyword scoring, and generates smart replies.
             Enhanced with external datasets from Kaggle:
             - Intent Recognition for Chatbots (intent.json)
             - Emotions Dataset for NLP (train.txt, test.txt, val.txt)
=============================================================================
"""

import re
import streamlit as st
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Smart Reply Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  – clean light theme, fully readable
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #1a1a2e;
}

/* ── Main background ── */
.stApp { background: #f4f6fb; color: #1a1a2e; }

/* ── Section cards ── */
.nlp-card {
    background: #ffffff;
    border: 1px solid #dde3ef;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    color: #1a1a2e;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* ── Score pill ── */
.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.03em;
    margin-left: 8px;
}
.pill-green  { background:#d4f5e2; color:#1a7a3c; border:1px solid #7dd9a3; }
.pill-yellow { background:#fff3cd; color:#8a5c00; border:1px solid #ffc107; }
.pill-red    { background:#fde8e8; color:#b91c1c; border:1px solid #f87171; }
.pill-blue   { background:#dbeafe; color:#1d4ed8; border:1px solid #93c5fd; }
.pill-purple { background:#ede9fe; color:#5b21b6; border:1px solid #a78bfa; }
.pill-gray   { background:#f1f5f9; color:#475569; border:1px solid #cbd5e1; }

/* ── Highlighted keyword ── */
mark {
    background: #fef9c3;
    color: #78350f;
    border-radius: 3px;
    padding: 1px 4px;
    font-weight: 600;
}
mark.toxic-kw {
    background: #fde8e8;
    color: #b91c1c;
}

/* ── Smart-reply bubble ── */
.reply-bubble {
    background: #eef4ff;
    border: 1px solid #93c5fd;
    border-radius: 0 14px 14px 14px;
    padding: 0.9rem 1.2rem;
    font-size: 1.05rem;
    line-height: 1.6;
    color: #1e3a5f;
    margin-top: 0.5rem;
}

/* ── Confidence bar ── */
.conf-bar-wrap { background:#e2e8f0; border-radius:6px; height:8px; width:100%; }
.conf-bar-fill { background: #2563eb; border-radius:6px; height:8px; }

/* ── Section headers ── */
h3 { font-family: 'Space Mono', monospace !important; color: #1d4ed8 !important;
     font-size:0.88rem !important; letter-spacing:0.06em; text-transform:uppercase; }
h1 { font-family: 'Space Mono', monospace; color: #1a1a2e; }
h2 { color: #1a1a2e; }

/* ── Card inner text colours ── */
.nlp-card span, .nlp-card div { color: #1a1a2e; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #dde3ef;
}
section[data-testid="stSidebar"] * { color: #1a1a2e !important; }

/* ── TextArea ── */
textarea {
    background: #ffffff !important;
    color: #1a1a2e !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
    transition: background 0.2s;
}
.stButton > button:hover { background: #1d4ed8; }

/* ── Expander ── */
.streamlit-expanderHeader { color: #1a1a2e !important; font-weight: 600; }

/* ── Divider ── */
hr { border-color: #dde3ef; }

/* ── JSON viewer readability ── */
.stJson { background: #f8fafc !important; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 1 – DATASET LOADING & TRAINING DATA
# =============================================================================

# Original training data (kept as fallback)
ORIGINAL_TRAINING_DATA = [
    # ── greeting ──────────────────────────────────────────────────────────
    ("hello there", "greeting"),
    ("hi how are you", "greeting"),
    ("hey good morning", "greeting"),
    ("good evening", "greeting"),
    ("howdy", "greeting"),
    ("what's up", "greeting"),
    ("greetings", "greeting"),
    ("nice to meet you", "greeting"),
    ("hey", "greeting"),
    ("hello", "greeting"),

    # ── question ──────────────────────────────────────────────────────────
    ("what is natural language processing", "question"),
    ("how does this work", "question"),
    ("can you explain sentiment analysis", "question"),
    ("why is my account not working", "question"),
    ("when will the feature be added", "question"),
    ("where can I find the documentation", "question"),
    ("who created this system", "question"),
    ("what time is it", "question"),
    ("how do I reset my password", "question"),
    ("could you tell me more about this", "question"),
    ("is this feature available", "question"),
    ("what are the benefits", "question"),

    # ── complaint ─────────────────────────────────────────────────────────
    ("this is not working properly", "complaint"),
    ("I am very frustrated with the service", "complaint"),
    ("the app keeps crashing", "complaint"),
    ("I've been waiting too long", "complaint"),
    ("terrible experience overall", "complaint"),
    ("nothing is functioning correctly", "complaint"),
    ("this is broken and useless", "complaint"),
    ("I can't login at all", "complaint"),
    ("the feature doesn't work", "complaint"),
    ("very disappointed with this product", "complaint"),
    ("this is so slow and buggy", "complaint"),
    ("error keeps appearing every time", "complaint"),

    # ── request ───────────────────────────────────────────────────────────
    ("please help me with this", "request"),
    ("I need assistance urgently", "request"),
    ("can you add this feature please", "request"),
    ("could you fix this issue", "request"),
    ("I would like to request support", "request"),
    ("please send me the details", "request"),
    ("I need you to update my account", "request"),
    ("kindly resolve this problem", "request"),
    ("please look into this matter", "request"),
    ("I want help with my subscription", "request"),
    ("can you walk me through it", "request"),

    # ── general ───────────────────────────────────────────────────────────
    ("I just wanted to share this", "general"),
    ("this is interesting", "general"),
    ("okay sounds good", "general"),
    ("alright thank you", "general"),
    ("noted", "general"),
    ("I understand", "general"),
    ("that makes sense", "general"),
    ("cool stuff", "general"),
    ("just checking in", "general"),
    ("this looks fine to me", "general"),
    ("nothing specific just browsing", "general"),
    
    # ── statement / announcement ─────────────────────────────────────────
    ("i am going to study", "statement"),
    ("i am going to work", "statement"),
    ("i am leaving now", "statement"),
    ("i am heading out", "statement"),
    ("i'm leaving now", "statement"),
    ("i'm heading out", "statement"),
    ("i'm going to study", "statement"),
    ("i'm going to work", "statement"),
    ("i will start soon", "statement"),
    ("i will begin shortly", "statement"),
    ("i'm about to start", "statement"),
    ("i'm starting now", "statement"),
    ("i'm taking a break", "statement"),
    ("i'm done for today", "statement"),
    ("i finished my work", "statement"),
    ("i completed it", "statement"),
    ("i have to go now", "statement"),
    ("i'm busy right now", "statement"),
    ("i'm working on it", "statement"),
    ("i'm eating lunch", "statement"),
    ("going to sleep now", "statement"),
    ("i'm going to the store", "statement"),
    ("i am starting my homework", "statement"),
    ("i will do it later", "statement"),
    ("i'm about to leave", "statement"),
    ("i'm getting ready", "statement"),
    ("i'm preparing for my exam", "statement"),
    ("i'm studying right now", "statement"),
    ("i'm at work", "statement"),
    ("i'm in a meeting", "statement"),
    ("i'm on my way", "statement"),
    ("i'll be there soon", "statement"),
    ("i'll call you later", "statement"),
    ("i'll get back to you", "statement"),
]

def load_intent_json_dataset(filepath="intent.json"):
    """Load intent dataset from intent.json"""
    samples = []
    
    if not os.path.exists(filepath):
        return samples
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "intents" in data:
            intents_list = data["intents"]
        else:
            return samples
        
        intent_mapping = {
            "greeting": "greeting", "greetingresponse": "greeting", "courtesygreeting": "greeting",
            "courtesygreetingresponse": "greeting",
            "namequery": "question", "realnamequery": "question", "timequery": "question",
            "understandquery": "question", "currenthumanquery": "question", "whoami": "question",
            "swearing": "complaint", "shutup": "complaint",
            "podbaydoor": "request", "podbaydoorresponse": "request",
            "statement": "statement", "announcement": "statement", "intention": "statement",
            "thanks": "general", "goodbye": "general", "courtesygoodbye": "general",
            "jokes": "general", "gossip": "general", "clever": "general", "selfaware": "general",
            "nottalking2u": "general",
        }
        
        for intent_obj in intents_list:
            intent_name = intent_obj.get("intent", "").lower()
            patterns = intent_obj.get("text", [])
            
            if intent_name and patterns:
                mapped_intent = intent_mapping.get(intent_name, "general")
                
                if "greeting" in intent_name:
                    mapped_intent = "greeting"
                elif "query" in intent_name:
                    mapped_intent = "question"
                elif "swearing" in intent_name or "shutup" in intent_name:
                    mapped_intent = "complaint"
                elif "podbay" in intent_name:
                    mapped_intent = "request"
                
                for pattern in patterns:
                    if pattern and len(pattern) > 2:
                        samples.append((pattern, mapped_intent))
        
        return samples
        
    except Exception as e:
        return samples

def load_emotions_dataset(train_file="train.txt", test_file="test.txt", val_file="val.txt"):
    """Load emotions dataset from text files"""
    samples = []
    
    emotion_to_intent = {
        "joy": "general", "love": "general", "surprise": "general", "excitement": "general",
        "admiration": "general", "approval": "general", "optimism": "general",
        "anger": "complaint", "sadness": "complaint", "fear": "complaint", 
        "disgust": "complaint", "annoyance": "complaint", "disapproval": "complaint",
        "disappointment": "complaint", "frustration": "complaint",
        "curiosity": "question", "confusion": "question",
        "desire": "request", "need": "request", "caring": "request",
        "anticipation": "statement", "intention": "statement", "announcement": "statement",
    }
    
    for filepath in [train_file, test_file, val_file]:
        if not os.path.exists(filepath):
            continue
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if ';' in line:
                    text, emotion = line.split(';', 1)
                elif ',' in line:
                    text, emotion = line.split(',', 1)
                elif '\t' in line:
                    text, emotion = line.split('\t', 1)
                else:
                    continue
                
                text = text.strip().lower()
                emotion = emotion.strip().lower()
                mapped_intent = emotion_to_intent.get(emotion, "general")
                
                if len(text) > 3:
                    samples.append((text, mapped_intent))
                
        except Exception as e:
            continue
    
    return samples

def combine_datasets():
    """Combine all datasets"""
    combined = ORIGINAL_TRAINING_DATA.copy()
    
    intent_samples = load_intent_json_dataset("intent.json")
    combined.extend(intent_samples)
    
    emotion_samples = load_emotions_dataset("train.txt", "test.txt", "val.txt")
    combined.extend(emotion_samples)
    
    # Remove duplicates
    unique_samples = []
    seen = set()
    for text, intent in combined:
        if text not in seen:
            seen.add(text)
            unique_samples.append((text, intent))
    
    return unique_samples

# Load datasets
TRAINING_DATA = combine_datasets()


# =============================================================================
# SECTION 2 – TOXICITY KEYWORD BANK (weighted)
# =============================================================================
TOXIC_KEYWORDS = {
    "fuck": 0.95, "fucker": 0.95, "fucking": 0.95,
    "motherfucker": 0.95, "mother fucker": 0.95,
    "bitch": 0.92, "bastard": 0.90, "asshole": 0.92, "ass hole": 0.92,
    "shit": 0.88, "bullshit": 0.90, "bull shit": 0.90,
    "piss off": 0.85, "pissed": 0.70,
    "cunt": 0.95, "dick": 0.85, "cock": 0.80,
    "whore": 0.92, "slut": 0.92,
    "screw you": 0.85, "go to hell": 0.80,
    "idiot": 0.9, "moron": 0.9, "stupid": 0.85, "loser": 0.85,
    "hate": 0.80, "kill": 0.90, "die": 0.80, "trash": 0.75,
    "garbage": 0.75, "worthless": 0.80, "pathetic": 0.75,
    "dumb": 0.70, "shut up": 0.72, "useless": 0.75,
    "imbecile": 0.88, "retard": 0.92, "scum": 0.85, "pig": 0.70,
    "damn": 0.50, "hell": 0.45, "crap": 0.50, "sucks": 0.55,
    "awful": 0.45, "horrible": 0.50, "terrible": 0.45,
    "annoying": 0.40, "disgusting": 0.55, "ridiculous": 0.40,
    "jerk": 0.55, "freak": 0.50, "creep": 0.55,
    "bad": 0.20, "wrong": 0.15, "lame": 0.25, "boring": 0.20,
    "broken": 0.18,
}


# =============================================================================
# SECTION 3 – NLP PIPELINE FUNCTIONS
# =============================================================================

def preprocess(text: str) -> str:
    text = text.lower()
    contractions = {
        "isn't": "is not", "aren't": "are not", "wasn't": "was not",
        "weren't": "were not", "don't": "do not", "doesn't": "does not",
        "didn't": "did not", "won't": "will not", "can't": "cannot",
        "couldn't": "could not", "wouldn't": "would not", "shouldn't": "should not",
        "i'm": "i am", "i've": "i have", "i'll": "i will", "it's": "it is",
        "that's": "that is", "they're": "they are", "you're": "you are",
        "we're": "we are", "he's": "he is", "she's": "she is",
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    text = text.encode("ascii", "ignore").decode()
    return text.strip()


def detect_emotion(text: str, toxicity_score: float = 0.0) -> dict:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if toxicity_score >= 0.60 and abs(polarity) < 0.15:
        emotion, emoji = "Disgust", "🤢"
        polarity = round(-0.55 - (toxicity_score - 0.60) * 0.5, 3)
        confidence = round(0.30 + 0.70 * min(toxicity_score, 1.0), 2)
        return {
            "emotion": emotion, "emoji": emoji, "polarity": polarity,
            "subjectivity": round(subjectivity, 3), "confidence": confidence,
            "note": "Emotion inferred from toxicity",
        }

    if polarity >= 0.55:
        emotion, emoji = "Joy", "😄"
    elif polarity >= 0.25:
        emotion, emoji = "Positive", "🙂"
    elif polarity >= -0.10:
        if subjectivity > 0.55:
            emotion, emoji = "Surprise", "😮"
        else:
            emotion, emoji = "Neutral", "😐"
    elif polarity >= -0.35:
        emotion, emoji = "Concern / Sadness", "😟"
    elif polarity >= -0.60:
        emotion, emoji = "Anger", "😠"
    else:
        emotion, emoji = "Disgust", "🤢"

    confidence = round(0.30 + 0.70 * min(abs(polarity) / 0.7, 1.0), 2)

    return {
        "emotion": emotion, "emoji": emoji, "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3), "confidence": confidence,
    }


@st.cache_resource
def build_intent_classifier():
    """Build and train the intent classifier with error handling"""
    texts, labels = zip(*TRAINING_DATA)
    
    # Ensure we have enough samples
    if len(texts) < 10:
        st.error(f"Insufficient training data. Only {len(texts)} samples found.")
        return None
    
    # Check class distribution
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        st.error(f"Need at least 2 intent classes. Found: {unique_labels}")
        return None
    
    # Use only compatible parameters that work across all scikit-learn versions
    try:
        # First attempt: Standard parameters that work with most versions
        clf = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), 
                max_features=1500,
                sublinear_tf=True,
                strip_accents="unicode",
                min_df=2,
            )),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )),
        ])
        
        clf.fit(texts, labels)
        
        # Show training accuracy
        accuracy = clf.score(texts, labels)
        st.sidebar.success(f"🎯 Model trained on {len(texts)} samples, Accuracy: {accuracy:.2%}")
        
        # Show class distribution
        from collections import Counter
        class_counts = Counter(labels)
        st.sidebar.info(f"📊 Classes: {', '.join([f'{k}({v})' for k,v in class_counts.items()])}")
        
        return clf
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        
        # Fallback: Even simpler model without class_weight
        try:
            st.warning("Retrying with simplified model...")
            clf = Pipeline([
                ("tfidf", TfidfVectorizer(
                    ngram_range=(1, 2), 
                    max_features=1000,
                )),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                )),
            ])
            clf.fit(texts, labels)
            accuracy = clf.score(texts, labels)
            st.sidebar.success(f"🎯 Simplified model trained, Accuracy: {accuracy:.2%}")
            return clf
        except Exception as e2:
            st.error(f"Fallback also failed: {str(e2)}")
            
            # Final fallback: Use Naive Bayes which is very stable
            try:
                from sklearn.naive_bayes import MultinomialNB
                st.warning("Using Naive Bayes as final fallback...")
                clf = Pipeline([
                    ("tfidf", TfidfVectorizer(
                        ngram_range=(1, 2), 
                        max_features=1000,
                    )),
                    ("clf", MultinomialNB()),
                ])
                clf.fit(texts, labels)
                accuracy = clf.score(texts, labels)
                st.sidebar.success(f"🎯 Naive Bayes model trained, Accuracy: {accuracy:.2%}")
                return clf
            except Exception as e3:
                st.error(f"All training attempts failed: {str(e3)}")
                return None


def classify_intent(text: str, clf) -> dict:
    if clf is None:
        return {"intent": "general", "label": "General", "emoji": "💬", 
                "confidence": 0.0, "scores": {}}
    
    INTENT_META = {
        "greeting": ("👋", "Greeting"),
        "question": ("❓", "Question"),
        "complaint": ("😤", "Complaint"),
        "request": ("🙏", "Request"),
        "general": ("💬", "General"),
        "statement": ("📢", "Statement"),
    }
    proba = clf.predict_proba([text])[0]
    classes = clf.classes_
    top_idx = int(np.argmax(proba))
    intent = classes[top_idx]
    confidence = round(float(proba[top_idx]), 3)
    scores = {cls: round(float(p), 3) for cls, p in zip(classes, proba)}
    emoji, label = INTENT_META.get(intent, ("💬", intent.title()))
    return {"intent": intent, "label": label, "emoji": emoji,
            "confidence": confidence, "scores": scores}


def analyze_toxicity(text: str) -> dict:
    tokens = text.lower().split()
    bigrams = [" ".join(tokens[i:i+2]) for i in range(len(tokens) - 1)]
    candidates = tokens + bigrams
    found = {}
    for token in candidates:
        clean_token = re.sub(r'[^a-z\s]', '', token).strip()
        if clean_token in TOXIC_KEYWORDS:
            found[clean_token] = TOXIC_KEYWORDS[clean_token]
    if not found:
        score = 0.0
    else:
        weights = sorted(found.values(), reverse=True)
        score = weights[0]
        for w in weights[1:]:
            score += w * 0.25
        score = min(score, 1.0)
    score = round(score, 3)
    if score < 0.30:
        level, emoji = "Safe", "✅"
    elif score < 0.60:
        level, emoji = "Mild", "⚠️"
    else:
        level, emoji = "Toxic", "🚫"
    return {"score": score, "level": level, "emoji": emoji, "keywords": found}


def highlight_keywords(text: str, toxic_kws: dict) -> str:
    emphasis = {"not", "never", "no", "very", "really", "extremely", "absolutely"}
    tokens = text.split()
    highlighted = []
    for token in tokens:
        clean = re.sub(r'[^a-z]', '', token.lower())
        if clean in toxic_kws:
            highlighted.append(f"<mark class='toxic-kw'>{token}</mark>")
        elif clean in emphasis:
            highlighted.append(f"<mark>{token}</mark>")
        else:
            highlighted.append(token)
    return " ".join(highlighted)


# =============================================================================
# SECTION 4 – SMART REPLY GENERATION
# =============================================================================

REPLIES = {
    ("greeting", "positive"): [
        "Hello! 😊 Great to see you in such high spirits! How can I assist you today?",
        "Hey there! You seem to be in a wonderful mood. What can I do for you?",
    ],
    ("greeting", "neutral"): [
        "Hello! 👋 Welcome. How can I help you today?",
        "Hi there! What can I assist you with?",
    ],
    ("greeting", "negative"): [
        "Hi there. I can sense something might be bothering you. I'm here to help 🤝",
        "Hello. I'm sorry if things aren't going well. Let me know how I can assist.",
    ],
    
    ("question", "positive"): [
        "Great question! 💡 I'd be happy to help. Could you provide a bit more context?",
        "Excellent curiosity! Let me help you find the right answer.",
    ],
    ("question", "neutral"): [
        "That's a good question. Let me help you with that. Could you share more details?",
        "Happy to help clarify! Could you elaborate a little?",
    ],
    ("question", "negative"): [
        "I understand you might be frustrated. Let's work through your question together 🤝",
        "I hear you. Let me do my best to answer and ease your concern.",
    ],
    
    ("complaint", "positive"): [
        "I'm glad you're staying positive despite the issue! 🙌 Let's resolve it quickly.",
        "Thanks for the constructive tone. We'll get this sorted right away.",
    ],
    ("complaint", "neutral"): [
        "Thank you for bringing this to our attention. We'll look into it immediately 🔍",
        "I'm sorry you're experiencing this. Our team will prioritise your concern.",
    ],
    ("complaint", "negative"): [
        "I sincerely apologise for the trouble you've faced 😔 We take this very seriously.",
        "I'm truly sorry about this experience. Let me escalate this right away.",
    ],
    
    ("request", "positive"): [
        "Absolutely! 😊 I'd be happy to assist with your request. Let's get started.",
        "Sure thing! I'll take care of this for you right away.",
    ],
    ("request", "neutral"): [
        "Of course! I'll process your request promptly. Please stand by.",
        "Understood. I'll handle this as soon as possible.",
    ],
    ("request", "negative"): [
        "I understand the urgency 🙏 Let me prioritise your request immediately.",
        "I'm sorry for any inconvenience. I'll do my best to fulfil your request right away.",
    ],
    
    ("general", "positive"): [
        "Sounds wonderful! 🌟 Thanks for sharing. Is there anything else I can do for you?",
        "That's great to hear! Feel free to reach out anytime.",
    ],
    ("general", "neutral"): [
        "Got it, thanks for letting me know. Let me know if you need anything.",
        "Understood. I'm here if you need further assistance.",
    ],
    ("general", "negative"): [
        "I'm sorry to hear that things are tough. Remember, I'm here to help whenever you need 💙",
        "That sounds difficult. If there's anything I can do to help, don't hesitate to ask.",
    ],
    
    ("statement", "positive"): [
        "That sounds great! 😊 Let me know if you need any help!",
        "Awesome! I'm here if you need any assistance with that!",
        "Wonderful! Wishing you all the best! 🌟",
    ],
    ("statement", "neutral"): [
        "Okay, noted! 📝 Let me know if you need anything.",
        "Got it! I'm here if you need any help.",
        "Thanks for letting me know! 👍",
    ],
    ("statement", "negative"): [
        "I understand. Take your time! 💙 Let me know if I can help.",
        "No worries! I'm here whenever you need me.",
        "I hear you. Hope things get better soon! 🌈",
    ],
}

TOXIC_REPLY = (
    "⚠️ Your message appears to contain some harsh language. "
    "We encourage respectful communication. "
    "I'm still here to help — please feel free to rephrase, and I'll do my best to assist you."
)

def generate_reply(intent: str, emotion: str, toxicity_level: str) -> str:
    if toxicity_level == "Toxic":
        return TOXIC_REPLY
    positive_emotions = {"Joy", "Positive"}
    negative_emotions = {"Anger", "Disgust", "Concern / Sadness"}
    if emotion in positive_emotions:
        bucket = "positive"
    elif emotion in negative_emotions:
        bucket = "negative"
    else:
        bucket = "neutral"
    key = (intent, bucket)
    options = REPLIES.get(key, REPLIES.get((intent, "neutral"), ["I'm here to help. Could you share more details?"]))
    return options[0]


# =============================================================================
# SECTION 5 – HELPER UI COMPONENTS
# =============================================================================

def pill(label: str, color: str) -> str:
    return f"<span class='pill pill-{color}'>{label}</span>"

def confidence_bar_html(value: float) -> str:
    pct = int(value * 100)
    color = "#16a34a" if value >= 0.7 else "#d97706" if value >= 0.4 else "#dc2626"
    return f"""
    <div style='display:flex;align-items:center;gap:10px;margin-top:4px'>
      <div class='conf-bar-wrap' style='flex:1'>
        <div class='conf-bar-fill' style='width:{pct}%;background:{color}'></div>
      </div>
      <span style='font-family:Space Mono,monospace;font-size:0.78rem;color:{color};min-width:36px'>{pct}%</span>
    </div>
    """


# =============================================================================
# SECTION 6 – SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## NLP Smart Reply Engine")
    st.markdown("---")
    st.markdown("""
**How It Works**

1. **Preprocessing** — Contractions expanded, text normalised  
2. **Emotion Detection** — TextBlob polarity + subjectivity → 7 emotion classes  
3. **Intent Classification** — TF-IDF (1–2 grams) + Logistic Regression → 6 intent classes  
4. **Toxicity Analysis** — Weighted keyword scoring with bigram support  
5. **Smart Reply** — Context-aware reply matrix (intent × emotion)
""")
    st.markdown("---")
    st.markdown("""
**Intent Classes**
- 👋 Greeting
- ❓ Question
- 😤 Complaint
- 🙏 Request
- 💬 General
- 📢 Statement

**Emotion Classes**
- 😄 Joy &nbsp; 🙂 Positive &nbsp; 😮 Surprise
- 😐 Neutral &nbsp; 😟 Concern &nbsp; 😠 Anger &nbsp; 🤢 Disgust
""")
    st.markdown("---")
    
    # Dataset statistics
    st.markdown("### 📊 Dataset Statistics")
    st.markdown(f"**Total samples:** {len(TRAINING_DATA)}")
    st.markdown(f"**Original samples:** {len(ORIGINAL_TRAINING_DATA)}")
    
    # Intent distribution
    intent_counts = {}
    for _, intent in TRAINING_DATA:
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    st.markdown("**Intent Distribution:**")
    for intent, count in sorted(intent_counts.items()):
        percentage = (count / len(TRAINING_DATA)) * 100
        st.markdown(f"- {intent}: {count} ({percentage:.1f}%)")
    
    st.markdown("---")
    st.caption("Built with Streamlit · TextBlob · scikit-learn")
    st.caption("📊 Enhanced with Kaggle datasets")


# =============================================================================
# SECTION 7 – MAIN UI
# =============================================================================

st.markdown("# Context-Aware Emotion, Intent and Toxicity Detection for Smart Reply Generation")
st.markdown("*NLP Pipeline: Preprocessing · Sentiment · Intent Classification · Toxicity Analysis*")
st.markdown("---")

# Load the model
clf = build_intent_classifier()

if clf is None:
    st.error("Failed to load the intent classifier. Please check your training data.")
    st.stop()

# Input area
st.markdown("### Enter Your Text")
user_input = st.text_area(
    label="",
    placeholder="Type anything — a question, a complaint, a greeting, or a statement…",
    height=110,
    max_chars=500,
    label_visibility="collapsed",
)

col_btn, col_clear, _ = st.columns([1, 1, 6])
analyse_btn = col_btn.button("Analyse")
clear_btn = col_clear.button("Clear")

if clear_btn:
    st.rerun()

# Run analysis
if analyse_btn:
    raw = user_input.strip()
    if not raw:
        st.warning("Please enter some text before analysing.")
        st.stop()
    
    clean_text = preprocess(raw)
    toxic_res = analyze_toxicity(clean_text)
    emotion_res = detect_emotion(clean_text, toxicity_score=toxic_res["score"])
    intent_res = classify_intent(clean_text, clf)
    smart_reply = generate_reply(intent_res["intent"], emotion_res["emotion"], toxic_res["level"])
    highlighted = highlight_keywords(raw, toxic_res["keywords"])
    
    st.markdown("---")
    
    # Preprocessed Input
    st.markdown("### Preprocessed Input")
    st.markdown(
        f"<div class='nlp-card'>"
        f"<span style='color:#64748b;font-size:0.78rem;font-family:Space Mono,monospace'>ORIGINAL</span><br>"
        f"<span style='color:#1a1a2e'>{highlighted}</span>"
        f"<br><br>"
        f"<span style='color:#64748b;font-size:0.78rem;font-family:Space Mono,monospace'>NORMALISED</span><br>"
        f"<span style='color:#334155'>{clean_text}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    
    # Three analysis columns
    c1, c2, c3 = st.columns(3)
    
    # Emotion
    with c1:
        st.markdown("### Emotion Detection")
        pol_color = "green" if emotion_res["polarity"] >= 0 else "red"
        subj_color = "purple" if emotion_res["subjectivity"] > 0.5 else "gray"
        st.markdown(
            f"<div class='nlp-card'>"
            f"<div style='font-size:2rem'>{emotion_res['emoji']}</div>"
            f"<div style='font-size:1.1rem;font-weight:600;margin:6px 0'>{emotion_res['emotion']}</div>"
            f"Polarity {pill(str(emotion_res['polarity']), pol_color)} "
            f"Subjectivity {pill(str(emotion_res['subjectivity']), subj_color)}"
            f"<div style='margin-top:10px;font-size:0.82rem;color:#64748b'>Confidence</div>"
            f"{confidence_bar_html(emotion_res['confidence'])}"
            f"</div>",
            unsafe_allow_html=True,
        )
    
    # Intent
    with c2:
        st.markdown("### Intent Classification")
        conf_color = "green" if intent_res["confidence"] >= 0.65 else "yellow" if intent_res["confidence"] >= 0.40 else "red"
        top5 = sorted(intent_res["scores"].items(), key=lambda x: -x[1])
        bars_html = ""
        for cls, score in top5:
            pct = int(score * 100)
            bars_html += (
                f"<div style='margin:5px 0'>"
                f"<div style='display:flex;justify-content:space-between;font-size:0.76rem;color:#475569'>"
                f"<span>{cls.title()}</span><span>{pct}%</span></div>"
                f"<div class='conf-bar-wrap'><div class='conf-bar-fill' style='width:{pct}%'></div></div>"
                f"</div>"
            )
        intent_conf_pct = str(int(intent_res['confidence'] * 100)) + "%"
        st.markdown(
            f"<div class='nlp-card'>"
            f"<div style='font-size:2rem'>{intent_res['emoji']}</div>"
            f"<div style='font-size:1.1rem;font-weight:600;margin:6px 0'>{intent_res['label']}</div>"
            f"Confidence {pill(intent_conf_pct, conf_color)}"
            f"<div style='margin-top:12px'>{bars_html}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    
    # Toxicity
    with c3:
        st.markdown("### Toxicity Analysis")
        tox_color = "green" if toxic_res["level"] == "Safe" else "yellow" if toxic_res["level"] == "Mild" else "red"
        kw_html = ""
        if toxic_res["keywords"]:
            kw_html = "<div style='margin-top:10px;font-size:0.80rem;color:#64748b'>Detected keywords:</div>"
            for kw, wt in toxic_res["keywords"].items():
                bar_pct = int(wt * 100)
                kw_html += (
                    f"<div style='margin:4px 0'>"
                    f"<div style='display:flex;justify-content:space-between;font-size:0.76rem;color:#dc2626'>"
                    f"<span>{kw}</span><span>{bar_pct}%</span></div>"
                    f"<div class='conf-bar-wrap'><div class='conf-bar-fill' style='width:{bar_pct}%;background:#dc2626'></div></div>"
                    f"</div>"
                )
        else:
            kw_html = "<div style='margin-top:10px;font-size:0.82rem;color:#16a34a'>No toxic keywords found ✓</div>"
        
        st.markdown(
            f"<div class='nlp-card'>"
            f"<div style='font-size:2rem'></div>"
            f"<div style='font-size:1.1rem;font-weight:600;margin:6px 0'>{toxic_res['level']}</div>"
            f"Score {pill(str(toxic_res['score']), tox_color)}"
            f"<div style='margin-top:10px;font-size:0.82rem;color:#64748b'>Severity</div>"
            f"{confidence_bar_html(toxic_res['score'])}"
            f"{kw_html}"
            f"</div>",
            unsafe_allow_html=True,
        )
    
    # Smart Reply
    st.markdown("### Smart Reply")
    ctx_tags = (
        f"{intent_res['emoji']} {intent_res['label']} &nbsp;·&nbsp; "
        f"{emotion_res['emoji']} {emotion_res['emotion']} &nbsp;·&nbsp; "
        f"Toxicity: {toxic_res['level']}"
    )
    st.markdown(
        f"<div class='nlp-card'>"
        f"<div style='font-size:0.78rem;color:#64748b;font-family:Space Mono,monospace;margin-bottom:8px'>{ctx_tags}</div>"
        f"<div class='reply-bubble'>{smart_reply}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    
    # Expandable details
    with st.expander("Raw NLP Scores (for evaluation)"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Sentiment (TextBlob)**")
            st.json({
                "polarity": emotion_res["polarity"],
                "subjectivity": emotion_res["subjectivity"],
                "mapped_emotion": emotion_res["emotion"],
                "confidence": emotion_res["confidence"],
            })
        with col_b:
            st.markdown("**Intent Class Probabilities (LR)**")
            st.json(intent_res["scores"])
        st.markdown("**Toxicity**")
        st.json({
            "score": toxic_res["score"],
            "level": toxic_res["level"],
            "keywords": toxic_res["keywords"],
        })

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:0.78rem;color:#484f58'>"
    "Context-Aware Emotion, Intent and Toxicity Detection · Built with Streamlit, TextBlob & scikit-learn<br>"
    "📊 Enhanced with Kaggle datasets: Chatbots Intent Recognition & Emotions Dataset for NLP"
    "</div>",
    unsafe_allow_html=True,
)
