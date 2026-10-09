"""
language.py — Detect the language of a bug report so the agent can answer in it.

Retrieval stays in English: the planner writes its diagnostic questions in English (the
incident corpus and crawled docs are English), and only the analysis and final report are
written in the reporter's language.
"""

import py3langid

LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
    "it": "Italian", "nl": "Dutch", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
    "hi": "Hindi", "ar": "Arabic", "ru": "Russian", "kn": "Kannada", "ta": "Tamil",
}


def detect_language(text: str) -> str:
    """Return an ISO 639-1 code. Falls back to 'en' for empty or very short input."""
    if len(text.strip()) < 12:
        return "en"
    code, _score = py3langid.classify(text)
    return code


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)
