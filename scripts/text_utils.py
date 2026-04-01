"""
Text preprocessing utilities for radiology report analysis.

Functions
---------
is_filler_token(token) -> bool
remove_filler_words(text, tokens, log_probs, entropies) -> dict
"""

import re
import string
from typing import Optional


# ---------------------------------------------------------------------------
# Filler word sets
# ---------------------------------------------------------------------------

ARTICLES = {"a", "an", "the"}

PREPOSITIONS = {
    "about", "above", "across", "after", "against", "along", "among", "around",
    "at", "before", "behind", "below", "beneath", "beside", "between", "beyond",
    "by", "down", "during", "except", "for", "from", "in", "inside", "into",
    "like", "near", "of", "off", "on", "onto", "out", "outside", "over",
    "past", "since", "through", "throughout", "to", "toward", "towards",
    "under", "until", "up", "upon", "with", "within", "without",
}

CONJUNCTIONS = {
    "and", "but", "or", "nor", "for", "yet", "so",
    "although", "because", "before", "if", "since",
    "though", "unless", "until", "when", "where",
    "whether", "while", "after", "as", "once", "that",
    "although", "even", "either", "neither", "both",
    "however", "therefore", "thus", "hence", "moreover",
    "furthermore", "additionally", "also", "then",
}

FILLER_WORDS = ARTICLES | PREPOSITIONS | CONJUNCTIONS


# ---------------------------------------------------------------------------
# Token-level filler filtering
# ---------------------------------------------------------------------------

def is_filler_token(token: str) -> bool:
    """Return True if the token (after stripping) is a filler word or punctuation."""
    clean = token.strip().lower().strip(string.punctuation)
    return not clean or clean in FILLER_WORDS or all(c in string.punctuation + " " for c in token)


def remove_filler_words(
    text: str,
    tokens: Optional[list] = None,
    log_probs: Optional[list] = None,
    entropies: Optional[list] = None,
) -> dict:
    """
    Remove punctuation, articles, prepositions, and conjunctions from report text
    and (optionally) from parallel token/log_prob/entropy lists.

    Parameters
    ----------
    text : str
        Raw report text.
    tokens : list[str], optional
        Token strings parallel to log_probs/entropies.
    log_probs : list[float], optional
        Per-token log probabilities.
    entropies : list[float], optional
        Per-token entropies.

    Returns
    -------
    dict with keys:
        'text'      : filtered text string
        'tokens'    : filtered token list (or None)
        'log_probs' : filtered log probs (or None)
        'entropies' : filtered entropies (or None)
        'mask'      : bool list indicating kept positions (only if tokens given)
    """
    words = text.split()
    filtered_words = [
        w for w in words
        if w.strip().lower().strip(string.punctuation) not in FILLER_WORDS
        and not all(c in string.punctuation for c in w.strip())
    ]
    filtered_text = " ".join(filtered_words)

    result = {"text": filtered_text, "tokens": None, "log_probs": None, "entropies": None, "mask": None}

    if tokens is not None:
        mask = [not is_filler_token(t) for t in tokens]
        result["mask"] = mask
        result["tokens"] = [t for t, m in zip(tokens, mask) if m]
        if log_probs is not None:
            result["log_probs"] = [lp for lp, m in zip(log_probs, mask) if m]
        if entropies is not None:
            result["entropies"] = [e for e, m in zip(entropies, mask) if m]

    return result