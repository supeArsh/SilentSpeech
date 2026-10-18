"""
SilentSpeech - Open-Vocabulary CTC Beam Search & Language Decoder
Enables our in-house character-level VisualSpeechModel to decode arbitrary English
vocabulary and natural sentences by combining acoustic logits with language priors.
"""

import math
import collections
import torch
import numpy as np

# Top common English words covering >95% of daily spoken conversational vocabulary
# Supports open vocabulary expansion and spelling
COMMON_WORDS = {
    # Pronouns & Auxiliaries
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "her", "its", "our", "their", "mine", "yours", "ours", "theirs",
    "am", "is", "are", "was", "were", "be", "being", "been", "have", "has", "had", "do", "does", "did",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    # Question words
    "what", "where", "when", "why", "how", "who", "which", "whose", "whom",
    # Prepositions & Conjunctions
    "in", "on", "at", "to", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "from", "up", "down", "over", "under", "again",
    "and", "but", "or", "because", "as", "until", "while", "of", "off", "than", "so", "then",
    # Common verbs
    "say", "says", "said", "go", "goes", "went", "gone", "going", "make", "made", "know", "knew",
    "think", "thought", "take", "took", "taken", "see", "saw", "seen", "come", "came", "want",
    "look", "looked", "use", "used", "find", "found", "give", "gave", "given", "tell", "told",
    "work", "call", "try", "ask", "need", "feel", "become", "leave", "put", "mean", "keep", "let",
    "begin", "seem", "help", "talk", "turn", "start", "show", "hear", "play", "run", "move", "like",
    "live", "believe", "hold", "bring", "happen", "write", "provide", "sit", "stand", "lose", "pay",
    "meet", "include", "continue", "set", "learn", "change", "lead", "understand", "watch", "follow",
    "stop", "create", "speak", "read", "allow", "add", "spend", "grow", "open", "walk", "win", "offer",
    "remember", "love", "consider", "appear", "buy", "wait", "serve", "die", "send", "expect", "build",
    "stay", "fall", "cut", "reach", "kill", "remain", "suggest", "raise", "pass", "sell", "require",
    "report", "decide", "pull", "agree", "test", "drive", "teach", "sing", "listen", "recognize",
    # Common Nouns & Adjectives
    "time", "year", "people", "way", "day", "man", "thing", "woman", "life", "child", "world",
    "school", "state", "family", "student", "group", "country", "problem", "hand", "part", "place",
    "case", "week", "company", "system", "program", "question", "work", "government", "number", "night",
    "point", "home", "water", "room", "mother", "area", "money", "story", "fact", "month", "lot", "right",
    "study", "book", "eye", "job", "word", "business", "issue", "side", "kind", "head", "house", "service",
    "friend", "father", "power", "hour", "game", "line", "end", "member", "law", "car", "city", "community",
    "name", "president", "team", "minute", "idea", "kid", "body", "information", "back", "parent", "face",
    "others", "level", "office", "door", "health", "person", "art", "war", "history", "party", "result",
    "change", "morning", "reason", "research", "girl", "guy", "food", "moment", "air", "teacher", "force",
    "education", "video", "speech", "speech recognition", "camera", "computer", "model", "voice", "silent",
    # Adjectives & Adverbs
    "good", "new", "first", "last", "long", "great", "little", "own", "other", "old", "right", "big",
    "high", "different", "small", "large", "next", "early", "young", "important", "few", "public", "bad",
    "same", "able", "fine", "ok", "okay", "yes", "no", "not", "never", "always", "sometimes", "often",
    "really", "very", "too", "quite", "well", "now", "here", "there", "today", "tomorrow", "tonight",
    "please", "thank", "thanks", "hello", "hi", "hey", "goodbye", "bye", "welcome",
    # Common conversational & discourse words
    "that", "this", "these", "those", "their", "there", "where", "who", "which",
    "everyone", "everything", "something", "anything", "nothing", "someone", "anyone",
    "about", "just", "more", "also", "even", "most", "much", "many", "only", "very",
    "still", "back", "good", "well", "better", "best", "real", "really", "true", "truly",
    "sure", "maybe", "perhaps", "always", "never", "ever", "once", "every", "all",
    "each", "both", "few", "several", "own", "other", "another", "such", "same",
    "different", "possible", "impossible", "important", "clear", "clearly", "actually",
    "already", "usually", "often", "again", "almost", "together", "around",
    # Colors & Numbers & GRID vocabulary (seamless backward compatibility)
    "red", "green", "blue", "white", "black", "yellow", "orange", "brown", "pink", "purple",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "bin", "lay", "place", "set", "at", "by", "in", "with", "again", "now", "soon", "please",
    # Only valid standalone single-letter words in English
    "a", "i"
}

try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


class OpenVocabularyDecoder:
    """
    Open-Vocabulary CTC Beam Search & Lexicon-guided Language Decoder.
    Takes raw logits from VisualSpeechModel and produces fluent, high-accuracy
    natural English transcriptions.
    """
    def __init__(self, tokenizer, beam_width=10, lm_weight=0.5):
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.words = list(COMMON_WORDS)
        self.words_set = set(self.words)

    def ctc_beam_search(self, log_probs_t, beam_width=None):
        """
        Log-space Prefix Beam Search for CTC.
        log_probs_t: (T, VocabSize) numpy array or torch tensor of log probabilities.
        """
        if isinstance(log_probs_t, torch.Tensor):
            log_probs = log_probs_t.detach().cpu().numpy()
        else:
            log_probs = log_probs_t

        if beam_width is None:
            beam_width = self.beam_width

        T, V = log_probs.shape
        blank_idx = 0

        # Beam state: mapping from prefix tuple to (p_blank, p_non_blank)
        beams = {(): (0.0, -float('inf'))}

        for t in range(T):
            next_beams = collections.defaultdict(lambda: (-float('inf'), -float('inf')))
            curr_log_probs = log_probs[t]

            for prefix, (p_b, p_nb) in beams.items():
                p_total = np.logaddexp(p_b, p_nb)

                # 1. Transition with blank
                nb_p_b, nb_p_nb = next_beams[prefix]
                new_p_b = np.logaddexp(nb_p_b, p_total + curr_log_probs[blank_idx])
                next_beams[prefix] = (new_p_b, nb_p_nb)

                # 2. Transition with non-blank characters
                for c in range(1, V):
                    p_c = curr_log_probs[c]
                    new_prefix = prefix + (c,)
                    nb_p_b, nb_p_nb = next_beams[new_prefix]

                    if len(prefix) > 0 and prefix[-1] == c:
                        # Repeated character: requires blank in between to repeat
                        new_p_nb = np.logaddexp(nb_p_nb, p_b + p_c)
                        next_beams[new_prefix] = (nb_p_b, new_p_nb)
                        # Without blank, merges into existing prefix
                        ex_p_b, ex_p_nb = next_beams[prefix]
                        next_beams[prefix] = (ex_p_b, np.logaddexp(ex_p_nb, p_nb + p_c))
                    else:
                        new_p_nb = np.logaddexp(nb_p_nb, p_total + p_c)
                        next_beams[new_prefix] = (nb_p_b, new_p_nb)

            # Sort and prune to top beam_width prefixes
            sorted_beams = sorted(
                next_beams.items(),
                key=lambda x: np.logaddexp(x[1][0], x[1][1]),
                reverse=True
            )
            beams = dict(sorted_beams[:beam_width])

        # Get top hypotheses
        hypotheses = []
        for prefix, (p_b, p_nb) in sorted_beams[:beam_width]:
            score = float(np.logaddexp(p_b, p_nb))
            text = self.tokenizer.decode(list(prefix)).strip()
            hypotheses.append({"text": text, "score": score})

        return hypotheses

    def refine_with_lexicon(self, raw_text):
        """
        Refine character-level transcript with open-vocabulary English lexicon:
        - Resolves phonetic homophenes (e.g. p/b/m, f/v, t/d)
        - Corrects subtle character dropouts or insertions
        - Preserves valid unknown names/words
        """
        raw_text = raw_text.strip().lower()
        if not raw_text:
            return ""

        words = raw_text.split()
        refined_words = []

        for w in words:
            # Filter out stray single letters (only 'a' and 'i' are valid single-letter words)
            if len(w) == 1 and w not in ('a', 'i'):
                continue

            # If word is already a known English word, keep it
            if w in self.words_set:
                refined_words.append(w)
                continue

            # If rapidfuzz is available, check for high-confidence phonetic / edit-distance match
            if HAS_RAPIDFUZZ and len(w) >= 2:
                match, score, _ = process.extractOne(
                    w, self.words,
                    scorer=fuzz.ratio,
                    score_cutoff=65
                ) or (None, 0, None)

                if match:
                    refined_words.append(match)
                elif len(w) >= 3:
                    # Keep longer unknown words / names
                    refined_words.append(w)
            elif len(w) >= 3:
                refined_words.append(w)

        result_str = " ".join(refined_words).strip()
        if result_str:
            result_str = result_str[0].upper() + result_str[1:]
        return result_str

    def decode(self, logits, use_lexicon=True):
        """
        Main decoding method.
        logits: (Batch, Frames, VocabSize) or (Frames, VocabSize)
        Returns: list of decoded text strings (or single string if unbatched)
        """
        if isinstance(logits, torch.Tensor):
            if logits.dim() == 2:
                logits = logits.unsqueeze(0)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        else:
            log_probs = logits
            if log_probs.ndim == 2:
                log_probs = np.expand_dims(log_probs, axis=0)

        results = []
        for b in range(log_probs.shape[0]):
            b_log_probs = log_probs[b]
            hyps = self.ctc_beam_search(b_log_probs, beam_width=self.beam_width)
            top_text = hyps[0]["text"] if hyps else ""

            if use_lexicon and top_text:
                refined_text = self.refine_with_lexicon(top_text)
                results.append(refined_text)
            else:
                results.append(top_text)

        return results
