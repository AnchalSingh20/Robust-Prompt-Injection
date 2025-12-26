"""Six text perturbations, at three levels.

character   typo_swap, space_insert, homoglyph
obfuscation leetspeak, random_case
context     benign_prefix

Every attack takes an explicit `random.Random` so a run is reproducible from
its seed. Attacks are label-preserving: they change the surface form of a
prompt, never its intent.
"""
import random

# Latin -> visually identical Cyrillic. These tokenize completely differently,
# which is why homoglyph is the held-out attack: it is the least like the rest.
HOMOGLYPHS = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "i": "і"}
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}

BENIGN_PREFIXES = [
    "Please help me with my homework on photosynthesis.",
    "I really enjoyed our conversation about travel yesterday.",
    "Thanks for the recipe, it turned out great!",
    "Can you summarise the history of the printing press?",
    "My favourite programming language is Python.",
]


def typo_swap(text, rng, rate=0.2):
    out = []
    for w in text.split(" "):
        if len(w) > 3 and rng.random() < rate:
            i = rng.randrange(1, len(w) - 2)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)


def space_insert(text, rng, rate=0.2):
    out = []
    for w in text.split(" "):
        if len(w) > 3 and rng.random() < rate:
            i = rng.randrange(1, len(w))
            w = w[:i] + " " + w[i:]
        out.append(w)
    return " ".join(out)


def homoglyph(text, rng, rate=0.3):
    return "".join(HOMOGLYPHS[c] if c in HOMOGLYPHS and rng.random() < rate else c
                   for c in text)


def leetspeak(text, rng, rate=0.5):
    return "".join(LEET[c.lower()] if c.lower() in LEET and rng.random() < rate else c
                   for c in text)


def random_case(text, rng):
    return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in text)


def benign_prefix(text, rng):
    return rng.choice(BENIGN_PREFIXES) + " " + text


ATTACKS = {
    "typo_swap": typo_swap,
    "space_insert": space_insert,
    "homoglyph": homoglyph,
    "leetspeak": leetspeak,
    "random_case": random_case,
    "benign_prefix": benign_prefix,
}

HELD_OUT = ["homoglyph"]
TRAIN_ATTACKS = {k: v for k, v in ATTACKS.items() if k not in HELD_OUT}
