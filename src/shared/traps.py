"""Systematic-error task families for the confident-wrong search.

Where the calculus task puts its difficulty in arithmetic and so produces scatter,
these tasks put the difficulty in method selection with light arithmetic, so a wrong
but deterministic method yields the same wrong value every trial. Each problem carries
both the correct answer and the predicted naive answer (the attractor), which lets the
harness measure a trap-capture rate directly rather than inferring it from dispersion.
"""

import re

from shared.cases import NO_THINK_DIRECTIVE
from shared.harvest import CORRECT, DEGENERATE, WRONG_COMPLETE, HarvestProblem

DEFAULTS = {
    "speed_min": 10,
    "speed_max": 80,
    "max_tokens": 1024,
}

AVERAGE_SPEED_PROMPT = (
    "A car drives from town A to town B at {a} mph and returns along the same road "
    "at {b} mph.\nWhat is the car's average speed for the whole round trip, in miles "
    "per hour?\nReply with only the integer value."
)


def extract_integer(text):
    """Last signed integer in the reply, after stripping any think block and commas."""
    if not text:
        return None
    stripped = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    matches = re.findall(r"-?\d+", stripped.replace(",", ""))
    return int(matches[-1]) if matches else None


def average_speed_answers(a, b):
    """Correct (harmonic) and naive (arithmetic-mean) average speed, or None if either is non-integer."""
    if a == b:
        return None
    if (2 * a * b) % (a + b) or (a + b) % 2:
        return None
    return (2 * a * b) // (a + b), (a + b) // 2


def build_average_speed_problem(a, b, no_think, max_tokens):
    answers = average_speed_answers(a, b)
    if answers is None:
        return None
    correct, naive = answers
    content = AVERAGE_SPEED_PROMPT.format(a=a, b=b)
    if no_think:
        content += NO_THINK_DIRECTIVE
    return HarvestProblem(
        label=f"avgspeed_{a}_{b}",
        messages=[{"role": "user", "content": content}],
        answer=correct,
        key=f"average_speed:{a}:{b}",
        max_tokens=max_tokens,
        detail={"family": "average_speed", "a": a, "b": b,
                "correct": correct, "naive": naive},
    )


def classify_trap(finish_reason, text, problem):
    """Bucket a trial and return its committed integer; the naive attractor lives in detail."""
    committed = extract_integer(text)
    if finish_reason in (None, "error", "length", "loop") or committed is None:
        return DEGENERATE, None
    if committed == problem.answer:
        return CORRECT, committed
    return WRONG_COMPLETE, committed


def sample_average_speed(pool, no_think, max_tokens, speed_min, speed_max, seed):
    """Build a deduped pool of valid average-speed problems, ordered for a stable seed."""
    import random
    pairs = []
    for a in range(speed_min, speed_max + 1):
        for b in range(a + 1, speed_max + 1):
            if average_speed_answers(a, b) is not None:
                pairs.append((a, b))
    random.Random(seed).shuffle(pairs)
    problems = []
    for a, b in pairs:
        problem = build_average_speed_problem(a, b, no_think, max_tokens)
        if problem is not None:
            problems.append(problem)
        if len(problems) >= pool:
            break
    return problems


NOVEL_OPERAND_RANGE = (1, 9)
NOVEL_TERM_COUNTS = (3, 4, 5, 6)

NOVEL_OPERATOR_PROMPT = (
    "Define a new operation written #, where a # b = 2*a - b.\n"
    "Evaluate chains of # from right to left, the rightmost # first.\n"
    "Compute {expr}.\nReply with only the integer value."
)


def novel_operator_answers(operands):
    """Correct (right-fold, as instructed) and naive (left-fold, the habitual default)."""
    right = operands[-1]
    for x in reversed(operands[:-1]):
        right = 2 * x - right
    left = operands[0]
    for x in operands[1:]:
        left = 2 * left - x
    return right, left


def build_novel_operator_problem(operands, no_think, max_tokens):
    correct, naive = novel_operator_answers(operands)
    expr = " # ".join(str(x) for x in operands)
    content = NOVEL_OPERATOR_PROMPT.format(expr=expr)
    if no_think:
        content += NO_THINK_DIRECTIVE
    return HarvestProblem(
        label="novelop_" + "_".join(str(x) for x in operands),
        messages=[{"role": "user", "content": content}],
        answer=correct,
        key=f"novel_operator:{operands}",
        max_tokens=max_tokens,
        detail={"family": "novel_operator", "operands": operands, "terms": len(operands),
                "correct": correct, "naive": naive},
    )


def sample_novel_operator(pool, no_think, max_tokens, lo, hi, seed):
    """Build a deduped pool of right-to-left operator chains where the two folds disagree."""
    import random
    rng = random.Random(seed)
    problems, seen, attempts = [], set(), 0
    while len(problems) < pool and attempts < pool * 100:
        attempts += 1
        n = rng.choice(NOVEL_TERM_COUNTS)
        operands = [rng.randint(*NOVEL_OPERAND_RANGE) for _ in range(n)]
        correct, naive = novel_operator_answers(operands)
        key = f"novel_operator:{operands}"
        if correct == naive or key in seen:
            continue
        seen.add(key)
        problems.append(build_novel_operator_problem(operands, no_think, max_tokens))
    return problems


COUNT_WORDS = [
    "accommodate", "accommodation", "address", "aggressive", "apparent", "appreciate",
    "arrangement", "assassin", "assessment", "assistant", "attention", "balloon", "banana",
    "beginner", "beginning", "believe", "blueberry", "bookkeeper", "bottle", "broccoli",
    "bubble", "bulletin", "business", "butter", "butterfly", "calendar", "carry", "ceiling",
    "channel", "cheese", "cherry", "chocolate", "cinnamon", "coffee", "collar", "collection",
    "college", "committee", "commitment", "communication", "community", "connection", "cookie",
    "correct", "corridor", "cotton", "currency", "current", "dessert", "different", "dilemma",
    "dinner", "discussion", "dribble", "eleven", "embarrass", "embarrassment", "equipment",
    "essential", "excellent", "exercise", "feedback", "follow", "football", "forgotten",
    "freedom", "fulfill", "gallery", "grammar", "guarantee", "hammer", "happiness", "hello",
    "hippopotamus", "hobby", "holiday", "hurry", "immediately", "immune", "innocent",
    "intelligent", "interesting", "kettle", "kitten", "ladder", "lesson", "letter", "lettuce",
    "little", "lottery", "mammal", "mattress", "mayonnaise", "meeting", "message", "middle",
    "millennium", "mirror", "mission", "mississippi", "moccasin", "motto", "muffin", "narrow",
    "necessary", "occasion", "occurrence", "office", "official", "opportunity", "opposite",
    "paddle", "parallel", "passenger", "pattern", "pepper", "personnel", "pizza", "pollen",
    "possess", "possible", "pottery", "pressure", "professional", "professor", "puzzle",
    "rabbit", "raccoon", "raspberry", "really", "recommend", "recommendation", "ribbon",
    "riddle", "roommate", "saddle", "scissors", "settle", "shuttle", "soccer", "sorry",
    "spaghetti", "squirrel", "stubborn", "success", "sudden", "suggest", "summer", "sunny",
    "support", "suppose", "surround", "swimming", "syllable", "tattoo", "tennessee", "terror",
    "territory", "tissue", "toddler", "tomorrow", "traffic", "tunnel", "unnecessary", "vacuum",
    "village", "wedding", "weekend", "willing", "winner", "wooden", "worry", "written", "yellow",
    "zucchini",
]

LETTER_COUNT_PROMPT = (
    "How many times does the letter '{letter}' appear in the word \"{word}\"?\n"
    "Reply with only the integer count."
)


def build_letter_count_problem(word, letter, no_think, max_tokens):
    content = LETTER_COUNT_PROMPT.format(letter=letter, word=word)
    if no_think:
        content += NO_THINK_DIRECTIVE
    return HarvestProblem(
        label=f"count_{letter}_{word}",
        messages=[{"role": "user", "content": content}],
        answer=word.count(letter),
        key=f"letter_count:{word}:{letter}",
        max_tokens=max_tokens,
        detail={"family": "letter_count", "word": word, "letter": letter,
                "count": word.count(letter)},
    )


def sample_letter_count(pool, no_think, max_tokens, lo, hi, seed):
    """Build a deduped pool of letter-count problems, biased to letters that recur."""
    import random
    candidates = []
    for word in COUNT_WORDS:
        for letter in sorted(set(word)):
            if word.count(letter) >= 2:
                candidates.append((word, letter))
    random.Random(seed).shuffle(candidates)
    return [build_letter_count_problem(w, c, no_think, max_tokens)
            for w, c in candidates[:pool]]


FAMILIES = {
    "average_speed": sample_average_speed,
    "novel_operator": sample_novel_operator,
    "letter_count": sample_letter_count,
}
