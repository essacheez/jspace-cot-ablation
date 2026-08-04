"""Read a final answer out of a model completion, one function per dataset."""

import re

BOXED = "\\boxed{"
MARK = r"####\s*([^\n]+)"
PROSE_ANSWER = r"answer\s*(?:is|:|=)\s*\**\s*([^\n]+?)\s*\**\s*\.?\s*$"
SPAN = r"\$([^$]+)\$"

JUNK = (
    "\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "$", "**",
    "^\\circ", "^{\\circ}", "\\%", "%",
)


def find_boxed(text):
    """Contents of the last \\boxed{...}, matching braces so nested ones survive."""
    start = text.rfind(BOXED)
    if start < 0:
        return None

    depth = 0
    for index in range(start + len(BOXED) - 1, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start + len(BOXED) : index]
    return None


def candidates(text):
    """Answer candidates in the order they should be trusted, most reliable first."""
    found = [find_boxed(text)]
    for pattern in (MARK, PROSE_ANSWER, SPAN):
        hits = re.findall(pattern, text, re.I | re.M)
        if hits:
            found.append(hits[-1])
    found.append(text)
    return [item for item in found if item is not None]


def clean_math(answer):
    """Normalize one candidate, or None if it does not look like a final answer."""
    for _ in range(4):
        answer = re.sub(
            r"\\(?:text|textbf|textit|textrm|mathrm|mathbf|mbox)\s*\{([^{}]*)\}",
            r"\1",
            answer,
        )
    for junk in JUNK:
        answer = answer.replace(junk, "")

    answer = answer.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    answer = answer.replace("√", "\\sqrt").replace("π", "\\pi")
    answer = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", answer)
    answer = re.sub(r"\\sqrt(\d+)", r"\\sqrt{\1}", answer)
    answer = re.sub(r"^\s*[a-zA-Z]\s*=\s*", "", answer.strip())
    answer = re.sub(r"\s+", "", answer).rstrip(".")

    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", answer):
        answer = answer.replace(",", "")
    if re.fullmatch(r"-?\d+\.\d+", answer):
        answer = answer.rstrip("0").rstrip(".")
    if len(answer) > 50 or not re.search(r"[0-9a-zA-Z]", answer):
        return None
    return answer


def extract_gsm8k(text):
    """The number after ####, or the last number in the completion."""
    if not text:
        return None

    marks = re.findall(MARK, text)
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", marks[-1] if marks else text)
    if not numbers:
        return None

    value = (numbers[0] if marks else numbers[-1]).replace(",", "")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value or None


def extract_math500(text):
    """The first candidate that normalizes to something answer-shaped."""
    if not text:
        return None

    for candidate in candidates(text):
        answer = clean_math(candidate)
        if answer is not None:
            return answer
    return None


def extract_aime2024(text):
    """The last integer in the best candidate, since every AIME answer is an integer."""
    if not text:
        return None

    numbers = re.findall(r"\d+", candidates(text)[0]) or re.findall(r"\d+", text)
    if not numbers:
        return None
    return str(int(numbers[-1]))
