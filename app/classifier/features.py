"""Feature extraction for the complexity classifier.

All features are cheap, regex/heuristic based — no model calls. The point is
a fast pre-routing signal, not a perfect understanding of the prompt.
"""
import re

FEATURE_NAMES = [
    "token_count",
    "has_analysis_keyword",
    "has_creative_keyword",
    "has_extraction_keyword",
    "has_summarization_keyword",
    "has_classification_keyword",
    "num_constraints",
    "has_context_block",
    "output_format_complexity",
    "num_questions",
    "has_multi_step",
]

_ANALYSIS_RE = re.compile(r"\b(analyz\w*|compar\w*|evaluat\w*|critiqu\w*|judg\w*|reason\w*|why|trade-?off)\b", re.I)
_CREATIVE_RE = re.compile(r"\b(write a (story|poem|song)|creative|imagine|invent|brainstorm)\b", re.I)
_EXTRACTION_RE = re.compile(r"\b(extract|find the|what is the|reformat|convert)\b", re.I)
_SUMMARY_RE = re.compile(r"\b(summar\w*|tl;?dr|condense|shorten)\b", re.I)
_CLASSIFY_RE = re.compile(r"\b(classify|categoriz\w*|label|sentiment)\b", re.I)
_CONSTRAINT_RE = re.compile(r"\b(must|should|only|at least|no more than|exactly|within)\b", re.I)
_NUMBERED_ITEM_RE = re.compile(r"(^|\n)\s*\d+[\.\)]", re.M)
_CONTEXT_RE = re.compile(r"['\"“].{15,}['\"”]|given the following|based on (the|this)", re.I)
_JSON_TABLE_RE = re.compile(r"\bjson\b|\btable\b|\bcsv\b|\bschema\b", re.I)
_LIST_RE = re.compile(r"\bbullet\w*|\blist\b|\benumerate\b", re.I)
_MULTISTEP_RE = re.compile(r"\bstep\s*\d|\bfirst\b.*\bthen\b|\bafter that\b", re.I | re.S)


def extract_features(prompt: str) -> dict[str, float]:
    token_count = len(prompt.split())
    num_constraints = len(_CONSTRAINT_RE.findall(prompt)) + len(_NUMBERED_ITEM_RE.findall(prompt))

    if _JSON_TABLE_RE.search(prompt):
        output_format_complexity = 2
    elif _LIST_RE.search(prompt):
        output_format_complexity = 1
    else:
        output_format_complexity = 0

    return {
        "token_count": token_count,
        "has_analysis_keyword": float(bool(_ANALYSIS_RE.search(prompt))),
        "has_creative_keyword": float(bool(_CREATIVE_RE.search(prompt))),
        "has_extraction_keyword": float(bool(_EXTRACTION_RE.search(prompt))),
        "has_summarization_keyword": float(bool(_SUMMARY_RE.search(prompt))),
        "has_classification_keyword": float(bool(_CLASSIFY_RE.search(prompt))),
        "num_constraints": float(num_constraints),
        "has_context_block": float(bool(_CONTEXT_RE.search(prompt))),
        "output_format_complexity": float(output_format_complexity),
        "num_questions": float(prompt.count("?")),
        "has_multi_step": float(bool(_MULTISTEP_RE.search(prompt))),
    }


def features_to_vector(features: dict[str, float]) -> list[float]:
    return [features[name] for name in FEATURE_NAMES]
