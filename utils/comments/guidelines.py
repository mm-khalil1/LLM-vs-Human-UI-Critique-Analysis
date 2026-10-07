"""Parsing for the guideline references LLM critiques attach to their comments.

Models cite Nielsen heuristics, Apple HIG concepts and CrowdCrit labels as free text with
inconsistent naming, separators and nesting. `parse_guideline_references` turns one raw
reference field into a list of {source, category, subcategory, evidence, raw_reference}
dicts; `build_parsed_long_df` explodes a frame of those into one row per reference.
"""

from __future__ import annotations

import ast
import json
import re

import pandas as pd

CANONICAL_SOURCES = {"Nielsen", "Apple HIG", "CrowdCrit"}
SOURCE_RE = re.compile(
    r"(Apple Human Interface Guidelines \(HIG\)|Apple Human Interface Guidelines|Apple HIG|Nielsen Norman 10 Heuristics|Nielsen Norman Group|Nielsen Norman Heuristics?|Nielsen Norman|NN/g|Nielsen|CrowdCrit Visual Design Critiques|CrowdCrit Visual Design|CrowdCrit|CrowCrit)",
    flags=re.I,
)
SOURCE_MAP = {
    "apple human interface guidelines": "Apple HIG",
    "apple human interface guidelines (hig)": "Apple HIG",
    "apple hig": "Apple HIG",
    "nielsen norman heuristic": "Nielsen",
    "nielsen norman heuristics": "Nielsen",
    "nielsen norman 10 heuristics": "Nielsen",
    "nielsen norman group": "Nielsen",
    "nielsen norman": "Nielsen",
    "nn/g": "Nielsen",
    "nielsen": "Nielsen",
    "crowdcrit visual design critiques": "CrowdCrit",
    "crowdcrit visual design": "CrowdCrit",
    "crowdcrit": "CrowdCrit",
    "crowcrit": "CrowdCrit",
}
CATEGORY_DELIMITER_RE = re.compile(r"\s*(?:—|–|>|/|\\|:)\s*")
LEADING_CATEGORY_DELIMITER_RE = re.compile(r"^(?:—|–|-|>|/|\\|:)\s*")

NIELSEN_CATEGORIES = [
    "Visibility of System Status",
    "Match Between the System and the Real World",
    "User Control and Freedom",
    "Consistency and Standards",
    "Error Prevention",
    "Recognition Rather than Recall",
    "Flexibility and Efficiency of Use",
    "Aesthetic and Minimalist Design",
    "Help Users Recognize, Diagnose, and Recover from Errors",
    "Help and Documentation",
]
NIELSEN_CANONICAL = {
    "minimize user memory load": "Recognition Rather than Recall",
    "error recovery": "Help Users Recognize, Diagnose, and Recover from Errors",
    "match between system and the real world": "Match Between the System and the Real World",
    "match between system and real world": "Match Between the System and the Real World",
    "feedback": "Visibility of System Status",
    "Prevent errors": "Error Prevention",
    "efficiency of use": "Flexibility and Efficiency of Use",
    "visibility of system state": "Visibility of System Status",
    "aesthetics and minimalist design": "Aesthetic and Minimalist Design",
    "focus and minimalist design": "Aesthetic and Minimalist Design",
    "help users recognize and recover from errors": "Help Users Recognize, Diagnose, and Recover from Errors",
    "help users recognize and recover": "Help Users Recognize, Diagnose, and Recover from Errors",
    "Help users recognize, diagnose information": "Help Users Recognize, Diagnose, and Recover from Errors",
    "Help users recognize and understand": "Help Users Recognize, Diagnose, and Recover from Errors",
    "Help users recognize and recover from ambiguity": "Help Users Recognize, Diagnose, and Recover from Errors",
    "Help users recognize, diagnose…": "Help Users Recognize, Diagnose, and Recover from Errors",
    "Help Users Recognize, Diagnose, and Recover": "Help Users Recognize, Diagnose, and Recover from Errors",
    "Help users recognize, diagnose actions": "Help Users Recognize, Diagnose, and Recover from Errors",
    "help users recover from errors": "Help Users Recognize, Diagnose, and Recover from Errors",
    "help users prevent errors": "Error Prevention",
}

APPLE_HIG_TREE = {
    "Foundations": [
        "App icons", "Branding", "Color", "Dark Mode", "Icons", "Images", "Immersive experiences",
        "Inclusion", "Layout", "Materials", "Motion", "Privacy", "Right to left", "SF Symbols",
        "Spatial layout", "Typography", "Writing",
    ],
    "Patterns": [
        "Charting Data", "Collaboration and Sharing", "Drag and Drop", "Entering Data", "Feedback",
        "File Management", "Going Full Screen", "Launching", "Live-Viewing Apps", "Loading",
        "Managing Accounts", "Managing Notifications", "Modality", "Multitasking", "Offering Help",
        "Onboarding", "Playing Audio", "Playing Haptics", "Playing Video", "Printing",
        "Ratings and Reviews", "Searching", "Settings", "Undo and Redo", "Workouts",
    ],
    "Components": {
        "Content": ["Charts", "Image Views", "Text Views", "Web Views"],
        "Layout and Organization": ["Boxes", "Collections", "Column Views", "Disclosure Controls", "Labels", "Lists and Tables", "Lockups", "Outline Views", "Split Views", "Tab Views"],
        "Menus and Actions": ["Activity Views", "Buttons", "Context Menus", "Dock Menus", "Edit Menus", "Home Screen Quick Actions", "Menus", "Going Full Screen", "Ornaments", "Pop-up Buttons", "Pull-down Buttons", "The Menu Bar", "Toolbars"],
        "Navigation and Search": ["Path Controls", "Search Fields", "Sidebars", "Tab Bars", "Token Fields"],
        "Presentation": ["Action Sheets", "Alerts", "Page Controls", "Panels", "Popovers", "Scroll Views", "Sheets", "Windows"],
        "Selection and Input": ["Color Wells", "Combo Boxes", "Digit Entry Views", "Image Wells", "Pickers", "Segmented Controls", "Sliders", "Steppers", "Text Fields", "Toggles", "Virtual Keyboards"],
        "Status": ["Activity Rings", "Gauges", "Progress Indicators", "Rating Indicators"],
        "System Experiences": ["App Shortcuts", "Complications", "Controls", "Live Activities", "Notifications", "Snippets", "Status Bars", "Top Shelf", "Watch Faces", "Widgets"],
    },
    "Inputs": [
        "Action Button", "Apple Pencil and Scribble", "Camera Control", "Digital Crown", "Eyes",
        "Focus and Selection", "Game Controls", "Gestures", "Gyroscope and Accelerometer", "Keyboards",
        "Nearby Interactions", "Pointing Devices", "Remotes",
    ],
    "Technologies": [
        "AirPlay", "Always On", "App Clips", "Apple Pay", "Augmented Reality", "CareKit", "CarPlay",
        "Game Center", "Generative AI", "HealthKit", "HomeKit", "iCloud", "ID Verifier",
        "iMessage Apps and Stickers", "In-App Purchase", "Live Photos", "Mac Catalyst", "Machine Learning",
        "Maps", "NFC", "Photo Editing", "ResearchKit", "SharePlay", "ShazamKit", "Sign in with Apple",
        "Siri", "Tap to Pay on iPhone", "VoiceOver", "Wallet",
    ],
}


def clean_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().strip(" .;"))


def clean_reference_text(text):
    return clean_text(LEADING_CATEGORY_DELIMITER_RE.sub("", clean_text(text)))


def normalize_key(text):
    text = clean_text(text).lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


NIELSEN_LOOKUP = {normalize_key(name): name for name in NIELSEN_CATEGORIES}
NIELSEN_LOOKUP.update({normalize_key(k): v for k, v in NIELSEN_CANONICAL.items()})
APPLE_TOP_LOOKUP = {normalize_key(k): k for k in APPLE_HIG_TREE}
APPLE_SUBCATEGORY_PARENT = {}
for parent, children in APPLE_HIG_TREE.items():
    if parent == "Components":
        for group, items in children.items():
            APPLE_SUBCATEGORY_PARENT[normalize_key(group)] = ("Components", group)
            for item in items:
                APPLE_SUBCATEGORY_PARENT[normalize_key(item)] = ("Components", f"{group} / {item}")
                APPLE_SUBCATEGORY_PARENT[normalize_key(f"{group} {item}")] = ("Components", f"{group} / {item}")
    else:
        for item in children:
            APPLE_SUBCATEGORY_PARENT[normalize_key(item)] = (parent, item)

APPLE_SUBCATEGORY_ALIASES = {
    "color and contrast": ("Foundations", "Color"),
    "visual design": ("Foundations", "Layout"),
    "touch targets": ("Inputs", "Gestures"),
    "navigation bars": ("Components", "Navigation and Search"),
    "clarity": ("Foundations", "Writing"),
    "search": ("Patterns", "Searching"),
    "navigation": ("Components", "Navigation and Search"),
    "typography and color": ("Foundations", "Typography"),
    "table views": ("Components", "Lists and Tables"),
    "writing and terminology": ("Foundations", "Writing"),
    "lists": ("Components", "Lists and Tables"),
}
VALIDISH_APPLE_HIG_ALIASES = {
    "Iconography": ("Foundations", "Icons"),
    "Terminology": ("Foundations", "Writing"),
    "Requesting Permission": ("Foundations", "Privacy"),
    "touch target": ("Foundations", "Layout"),
    "touch targets": ("Foundations", "Layout"),
    "writing and terminology": ("Foundations", "Writing"),
    "typography and color": ("Foundations", "Typography"),
    "color and contrast": ("Foundations", "Color"),
    "Data Entry": ("Patterns", "Entering Data"),
    "search": ("Patterns", "Searching"),
    "Authentication": ("Patterns", "Managing Accounts"),
    "Sharing": ("Patterns", "Collaboration and Sharing"),
    "text input": ("Components", "Selection and Input / Text Fields"),
    "Secure Text Entry": ("Components", "Selection and Input / Text Fields"),
    "Secure Text Field": ("Components", "Selection and Input / Text Fields"),
    "Switches": ("Components", "Selection and Input / Toggles"),
    "Tables": ("Components", "Layout and Organization / Lists and Tables"),
    "table views": ("Components", "Layout and Organization / Lists and Tables"),
    "navigation bars": ("Components", "Navigation and Search"),
    "navigation": ("Components", "Navigation and Search"),
    "Lists": ("Components", "Layout and Organization / Lists and Tables"),
    "Table views": ("Components", "Layout and Organization / Lists and Tables"),
    "Page Control": ("Components", "Presentation / Page Controls"),
    "Disclosure Indicators": ("Components", "Layout and Organization / Disclosure Controls"),
}

for _alias_map in (APPLE_SUBCATEGORY_ALIASES, VALIDISH_APPLE_HIG_ALIASES):
    APPLE_SUBCATEGORY_PARENT.update({normalize_key(k): v for k, v in _alias_map.items()})


def lookup_exact_or_contains(text, lookup):
    key = normalize_key(text)
    if key in lookup:
        return lookup[key]
    matches = [(k, v) for k, v in lookup.items() if k and k in key]
    if matches:
        return max(matches, key=lambda kv: len(kv[0]))[1]
    return None


def is_recognized_reference_start(text, source):
    if source == "Nielsen":
        return lookup_exact_or_contains(text, NIELSEN_LOOKUP) is not None
    if source == "Apple HIG":
        return (
            lookup_exact_or_contains(text, APPLE_TOP_LOOKUP) is not None
            or lookup_exact_or_contains(text, APPLE_SUBCATEGORY_PARENT) is not None
        )
    return True


def split_source_chunk(chunk, source):
    parts = [clean_text(part) for part in re.split(r"[|]", clean_text(chunk)) if clean_text(part)]
    period_parts = []
    for part in parts:
        period_parts.extend(re.split(r"\.\s+(?=[A-Z])", part))

    results = []
    for part in (clean_text(part) for part in period_parts if clean_text(part)):
        semicolon_parts = [clean_text(piece) for piece in part.split(";") if clean_text(piece)]
        if not semicolon_parts:
            continue
        current = semicolon_parts[0]
        for candidate in semicolon_parts[1:]:
            if is_recognized_reference_start(candidate, source):
                results.append(current)
                current = candidate
            else:
                current = f"{current}; {candidate}"
        results.append(current)
    return results


def split_category_segments(text):
    return [clean_text(seg) for seg in CATEGORY_DELIMITER_RE.split(clean_text(text)) if clean_text(seg)]


def blank_reference(source, raw_text):
    return {"source": source, "category": None, "subcategory": None, "evidence": None, "raw_reference": raw_text}


def parse_nielsen_reference(text):
    text = clean_reference_text(text)
    numbered_match = re.match(r"^Heuristic\s+\d+\s*\(\s*([^)]*?)\s*\)\s*(.*)$", text, flags=re.I)
    if numbered_match:
        category_text, trailing_text = numbered_match.groups()
        text = clean_text(f"{category_text} — {trailing_text}" if trailing_text else category_text)
    ref = blank_reference("Nielsen", text)
    normalized_text = normalize_key(text)
    matching_keys = [key for key in NIELSEN_LOOKUP if key and key in normalized_text]
    matched_key = max(matching_keys, key=len) if matching_keys else None
    category = NIELSEN_LOOKUP.get(matched_key)
    ref["category"] = category or text
    if category and matched_key:
        matched_phrase_re = r"[^a-zA-Z0-9]+".join(re.escape(token) for token in matched_key.split())
        remainder = clean_text(re.sub(matched_phrase_re, "", clean_text(text), count=1, flags=re.I))
        remainder = clean_text(CATEGORY_DELIMITER_RE.sub(" ", remainder, count=1))
        ref["evidence"] = remainder or None
    return ref


def parse_apple_reference(text):
    text = clean_reference_text(text)
    ref = blank_reference("Apple HIG", text)
    segments = split_category_segments(text) or [clean_text(text)]
    candidates = []
    for index, segment in enumerate(segments):
        top = lookup_exact_or_contains(segment, APPLE_TOP_LOOKUP)
        sub_parent = lookup_exact_or_contains(segment, APPLE_SUBCATEGORY_PARENT)
        if sub_parent:
            parent, subcategory = sub_parent
            candidates.append({"index": index, "specificity": 2, "category": parent, "subcategory": subcategory})
        elif top:
            candidates.append({"index": index, "specificity": 1, "category": top, "subcategory": None})

    if candidates:
        selected = max(candidates, key=lambda candidate: (candidate["specificity"], candidate["index"]))
        ref["category"] = selected["category"]
        ref["subcategory"] = selected["subcategory"]
        evidence_parts = [segment for index, segment in enumerate(segments) if index != selected["index"]]
    else:
        ref["category"] = segments[0]
        evidence_parts = segments[1:]

    ref["evidence"] = "; ".join(evidence_parts) or None
    return ref


def parse_crowdcrit_reference(text):
    text = clean_reference_text(text)
    ref = blank_reference("CrowdCrit", text)
    segments = split_category_segments(text)
    ref["category"] = segments[0] if segments else clean_text(text)
    ref["evidence"] = "; ".join(segments[1:]) or None
    return ref


def parse_reference_piece(source, text):
    if source == "Nielsen":
        return parse_nielsen_reference(text)
    if source == "Apple HIG":
        return parse_apple_reference(text)
    if source == "CrowdCrit":
        return parse_crowdcrit_reference(text)
    return blank_reference(source, text)


def normalize_existing_reference(ref):
    if not isinstance(ref, dict):
        return parse_reference_piece(None, clean_text(ref))
    source = SOURCE_MAP.get(normalize_key(ref.get("source")), clean_text(ref.get("source")) or None)
    raw_text = clean_text(ref.get("category"))
    parsed = parse_reference_piece(source, raw_text) if source in CANONICAL_SOURCES else blank_reference(source, raw_text)
    parsed["evidence"] = ref.get("evidence") or parsed.get("evidence")
    return parsed


def parse_structured_reference_string(text):
    stripped = clean_text(text)
    if not stripped.startswith(("[", "{")):
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(stripped)
            if isinstance(parsed, list):
                return [normalize_existing_reference(ref) for ref in parsed]
            if isinstance(parsed, dict):
                return [normalize_existing_reference(parsed)]
        except Exception:
            pass
    return None


def coerce_reference_sequence(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        coerced = value.tolist()
        return coerced if isinstance(coerced, list) else [coerced]
    return None


def parse_guideline_references(value):
    sequence = coerce_reference_sequence(value)
    if sequence is not None:
        return [normalize_existing_reference(ref) for ref in sequence]
    if isinstance(value, dict):
        return [normalize_existing_reference(value)]
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    structured_refs = parse_structured_reference_string(value)
    if structured_refs is not None:
        return structured_refs

    text = clean_text(value)
    matches = list(SOURCE_RE.finditer(text))
    refs = []
    for i, match in enumerate(matches):
        source = SOURCE_MAP[match.group(1).lower()]
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        for piece in split_source_chunk(text[start:end], source):
            pieces = re.split(r",\s*(?=Heuristic\s+\d+\b)", piece, flags=re.I) if source == "Nielsen" else [piece]
            refs.extend(parse_reference_piece(source, subpiece) for subpiece in pieces if clean_text(subpiece))
    return refs



def build_parsed_long_df(refs):
    return (
        refs.explode("parsed_guideline_references")
        .dropna(subset=["parsed_guideline_references"])
        .reset_index(drop=True)
        .assign(
            source=lambda d: d["parsed_guideline_references"].apply(lambda r: r.get("source")),
            category=lambda d: d["parsed_guideline_references"].apply(lambda r: r.get("category")),
            subcategory=lambda d: d["parsed_guideline_references"].apply(lambda r: r.get("subcategory")),
            evidence=lambda d: d["parsed_guideline_references"].apply(lambda r: r.get("evidence")),
            raw_reference=lambda d: d["parsed_guideline_references"].apply(lambda r: r.get("raw_reference")),
        )
    )
