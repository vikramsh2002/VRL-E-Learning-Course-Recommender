import base64
from collections import Counter
from html import escape
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
import streamlit as st
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "CourseDetails.csv"
INDEX_PATH = APP_DIR / "artifacts" / "recommendation_index.joblib"
LOGO_PATH = APP_DIR / "VRLLogoTransparentSharp.png"

ALL_OPTION = "All"
MAX_FEATURES = 25000
MAX_RECOMMENDATIONS = 12
SORT_SIMILARITY = "Similarity first"
SORT_HIGH_TO_LOW = "Rating high to low"
SORT_LOW_TO_HIGH = "Rating low to high"

SMART_FILTER_STOPWORDS = {
    "a",
    "about",
    "and",
    "any",
    "are",
    "best",
    "can",
    "coursera",
    "course",
    "courses",
    "courseware",
    "find",
    "for",
    "from",
    "futurelearn",
    "give",
    "good",
    "i",
    "in",
    "learn",
    "learning",
    "me",
    "microsoft",
    "mit",
    "need",
    "on",
    "online",
    "please",
    "recommend",
    "show",
    "that",
    "the",
    "to",
    "training",
    "want",
    "with",
}

REQUIRED_COLUMNS = [
    "Course Name",
    "University",
    "Difficulty Level",
    "Rating",
    "Course URL",
    "Course Description",
    "Skills",
    "Tags",
]

CATALOG_COLUMNS = [
    *REQUIRED_COLUMNS,
    "Provider",
    "Category",
    "Course Key",
    "Last Verified",
]


st.set_page_config(
    page_title="VRL Course Recommender",
    page_icon=str(LOGO_PATH),
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --vrl-bg: #05070b;
            --vrl-bg-2: #0a111a;
            --vrl-border: #263344;
            --vrl-border-strong: #35516d;
            --vrl-muted: #a3b2c6;
            --vrl-text: #f2f6fb;
            --vrl-panel: #101821;
            --vrl-panel-2: #0c131c;
            --vrl-soft: #172332;
            --vrl-blue: #2f7de1;
            --vrl-blue-bright: #5ba9ff;
            --vrl-silver: #c7d0db;
            --vrl-platinum: #e7e1d4;
            --vrl-gold: #d7b56d;
        }

        .stApp {
            background:
                linear-gradient(135deg, rgba(47, 125, 225, 0.13) 0%, rgba(47, 125, 225, 0.03) 36%, rgba(5, 7, 11, 0) 64%),
                linear-gradient(180deg, #070b11 0%, var(--vrl-bg-2) 48%, #05070b 100%);
            color: var(--vrl-text);
        }

        .stApp,
        .stApp p,
        .stApp label,
        .stApp span,
        .stApp div {
            color: var(--vrl-text);
        }

        .main .block-container {
            max-width: 1440px;
            padding-top: 1.3rem;
            padding-bottom: 2.4rem;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(12, 19, 28, 0.99), rgba(6, 9, 14, 0.99));
            border-right: 1px solid var(--vrl-border);
        }

        [data-testid="stSidebar"] * {
            color: var(--vrl-text);
        }

        [data-testid="stSidebar"] [data-baseweb="input"],
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="popover"] {
            background: var(--vrl-panel-2);
            border-color: var(--vrl-border);
        }

        .vrl-sidebar-brand {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 0.85rem 0.4rem;
            margin: 0 0 1rem;
            border: 1px solid rgba(199, 208, 219, 0.12);
            border-radius: 8px;
            background:
                linear-gradient(180deg, rgba(242, 246, 251, 0.055), rgba(47, 125, 225, 0.045));
        }

        .vrl-sidebar-brand img {
            width: min(188px, 88%);
            height: auto;
            display: block;
        }

        [data-testid="stMetric"] {
            background: var(--vrl-panel);
            border: 1px solid var(--vrl-border);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.28);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--vrl-border);
            border-radius: 8px;
            box-shadow: 0 18px 38px rgba(0, 0, 0, 0.26);
            background:
                linear-gradient(180deg, rgba(16, 24, 33, 0.97), rgba(10, 16, 24, 0.97));
        }

        .vrl-header {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--vrl-border);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(47, 125, 225, 0.18), rgba(16, 24, 33, 0.92) 48%, rgba(8, 12, 18, 0.96));
            padding: 1.05rem 1.25rem 1.05rem 1.45rem;
            margin-bottom: 1rem;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.26);
        }

        .vrl-header::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            background: linear-gradient(180deg, var(--vrl-blue-bright), var(--vrl-gold));
        }

        .vrl-header-content {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.25rem;
        }

        .vrl-header-copy {
            min-width: 0;
        }

        .vrl-eyebrow {
            color: var(--vrl-gold);
            font-size: 0.78rem;
            font-weight: 760;
            margin: 0 0 0.28rem;
        }

        .vrl-title {
            font-size: 1.58rem;
            font-weight: 760;
            line-height: 1.14;
            letter-spacing: 0;
            margin: 0;
            color: var(--vrl-text);
        }

        .vrl-subtitle {
            color: var(--vrl-muted);
            font-size: 0.95rem;
            line-height: 1.42;
            margin: 0.32rem 0 0;
            max-width: 42rem;
        }

        .vrl-header-pills {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 0.45rem;
            min-width: 15rem;
        }

        .vrl-header-pill {
            border: 1px solid rgba(91, 169, 255, 0.3);
            border-radius: 6px;
            background: rgba(5, 7, 11, 0.34);
            color: #d8e8fb;
            font-size: 0.78rem;
            font-weight: 680;
            padding: 0.3rem 0.52rem;
            white-space: nowrap;
        }

        @media (max-width: 760px) {
            .vrl-header-content {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.85rem;
            }

            .vrl-header-pills {
                justify-content: flex-start;
                min-width: 0;
            }
        }

        .vrl-section-title {
            font-size: 1.05rem;
            font-weight: 720;
            margin: 0.25rem 0 0.6rem;
            color: var(--vrl-text);
        }

        .vrl-card-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.6rem;
            margin-bottom: 0.35rem;
        }

        .vrl-rank {
            color: var(--vrl-gold);
            font-weight: 760;
            font-size: 0.9rem;
        }

        .vrl-badge {
            border: 1px solid var(--vrl-border);
            border-radius: 6px;
            color: var(--vrl-platinum);
            background: var(--vrl-soft);
            font-size: 0.78rem;
            font-weight: 650;
            padding: 0.18rem 0.45rem;
            white-space: nowrap;
        }

        .vrl-course-title {
            font-size: 1.02rem;
            font-weight: 730;
            line-height: 1.32;
            letter-spacing: 0;
            margin: 0 0 0.35rem;
            overflow-wrap: anywhere;
            color: var(--vrl-text);
        }

        .vrl-meta {
            color: var(--vrl-muted);
            font-size: 0.86rem;
            line-height: 1.35;
            margin-bottom: 0.62rem;
        }

        .vrl-description {
            color: #cad4df;
            font-size: 0.9rem;
            line-height: 1.45;
            min-height: 3.9rem;
            margin: 0.65rem 0 0.75rem;
        }

        .vrl-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin: 0.25rem 0 0.8rem;
        }

        .vrl-chip {
            border: 1px solid rgba(91, 169, 255, 0.34);
            border-radius: 6px;
            color: #cfe7ff;
            background: rgba(47, 125, 225, 0.15);
            font-size: 0.78rem;
            font-weight: 620;
            padding: 0.18rem 0.45rem;
            max-width: 100%;
        }

        .vrl-empty {
            border: 1px dashed var(--vrl-border);
            border-radius: 8px;
            background: rgba(16, 24, 39, 0.72);
            color: var(--vrl-muted);
            padding: 1.15rem;
        }

        .vrl-small-note {
            color: var(--vrl-muted);
            font-size: 0.86rem;
            margin-top: -0.15rem;
        }

        .vrl-chat-bubble {
            border: 1px solid var(--vrl-border);
            border-radius: 8px;
            background: rgba(16, 24, 33, 0.82);
            color: var(--vrl-text);
            font-size: 0.86rem;
            line-height: 1.4;
            margin: 0.35rem 0;
            padding: 0.62rem 0.72rem;
        }

        .vrl-chat-user {
            border-color: rgba(91, 169, 255, 0.38);
            background: rgba(47, 125, 225, 0.13);
        }

        .vrl-chat-assistant {
            border-color: rgba(215, 181, 109, 0.32);
        }

        .vrl-advisor-head {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            border: 1px solid rgba(91, 169, 255, 0.28);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(47, 125, 225, 0.16), rgba(16, 24, 33, 0.92));
            padding: 0.82rem 0.95rem;
            margin-bottom: 0.72rem;
        }

        .vrl-bot-avatar {
            display: grid;
            place-items: center;
            width: 2.45rem;
            height: 2.45rem;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--vrl-blue), var(--vrl-gold));
            color: #05070b;
            font-weight: 800;
            box-shadow: 0 10px 22px rgba(47, 125, 225, 0.24);
            flex: 0 0 auto;
        }

        .vrl-advisor-name {
            margin: 0;
            font-weight: 760;
            font-size: 1.05rem;
        }

        .vrl-advisor-status {
            color: var(--vrl-muted);
            font-size: 0.84rem;
            margin: 0.15rem 0 0;
        }

        .vrl-roadmap {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.75rem 0 1rem;
        }

        .vrl-roadmap-step {
            border: 1px solid var(--vrl-border);
            border-radius: 8px;
            background: rgba(16, 24, 33, 0.82);
            padding: 0.78rem;
            min-height: 8rem;
        }

        .vrl-roadmap-label {
            color: var(--vrl-gold);
            font-size: 0.78rem;
            font-weight: 760;
            margin-bottom: 0.28rem;
        }

        .vrl-roadmap-title {
            font-size: 0.94rem;
            font-weight: 730;
            line-height: 1.28;
            margin-bottom: 0.35rem;
        }

        .vrl-match-reason {
            border-left: 3px solid var(--vrl-gold);
            background: rgba(215, 181, 109, 0.1);
            color: #efe5ce;
            border-radius: 6px;
            font-size: 0.84rem;
            line-height: 1.4;
            margin: 0.2rem 0 0.72rem;
            padding: 0.5rem 0.62rem;
        }

        @media (max-width: 920px) {
            .vrl-roadmap {
                grid-template-columns: 1fr;
            }
        }

        div[data-testid="stChatMessage"] {
            border: 1px solid var(--vrl-border);
            border-radius: 8px;
            background: rgba(16, 24, 33, 0.78);
            padding: 0.45rem 0.65rem;
            margin-bottom: 0.45rem;
        }

        div[data-testid="stChatInput"] {
            border-radius: 8px;
        }

        div[data-testid="stChatInput"] textarea {
            background: var(--vrl-panel);
            border-color: var(--vrl-border);
            color: var(--vrl-text);
        }

        h1, h2, h3, h4 {
            letter-spacing: 0;
            color: var(--vrl-text);
        }

        div[data-testid="stTabs"] button p {
            color: var(--vrl-muted);
            font-weight: 650;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] p {
            color: var(--vrl-blue-bright);
        }

        div[data-testid="stExpander"] {
            background: var(--vrl-panel);
            border-color: var(--vrl-border);
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--vrl-border);
            border-radius: 8px;
            overflow: hidden;
        }

        div[data-testid="stAlert"] {
            background: rgba(251, 191, 36, 0.12);
            border-color: rgba(251, 191, 36, 0.28);
            color: #fde68a;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] {
            background: var(--vrl-panel);
            border-color: var(--vrl-border);
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="input"] input {
            color: var(--vrl-text);
        }

        div[data-baseweb="popover"] {
            background: var(--vrl-panel);
        }

        div[data-testid="stProgress"] > div > div {
            background: rgba(91, 169, 255, 0.18);
        }

        div.stButton > button,
        div[data-testid="stLinkButton"] > a {
            border-radius: 6px;
            font-weight: 650;
            border-color: var(--vrl-border);
        }

        div.stButton > button[kind="primary"],
        div[data-testid="stLinkButton"] > a[kind="primary"] {
            background: linear-gradient(135deg, #1f5fb8, #2f7de1);
            border: 1px solid rgba(91, 169, 255, 0.5);
            color: #ffffff;
        }

        div.stButton > button:hover,
        div[data-testid="stLinkButton"] > a:hover {
            border-color: var(--vrl-blue-bright);
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def parse_skills(skills: str) -> tuple[str, ...]:
    return tuple(
        token.strip().lower()
        for token in str(skills).split()
        if token.strip()
    )


def format_skill_label(skill: str) -> str:
    return skill.replace("-", " ").replace("_", " ").title()


def normalize_phrase(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9+#.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def phrase_in_text(phrase: str, text: str) -> bool:
    if not phrase:
        return False
    return f" {phrase} " in f" {text} "


def truncate_text(text: str, max_chars: int = 230) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_chars:
        return normalized

    truncated = normalized[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{truncated}..."


def normalize_course_key(value: object) -> str:
    normalized = " ".join(str(value or "").lower().split())
    return "".join(char if char.isalnum() else "-" for char in normalized).strip("-")


def course_key_from_row(row: pd.Series) -> str:
    provider = normalize_course_key(row.get("Provider", "Coursera")) or "coursera"
    source = row.get("Course URL") or row.get("Course Name")
    return f"{provider}:{normalize_course_key(source)}"


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def logo_data_uri() -> str | None:
    if not LOGO_PATH.exists():
        return None

    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@st.cache_data(show_spinner="Loading course catalog...")
def load_courses(data_mtime: float) -> pd.DataFrame:
    courses = pd.read_csv(DATA_PATH, index_col=0)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in courses.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"CourseDetails.csv is missing required columns: {missing}")

    for column in CATALOG_COLUMNS:
        if column not in courses.columns:
            if column == "Provider":
                courses[column] = "Coursera"
            elif column == "Category":
                courses[column] = "Coursera"
            else:
                courses[column] = ""

    courses = courses[CATALOG_COLUMNS].copy()
    courses["Rating"] = pd.to_numeric(courses["Rating"], errors="coerce")
    courses["Rating"] = courses["Rating"].fillna(0.0)
    courses = courses.dropna(subset=["Course Name", "Tags", "Course URL"])

    text_columns = [
        "Course Name",
        "University",
        "Difficulty Level",
        "Course URL",
        "Course Description",
        "Skills",
        "Tags",
        "Provider",
        "Category",
        "Course Key",
        "Last Verified",
    ]
    for column in text_columns:
        courses[column] = courses[column].fillna("").astype(str).str.strip()

    courses["Provider"] = courses["Provider"].replace("", "Coursera")
    courses["Category"] = courses["Category"].replace("", "Coursera")
    courses = courses.reset_index(drop=True)
    empty_keys = courses["Course Key"] == ""
    courses.loc[empty_keys, "Course Key"] = courses[empty_keys].apply(course_key_from_row, axis=1)
    courses["Skill Tokens"] = courses["Skills"].apply(parse_skills)
    courses["Search Text"] = (
        courses["Course Name"]
        + " "
        + courses["University"]
        + " "
        + courses["Provider"]
        + " "
        + courses["Category"]
        + " "
        + courses["Difficulty Level"]
        + " "
        + courses["Course Description"]
        + " "
        + courses["Skills"]
    ).str.lower()

    return courses


@st.cache_resource(show_spinner="Loading recommendation index...")
def load_recommendation_resources(
    course_keys: tuple[str, ...],
    search_texts: tuple[str, ...],
    index_mtime: float,
):
    key_list = list(course_keys)
    if INDEX_PATH.exists():
        index_data = load(INDEX_PATH)
        if index_data.get("course_keys") == key_list:
            index_data["mode"] = "precomputed"
            index_data["key_to_position"] = {key: index for index, key in enumerate(key_list)}
            return index_data

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 2),
    )
    vectors = vectorizer.fit_transform(search_texts)
    return {
        "mode": "runtime",
        "course_keys": key_list,
        "key_to_position": {key: index for index, key in enumerate(key_list)},
        "vectorizer": vectorizer,
        "vectors": vectors,
    }


@st.cache_data(show_spinner=False)
def top_skill_options(skill_rows: tuple[tuple[str, ...], ...], limit: int = 18) -> list[str]:
    counter: Counter[str] = Counter()
    for skills in skill_rows:
        counter.update(
            skill
            for skill in skills
            if skill not in SMART_FILTER_STOPWORDS and (len(skill) > 2 or skill in {"ai", "ml", "ui", "ux"})
        )
    return [skill for skill, _ in counter.most_common(limit)]


def ensure_session_state() -> None:
    st.session_state.setdefault("shortlist", [])
    st.session_state.setdefault("recommendations", None)
    st.session_state.setdefault("recommendation_context", None)
    st.session_state.setdefault("advisor_recommendations", None)
    st.session_state.setdefault("advisor_context", None)
    st.session_state.setdefault("pending_toast", None)
    st.session_state.setdefault(
        "advisor_messages",
        [
            {
                "role": "assistant",
                "content": "Tell me your background, goal, and constraints. I will recommend courses from the catalog.",
            }
        ],
    )


def show_pending_toast() -> None:
    message = st.session_state.pop("pending_toast", None)
    if message:
        st.toast(message)


def reset_filters() -> None:
    for key in (
        "catalog_search",
        "course_provider",
        "difficulty_level",
        "university",
        "rating_sort",
        "selected_skills",
        "advisor_prompt",
        "course_name",
        "recommendations",
        "recommendation_context",
        "advisor_recommendations",
        "advisor_context",
    ):
        st.session_state.pop(key, None)


def toggle_shortlist(course_name: str) -> None:
    shortlist = list(st.session_state.get("shortlist", []))
    if course_name in shortlist:
        shortlist.remove(course_name)
        st.session_state["pending_toast"] = "Removed from shortlist"
    else:
        shortlist.append(course_name)
        st.session_state["pending_toast"] = "Saved to shortlist"
    st.session_state["shortlist"] = shortlist


def options_from(series: pd.Series) -> list[str]:
    values = sorted(value for value in series.dropna().unique() if str(value).strip())
    return [ALL_OPTION, *values]


def difficulty_options(courses: pd.DataFrame) -> list[str]:
    preferred = ["Beginner", "Intermediate", "Advanced", "Mixed_Difficulty"]
    available = set(courses["Difficulty Level"].dropna().unique())
    ordered = [level for level in preferred if level in available]
    ordered.extend(sorted(available.difference(ordered)))
    return [ALL_OPTION, *ordered]


def smart_filter_provider(request_text: str, providers: list[str]) -> str:
    aliases = {
        "Coursera": ("coursera",),
        "FutureLearn": ("future learn", "futurelearn"),
        "Kaggle Learn": ("kaggle", "kaggle learn"),
        "MIT OpenCourseWare": ("mit", "ocw", "open courseware", "opencourseware"),
        "Microsoft Learn": ("microsoft", "microsoft learn", "ms learn", "azure"),
    }
    available = set(providers)
    for provider, provider_aliases in aliases.items():
        if provider in available and any(phrase_in_text(alias, request_text) for alias in provider_aliases):
            return provider

    matches = [
        provider
        for provider in providers
        if provider != ALL_OPTION and phrase_in_text(normalize_phrase(provider), request_text)
    ]
    return max(matches, key=len) if matches else ALL_OPTION


def smart_filter_difficulty(request_text: str, difficulties: list[str]) -> str:
    difficulty_aliases = (
        ("Beginner", ("beginner", "basic", "foundation", "foundational", "intro", "introductory", "starter")),
        ("Intermediate", ("intermediate", "mid level", "practical")),
        ("Advanced", ("advanced", "expert", "deep", "senior")),
        ("Mixed_Difficulty", ("mixed difficulty", "mixed")),
    )
    available = set(difficulties)
    for difficulty, aliases in difficulty_aliases:
        if difficulty in available and any(phrase_in_text(alias, request_text) for alias in aliases):
            return difficulty
    return ALL_OPTION


def smart_filter_university(request_text: str, courses: pd.DataFrame) -> str:
    matches: list[str] = []
    for university in courses["University"].dropna().unique():
        normalized = normalize_phrase(university)
        if len(normalized) >= 4 and phrase_in_text(normalized, request_text):
            matches.append(str(university))
    return max(matches, key=len) if matches else ALL_OPTION


def smart_filter_skills(request_text: str, courses: pd.DataFrame, limit: int = 8) -> list[str]:
    skill_counter: Counter[str] = Counter()
    for skills in courses["Skill Tokens"]:
        skill_counter.update(skills)

    matches: list[str] = []
    for skill, _ in skill_counter.most_common(240):
        normalized = normalize_phrase(format_skill_label(skill))
        if normalized in SMART_FILTER_STOPWORDS:
            continue
        if len(normalized) <= 2 and normalized not in {"ai", "ml", "ui", "ux"}:
            continue
        if phrase_in_text(normalized, request_text):
            matches.append(skill)
        if len(matches) >= limit:
            break
    return matches


def smart_filter_search_terms(
    request: str,
    provider: str,
    difficulty_level: str,
    university: str,
    selected_skills: Iterable[str],
) -> str:
    tokens = re.findall(r"[a-z0-9+#.]+", request.lower())
    blocked = set(SMART_FILTER_STOPWORDS)
    for value in (provider, difficulty_level, university, *selected_skills):
        if value != ALL_OPTION:
            blocked.update(normalize_phrase(value).split())

    retained = [
        token
        for token in tokens
        if token not in blocked and len(token) > 1
    ]
    return " ".join(dict.fromkeys(retained))


def interpret_smart_filter(courses: pd.DataFrame, request: str) -> dict[str, object]:
    request_text = normalize_phrase(request)
    providers = options_from(courses["Provider"])
    difficulties = difficulty_options(courses)
    provider = smart_filter_provider(request_text, providers)
    difficulty_level = smart_filter_difficulty(request_text, difficulties)

    scoped = apply_filters(
        courses,
        "",
        provider,
        difficulty_level,
        ALL_OPTION,
        [],
    )
    university = smart_filter_university(request_text, scoped if not scoped.empty else courses)
    selected_skills = smart_filter_skills(request_text, scoped if not scoped.empty else courses)
    search_query = smart_filter_search_terms(
        request,
        provider,
        difficulty_level,
        university,
        selected_skills,
    )

    return {
        "search_query": search_query,
        "provider": provider,
        "difficulty_level": difficulty_level,
        "university": university,
        "selected_skills": selected_skills,
    }


def apply_smart_filter(courses: pd.DataFrame, request: str) -> str:
    lowered = normalize_phrase(request)
    if any(phrase_in_text(word, lowered) for word in ("clear", "reset", "start over")):
        for key in (
            "catalog_search",
            "course_provider",
            "difficulty_level",
            "university",
            "rating_sort",
            "selected_skills",
            "course_name",
            "recommendations",
            "recommendation_context",
        ):
            st.session_state.pop(key, None)
        return "Cleared the active filters."

    result = interpret_smart_filter(courses, request)
    st.session_state["catalog_search"] = result["search_query"]
    st.session_state["course_provider"] = result["provider"]
    st.session_state["difficulty_level"] = result["difficulty_level"]
    st.session_state["university"] = result["university"]
    st.session_state["selected_skills"] = list(result["selected_skills"])
    st.session_state.pop("recommendations", None)
    st.session_state.pop("recommendation_context", None)

    summary_parts: list[str] = []
    if result["provider"] != ALL_OPTION:
        summary_parts.append(f"provider {result['provider']}")
    if result["difficulty_level"] != ALL_OPTION:
        summary_parts.append(f"difficulty {result['difficulty_level']}")
    if result["university"] != ALL_OPTION:
        summary_parts.append(f"organization {result['university']}")
    if result["selected_skills"]:
        labels = ", ".join(format_skill_label(skill) for skill in result["selected_skills"])
        summary_parts.append(f"skills {labels}")
    if result["search_query"]:
        summary_parts.append(f"search {result['search_query']}")

    if not summary_parts:
        st.session_state["catalog_search"] = request.strip()
        return "I used your full request as the catalog search."
    return "Applied " + "; ".join(summary_parts) + "."


def expand_goal_text(goal: str) -> str:
    normalized = normalize_phrase(goal)
    expansions: list[str] = []
    if any(phrase_in_text(term, normalized) for term in ("data analyst", "analytics", "business analyst")):
        expansions.append("data analysis analytics sql python statistics visualization dashboard business intelligence")
    if any(phrase_in_text(term, normalized) for term in ("software engineer", "developer", "programmer")):
        expansions.append("software engineering programming algorithms data structures python java git github")
    if any(phrase_in_text(term, normalized) for term in ("cloud", "azure", "devops", "kubernetes")):
        expansions.append("cloud azure devops kubernetes infrastructure deployment containers security")
    if any(phrase_in_text(term, normalized) for term in ("web developer", "frontend", "front end", "backend", "back end", "full stack", "fullstack")):
        expansions.append("web development frontend backend full stack javascript html css react api application development")
    if any(phrase_in_text(term, normalized) for term in ("cyber", "security", "secure")):
        expansions.append("cybersecurity security network risk identity protection threat")
    if any(phrase_in_text(term, normalized) for term in ("ai", "ml", "machine learning", "deep learning", "generative", "gen ai", "genai")):
        expansions.append("artificial intelligence generative ai genai large language models llm prompt engineering azure openai machine learning deep learning neural networks python")
    if any(phrase_in_text(term, normalized) for term in ("finance", "financial", "banking")):
        expansions.append("finance financial risk accounting investment fintech")
    if any(phrase_in_text(term, normalized) for term in ("healthcare", "health", "medical")):
        expansions.append("healthcare health medical clinical public health")
    if any(phrase_in_text(term, normalized) for term in ("project manager", "product manager", "management", "leadership")):
        expansions.append("project management product management leadership agile scrum strategy communication")
    if any(phrase_in_text(term, normalized) for term in ("database", "sql", "data engineer", "data engineering")):
        expansions.append("database sql data engineering pipelines data warehouse postgresql mysql big data")
    if any(phrase_in_text(term, normalized) for term in ("career switch", "switch career", "job ready", "job-ready")):
        expansions.append("beginner professional certificate career skills hands on project portfolio")
    if any(phrase_in_text(term, normalized) for term in ("beginner", "new", "start", "foundation")):
        expansions.append("beginner introductory foundations fundamentals")
    if any(phrase_in_text(term, normalized) for term in ("advanced", "expert", "senior")):
        expansions.append("advanced expert architecture optimization")
    return " ".join([goal, *expansions]).strip()


def advisor_query_text(messages: list[dict[str, str]], request: str) -> str:
    previous_user_context = [
        str(message.get("content", ""))
        for message in messages[-6:]
        if message.get("role") == "user"
    ]
    return expand_goal_text(" ".join([*previous_user_context, request]))


def goal_terms(goal_text: str) -> list[str]:
    normalized = normalize_phrase(goal_text)
    terms = [
        token
        for token in re.findall(r"[a-z0-9+#.]+", normalized)
        if token not in SMART_FILTER_STOPWORDS and len(token) > 2
    ]
    phrase_terms = [
        "generative ai",
        "large language",
        "prompt engineering",
        "machine learning",
        "deep learning",
        "data analyst",
        "data analytics",
        "cybersecurity",
        "cloud",
        "devops",
        "azure",
        "sql",
        "python",
    ]
    for phrase in phrase_terms:
        if phrase_in_text(phrase, normalized):
            terms.insert(0, phrase)
    unique_terms = list(dict.fromkeys(terms))
    return [
        term
        for term in unique_terms
        if not any(term != other and phrase_in_text(term, other) for other in unique_terms)
    ]


def roadmap_stage(difficulty: object) -> str:
    normalized = str(difficulty or "").lower()
    if "beginner" in normalized:
        return "Foundation"
    if "advanced" in normalized:
        return "Specialize"
    return "Build"


def explain_course_match(course: pd.Series, query_text: str) -> str:
    course_text = normalize_phrase(
        " ".join(
            str(course.get(column, ""))
            for column in ("Course Name", "Course Description", "Skills", "Category", "Provider")
        )
    )
    matched_terms = [
        term
        for term in goal_terms(query_text)
        if phrase_in_text(normalize_phrase(term), course_text)
    ][:3]
    matched_skills = list(dict.fromkeys(
        format_skill_label(skill)
        for skill in course.get("Skill Tokens", ())
        if normalize_phrase(format_skill_label(skill)) in {normalize_phrase(term) for term in matched_terms}
    ))[:2]

    reason_bits: list[str] = []
    if matched_terms:
        reason_bits.append("matches " + ", ".join(term.title() for term in matched_terms))
    if matched_skills:
        reason_bits.append("skill signals " + ", ".join(matched_skills))
    reason_bits.append(f"{course.get('Difficulty Level', 'Mixed')} level")
    reason_bits.append(str(course.get("Provider", "Catalog")))
    return "; ".join(reason_bits) + "."


def add_advisor_context(recommendations: pd.DataFrame, query_text: str) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations

    enriched = recommendations.copy()
    enriched["Roadmap Stage"] = enriched["Difficulty Level"].apply(roadmap_stage)
    enriched["Match Reason"] = enriched.apply(
        lambda row: explain_course_match(row, query_text),
        axis=1,
    )
    return enriched


def recommend_for_goal(
    goal_text: str,
    candidate_courses: pd.DataFrame,
    recommendation_resources: dict,
    limit: int = MAX_RECOMMENDATIONS,
) -> pd.DataFrame:
    if candidate_courses.empty or not goal_text.strip():
        return pd.DataFrame()

    key_to_position = recommendation_resources["key_to_position"]
    candidate_pairs = [
        (key, key_to_position[key])
        for key in candidate_courses["Course Key"].astype(str)
        if key in key_to_position
    ]
    if not candidate_pairs:
        return pd.DataFrame()

    candidate_keys = [key for key, _ in candidate_pairs]
    candidate_positions = [position for _, position in candidate_pairs]
    query_vector = recommendation_resources["vectorizer"].transform([goal_text])
    similarities = cosine_similarity(
        query_vector,
        recommendation_resources["vectors"][candidate_positions],
    ).ravel()

    normalized_goal = normalize_phrase(goal_text)
    genai_intent = any(
        phrase_in_text(term, normalized_goal)
        for term in ("gen ai", "genai", "generative ai", "llm", "large language", "prompt engineering", "openai")
    )
    genai_terms = (
        "generative ai",
        "generative artificial intelligence",
        "large language",
        "llm",
        "prompt",
        "openai",
        "copilot",
        "foundation model",
        "transformer",
    )
    ai_gate_terms = (
        "artificial intelligence",
        "generative",
        "machine learning",
        "deep learning",
        "neural",
        "language model",
        "prompt",
        "openai",
        "copilot",
        "transformer",
    )
    candidate_meta = candidate_courses.set_index("Course Key")[
        ["Search Text", "Difficulty Level", "Provider"]
    ].to_dict("index")
    text_by_key = {
        key: str(meta.get("Search Text", ""))
        for key, meta in candidate_meta.items()
    }
    similarity_by_key = {
        key: min(float(score), 1.0)
        for key, score in zip(candidate_keys, similarities)
        if float(score) > 0
    }

    difficulty_intent = smart_filter_difficulty(
        normalized_goal,
        difficulty_options(candidate_courses),
    )
    provider_intent = smart_filter_provider(
        normalized_goal,
        options_from(candidate_courses["Provider"]),
    )
    if difficulty_intent != ALL_OPTION or provider_intent != ALL_OPTION:
        adjusted_scores: dict[str, float] = {}
        for key, score in similarity_by_key.items():
            meta = candidate_meta.get(key, {})
            if difficulty_intent != ALL_OPTION:
                difficulty = str(meta.get("Difficulty Level", ""))
                if difficulty == difficulty_intent:
                    score += 0.12
                elif difficulty == "Mixed_Difficulty":
                    score += 0.03
                else:
                    score *= 0.82
            if provider_intent != ALL_OPTION:
                if str(meta.get("Provider", "")) == provider_intent:
                    score += 0.09
                else:
                    score *= 0.9
            adjusted_scores[key] = min(score, 1.0)
        similarity_by_key = adjusted_scores

    if genai_intent:
        boosted_scores: dict[str, float] = {}
        for key, score in similarity_by_key.items():
            course_text = str(text_by_key.get(key, "")).lower()
            if not any(term in course_text for term in ai_gate_terms):
                continue
            if any(term in course_text for term in genai_terms):
                score += 0.18
            else:
                score += 0.035
            boosted_scores[key] = min(score, 1.0)
        similarity_by_key = boosted_scores

    if not similarity_by_key:
        return pd.DataFrame()

    recommendations = candidate_courses[
        candidate_courses["Course Key"].isin(similarity_by_key)
    ].copy()
    recommendations["Similarity"] = recommendations["Course Key"].map(similarity_by_key)
    return recommendations.sort_values(
        by=["Similarity", "Rating"],
        ascending=[False, False],
    ).head(limit)


def advisor_reply(recommendations: pd.DataFrame, scoped_count: int) -> str:
    if recommendations.empty:
        return "I could not find a strong match in the current catalog scope. Try adding a role, skill, level, or remove some manual filters."

    top = recommendations.iloc[0]
    providers = ", ".join(recommendations["Provider"].dropna().astype(str).unique()[:3])
    return (
        f"I found {len(recommendations)} recommendations from {scoped_count:,} in-scope courses. "
        f"Top match: {top['Course Name']} from {top['University']}. "
        f"Provider mix: {providers}."
    )


def apply_filters(
    courses: pd.DataFrame,
    search_query: str,
    provider: str,
    difficulty_level: str,
    university: str,
    selected_skills: Iterable[str],
) -> pd.DataFrame:
    filtered = courses
    query = search_query.strip().lower()
    skills = tuple(selected_skills or ())

    if query:
        filtered = filtered[filtered["Search Text"].str.contains(query, regex=False)]

    if provider != ALL_OPTION:
        filtered = filtered[filtered["Provider"] == provider]

    if difficulty_level != ALL_OPTION:
        filtered = filtered[filtered["Difficulty Level"] == difficulty_level]

    if university != ALL_OPTION:
        filtered = filtered[filtered["University"] == university]

    if skills:
        selected = set(skills)
        filtered = filtered[
            filtered["Skill Tokens"].apply(lambda row_skills: bool(selected.intersection(row_skills)))
        ]

    return filtered


def recommend_courses(
    course_key: str,
    candidate_courses: pd.DataFrame,
    recommendation_resources: dict,
    limit: int = MAX_RECOMMENDATIONS,
) -> pd.DataFrame:
    key_to_position = recommendation_resources["key_to_position"]
    selected_index = key_to_position.get(course_key)
    if selected_index is None:
        return pd.DataFrame()

    candidates = candidate_courses[candidate_courses["Course Key"] != course_key].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidate_keys = set(candidates["Course Key"])
    candidate_positions = [
        key_to_position[key]
        for key in candidate_keys
        if key in key_to_position and key != course_key
    ]

    if not candidate_positions:
        return pd.DataFrame()

    vectors = recommendation_resources["vectors"]
    ranked_rows: list[tuple[str, float]] = []
    if recommendation_resources.get("mode") == "precomputed":
        neighbor_indices = recommendation_resources.get("neighbor_indices")
        neighbor_similarities = recommendation_resources.get("neighbor_similarities")
        course_keys = recommendation_resources["course_keys"]
        for neighbor_index, similarity in zip(
            neighbor_indices[selected_index],
            neighbor_similarities[selected_index],
        ):
            neighbor_key = course_keys[int(neighbor_index)]
            if neighbor_key == course_key or neighbor_key not in candidate_keys:
                continue
            ranked_rows.append((neighbor_key, float(similarity)))
            if len(ranked_rows) >= limit:
                break

    if len(ranked_rows) < limit:
        similarities = cosine_similarity(
            vectors[selected_index],
            vectors[candidate_positions],
        ).ravel()
        ranked_rows = [
            (recommendation_resources["course_keys"][position], float(score))
            for position, score in zip(candidate_positions, similarities)
        ]

    similarity_by_key = dict(ranked_rows)
    candidates = candidates[candidates["Course Key"].isin(similarity_by_key)].copy()
    candidates["Similarity"] = candidates["Course Key"].map(similarity_by_key)

    return candidates.sort_values(
        by=["Similarity", "Rating"],
        ascending=[False, False],
    ).head(limit)


def sort_recommendations(recommendations: pd.DataFrame, sort_order: str) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations

    if sort_order == SORT_HIGH_TO_LOW:
        return recommendations.sort_values(
            by=["Rating", "Similarity"],
            ascending=[False, False],
        )

    if sort_order == SORT_LOW_TO_HIGH:
        return recommendations.sort_values(
            by=["Rating", "Similarity"],
            ascending=[True, False],
        )

    return recommendations.sort_values(
        by=["Similarity", "Rating"],
        ascending=[False, False],
    )


def filter_key(
    search_query: str,
    provider: str,
    difficulty_level: str,
    university: str,
    selected_skills: Iterable[str],
) -> tuple[str, str, str, str, tuple[str, ...]]:
    return (
        search_query.strip().lower(),
        provider,
        difficulty_level,
        university,
        tuple(sorted(selected_skills or ())),
    )


def render_header() -> None:
    st.markdown(
        """
        <section class="vrl-header">
            <div class="vrl-header-content">
                <div class="vrl-header-copy">
                    <p class="vrl-eyebrow">Learning Intelligence</p>
                    <p class="vrl-title">VRL Course Recommender</p>
                    <p class="vrl-subtitle">Find a stronger next course from the catalog in view.</p>
                </div>
                <div class="vrl-header-pills" aria-label="Recommendation context">
                    <span class="vrl-header-pill">Catalog</span>
                    <span class="vrl-header-pill">Skill Match</span>
                    <span class="vrl-header-pill">Ratings</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(courses: pd.DataFrame, filtered_courses: pd.DataFrame) -> None:
    total_courses = len(courses)
    filtered_count = len(filtered_courses)
    universities = filtered_courses["University"].nunique() if filtered_count else 0
    avg_rating = filtered_courses["Rating"].mean() if filtered_count else 0

    cols = st.columns(4)
    cols[0].metric("Total courses", f"{total_courses:,}", border=True)
    cols[1].metric("Filtered catalog", f"{filtered_count:,}", border=True)
    cols[2].metric("Universities", f"{universities:,}", border=True)
    cols[3].metric("Average rating", f"{avg_rating:.2f}", border=True)


def render_skill_chips(skills: Iterable[str], limit: int = 5) -> None:
    visible = list(skills or ())[:limit]
    if not visible:
        return

    chip_html = "".join(
        f'<span class="vrl-chip">{escape(format_skill_label(skill))}</span>'
        for skill in visible
    )
    extra = max(0, len(list(skills or ())) - limit)
    if extra:
        chip_html += f'<span class="vrl-chip">+{extra}</span>'

    st.markdown(f'<div class="vrl-chip-row">{chip_html}</div>', unsafe_allow_html=True)


def render_course_card(
    course: pd.Series,
    *,
    rank: int | None = None,
    show_similarity: bool = False,
    show_reason: bool = False,
    key_prefix: str = "course",
) -> None:
    course_name = str(course["Course Name"])
    course_key = normalize_course_key(course.get("Course Key", course.name))
    saved = course_name in st.session_state.get("shortlist", [])
    similarity = course.get("Similarity", None)
    similarity_value = None
    if similarity is not None and pd.notna(similarity):
        similarity_value = min(max(float(similarity), 0.0), 1.0)

    rank_text = f"#{rank}" if rank is not None else "Course"
    difficulty = escape(str(course["Difficulty Level"]))

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="vrl-card-top">
                <span class="vrl-rank">{escape(rank_text)}</span>
                <span class="vrl-badge">{difficulty}</span>
            </div>
            <div class="vrl-course-title">{escape(course_name)}</div>
            <div class="vrl-meta">
                {escape(str(course["University"]))} | {float(course["Rating"]):.1f}/5
            </div>
            """,
            unsafe_allow_html=True,
        )

        if show_similarity and similarity_value is not None:
            st.progress(similarity_value, text=f"{similarity_value:.0%} match")

        if show_reason and str(course.get("Match Reason", "")).strip():
            st.markdown(
                f'<div class="vrl-match-reason">{escape(str(course["Match Reason"]))}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="vrl-description">{escape(truncate_text(course["Course Description"]))}</div>',
            unsafe_allow_html=True,
        )
        render_skill_chips(course.get("Skill Tokens", ()))

        open_col, save_col = st.columns([0.58, 0.42])
        with open_col:
            st.link_button(
                "Open course",
                str(course["Course URL"]),
                type="primary",
                width="stretch",
            )
        with save_col:
            st.button(
                "Saved" if saved else "Save",
                key=f"{key_prefix}_save_{course_key}",
                width="stretch",
                on_click=toggle_shortlist,
                args=(course_name,),
            )


def render_course_grid(
    courses: pd.DataFrame,
    *,
    show_similarity: bool = False,
    show_reason: bool = False,
    key_prefix: str = "grid",
) -> None:
    if courses.empty:
        st.markdown(
            '<div class="vrl-empty">No courses match the current selection.</div>',
            unsafe_allow_html=True,
        )
        return

    for start in range(0, len(courses), 3):
        columns = st.columns(3)
        chunk = courses.iloc[start : start + 3]
        for offset, (index, course) in enumerate(chunk.iterrows(), start=1):
            with columns[offset - 1]:
                render_course_card(
                    course,
                    rank=start + offset,
                    show_similarity=show_similarity,
                    show_reason=show_reason,
                    key_prefix=f"{key_prefix}_{index}",
                )


def render_advisor_roadmap(recommendations: pd.DataFrame) -> None:
    if recommendations.empty or "Roadmap Stage" not in recommendations.columns:
        return

    stage_order = ["Foundation", "Build", "Specialize"]
    stage_copy = {
        "Foundation": "Start here",
        "Build": "Practice next",
        "Specialize": "Go deeper",
    }
    selected_indexes: set[object] = set()
    roadmap_rows: list[pd.Series] = []
    for stage in stage_order:
        stage_rows = recommendations[
            (recommendations["Roadmap Stage"] == stage)
            & (~recommendations.index.isin(selected_indexes))
        ]
        if stage_rows.empty:
            continue
        row = stage_rows.iloc[0]
        roadmap_rows.append(row)
        selected_indexes.add(row.name)

    if len(roadmap_rows) < 3:
        for _, row in recommendations.iterrows():
            if row.name in selected_indexes:
                continue
            roadmap_rows.append(row)
            selected_indexes.add(row.name)
            if len(roadmap_rows) >= 3:
                break

    if not roadmap_rows:
        return

    cards = []
    for row in roadmap_rows[:3]:
        stage = str(row.get("Roadmap Stage", "Build"))
        label = stage_copy.get(stage, "Recommended")
        cards.append(
            f"""
            <div class="vrl-roadmap-step">
                <div class="vrl-roadmap-label">{escape(label)} | {escape(stage)}</div>
                <div class="vrl-roadmap-title">{escape(str(row.get("Course Name", "")))}</div>
                <div class="vrl-meta">{escape(str(row.get("Provider", "")))} | {escape(str(row.get("Difficulty Level", "")))}</div>
            </div>
            """
        )

    st.markdown('<div class="vrl-section-title">Suggested learning roadmap</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="vrl-roadmap">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_smart_filter(courses: pd.DataFrame) -> None:
    st.sidebar.markdown("### Smart filter")
    for message in st.session_state.get("smart_filter_messages", [])[-3:]:
        role = "user" if message.get("role") == "user" else "assistant"
        st.sidebar.markdown(
            f'<div class="vrl-chat-bubble vrl-chat-{role}">{escape(str(message.get("content", "")))}</div>',
            unsafe_allow_html=True,
        )

    request = st.sidebar.text_input(
        "Ask smart filter",
        placeholder="Beginner Azure security from Microsoft",
        key="smart_filter_prompt",
    )
    submitted = st.sidebar.button("Apply smart filter", width="stretch")

    if submitted and request.strip():
        reply = apply_smart_filter(courses, request.strip())
        st.session_state["smart_filter_messages"] = [
            *st.session_state.get("smart_filter_messages", [])[-4:],
            {"role": "user", "content": request.strip()},
            {"role": "assistant", "content": reply},
        ]
        st.rerun()


def submit_advisor_request(
    request: str,
    filtered_courses: pd.DataFrame,
    recommendation_resources: dict,
    active_filter_key: tuple[str, str, str, str, tuple[str, ...]],
    rating_sort: str,
) -> None:
    query_text = advisor_query_text(
        st.session_state.get("advisor_messages", []),
        request.strip(),
    )
    with st.status("Searching the course catalog", expanded=False) as status:
        status.write("Reading your goal")
        recommendations = recommend_for_goal(
            query_text,
            filtered_courses,
            recommendation_resources,
            limit=MAX_RECOMMENDATIONS,
        )
        status.write("Ranking best-fit courses")
        recommendations = sort_recommendations(recommendations, rating_sort)
        recommendations = add_advisor_context(recommendations, query_text)
        reply = advisor_reply(recommendations, len(filtered_courses))
        status.update(label="Recommendations ready", state="complete", expanded=False)

    st.session_state["advisor_recommendations"] = recommendations
    st.session_state["advisor_context"] = active_filter_key
    st.session_state["advisor_messages"] = [
        *st.session_state.get("advisor_messages", [])[-6:],
        {"role": "user", "content": request.strip()},
        {"role": "assistant", "content": reply},
    ]
    st.session_state["pending_toast"] = "Recommendations ready"
    st.rerun()


def render_advisor_chat(
    filtered_courses: pd.DataFrame,
    recommendation_resources: dict,
    active_filter_key: tuple[str, str, str, str, tuple[str, ...]],
    rating_sort: str,
) -> None:
    st.markdown(
        """
        <div class="vrl-advisor-head">
            <div class="vrl-bot-avatar">AI</div>
            <div>
                <p class="vrl-advisor-name">Course Advisor</p>
                <p class="vrl-advisor-status">Searching the catalog locally</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    quick_prompts = [
        ("Gen AI", "I am looking for Gen AI courses"),
        ("Cybersecurity", "Beginner cybersecurity courses"),
        ("Data analyst", "Python data analyst SQL dashboard courses"),
        ("Cloud DevOps", "Cloud DevOps Azure courses"),
    ]
    quick_cols = st.columns(4)
    quick_request = ""
    for index, (label, prompt) in enumerate(quick_prompts):
        with quick_cols[index]:
            if st.button(label, key=f"advisor_quick_{index}", width="stretch"):
                quick_request = prompt

    clear_advisor = st.button("Clear conversation", width="content")

    if clear_advisor:
        st.session_state["advisor_messages"] = [
            {
                "role": "assistant",
                "content": "Tell me your background, goal, and constraints. I will recommend courses from the catalog.",
            }
        ]
        st.session_state["advisor_recommendations"] = None
        st.session_state["advisor_context"] = None
        st.rerun()

    if quick_request:
        submit_advisor_request(
            quick_request,
            filtered_courses,
            recommendation_resources,
            active_filter_key,
            rating_sort,
        )

    for message in st.session_state.get("advisor_messages", [])[-6:]:
        role = "user" if message.get("role") == "user" else "assistant"
        avatar = "user" if role == "user" else "assistant"
        with st.chat_message(role, avatar=avatar):
            st.write(str(message.get("content", "")))

    advisor_recommendations = st.session_state.get("advisor_recommendations")
    advisor_context = st.session_state.get("advisor_context")
    if advisor_recommendations is not None:
        if advisor_context != active_filter_key:
            st.info("Manual filters changed after the last advisor answer. Ask again to refresh these recommendations.")
        else:
            render_advisor_roadmap(advisor_recommendations)
            render_course_grid(
                advisor_recommendations,
                show_similarity=True,
                show_reason=True,
                key_prefix="advisor",
            )

    chat_request = st.chat_input(
        "Tell me what you want to learn",
        key="advisor_chat_input",
    )
    if chat_request and str(chat_request).strip():
        submit_advisor_request(
            str(chat_request).strip(),
            filtered_courses,
            recommendation_resources,
            active_filter_key,
            rating_sort,
        )


def render_sidebar(courses: pd.DataFrame) -> tuple[str, str, str, str, str, list[str]]:
    logo_uri = logo_data_uri()
    if logo_uri:
        st.sidebar.markdown(
            f"""
            <div class="vrl-sidebar-brand">
                <img src="{logo_uri}" alt="VRL logo" />
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("### Filters")
    search_query = st.sidebar.text_input(
        "Search catalog",
        placeholder="Python, leadership, finance",
        key="catalog_search",
    )

    providers = options_from(courses["Provider"])
    if st.session_state.get("course_provider") not in providers:
        st.session_state["course_provider"] = ALL_OPTION
    provider = st.sidebar.selectbox(
        "Course provider",
        providers,
        key="course_provider",
    )

    difficulties = difficulty_options(courses)
    if st.session_state.get("difficulty_level") not in difficulties:
        st.session_state["difficulty_level"] = ALL_OPTION
    difficulty_level = st.sidebar.segmented_control(
        "Difficulty",
        difficulties,
        key="difficulty_level",
        width="stretch",
    )

    university_base = apply_filters(
        courses,
        search_query,
        provider,
        difficulty_level,
        ALL_OPTION,
        st.session_state.get("selected_skills", []),
    )
    universities = options_from(university_base["University"])
    if st.session_state.get("university") not in universities:
        st.session_state["university"] = ALL_OPTION
    university = st.sidebar.selectbox(
        "University / company",
        universities,
        key="university",
    )

    sort_options = [SORT_SIMILARITY, SORT_HIGH_TO_LOW, SORT_LOW_TO_HIGH]
    if st.session_state.get("rating_sort") not in sort_options:
        st.session_state["rating_sort"] = SORT_SIMILARITY
    rating_sort = st.sidebar.selectbox(
        "Recommendation sort",
        sort_options,
        key="rating_sort",
    )

    base_skill_options = top_skill_options(tuple(courses["Skill Tokens"]))
    stored_skills = list(st.session_state.get("selected_skills", []))
    skill_options = [
        *base_skill_options,
        *[skill for skill in stored_skills if skill not in base_skill_options],
    ]
    selected_skills = [
        skill
        for skill in stored_skills
        if skill in skill_options
    ]
    if st.session_state.get("selected_skills") != selected_skills:
        st.session_state["selected_skills"] = selected_skills

    selected_skills = st.sidebar.pills(
        "Skills",
        skill_options,
        selection_mode="multi",
        format_func=format_skill_label,
        key="selected_skills",
        width="stretch",
    )
    selected_skills = list(selected_skills or [])

    st.sidebar.button(
        "Reset filters",
        width="stretch",
        on_click=reset_filters,
    )

    return search_query, provider, difficulty_level, university, rating_sort, selected_skills


def render_recommend_tab(
    filtered_courses: pd.DataFrame,
    recommendation_resources: dict,
    active_filter_key: tuple[str, str, str, str, tuple[str, ...]],
    rating_sort: str,
) -> None:
    render_advisor_chat(
        filtered_courses,
        recommendation_resources,
        active_filter_key,
        rating_sort,
    )

    st.divider()
    st.markdown('<div class="vrl-section-title">Course similarity</div>', unsafe_allow_html=True)

    course_label_map = filtered_courses.set_index("Course Key")["Course Name"].to_dict()
    course_options = (
        filtered_courses.sort_values("Course Name")["Course Key"].astype(str).to_list()
    )
    if not course_options:
        st.warning("No courses match the selected filters.")
        return

    if st.session_state.get("course_name") not in course_options:
        st.session_state["course_name"] = course_options[0]

    selected_course = st.selectbox(
        "Completed, liked, or target course",
        course_options,
        format_func=lambda key: course_label_map.get(key, key),
        key="course_name",
    )

    active_context = (selected_course, active_filter_key)
    if st.session_state.get("recommendation_context") != active_context:
        st.session_state["recommendations"] = None

    action_col, count_col = st.columns([0.22, 0.78], vertical_alignment="center")
    with action_col:
        find_matches = st.button(
            "Find matches",
            type="primary",
            width="stretch",
        )
    with count_col:
        st.markdown(
            f'<div class="vrl-small-note">{len(filtered_courses):,} courses in scope</div>',
            unsafe_allow_html=True,
        )

    if find_matches:
        with st.status("Finding similar courses", expanded=False) as status:
            with st.spinner("Comparing course signals", show_time=True):
                recommendations = recommend_courses(
                    selected_course,
                    filtered_courses,
                    recommendation_resources,
                )
            st.session_state["recommendations"] = recommendations
            st.session_state["recommendation_context"] = active_context
            status.update(label="Recommendations ready", state="complete", expanded=False)
        st.toast("Recommendations ready")

    recommendations = st.session_state.get("recommendations")
    if recommendations is None:
        st.markdown(
            '<div class="vrl-empty">Select a course and run matching to populate this view.</div>',
            unsafe_allow_html=True,
        )
        return

    if recommendations.empty:
        st.warning("No similar course is available under the selected filters.")
        return

    sorted_recommendations = sort_recommendations(recommendations, rating_sort)
    render_course_grid(
        sorted_recommendations,
        show_similarity=True,
        key_prefix="recommend",
    )


def render_explore_tab(filtered_courses: pd.DataFrame) -> None:
    st.markdown('<div class="vrl-section-title">Catalog</div>', unsafe_allow_html=True)

    preview = filtered_courses.sort_values(
        by=["Rating", "Course Name"],
        ascending=[False, True],
    ).head(12)
    render_course_grid(preview, key_prefix="explore")

    table = filtered_courses[
        ["Course Name", "University", "Difficulty Level", "Rating", "Course URL"]
    ].sort_values(by=["Rating", "Course Name"], ascending=[False, True])
    st.dataframe(
        table.head(200),
        hide_index=True,
        width="stretch",
    )


def render_shortlist_tab(courses: pd.DataFrame) -> None:
    st.markdown('<div class="vrl-section-title">Shortlist</div>', unsafe_allow_html=True)

    shortlist = st.session_state.get("shortlist", [])
    if not shortlist:
        st.markdown(
            '<div class="vrl-empty">Saved courses will appear here for this session.</div>',
            unsafe_allow_html=True,
        )
        return

    shortlisted_courses = courses[courses["Course Name"].isin(shortlist)]
    render_course_grid(shortlisted_courses, key_prefix="shortlist")


def top_skills_frame(courses: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for skills in courses["Skill Tokens"]:
        counter.update(skills)

    return pd.DataFrame(
        [
            {"Skill": format_skill_label(skill), "Courses": count}
            for skill, count in counter.most_common(limit)
        ]
    )


def render_insights_tab(courses: pd.DataFrame, filtered_courses: pd.DataFrame) -> None:
    st.markdown('<div class="vrl-section-title">Insights</div>', unsafe_allow_html=True)

    if filtered_courses.empty:
        st.warning("No data available for the selected filters.")
        return

    difficulty_frame = (
        filtered_courses["Difficulty Level"]
        .value_counts()
        .rename_axis("Difficulty")
        .reset_index(name="Courses")
    )
    provider_frame = (
        filtered_courses["University"]
        .value_counts()
        .head(10)
        .rename_axis("Provider")
        .reset_index(name="Courses")
    )
    skills_frame = top_skills_frame(filtered_courses)

    chart_col, provider_col = st.columns(2)
    with chart_col:
        st.markdown("##### Difficulty distribution")
        st.bar_chart(difficulty_frame, x="Difficulty", y="Courses", width="stretch")
    with provider_col:
        st.markdown("##### Top providers")
        st.bar_chart(provider_frame, x="Provider", y="Courses", width="stretch")

    skill_col, rating_col = st.columns(2)
    with skill_col:
        st.markdown("##### Top skills")
        st.bar_chart(skills_frame, x="Skill", y="Courses", width="stretch")
    with rating_col:
        st.markdown("##### Rating profile")
        rating_summary = pd.DataFrame(
            {
                "Metric": ["Filtered average", "Catalog average"],
                "Rating": [
                    filtered_courses["Rating"].mean(),
                    courses["Rating"].mean(),
                ],
            }
        )
        st.bar_chart(rating_summary, x="Metric", y="Rating", width="stretch")


def main() -> None:
    inject_styles()
    ensure_session_state()
    show_pending_toast()

    courses = load_courses(file_mtime(DATA_PATH))
    recommendation_resources = load_recommendation_resources(
        tuple(courses["Course Key"]),
        tuple(courses["Search Text"]),
        file_mtime(INDEX_PATH),
    )

    (
        search_query,
        provider,
        difficulty_level,
        university,
        rating_sort,
        selected_skills,
    ) = render_sidebar(courses)

    filtered_courses = apply_filters(
        courses,
        search_query,
        provider,
        difficulty_level,
        university,
        selected_skills,
    )
    active_filter_key = filter_key(
        search_query,
        provider,
        difficulty_level,
        university,
        selected_skills,
    )

    render_header()
    render_metrics(courses, filtered_courses)

    if filtered_courses.empty:
        st.warning("No courses match the selected filters. Try a different combination.")
        st.stop()

    recommend_tab, explore_tab, shortlist_tab, insights_tab = st.tabs(
        ["Recommend", "Explore", "Shortlist", "Insights"]
    )

    with recommend_tab:
        render_recommend_tab(
            filtered_courses,
            recommendation_resources,
            active_filter_key,
            rating_sort,
        )

    with explore_tab:
        render_explore_tab(filtered_courses)

    with shortlist_tab:
        render_shortlist_tab(courses)

    with insights_tab:
        render_insights_tab(courses, filtered_courses)


if __name__ == "__main__":
    main()
