from collections import Counter
from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "CourseDetails.csv"
LOGO_PATH = APP_DIR / "MyLogoLight.png"

ALL_OPTION = "All"
MAX_FEATURES = 4000
MAX_RECOMMENDATIONS = 12
SORT_SIMILARITY = "Similarity first"
SORT_HIGH_TO_LOW = "Rating high to low"
SORT_LOW_TO_HIGH = "Rating low to high"

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
            --vrl-bg: #070b14;
            --vrl-bg-2: #0b1220;
            --vrl-border: #243348;
            --vrl-muted: #99a8bd;
            --vrl-text: #edf4ff;
            --vrl-panel: #101827;
            --vrl-panel-2: #0d1524;
            --vrl-soft: #162235;
            --vrl-blue: #60a5fa;
            --vrl-cyan: #22d3ee;
            --vrl-green: #34d399;
            --vrl-amber: #fbbf24;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.22), transparent 30rem),
                radial-gradient(circle at top right, rgba(34, 211, 238, 0.14), transparent 24rem),
                linear-gradient(180deg, var(--vrl-bg) 0%, #0a1020 46%, #060a12 100%);
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
                linear-gradient(180deg, rgba(16, 24, 39, 0.98), rgba(9, 14, 24, 0.98));
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
                linear-gradient(180deg, rgba(16, 24, 39, 0.96), rgba(13, 21, 36, 0.96));
        }

        .vrl-header {
            border: 1px solid var(--vrl-border);
            border-radius: 8px;
            background: rgba(16, 24, 39, 0.92);
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.26);
        }

        .vrl-title {
            font-size: 2.15rem;
            font-weight: 760;
            line-height: 1.08;
            letter-spacing: 0;
            margin: 0;
            color: var(--vrl-text);
        }

        .vrl-subtitle {
            color: var(--vrl-muted);
            font-size: 0.98rem;
            line-height: 1.45;
            margin: 0.35rem 0 0;
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
            color: var(--vrl-blue);
            font-weight: 760;
            font-size: 0.9rem;
        }

        .vrl-badge {
            border: 1px solid var(--vrl-border);
            border-radius: 6px;
            color: #dbeafe;
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
            color: #c5d2e5;
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
            border: 1px solid rgba(96, 165, 250, 0.36);
            border-radius: 6px;
            color: #bfdbfe;
            background: rgba(37, 99, 235, 0.16);
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

        h1, h2, h3, h4 {
            letter-spacing: 0;
            color: var(--vrl-text);
        }

        div[data-testid="stTabs"] button p {
            color: var(--vrl-muted);
            font-weight: 650;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] p {
            color: var(--vrl-cyan);
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
            background: rgba(96, 165, 250, 0.18);
        }

        div.stButton > button,
        div[data-testid="stLinkButton"] > a {
            border-radius: 6px;
            font-weight: 650;
            border-color: var(--vrl-border);
        }

        div.stButton > button[kind="primary"],
        div[data-testid="stLinkButton"] > a[kind="primary"] {
            background: linear-gradient(135deg, #2563eb, #0891b2);
            border: 1px solid rgba(147, 197, 253, 0.45);
            color: #ffffff;
        }

        div.stButton > button:hover,
        div[data-testid="stLinkButton"] > a:hover {
            border-color: var(--vrl-cyan);
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


def truncate_text(text: str, max_chars: int = 230) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_chars:
        return normalized

    truncated = normalized[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{truncated}..."


@st.cache_data(show_spinner="Loading course catalog...")
def load_courses() -> pd.DataFrame:
    courses = pd.read_csv(DATA_PATH, index_col=0)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in courses.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"CourseDetails.csv is missing required columns: {missing}")

    courses = courses[REQUIRED_COLUMNS].copy()
    courses["Rating"] = pd.to_numeric(courses["Rating"], errors="coerce")
    courses = courses.dropna(subset=["Course Name", "Tags", "Rating"])

    text_columns = [
        "Course Name",
        "University",
        "Difficulty Level",
        "Course URL",
        "Course Description",
        "Skills",
        "Tags",
    ]
    for column in text_columns:
        courses[column] = courses[column].fillna("").astype(str).str.strip()

    courses = courses.reset_index(drop=True)
    courses["Skill Tokens"] = courses["Skills"].apply(parse_skills)
    courses["Search Text"] = (
        courses["Course Name"]
        + " "
        + courses["University"]
        + " "
        + courses["Difficulty Level"]
        + " "
        + courses["Course Description"]
        + " "
        + courses["Skills"]
    ).str.lower()

    return courses


@st.cache_resource(show_spinner="Building recommendation index...")
def build_vector_index(course_names: tuple[str, ...], tags: tuple[str, ...]):
    vectorizer = CountVectorizer(max_features=MAX_FEATURES, stop_words="english")
    vectors = vectorizer.fit_transform(tags)
    course_index = {name: index for index, name in enumerate(course_names)}
    return vectors, course_index


@st.cache_data(show_spinner=False)
def top_skill_options(skill_rows: tuple[tuple[str, ...], ...], limit: int = 18) -> list[str]:
    counter: Counter[str] = Counter()
    for skills in skill_rows:
        counter.update(skills)
    return [skill for skill, _ in counter.most_common(limit)]


def ensure_session_state() -> None:
    st.session_state.setdefault("shortlist", [])
    st.session_state.setdefault("recommendations", None)
    st.session_state.setdefault("recommendation_context", None)
    st.session_state.setdefault("pending_toast", None)


def show_pending_toast() -> None:
    message = st.session_state.pop("pending_toast", None)
    if message:
        st.toast(message)


def reset_filters() -> None:
    for key in (
        "catalog_search",
        "difficulty_level",
        "university",
        "rating_sort",
        "selected_skills",
        "course_name",
        "recommendations",
        "recommendation_context",
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


def apply_filters(
    courses: pd.DataFrame,
    search_query: str,
    difficulty_level: str,
    university: str,
    selected_skills: Iterable[str],
) -> pd.DataFrame:
    filtered = courses
    query = search_query.strip().lower()
    skills = tuple(selected_skills or ())

    if query:
        filtered = filtered[filtered["Search Text"].str.contains(query, regex=False)]

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
    course_name: str,
    candidate_courses: pd.DataFrame,
    vectors,
    course_index: dict[str, int],
    limit: int = MAX_RECOMMENDATIONS,
) -> pd.DataFrame:
    selected_index = course_index.get(course_name)
    if selected_index is None:
        return pd.DataFrame()

    candidates = candidate_courses[candidate_courses["Course Name"] != course_name].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidate_indexes = candidates.index.to_list()
    similarities = cosine_similarity(
        vectors[selected_index],
        vectors[candidate_indexes],
    ).ravel()

    candidates["Similarity"] = similarities
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
    difficulty_level: str,
    university: str,
    selected_skills: Iterable[str],
) -> tuple[str, str, str, tuple[str, ...]]:
    return (
        search_query.strip().lower(),
        difficulty_level,
        university,
        tuple(sorted(selected_skills or ())),
    )


def render_header() -> None:
    with st.container(border=True):
        logo_col, text_col = st.columns([0.09, 0.91], vertical_alignment="center")
        with logo_col:
            if LOGO_PATH.exists():
                st.image(str(LOGO_PATH), width=84)
        with text_col:
            st.markdown(
                """
                <p class="vrl-title">VRL Course Intelligence</p>
                <p class="vrl-subtitle">Analytics dashboard for Coursera course discovery.</p>
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
    key_prefix: str = "course",
) -> None:
    course_name = str(course["Course Name"])
    course_key = str(course.name)
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
                    key_prefix=f"{key_prefix}_{index}",
                )


def render_sidebar(courses: pd.DataFrame) -> tuple[str, str, str, str, list[str]]:
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), width=140)

    st.sidebar.markdown("### Filters")
    search_query = st.sidebar.text_input(
        "Search catalog",
        placeholder="Python, leadership, finance",
        key="catalog_search",
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

    skill_options = top_skill_options(tuple(courses["Skill Tokens"]))
    selected_skills = [
        skill
        for skill in st.session_state.get("selected_skills", [])
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

    return search_query, difficulty_level, university, rating_sort, selected_skills


def render_recommend_tab(
    filtered_courses: pd.DataFrame,
    vectors,
    course_index: dict[str, int],
    active_filter_key: tuple[str, str, str, tuple[str, ...]],
    rating_sort: str,
) -> None:
    st.markdown('<div class="vrl-section-title">Recommendations</div>', unsafe_allow_html=True)

    course_options = filtered_courses["Course Name"].sort_values().to_list()
    if not course_options:
        st.warning("No courses match the selected filters.")
        return

    if st.session_state.get("course_name") not in course_options:
        st.session_state["course_name"] = course_options[0]

    selected_course = st.selectbox(
        "Completed, liked, or target course",
        course_options,
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
                    vectors,
                    course_index,
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

    courses = load_courses()
    vectors, course_index = build_vector_index(
        tuple(courses["Course Name"]),
        tuple(courses["Tags"]),
    )

    (
        search_query,
        difficulty_level,
        university,
        rating_sort,
        selected_skills,
    ) = render_sidebar(courses)

    filtered_courses = apply_filters(
        courses,
        search_query,
        difficulty_level,
        university,
        selected_skills,
    )
    active_filter_key = filter_key(
        search_query,
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
            vectors,
            course_index,
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
