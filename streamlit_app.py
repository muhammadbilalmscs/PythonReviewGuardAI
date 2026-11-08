"""ReviewGuard AI frontend (Streamlit).

Streamlit runs this whole file from top to bottom every time the user clicks
something. Values we need to keep between those runs go in st.session_state.

This page contains no business logic and no API keys. It sends HTTP requests
to the FastAPI backend (main.py) and displays the JSON that comes back.

Run it with:  streamlit run streamlit_app.py
"""

import csv
import io
import json
import os

import httpx
import pandas as pd
import streamlit as st

BACKEND_URL = os.getenv("REVIEWGUARD_BACKEND_URL", "http://127.0.0.1:8000")
MAX_REVIEWS = 100


# -----------------------------------------------------------------------------
# Talking to the backend
# -----------------------------------------------------------------------------


def error_text(response: httpx.Response) -> str:
    """Turn an error response from FastAPI into one readable sentence."""
    try:
        detail = response.json().get("detail", "")
    except ValueError:
        return f"The backend returned status {response.status_code}."

    # Our own errors have a text detail. Pydantic's validation errors (422)
    # have a list of problems, so we join their messages.
    if isinstance(detail, list):
        return "Invalid input: " + "; ".join(item.get("msg", "") for item in detail)
    return str(detail)


def call_backend(method: str, path: str, timeout: float = 300.0, **options):
    """Send one request to FastAPI and return (data, error_message).

    One of the two values is always None. The caller checks for an error first.
    response.json() converts the JSON text from FastAPI into Python dicts/lists.
    """
    try:
        response = httpx.request(method, BACKEND_URL + path, timeout=timeout, **options)
    except httpx.TimeoutException:
        return None, "The backend took too long to answer. Try fewer reviews."
    except httpx.RequestError:
        return None, "Cannot reach the backend. Start it with: uvicorn main:app --reload"

    if response.status_code == 200:
        return response.json(), None
    return None, error_text(response)


# -----------------------------------------------------------------------------
# Reading reviews typed in or uploaded by the user
# -----------------------------------------------------------------------------


def read_star_rating(value, position: int) -> int:
    """Convert a rating such as "4" or 4.0 to a whole number from 1 to 5."""
    try:
        rating = int(float(str(value).strip().replace(",", ".")))
    except ValueError:
        raise ValueError(f"Review {position}: the rating '{value}' is not a number.")
    if rating < 1 or rating > 5:
        raise ValueError(f"Review {position}: the rating must be from 1 to 5.")
    return rating


def build_review(row: dict, position: int, source: str) -> dict:
    """Turn one CSV row or JSON object into the format the backend expects."""
    fields = {str(key).strip().lower(): value for key, value in row.items() if key is not None}

    text = ""
    for name in ["text", "review_text", "review", "comment"]:
        if fields.get(name):
            text = str(fields[name]).strip()
            break
    if not text:
        raise ValueError(f"Review {position}: the review text is empty or missing.")

    rating = None
    for name in ["star_rating", "rating", "stars"]:
        if fields.get(name) not in (None, ""):
            rating = fields[name]
            break
    if rating is None:
        raise ValueError(f"Review {position}: the star rating is missing.")

    review_id = str(fields.get("review_id") or fields.get("id") or f"{source}-{position}")
    return {
        "review_id": review_id.strip(),
        "star_rating": read_star_rating(rating, position),
        "text": text,
        "source": source,
        "owner_evidence": str(fields.get("owner_evidence") or "").strip(),
    }


def parse_manual_reviews(text: str) -> list[dict]:
    """Read one review per line, written as: stars | review text."""
    reviews = []
    lines = [line for line in text.splitlines() if line.strip()]
    for position, line in enumerate(lines, start=1):
        if "|" not in line:
            raise ValueError(f"Line {position}: write it as  stars | review text")
        stars, review_text = line.split("|", 1)
        reviews.append(build_review({"rating": stars, "text": review_text}, position, "manual"))
    return reviews


def decode_file(data: bytes) -> str:
    """Read file bytes as text. utf-8-sig also handles the marker Excel adds."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def parse_csv_reviews(data: bytes) -> list[dict]:
    """Read a CSV file. German Excel uses ';' between columns, so we allow both."""
    text = decode_file(data)
    first_line = text.split("\n", 1)[0]
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","

    try:
        rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    except csv.Error:
        raise ValueError("This CSV file could not be read.")
    if not rows:
        raise ValueError("The CSV file has no review rows.")
    return [build_review(row, position, "upload") for position, row in enumerate(rows, start=1)]


def parse_json_reviews(data: bytes) -> list[dict]:
    """Read a JSON file: a list of reviews, or {"reviews": [...]}."""
    try:
        content = json.loads(decode_file(data))
    except json.JSONDecodeError:
        raise ValueError("This is not a valid JSON file.")

    if isinstance(content, dict):
        content = content.get("reviews")
    if not isinstance(content, list) or not content:
        raise ValueError('The JSON must be a list of reviews or {"reviews": [...]}.')
    if not all(isinstance(item, dict) for item in content):
        raise ValueError("Each review in the JSON file must be an object.")
    return [build_review(item, position, "upload") for position, item in enumerate(content, start=1)]


def parse_uploaded_file(name: str, data: bytes) -> list[dict]:
    if name.lower().endswith(".csv"):
        return parse_csv_reviews(data)
    if name.lower().endswith(".json"):
        return parse_json_reviews(data)
    raise ValueError("Unsupported file type. Please upload a .csv or .json file.")


# -----------------------------------------------------------------------------
# Small display helper
# -----------------------------------------------------------------------------


def show_prioritized_review(item: dict, rank: int) -> None:
    """Show one review from the top five."""
    label = f"{rank}. Priority {item['priority_score']}/100 · {item['star_rating']} star · {item['sentiment']}"
    with st.expander(label, expanded=rank == 1):
        # st.text shows the review exactly as written, so symbols in a
        # customer's text are not treated as Markdown formatting.
        st.text(item["review_text"])
        col1, col2, col3 = st.columns(3)
        col1.metric("Priority score", item["priority_score"])
        col2.metric("Customer service", item["customer_service_score"])
        col3.metric("Policy evidence", item["policy_reporting_score"])
        st.markdown(f"**Why selected:** {item['why_selected']}")
        if item["gemini_comment"]:
            st.markdown(f"**Gemini's comment:** {item['gemini_comment']}")
        st.markdown(f"**Assessment:** {item['assessment']}")
        st.markdown(f"**Policy concern:** {item['policy_category']}")
        st.markdown(f"**Explanation:** {item['explanation']}")
        st.markdown(f"**Recommended action:** {item['recommended_action']}")


# -----------------------------------------------------------------------------
# The page
# -----------------------------------------------------------------------------

st.set_page_config(page_title="ReviewGuard AI", page_icon=":material/shield:", layout="wide")

# Everything that must survive a rerun is created once, here.
st.session_state.setdefault("search_response", None)
st.session_state.setdefault("analysis", None)
st.session_state.setdefault("priority", None)
st.session_state.setdefault("analyzed_reviews", [])
st.session_state.setdefault("reports", {})

st.title("ReviewGuard AI")
st.caption("Google Business Review Intelligence")
st.info(
    "An educational tool. It never reports a review, cannot guarantee removal, "
    "and a negative review is not a reason to report one.",
    icon=":material/school:",
)

health, health_error = call_backend("GET", "/health", timeout=5.0)
if health_error:
    st.error(health_error, title="Backend not running")
    st.stop()

status_col1, status_col2 = st.columns(2)
status_col1.caption(
    ":green[Google key found]" if health["google_api_key_configured"] else ":red[Google key missing]"
)
status_col2.caption(
    f":green[Gemini key found] · {health['gemini_model']}"
    if health["gemini_api_key_configured"]
    else ":red[Gemini key missing]: add GEMINI_API_KEY to .env"
)

# --- Business search ----------------------------------------------------------

st.header("Business search")
with st.form("search_form"):
    col1, col2 = st.columns(2)
    business_name = col1.text_input("Business name")
    business_address = col2.text_input("Address")
    profile_url = col1.text_input("Google profile URL")
    place_id = col2.text_input("Place ID")
    search_clicked = st.form_submit_button("Search Business", icon=":material/search:")

if search_clicked:
    with st.spinner("Searching Google Places..."):
        data, error = call_backend(
            "GET",
            "/search-business",
            timeout=30.0,
            params={
                "name": business_name,
                "address": business_address,
                "url": profile_url,
                "place_id": place_id,
            },
        )
    if error:
        st.error(error, title="Search failed")
    else:
        st.session_state["search_response"] = data

selected_business = None
search_response = st.session_state["search_response"]
if search_response:
    st.subheader("Business overview")
    results = search_response["results"]
    if len(results) > 1:
        names = [f"{item['name']}, {item['address']}" for item in results]
        chosen_name = st.selectbox("Matching businesses", names)
        selected_business = results[names.index(chosen_name)]
    else:
        selected_business = results[0]

    with st.container(border=True):
        st.markdown(f"**{selected_business['name']}**")
        st.text(selected_business["address"])
        col1, col2, col3 = st.columns(3)
        rating = selected_business["rating"]
        col1.metric("Google rating", f"{rating:.1f}" if rating is not None else "n/a")
        col2.metric("Total review count", f"{selected_business['total_review_count']:,}")
        col3.metric("Reviews returned", len(selected_business["reviews"]))
        st.caption(f"Data source: {search_response['data_source']} · Google Maps")
        if selected_business["google_maps_uri"]:
            st.link_button("Open in Google Maps", selected_business["google_maps_uri"])
    st.warning(search_response["limitation_note"], icon=":material/info:")

# --- Review input -------------------------------------------------------------

st.header("Review input")
input_mode = st.segmented_control(
    "Where do the reviews come from?",
    ["Manual text", "CSV or JSON upload", "Google Places"],
    default="Manual text",
)

reviews: list[dict] = []
input_error = ""
google_terms_confirmed = False

if input_mode == "Manual text":
    manual_text = st.text_area(
        "One review per line, written as: stars | review text",
        placeholder="1 | Wir haben 40 Minuten gewartet und das Essen war kalt.\n4 | Nice staff, good coffee.",
        height=160,
    )
    try:
        reviews = parse_manual_reviews(manual_text)
    except ValueError as error:
        input_error = str(error)

elif input_mode == "CSV or JSON upload":
    uploaded_file = st.file_uploader(
        "Columns: text and rating. Optional: review_id, owner_evidence.",
        type=["csv", "json"],
    )
    if uploaded_file is not None:
        try:
            reviews = parse_uploaded_file(uploaded_file.name, uploaded_file.getvalue())
        except ValueError as error:
            input_error = str(error)

elif input_mode == "Google Places":
    if selected_business is None:
        st.info("Search for a business first.")
    elif not selected_business["reviews"]:
        st.info("Google returned no reviews with text for this business.")
    else:
        reviews = selected_business["reviews"]
        st.caption("Up to 5 reviews from Google Maps, ordered by relevance. Google does not verify reviews.")
        for review in reviews:
            with st.container(border=True):
                # Google requires each displayed review to credit its author.
                if review["author_photo_uri"]:
                    st.image(review["author_photo_uri"], width=32, alt=f"Avatar of {review['author_name']}")
                if review["author_uri"]:
                    st.link_button(review["author_name"] or "Author profile", review["author_uri"])
                else:
                    st.text(review["author_name"])
                st.caption(f"{review['star_rating']} star · {review['relative_time']}")
                st.text(review["text"])
                if review["google_maps_uri"]:
                    st.link_button("View on Google Maps", review["google_maps_uri"])
        st.warning(
            "Google's Places policies say not to create content based on Google "
            "Maps content, so sending these reviews to Gemini may not be allowed. "
            "This app keeps reviews in memory only, never in a file or database.",
            title="Check the data-use terms first",
        )
        google_terms_confirmed = st.checkbox(
            "I have checked Google's terms and confirm this use is allowed."
        )

if input_error:
    st.error(input_error, title="Could not read the reviews")
    reviews = []
elif len(reviews) > MAX_REVIEWS:
    st.error(f"Please use at most {MAX_REVIEWS} reviews at a time.", title="Too many reviews")
    reviews = []
elif reviews:
    st.caption(f"{len(reviews)} review(s) ready · Data source: {input_mode}")

analyze_clicked = st.button(
    "Analyze Reviews with Gemini",
    type="primary",
    icon=":material/psychology:",
    disabled=not reviews,
)

if analyze_clicked:
    body = {"reviews": reviews, "google_terms_confirmed": google_terms_confirmed}

    # Clear old results first, so a failed run cannot show stale answers.
    st.session_state["analysis"] = None
    st.session_state["priority"] = None
    st.session_state["reports"] = {}

    with st.spinner("Step 1 of 2: Gemini is reading the reviews..."):
        analysis, error = call_backend("POST", "/analyze-reviews", json=body)

    if error:
        st.error(error, title="Analysis failed")
    else:
        st.session_state["analysis"] = analysis
        st.session_state["analyzed_reviews"] = reviews
        with st.spinner("Step 2 of 2: Gemini is calling the Python priority tool..."):
            priority, error = call_backend(
                "POST",
                "/prioritize-reviews",
                json={**body, "analyses": analysis["analyses"]},
            )
        if error:
            st.error(error, title="Prioritization failed")
        else:
            st.session_state["priority"] = priority

# --- Results dashboard --------------------------------------------------------

analysis = st.session_state["analysis"]
priority = st.session_state["priority"]

if analysis:
    st.header("Results dashboard")
    st.caption(
        f"{len(analysis['analyses'])} reviews · Data source: "
        f"{', '.join(analysis['data_sources'])} · Model: {analysis['model_used']}"
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Positive", f"{analysis['sentiment_percentages']['Positive']}%")
    col2.metric("Neutral", f"{analysis['sentiment_percentages']['Neutral']}%")
    col3.metric("Negative", f"{analysis['sentiment_percentages']['Negative']}%")

    st.subheader("Main customer complaints")
    if analysis["main_complaints"]:
        complaints = pd.DataFrame(analysis["main_complaints"])
        st.bar_chart(
            complaints,
            x="topic",
            y="count",
            horizontal=True,
            sort="-count",
            x_label="Number of reviews",
            y_label="Complaint",
            alt="Bar chart of the most common complaints",
        )
    else:
        st.text("No complaints were found.")

if priority:
    st.header("Top five reviews requiring attention")
    if priority["overall_summary"]:
        st.markdown(priority["overall_summary"])
    for note in priority["notes"]:
        st.caption(note)
    if not priority["top_five"]:
        st.success("None of these reviews needs attention.")
    for rank, item in enumerate(priority["top_five"], start=1):
        show_prioritized_review(item, rank)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True, height="stretch"):
            st.subheader("A. Customer service priority")
            st.caption("Reviews that need a response or an investigation.")
            for item in priority["customer_service_queue"]:
                st.markdown(f"**Score {item['customer_service_score']}** · {item['star_rating']} star")
                st.text(item["review_text"][:160])
            if not priority["customer_service_queue"]:
                st.text("Nothing here.")
    with col2:
        with st.container(border=True, height="stretch"):
            st.subheader("B. Policy reporting priority")
            st.caption("Only reviews with specific evidence of a possible violation.")
            for item in priority["policy_reporting_queue"]:
                st.markdown(
                    f"**Evidence {item['policy_reporting_score']}/100 "
                    f"({item['policy_evidence_label']})** · {item['policy_category']}"
                )
                st.text(item["review_text"][:160])
            if not priority["policy_reporting_queue"]:
                st.text("No review has specific evidence of a policy violation.")

    with st.expander("See how Gemini called the Python tool"):
        st.markdown(
            "User input → FastAPI → Pydantic validation → Gemini analysis → "
            "**Gemini requests tool → Python runs tool → result goes back to Gemini** "
            "→ structured final analysis → Streamlit"
        )
        for number, entry in enumerate(priority["tool_log"], start=1):
            st.markdown(f"**Call {number}** · requested by `{entry['requested_by']}` · `{entry['tool_name']}`")
            st.json({"arguments": entry["arguments"], "result": entry["result"]}, expanded=False)

# --- Reporting assistant ------------------------------------------------------

if analysis:
    st.header("Reporting assistant")
    st.caption("Prepares text for you to check. You decide whether to submit anything to Google.")

    reviews_by_id = {item["review_id"]: item for item in st.session_state["analyzed_reviews"]}

    # Reviews with a possible concern are listed first. False sorts before True.
    report_options = sorted(
        analysis["analyses"],
        key=lambda item: item["policy_category"] == "No apparent policy violation",
    )
    labels = [
        f"{item['policy_category']} · {item['star_rating']} star · "
        f"{reviews_by_id[item['review_id']]['text'][:60]}"
        for item in report_options
    ]
    selected_label = st.selectbox("Choose a review", labels)
    selected_analysis = report_options[labels.index(selected_label)]
    selected_id = selected_analysis["review_id"]
    selected_review = reviews_by_id[selected_id]

    st.text(selected_review["text"])
    st.markdown(f"**Assessment:** {selected_analysis['assessment']}")
    st.markdown(f"**What cannot be known:** {selected_analysis['uncertainty']}")

    owner_evidence = st.text_area(
        "Evidence from your own records (optional)",
        value=selected_review.get("owner_evidence", ""),
        help="Write only what you can truly show. The app cannot check it.",
        key=f"owner_evidence_{selected_id}",
    )

    if st.button("Prepare report suggestion", icon=":material/description:"):
        with st.spinner("Preparing..."):
            report, error = call_backend(
                "POST",
                "/generate-report",
                json={
                    "review": selected_review,
                    "analysis": selected_analysis,
                    "owner_evidence": owner_evidence,
                    "google_terms_confirmed": google_terms_confirmed,
                },
            )
        if error:
            st.error(error, title="Could not prepare the report")
        else:
            st.session_state["reports"][selected_id] = report

    report = st.session_state["reports"].get(selected_id)
    if report:
        evidence = report["policy_evidence"]
        col1, col2 = st.columns(2)
        col1.metric("Policy Evidence Strength Score", f"{evidence['score']}/100")
        col2.metric("Evidence level", evidence["label"])
        st.warning(evidence["disclaimer"], icon=":material/info:")
        with st.expander("How the score was calculated"):
            for step in evidence["how_calculated"]:
                st.text(step)

        st.markdown(f"**Policy category:** {report['suspected_policy_category']}")
        if report["report_recommended"]:
            st.markdown(f"**Exact reason for concern:** {report['exact_reason_for_concern']}")
            st.markdown(f"**Evidence assessment:** {report['evidence_assessment']}")
            st.markdown("**Supporting evidence**")
            for item in report["supporting_evidence"]:
                st.text(item)
            st.markdown("**Suggested report text** (use the copy icon, and check every sentence first)")
            # st.code gives a copy-to-clipboard button for free.
            st.code(report["report_text"], language=None, wrap_lines=True)
        else:
            st.error(report["reason_not_recommended"], title="Reporting is not recommended")

        if report["missing_evidence"]:
            st.markdown("**Missing evidence**")
            for item in report["missing_evidence"]:
                st.text(item)
        st.markdown(f"**Recommended next step:** {report['recommended_next_step']}")

        with st.expander("Official reporting instructions", expanded=True):
            for instruction in report["reporting_instructions"]:
                st.text(instruction)
            with st.container(horizontal=True):
                for name, link in report["official_links"].items():
                    st.link_button(name, link)
