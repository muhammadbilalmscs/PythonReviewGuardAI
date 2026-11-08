# ReviewGuard AI

Google Business Review Intelligence: a small learning project using Python, FastAPI, Streamlit, Pydantic, Google Places and Gemini tool calling.

## 1. What the application does

A business owner can look up a business, enter or upload reviews, and get:

- sentiment and complaint analysis, for German and English reviews
- the five reviews needing the most attention, ranked by a Python tool
- two separate lists: reviews needing a **customer service** response, and reviews with **policy evidence**
- a factual report draft to check and submit by hand, or an explanation of why reporting is not recommended

It never reports a review, never promises removal and never invents evidence. A negative review is not a reason to report one.

## 2. Python concepts used

| Concept | Where to look in `main.py` |
| --- | --- |
| Functions and type hints | every section |
| Classes and inheritance | `ReviewAnalysis(GeminiReviewAnalysis)` |
| Dictionaries and list comprehensions | `build_tool_signals`, `apply_safety_rules` |
| `async` / `await` | `call_places_api`, `call_gemini` |
| Exceptions | `HTTPException` in `call_places_api` |
| Regular expressions | `read_details_from_google_url` |
| `**kwargs` | `call_gemini(**request)` |

## 3. FastAPI architecture

```text
Streamlit page  --HTTP/JSON-->  FastAPI (main.py)  --HTTPS-->  Google Places
                                                  --HTTPS-->  Gemini
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Is the backend running, and are the keys set? |
| `GET /search-business` | Find a business |
| `POST /analyze-reviews` | Gemini semantic analysis |
| `POST /prioritize-reviews` | Tool calling and the top five |
| `POST /generate-report` | Factual report suggestion |

`async def` lets the server handle other requests while it waits for Google. With the backend running, interactive API docs are at <http://127.0.0.1:8000/docs>.

## 4. Pydantic validation

A Pydantic class describes the shape of the data. FastAPI uses it to check every request, so a rating of 6 or an empty review gets a `422` error before your code runs. The same idea checks Gemini's answers, because an LLM can return the wrong shape too.

## 5. Google API integration

`call_places_api` sends one HTTP request. The API key travels in the `X-Goog-Api-Key` header, and `X-Goog-FieldMask` lists the fields we want (Google requires this). The app never opens or scrapes Google Maps pages. Short links such as `maps.app.goo.gl` cannot be read, so search by name or Place ID instead.

## 6. Gemini LLM calls

The official `google-genai` SDK is used through its Interactions API. `call_gemini` is the only place that talks to it.

The client receives `GEMINI_API_KEY` explicitly. If no key is passed, the SDK prefers `GOOGLE_API_KEY`, which here is the Maps key.

**Structured output:** `response_format` sends a JSON schema made from a Pydantic class. Gemini answers in that shape, and `model_validate_json` checks it.

## 7. Semantic analysis

Semantic analysis judges meaning rather than matching words, so a polite review can still be off-topic and an angry one can be legitimate. Python then checks what it can in `apply_safety_rules`:

- The real star rating replaces Gemini's copy.
- An evidence quote is kept only if it really appears in the review.
- A policy concern with no evidence becomes "Insufficient evidence".

## 8. Function calling and tool execution

An LLM cannot run Python. It can only ask. The loop is in `run_priority_agent`:

```text
User input → FastAPI → Pydantic validation → Gemini semantic analysis
→ Gemini requests the tool → Python runs the tool → result goes back to Gemini
→ structured final analysis → Streamlit
```

1. **Declaration:** `PRIORITY_TOOL_DECLARATION` describes the tool to Gemini.
2. **Request:** Gemini's answer contains `function_call` steps with arguments.
3. **Execution:** `execute_tool_call` runs the Python function.
4. **Response:** a `function_result` with the matching `call_id` goes back to Gemini.

Gemini's arguments are treated as untrusted. They only say which review to score, and the numbers come from Python's validated copy. If Gemini skips a review, Python scores it and the tool log says so. Open "See how Gemini called the Python tool" on the page to watch this.

### The scores

**Priority score (0 to 100):** negative sentiment 30, star rating 15, policy indicators 25, supporting evidence 20, business relevance 10.

**Policy Evidence Strength Score (0 to 100):** flagged concern 15 or 25, confirmed quotes 20 or 35, owner's evidence 30, off-topic signal 10. Below 40 is Weak, 40 to 69 is Moderate, 70 or more is Strong. Spam, conflict of interest, impersonation and "not a genuine experience" are capped at 39 unless the owner supplies evidence.

This score measures the evidence the app holds. **It is not the probability of removal.** Google publishes no formula, and its final decision is unknown.

## 9. How Streamlit communicates with FastAPI

`call_backend` in `streamlit_app.py` sends an HTTP request and reads the JSON reply. Streamlit reruns the whole script after every click, so results are kept in `st.session_state`. The page never sees your API keys.

## 10. Configure API credentials

Copy the example file if you do not have a `.env` yet:

```bash
copy .env.example .env
```

| Variable | What it must be |
| --- | --- |
| `GOOGLE_API_KEY` | A Google Maps Platform key with Places API (New) and billing enabled |
| `GEMINI_API_KEY` | A key from Google AI Studio |
| `GEMINI_MODEL` | A model your key can use, such as `gemini-3.6-flash` |

## 11. Run the application

Install once, from this folder in PowerShell:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

Terminal 1, the backend:

```bash
uvicorn main:app --reload
```

Terminal 2, the page (activate `.venv` here too):

```bash
streamlit run streamlit_app.py
```

Open <http://localhost:8501>, and check the backend at <http://127.0.0.1:8000/health>.

To try it without Google Places, choose **Manual text** and enter one review per line:

```text
1 | Wir haben 40 Minuten gewartet und das Essen war kalt.
4 | Nice staff, good coffee.
```

CSV and JSON uploads need `text` and `rating` columns. `review_id` and `owner_evidence` are optional.

## 12. Known limitations and data use

- Places returns **at most 5 reviews per business**, chosen by relevance. They are not all the reviews.
- Reading every review needs the Google Business Profile API, with OAuth 2.0 and access to an eligible, verified location. This version does not include it.
- Google's Places policies say not to store user reviews and not to create content based on Google Maps Content. Analyzing Places reviews with Gemini may fall under that, so it is **blocked unless you confirm you have checked the terms**. Read the current [Google Maps Platform Terms](https://cloud.google.com/maps-platform/terms) yourself.
- Manually entered and uploaded reviews are the intended learning path. Only upload data you have the right to use.
- Displayed Google reviews show the author's name, profile link and photo, as Google requires.
- The app has no database and writes no reviews to disk.
- The owner's evidence is recorded but cannot be verified by the app. Check every sentence of a report before using it.
- Official guidance: [review policy](https://support.google.com/contributionpolicy/answer/7400114) and [how to report a review](https://support.google.com/business/answer/4596773).
