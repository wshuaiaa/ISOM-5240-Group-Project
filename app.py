import os
from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo
from uuid import uuid4

import streamlit as st
from transformers import pipeline


# ---------------------------------------------------------------------------
# 1. Model configuration
# ---------------------------------------------------------------------------
HEADLINE_MODEL = os.getenv(
    "HEADLINE_MODEL",
    "JulesBelveze/t5-small-headline-generator",
)
NEWS_CLASSIFIER_MODEL = os.getenv(
    "NEWS_CLASSIFIER_MODEL",
    "wshuaiaa/News_classifier_Finetuned",
)
SENTIMENT_MODEL = os.getenv(
    "SENTIMENT_MODEL",
    "wshuaiaa/News_sentiment_Finetuned",
)

CATEGORY_OPTIONS = ["World", "Business", "Sci/Tech", "Sports"]

CATEGORY_LABELS = {
    "LABEL_0": "World",
    "LABEL_1": "Sports",
    "LABEL_2": "Business",
    "LABEL_3": "Sci/Tech",
    "label_0": "World",
    "label_1": "Sports",
    "label_2": "Business",
    "label_3": "Sci/Tech",
    "World": "World",
    "Sports": "Sports",
    "Business": "Business",
    "Sci/Tech": "Sci/Tech",
}

SENTIMENT_LABELS = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
    "label_0": "Negative",
    "label_1": "Neutral",
    "label_2": "Positive",
    "NEGATIVE": "Negative",
    "NEUTRAL": "Neutral",
    "POSITIVE": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
}

MARKET_TONES = {
    "Positive": "Bullish",
    "Neutral": "Neutral",
    "Negative": "Bearish",
}

SENTIMENT_COLORS = {
    "Positive": {"accent": "#2e7d32", "background": "#e8f5e9"},
    "Neutral": {"accent": "#f9a825", "background": "#fff8e1"},
    "Negative": {"accent": "#c62828", "background": "#ffebee"},
}

BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")


# ---------------------------------------------------------------------------
# 2. Manual text input for raw news
# ---------------------------------------------------------------------------
def get_news_input():
    """Collect raw news text from a text area."""
    st.subheader("Raw News Input")
    return st.text_area(
        "Paste the latest news content",
        height=180,
        placeholder="Enter a full raw news paragraph or article text here...",
    )


# ---------------------------------------------------------------------------
# 3. Load and cache the three pipelines
# ---------------------------------------------------------------------------
@st.cache_resource
def get_headline_generator():
    return pipeline("text2text-generation", model=HEADLINE_MODEL)


@st.cache_resource
def get_news_classifier():
    return pipeline("text-classification", model=NEWS_CLASSIFIER_MODEL)


@st.cache_resource
def get_sentiment_classifier():
    return pipeline("text-classification", model=SENTIMENT_MODEL)


# ---------------------------------------------------------------------------
# 4. Headline generation, classification, and sentiment scoring
# ---------------------------------------------------------------------------
def normalize_category(raw_label):
    return CATEGORY_LABELS.get(raw_label, raw_label)



def normalize_sentiment(raw_label):
    return SENTIMENT_LABELS.get(raw_label, raw_label)



def ensure_session_state():
    st.session_state.setdefault("news_items", [])



def generate_headline(news_text):
    """Generate a short title-like headline from raw news text."""
    generator = get_headline_generator()
    result = generator(news_text, max_length=30, min_length=6, do_sample=False)[0]["generated_text"]
    return " ".join(result.split()).strip()



def build_formatted_title(category, sentiment, headline):
    market_tone = MARKET_TONES.get(sentiment, "Neutral")
    return f"[{category} {market_tone} News] {headline}"



def analyze_news(news_text):
    generated_headline = generate_headline(news_text)

    classifier = get_news_classifier()
    sentiment_classifier = get_sentiment_classifier()

    category_result = classifier(generated_headline, truncation=True, max_length=512)[0]
    category = normalize_category(category_result["label"])
    category_confidence = float(category_result["score"])

    sentiment_result = sentiment_classifier(generated_headline, truncation=True, max_length=512)[0]
    sentiment = normalize_sentiment(sentiment_result["label"])
    sentiment_confidence = float(sentiment_result["score"])

    created_at = datetime.now(BEIJING_TIMEZONE)
    timestamp = created_at.strftime("%Y/%m/%d %H:%M")

    return {
        "id": str(uuid4()),
        "created_at": created_at.isoformat(),
        "timestamp": timestamp,
        "original_news": news_text,
        "headline": generated_headline,
        "category": category,
        "category_confidence": round(category_confidence, 4),
        "sentiment": sentiment,
        "sentiment_confidence": round(sentiment_confidence, 4),
    }


# ---------------------------------------------------------------------------
# 5. News board rendering
# ---------------------------------------------------------------------------
def render_sentiment_legend():
    legend_html = """
    <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:6px;">
        <span style="background:#e8f5e9; color:#2e7d32; padding:6px 10px; border-radius:999px; font-weight:600;">Bullish / Positive</span>
        <span style="background:#fff8e1; color:#f9a825; padding:6px 10px; border-radius:999px; font-weight:600;">Neutral</span>
        <span style="background:#ffebee; color:#c62828; padding:6px 10px; border-radius:999px; font-weight:600;">Bearish / Negative</span>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)



def render_news_card(item):
    sentiment = str(item["sentiment"])
    colors = SENTIMENT_COLORS.get(sentiment, SENTIMENT_COLORS["Neutral"])
    formatted_title = build_formatted_title(
        str(item["category"]),
        str(item["sentiment"]),
        str(item["headline"]),
    )

    card_html = f"""
    <div style="
        border-left: 8px solid {colors['accent']};
        background: {colors['background']};
        padding: 14px 16px;
        border-radius: 10px;
        margin: 10px 0 16px 0;
    ">
        <div style="display:flex; justify-content:space-between; gap:16px; align-items:flex-start;">
            <div style="font-weight:700; color:{colors['accent']}; font-size:1rem;">
                {escape(formatted_title)}
            </div>
            <div style="white-space:nowrap; color:#555; font-size:0.9rem;">
                {escape(str(item['timestamp']))} Beijing Time
            </div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    with st.expander("View details / Edit headline", expanded=False):
        st.write(f"**Category:** {item['category']}")
        st.write(f"**Sentiment:** {item['sentiment']}")
        st.write(f"**Original news:** {item['original_news']}")
        st.caption("Administrator override: manually edit the generated headline if the model output needs correction.")
        edited_title = st.text_input(
            "Edit headline",
            value=str(item["headline"]),
            key=f"headline_{item['id']}",
            label_visibility="collapsed",
        )
        if st.button("Save headline", key=f"save_{item['id']}"):
            cleaned_title = edited_title.strip()
            if not cleaned_title:
                st.warning("The edited headline cannot be empty.")
            else:
                item["headline"] = cleaned_title
                st.success("Headline updated.")
                st.rerun()



def render_board(items):
    if not items:
        st.info("No news items match the current filter yet.")
        return

    for item in items:
        render_news_card(item)


# ---------------------------------------------------------------------------
# Main: Streamlit UI – administrator news board demo
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Administrator News Board Demo",
        page_icon=":newspaper:",
        layout="wide",
    )
    ensure_session_state()

    st.title("Administrator News Board Demo")
    st.markdown(
        "This demo lets staff paste raw incoming news, generate a short headline, "
        "classify the headline, detect sentiment, and publish a formatted administrator news board item."
    )

    top_left, top_right = st.columns([3, 2])

    with top_left:
        news_text = get_news_input()
        if st.button("Generate and publish news", type="primary"):
            cleaned_news = news_text.strip()
            if not cleaned_news:
                st.warning("Please enter a news article before submitting.")
            else:
                with st.spinner("Running headline generation, classification, and sentiment analysis..."):
                    processed_item = analyze_news(cleaned_news)
                st.session_state["news_items"].insert(0, processed_item)
                st.success("News item added to the administrator board.")

    with top_right:
        st.subheader("Board Filter")
        selected_categories = st.multiselect(
            "Filter categories",
            options=CATEGORY_OPTIONS,
            default=CATEGORY_OPTIONS,
        )
        st.subheader("Sentiment Color Legend")
        render_sentiment_legend()
        with st.expander("Models used", expanded=False):
            st.caption(f"Headline generator: `{HEADLINE_MODEL}`")
            st.caption(f"News classifier: `{NEWS_CLASSIFIER_MODEL}`")
            st.caption(f"Sentiment model: `{SENTIMENT_MODEL}`")

    all_items = st.session_state["news_items"]
    filtered_items = [item for item in all_items if str(item["category"]) in selected_categories]

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Total news items", len(all_items))
    metric_col2.metric("Visible after filter", len(filtered_items))
    latest_time = filtered_items[0]["timestamp"] if filtered_items else "-"
    metric_col3.metric("Latest submission time", str(latest_time))

    st.subheader("Administrator News Board")
    render_board(filtered_items)


if __name__ == "__main__":
    main()
