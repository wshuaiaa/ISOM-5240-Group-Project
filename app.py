import os
from datetime import date

import pandas as pd
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
BRIEFING_MODEL = os.getenv(
    "BRIEFING_MODEL",
    "google/flan-t5-small",
)

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

SENTIMENT_SCORES = {
    "Negative": -1,
    "Neutral": 0,
    "Positive": 1,
}

BRIEFING_CATEGORIES = ["World", "Business", "Sci/Tech"]


# ---------------------------------------------------------------------------
# 2. File input: upload Excel daily news file
# ---------------------------------------------------------------------------
def get_uploaded_news():
    """
    Upload a daily Excel file. The first column should be the raw news text
    and the second column should be the country. Returns a cleaned DataFrame
    or None.
    """
    st.subheader("Daily News Input")
    uploaded_file = st.file_uploader(
        "Upload an Excel file (.xlsx)",
        type=["xlsx"],
        help="Column 1 = raw news text, Column 2 = country.",
    )
    if uploaded_file is None:
        return None

    df = pd.read_excel(uploaded_file)
    if df.shape[1] < 2:
        st.error("The uploaded Excel file must have at least two columns: raw news text and country.")
        return None

    working_df = df.iloc[:, :2].copy()
    working_df.columns = ["news_text", "country"]
    working_df["news_text"] = working_df["news_text"].astype(str).str.strip()
    working_df["country"] = working_df["country"].astype(str).str.strip()
    working_df = working_df[(working_df["news_text"] != "") & (working_df["country"] != "")]
    working_df = working_df.reset_index(drop=True)

    if working_df.empty:
        st.error("The uploaded file does not contain usable news text and country values.")
        return None

    return working_df


# ---------------------------------------------------------------------------
# 3. Load and cache the four pipelines
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


@st.cache_resource
def get_briefing_generator():
    return pipeline("text2text-generation", model=BRIEFING_MODEL)


# ---------------------------------------------------------------------------
# 4. Headline generation, classification, and sentiment scoring
# ---------------------------------------------------------------------------
def normalize_category(raw_label):
    return CATEGORY_LABELS.get(raw_label, raw_label)



def normalize_sentiment(raw_label):
    return SENTIMENT_LABELS.get(raw_label, raw_label)



def generate_headline(news_text):
    """Generate a short headline from raw news text."""
    generator = get_headline_generator()
    result = generator(news_text, max_length=30, min_length=6, do_sample=False)[0]["generated_text"]
    return " ".join(result.split()).strip()



def classify_and_score_news(news_df):
    """
    First generate a short headline from raw news text, then classify the
    generated headline, remove Sports items, and score sentiment.
    """
    classifier = get_news_classifier()
    sentiment_classifier = get_sentiment_classifier()

    rows = []
    for _, row in news_df.iterrows():
        generated_headline = generate_headline(row["news_text"])
        category_result = classifier(generated_headline, truncation=True, max_length=512)[0]
        category = normalize_category(category_result["label"])

        if category == "Sports":
            continue

        sentiment_result = sentiment_classifier(generated_headline, truncation=True, max_length=512)[0]
        sentiment = normalize_sentiment(sentiment_result["label"])

        rows.append(
            {
                "original_news_text": row["news_text"],
                "generated_headline": generated_headline,
                "country": row["country"],
                "category": category,
                "category_confidence": float(category_result["score"]),
                "sentiment": sentiment,
                "sentiment_confidence": float(sentiment_result["score"]),
                "sentiment_value": SENTIMENT_SCORES.get(sentiment, 0),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Build prompt for the daily country briefing
# ---------------------------------------------------------------------------
def build_section_summary(processed_df, category_name):
    category_df = processed_df[processed_df["category"] == category_name]
    if category_df.empty:
        return f"{category_name}: No major relevant headlines."

    positive_items = category_df[category_df["sentiment"] == "Positive"]["generated_headline"].head(3).tolist()
    negative_items = category_df[category_df["sentiment"] == "Negative"]["generated_headline"].head(3).tolist()
    neutral_items = category_df[category_df["sentiment"] == "Neutral"]["generated_headline"].head(2).tolist()

    return (
        f"{category_name}: "
        f"Positive headlines: {positive_items}. "
        f"Negative headlines: {negative_items}. "
        f"Neutral headlines: {neutral_items}."
    )



def interpret_average_sentiment(value):
    if value >= 0.25:
        return "Overall sentiment is positive."
    if value <= -0.25:
        return "Overall sentiment is negative."
    return "Overall sentiment is neutral."



def build_briefing_prompt(processed_df, selected_country, report_date):
    average_sentiment = round(processed_df["sentiment_value"].mean(), 4) if not processed_df.empty else 0.0

    world_summary = build_section_summary(processed_df, "World")
    business_summary = build_section_summary(processed_df, "Business")
    sci_tech_summary = build_section_summary(processed_df, "Sci/Tech")
    overall_sentiment = interpret_average_sentiment(average_sentiment)

    prompt = f"""
Write a daily market briefing for {selected_country} on {report_date.isoformat()}.
Average sentiment score: {average_sentiment}.
{world_summary}
{business_summary}
{sci_tech_summary}
Instructions:
- Write exactly three short paragraphs.
- Paragraph 1 must focus on World and politics.
- Paragraph 2 must focus on Business and market developments.
- Paragraph 3 must focus on Sci/Tech developments.
- In each paragraph explain the good news and the bad news when available.
- End the last paragraph with the overall market sentiment.
- The overall market sentiment is: {overall_sentiment}
""".strip()

    return prompt, average_sentiment


# ---------------------------------------------------------------------------
# 6. Daily briefing generation
# ---------------------------------------------------------------------------
def generate_daily_briefing(prompt):
    generator = get_briefing_generator()
    result = generator(prompt, max_new_tokens=260, do_sample=False)[0]["generated_text"]
    return result.strip()


# ---------------------------------------------------------------------------
# Main: Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Country Market Sentiment Briefing",
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.title("Country Market Sentiment Briefing")
    st.markdown(
        "Upload a daily news Excel file. The app will first generate a short headline, "
        "then classify the headline, score sentiment, and generate a country-level daily briefing."
    )

    with st.sidebar:
        st.subheader("Models Used")
        st.caption(f"Headline generator: `{HEADLINE_MODEL}`")
        st.caption(f"News classifier: `{NEWS_CLASSIFIER_MODEL}`")
        st.caption(f"Sentiment model: `{SENTIMENT_MODEL}`")
        st.caption(f"Briefing model: `{BRIEFING_MODEL}`")

    news_df = get_uploaded_news()
    if news_df is None:
        st.info("Please upload a daily news Excel file to continue.")
        return

    st.subheader("Uploaded Data Preview")
    st.dataframe(news_df.head(10), use_container_width=True)

    countries = sorted(news_df["country"].dropna().unique().tolist())
    selected_country = st.selectbox("Select country", countries)
    report_date = st.date_input("Select report date", value=date.today())

    if st.button("Generate daily briefing", type="primary"):
        country_df = news_df[news_df["country"] == selected_country].reset_index(drop=True)
        if country_df.empty:
            st.warning("No news items are available for the selected country.")
            return

        with st.spinner("Running headline generation, classification, and sentiment analysis..."):
            processed_df = classify_and_score_news(country_df)

        if processed_df.empty:
            st.warning("All generated headlines were filtered out. No non-sports headlines remain for this country.")
            return

        with st.spinner("Generating the daily briefing..."):
            prompt, average_sentiment = build_briefing_prompt(processed_df, selected_country, report_date)
            briefing = generate_daily_briefing(prompt)

        st.subheader("Pipeline Result")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Uploaded news items", len(country_df))
        metric_col2.metric("Relevant headlines kept", len(processed_df))
        metric_col3.metric("Average sentiment", f"{average_sentiment:.2f}")

        st.subheader("Generated Headlines and Analysis")
        st.dataframe(processed_df, use_container_width=True)

        st.subheader("Category Breakdown")
        st.bar_chart(processed_df["category"].value_counts().reindex(BRIEFING_CATEGORIES, fill_value=0))

        st.subheader("Sentiment Breakdown")
        st.bar_chart(
            processed_df["sentiment"].value_counts().reindex(
                ["Positive", "Neutral", "Negative"],
                fill_value=0,
            )
        )

        st.subheader("Generated Daily Briefing")
        st.write(briefing)

        with st.expander("Prompt used for generation", expanded=False):
            st.code(prompt)


if __name__ == "__main__":
    main()
