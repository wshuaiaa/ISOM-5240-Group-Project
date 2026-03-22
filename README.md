# Country Market Sentiment Briefing for ISOM5240

This project builds a four-stage Hugging Face pipeline for investment firms that want to understand the daily sentiment of a country's financial market from raw international news.

## Business problem

Investment firms read news from many countries and many news sources every day. Raw news is often too long and noisy for fast market assessment. The business need is to convert raw news into short usable headlines, classify the headlines, estimate sentiment, and summarize one country's daily market mood in a concise briefing.

## Final pipeline

1. `headline-generation`
   - Convert full raw news text into a short headline-style summary
   - Selected model: `JulesBelveze/t5-small-headline-generator`

2. `text-classification`
   - Classify each generated headline into `World`, `Sports`, `Business`, or `Sci/Tech`
   - Selected fine-tuned model: `wshuaiaa/News_classifier_Finetuned`
   - Remove `Sports` headlines from the downstream market pipeline

3. `sentiment-analysis`
   - Score the generated headline as `Negative`, `Neutral`, or `Positive`
   - Selected fine-tuned model: `wshuaiaa/News_sentiment_Finetuned`

4. `text2text-generation`
   - Generate a daily country briefing from processed headlines and average sentiment
   - Current model: `google/flan-t5-small`

## Dataset

- Dataset: `rajkumar4466/ag-news-sentiment`
- Features:
  - `text`
  - `label`
  - `sentiment`
- Topic labels:
  - `World`
  - `Sports`
  - `Business`
  - `Sci/Tech`
- Sentiment labels:
  - `0 = Negative`
  - `1 = Neutral`
  - `2 = Positive`

## Selected models

- Headline generation model: `JulesBelveze/t5-small-headline-generator`
- Fine-tuned news classifier: `wshuaiaa/News_classifier_Finetuned`
- Fine-tuned sentiment model: `wshuaiaa/News_sentiment_Finetuned`
- Daily briefing model: `google/flan-t5-small`

## Project files

```text
project/
├── 01_finetune_news_classifier.ipynb
├── 02_finetune_sentiment_model.ipynb
├── 03_pipeline_experiments_and_daily_briefing.ipynb
├── debug_summary_model.ipynb
├── app.py
├── requirements.txt
├── README.md
└── Project_Report_Guide.md
```

## Notebook summary

### `01_finetune_news_classifier.ipynb`

- Fine-tunes the news topic classification model
- Uses the `label` column from `rajkumar4466/ag-news-sentiment`
- Uses `train[:5000]` and `test[:1000]`
- Saves the final model locally
- Includes optional Hugging Face Hub upload steps

### `02_finetune_sentiment_model.ipynb`

- Fine-tunes the sentiment model
- Uses the `sentiment` column from `rajkumar4466/ag-news-sentiment`
- Rebalances the training split before fine-tuning
- Uses `train[:5000]` and `test[:1000]`
- Saves the final model locally
- Includes optional Hugging Face Hub upload steps

### `03_pipeline_experiments_and_daily_briefing.ipynb`

- Evaluates saved models on test data
- Measures accuracy and runtime
- Demonstrates the full end-to-end pipeline

### `debug_summary_model.ipynb`

- Tests the selected headline generation model
- Generates a short headline from raw news text
- Measures runtime
- Helps compare output quality for the first stage of the pipeline

## Streamlit app behavior

The app expects an Excel file (`.xlsx`) with:

- Column 1: raw news text
- Column 2: country

The app will:

1. Generate a short headline from the raw news text
2. Classify the generated headline into one of four topic categories
3. Remove `Sports` items from the downstream pipeline
4. Predict sentiment from the generated headline
5. Aggregate the results for a selected country
6. Generate a daily country briefing

## Run the Streamlit app locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Environment variables for deployment

You can override the default models with environment variables:

```bash
export HEADLINE_MODEL="JulesBelveze/t5-small-headline-generator"
export NEWS_CLASSIFIER_MODEL="wshuaiaa/News_classifier_Finetuned"
export SENTIMENT_MODEL="wshuaiaa/News_sentiment_Finetuned"
export BRIEFING_MODEL="google/flan-t5-small"
```

## Deploy to Streamlit Cloud

Recommended method:

1. Create a new GitHub repository using the contents of this `project/` folder as the repository root.
2. Upload the fine-tuned news classifier and sentiment model to Hugging Face Hub.
3. In Streamlit Cloud, connect the GitHub repository.
4. Set the main file to `app.py`.
5. Add `HEADLINE_MODEL`, `NEWS_CLASSIFIER_MODEL`, `SENTIMENT_MODEL`, and `BRIEFING_MODEL` in Streamlit Cloud settings or secrets.

## Notes

- The first-stage headline generation model is a selected pretrained model and is not fine-tuned in the current project.
- The two fine-tuned models are the news classification model and the sentiment model.
- Keep the Hugging Face model URLs, GitHub URL, and Streamlit app URL consistent in the final report.
