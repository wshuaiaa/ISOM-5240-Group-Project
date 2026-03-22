# Administrator News Board Demo for ISOM5240

This project builds a three-stage Hugging Face pipeline for a Bloomberg-style administrator news board. Staff paste raw incoming news into a text box, the system generates a short headline, classifies the headline, detects market sentiment, and publishes a formatted board item.

## Business problem

Financial media staff often receive long raw news text that is not immediately suitable for publication on a fast-moving news board. The business need is to quickly convert raw news into a concise market-facing headline, assign the correct category, estimate market tone, and display the result in a readable board format.

## Final pipeline

1. `headline-generation`
   - Convert full raw news text into a short headline-style summary
   - Selected model: `JulesBelveze/t5-small-headline-generator`

2. `text-classification`
   - Classify each generated headline into `World`, `Sports`, `Business`, or `Sci/Tech`
   - Selected fine-tuned model: `wshuaiaa/News_classifier_Finetuned`

3. `sentiment-analysis`
   - Score the generated headline as `Negative`, `Neutral`, or `Positive`
   - Selected fine-tuned model: `wshuaiaa/News_sentiment_Finetuned`
   - Convert sentiment into board labels:
     - `Negative -> Bearish`
     - `Neutral -> Neutral`
     - `Positive -> Bullish`

## Final administrator board format

Each published item is shown in this style:

```text
[Business Bearish News] Generated headline here
```

The board uses sentiment colors:

- Green for bullish / positive
- Yellow for neutral
- Red for bearish / negative

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

- Keeps supporting experiment code and pipeline testing
- Can still be used as a scratch notebook for additional testing if needed

### `debug_summary_model.ipynb`

- Tests the selected headline generation model
- Generates a short headline from raw news text
- Measures runtime
- Helps compare output quality for the first stage of the pipeline

## Streamlit app behavior

The app uses a manual text input box, not file upload.

The app will:

1. Accept raw incoming news text from a text area
2. Generate a short headline from the raw text
3. Classify the generated headline into one of four topic categories
4. Predict sentiment from the generated headline
5. Display the result in the administrator news board
6. Format the title as `[Category MarketTone News] Headline`
7. Allow manual headline editing by the administrator

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
```

## Deploy to Streamlit Cloud

Recommended method:

1. Create a new GitHub repository using the contents of this `project/` folder as the repository root.
2. Upload the fine-tuned news classifier and sentiment model to Hugging Face Hub.
3. In Streamlit Cloud, connect the GitHub repository.
4. Set the main file to `app.py`.
5. Add `HEADLINE_MODEL`, `NEWS_CLASSIFIER_MODEL`, and `SENTIMENT_MODEL` in Streamlit Cloud settings or secrets.

## Notes

- The first-stage headline generation model is a selected pretrained model and is not fine-tuned in the current project.
- The two fine-tuned models are the news classification model and the sentiment model.
- Keep the Hugging Face model URLs, GitHub URL, and Streamlit app URL consistent in the final report.
