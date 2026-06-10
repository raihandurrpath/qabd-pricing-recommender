# Quotation Pricing Decision Support Dashboard

This Streamlit dashboard supports the **Gross Profit vs. Competitor, Pricing Competitiveness, and Conversion Analysis** final project.

## Main purpose
The dashboard helps the company review quotation pricing scenarios by combining:

- proposed gross-margin rate;
- model-estimated win probability;
- historical win-rate benchmarks;
- competitor-price positioning;
- expected gross profit;
- product-specific margin recommendations;
- quotation review flags;
- data-quality and model-validation notes.

The system is intended for **price review**, not automatic approval. Managerial judgement remains necessary, especially for low-margin quotations, missing competitor information, and multi-product quotations.

## Files

- `app.py`: Streamlit dashboard
- `df_preprocessed.csv`: preprocessed project dataset
- `requirements.txt`: Python dependencies
- `run_dashboard.bat`: optional Windows launcher

## Run locally

1. Install Python 3.10 or later.
2. Open a terminal in this folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app.py
```

5. Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Dashboard sections

### 1. Executive Overview
Provides historical conversion summaries, product-level win rates, gross-margin bands, and potential lost-tender review signals.

### 2. Quote Recommender
Allows the user to input product, quantity, estimated cost, proposed gross-margin rate, energy grant, and optional competitor prices. The app calculates:

- proposed unit price;
- estimated win probability;
- gross profit if won;
- expected gross profit;
- recommended gross-margin rate;
- recommended unit price;
- rule-based review flags;
- competitor positioning;
- a what-if curve across margin scenarios.

### 3. Margin Sweet Spot
Shows historical win rates and a historical expected-gross-profit proxy by margin range, overall or by product.

### 4. Competitor Positioning
Compares historical outcomes when the company price is below, approximately aligned with, or above the minimum available competitor benchmark.

### 5. Data Quality & Model Notes
Explains preprocessing decisions, competitor-information gaps, model validation metrics, limitations, and recommended future data improvements.

## Important analytical notes

- The prediction model is an explainable logistic-regression baseline.
- Evaluation uses a Quote-ID grouped holdout split.
- Each Quote ID is weighted equally during training.
- Quantity values less than or equal to zero are excluded as system errors.
- Negative-margin records are retained because they may reflect strategic relationship-building deals.
- Missing competitor prices are not imputed as zero.
- Competitor analysis is retrospective and should only be used operationally when competitor information is available during price review.
- Product-line recommendations do not replace full quotation-level managerial review for multi-product quotations.
