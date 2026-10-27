# Bitcoin Price and Tweet Sentiment Analysis

## Project Overview

This project analyzes the relationship between Bitcoin market prices and sentiment expressed in Bitcoin-related tweets. It combines historical tweet data with Bitcoin price data to calculate sentiment scores, examine daily price returns, and visualize potential relationships between market activity and public sentiment.

## Technologies Used

- Python 3.12
- Pandas — data processing and analysis
- Matplotlib — data visualization
- VADER Sentiment — text sentiment analysis
- yfinance — Bitcoin price data retrieval

## Project Structure

```text
CrowdWisdomTrading/
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   └── figures/
├── src/
├── main.py
├── requirements.txt
└── README.md
```

## Installation

1. Install Python 3.12 or a compatible Python version.
2. Open a terminal in the project directory.
3. Install the required packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

## Running the Project

Run the main program:

```bash
python main.py
```

To run the existing price and sentiment relationship analysis:

```bash
python src/analyze_relationship.py
```

## Outputs

The project generates processed datasets and visualizations, including:

- Bitcoin closing-price chart
- Bitcoin daily-returns chart
- Daily tweet-sentiment chart
- Daily tweet-volume chart
- Sentiment versus Bitcoin-returns chart

Generated datasets are stored in `data/processed/`, and visualizations are stored in `reports/figures/`.

## Analysis Note

Correlation measures statistical association, not causation. A relationship between tweet sentiment and Bitcoin returns does not by itself establish that sentiment predicts or causes price movements.