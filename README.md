market_analysis is a collection of Python tools for screening and tracking Indian equities using technical and fundamental signals. It includes command-line scanners and backtesters for mid- and small-cap stocks, plus a small web dashboard for managing a watchlist and viewing scored “buy zones.” These scores are heuristic analysis, not investment advice; the scanner code also warns that backtests have data and survivorship-bias limitations.

Stack
Language(s): Python, with JavaScript, HTML, and CSS for the dashboard
Framework / runtime: FastAPI and Uvicorn for the web app; Python CLI scripts for scanning and backtesting
Notable libraries: yfinance for market data, pandas and numpy for analysis, SQLite for the dashboard watchlist
How it's organized
Text
midcap_momentum_scanner.py   Configurable mid/small-cap scanner emphasizing momentum and volume
midcap_scanner1.py           Stateless Nifty Midcap 100 scanner and backtester
midcap_scanner2.py           Earlier midcap scanner with SQLite-backed signals and backtest results
smallmom.py                  Small-cap momentum/volume scanner with additional liquidity and volatility filters
stock_tracker/
  app.py                     FastAPI routes and dashboard entry point
  analysis.py                Indicators, market structure, scoring, and chart data
  market.py                  Yahoo Finance fetches, ticker resolution, and in-memory caching
  store.py                   SQLite watchlist storage
  static/                    Browser dashboard (HTML, JavaScript, CSS)
  requirements.txt           Dashboard dependencies
  watchlist.db               Checked-in watchlist database
How it fits together: The scanner scripts independently fetch NSE constituents and Yahoo Finance data, calculate indicators and fundamental scores, then print ranked results; some support historical backtests. The dashboard follows a separate flow: app.py serves the static UI and API, store.py persists its watchlist, and market.py fetches/caches data for analysis.py to score and return to the browser.

How to run it
For the dashboard, install its declared dependencies and start the app from its directory:

bash
cd stock_tracker
pip install -r requirements.txt
python app.py
The app runs locally at 127.0.0.1:5000. For the standalone scanners, the repository has no root-level dependency manifest; install pandas, numpy, yfinance, and requests first (the scripts also use tqdm and colorama if installed). Then, from the repository root:

bash
python midcap_momentum_scanner.py
python smallmom.py
python midcap_scanner1.py --backtest
The scanners fetch market data over the network. Their backtests are explicitly caveated in the code: current Yahoo Finance fundamentals and today’s constituent lists can make historical results optimistic.

Try asking
How does smallmom.py combine its momentum and volume scores?
What are the main differences between midcap_scanner2.py and midcap_momentum_scanner.py?
How does stock_tracker/analysis.py determine a stock’s buy band and invalidation level?
