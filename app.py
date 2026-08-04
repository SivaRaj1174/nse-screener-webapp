from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd
import numpy as np
import os
import time
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

CACHE_FILE = "nse_data_cache.csv"

# Sector Mapping
SECTORS = {
    "NIFTY BANK": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "BANKBARODA", "PNB", "AUBANK", "FEDERALBNK"],
    "NIFTY IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "PERSISTENT", "COFORGE"],
    "NIFTY AUTO": ["TATAMOTORS", "MARUTI", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "TVSMOTOR"],
    "NIFTY PHARMA": ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN", "TORNTPHARM", "ALKEM"],
    "NIFTY FMCG": ["ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR", "GODREJCP"],
    "NIFTY METAL": ["TATASTEEL", "JINDALSTEL", "HINDALCO", "NMDC", "SAIL", "NATIONALUM", "VEDL"],
    "NIFTY ENERGY": ["RELIANCE", "NTPC", "POWERGRID", "ONGC", "BPCL", "IOC", "GAIL"],
    "NIFTY REALTY": ["DLF", "GODREJPROP", "OBEROIRLTY", "PHOENIXLTD", "BRIGADE"]
}

# Browser Headers to bypass NSE IP Blocking
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br'
}

def fetch_nse_stock_history(symbol):
    """ Direct Scraping from NSE India API with Session Cookies """
    session = requests.Session()
    session.headers.update(HEADERS)
    
    try:
        # Step 1: Initial call to capture NSE session cookies
        session.get("https://www.nseindia.com", timeout=10)
        
        # Step 2: Fetch Stock Historical Data directly from NSE
        url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}"
        response = session.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                df = pd.DataFrame(data['data'])
                df = df[['CH_TIMESTAMP', 'CH_OPENING_PRICE', 'CH_TRADE_HIGH_PRICE', 'CH_TRADE_LOW_PRICE', 'CH_CLOSING_PRICE']]
                df.columns = ['Date', 'Open', 'High', 'Low', 'Close']
                df['Date'] = pd.to_datetime(df['Date'])
                df.sort_values('Date', inplace=True)
                return df
    except Exception as e:
        print(f"Error fetching {symbol} from NSE: {e}")
    return pd.DataFrame()

def calculate_signals(df):
    try:
        df = df.copy()
        ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        ha_open = pd.Series(index=df.index, dtype=float)
        ha_open.iloc[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
        
        for i in range(1, len(df)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
            
        ha_high = np.maximum(df['High'], np.maximum(ha_open, ha_close))
        ha_body = np.abs(ha_close - ha_open)
        nrml_body = np.abs(df['Close'] - df['Open'])
        
        buy_cond = (ha_body.shift(2) > ha_body.shift(1)) & (nrml_body.shift(2) < nrml_body.shift(1)) & (ha_open.shift(2) < ha_close.shift(2))
        sell_cond = (ha_body.shift(2) > ha_body.shift(1)) & (nrml_body.shift(2) < nrml_body.shift(1)) & (ha_open.shift(2) > ha_close.shift(2))
        
        saved_buy_price = pd.Series(np.where(buy_cond, ha_high.shift(1), np.nan), index=df.index).ffill()
        saved_sell_exit_price = pd.Series(np.where(sell_cond, ha_high.shift(1), np.nan), index=df.index).ffill()
        
        raw_buy = (df['Close'] > saved_buy_price) & (df['Close'] > saved_sell_exit_price)
        return raw_buy.fillna(False)
    except Exception:
        return pd.Series([False]*len(df), index=df.index)

def update_nse_cache():
    print("Collecting direct NSE data into local storage...")
    records = []
    
    for sector, tickers in SECTORS.items():
        for symbol in tickers:
            df = fetch_nse_stock_history(symbol)
            if not df.empty and len(df) > 5:
                sigs = calculate_signals(df)
                for dt, buy_val in sigs.items():
                    records.append({
                        "Date": dt.strftime('%Y-%m-%d'),
                        "Sector": sector,
                        "Symbol": symbol,
                        "BuySignal": int(buy_val)
                    })
            time.sleep(0.5)  # Delay to respect NSE rate limits
            
    if records:
        cache_df = pd.DataFrame(records)
        cache_df.to_csv(CACHE_FILE, index=False)
        print("NSE Local Cache updated successfully!")

if not os.path.exists(CACHE_FILE):
    update_nse_cache()

scheduler = BackgroundScheduler()
scheduler.add_job(func=update_nse_cache, trigger="interval", hours=6)
scheduler.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/screener-data')
def screener_data():
    if not os.path.exists(CACHE_FILE):
        return jsonify({"dates": [], "matrix": [], "total_scanned": 0})
    
    df = pd.read_csv(CACHE_FILE)
    if df.empty:
        return jsonify({"dates": [], "matrix": [], "total_scanned": 0})

    recent_dates = sorted(df['Date'].unique())[-5:]
    formatted_dates = [pd.to_datetime(d).strftime('%b %d') for d in reversed(recent_dates)]
    
    matrix = []
    total_scanned_count = len(df['Symbol'].unique())

    for sector, tickers in SECTORS.items():
        sec_df = df[df['Sector'] == sector]
        signals_pct = []
        
        for dt in reversed(recent_dates):
            day_df = sec_df[sec_df['Date'] == dt]
            buy_count = day_df['BuySignal'].sum()
            total_stocks = len(tickers)
            pct = int((buy_count / total_stocks) * 100) if total_stocks > 0 else 0
            signals_pct.append(pct)
            
        matrix.append({
            "Sector": sector,
            "CompanyCount": len(tickers),
            "Signals": signals_pct
        })

    return jsonify({
        "dates": formatted_dates,
        "matrix": matrix,
        "total_scanned": total_scanned_count
    })

@app.route('/details')
def details():
    sector_name = request.args.get('sector', '')
    stocks_details = []
    
    if os.path.exists(CACHE_FILE):
        df = pd.read_csv(CACHE_FILE)
        sec_df = df[df['Sector'] == sector_name]
        
        if not sec_df.empty:
            latest_date = sec_df['Date'].max()
            day_df = sec_df[sec_df['Date'] == latest_date]
            
            for _, row in day_df.iterrows():
                stocks_details.append({
                    "symbol": row['Symbol'],
                    "status": "🟢 BUY SIGNAL" if row['BuySignal'] == 1 else "⚪ NO SIGNAL",
                    "tv_link": f"https://in.tradingview.com/chart/?symbol=NSE:{row['Symbol']}"
                })

    return render_template('details.html', sector=sector_name, stocks=stocks_details)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
