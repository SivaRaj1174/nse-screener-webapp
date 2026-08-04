from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf
import threading
import time

app = Flask(__name__)

# Cache object to store scanned results instantly in memory
CACHE = {
    "dates": [],
    "matrix": [],
    "total_scanned": 0,
    "stock_details": {}
}

SECTORS = {
    "NIFTY BANK": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "AUBANK.NS", "FEDERALBNK.NS"],
    "NIFTY IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS"],
    "NIFTY AUTO": ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "TVSMOTOR.NS"],
    "NIFTY PHARMA": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "LUPIN.NS", "TORNTPHARM.NS", "ALKEM.NS"],
    "NIFTY FMCG": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS", "DABUR.NS", "GODREJCP.NS"],
    "NIFTY METAL": ["TATASTEEL.NS", "JINDALSTEL.NS", "HINDALCO.NS", "NMDC.NS", "SAIL.NS", "NATIONALUM.NS", "VEDL.NS"],
    "NIFTY ENERGY": ["RELIANCE.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS"],
    "NIFTY REALTY": ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS", "BRIGADE.NS"]
}

ALL_TICKERS = [t for tickers in SECTORS.values() for t in tickers]

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

def refresh_scanner():
    global CACHE
    try:
        print("Fetching data via bulk yfinance...")
        data = yf.download(ALL_TICKERS, period="1mo", interval="1d", group_by='ticker', progress=False)
        
        matrix = []
        dates_list = []
        stock_details_map = {}
        
        for sector, tickers in SECTORS.items():
            sector_signals = []
            for ticker in tickers:
                try:
                    df = data[ticker].dropna() if len(ALL_TICKERS) > 1 else data.dropna()
                    if not df.empty and len(df) > 5:
                        sig = calculate_signals(df)
                        clean_sym = ticker.replace(".NS", "")
                        sig.name = clean_sym
                        sector_signals.append(sig)
                except Exception:
                    continue
            
            if sector_signals:
                combined = pd.concat(sector_signals, axis=1).fillna(False)
                daily_counts = combined.sum(axis=1)
                recent_counts = daily_counts.tail(5).iloc[::-1]
                dates_list = [d.strftime('%b %d') for d in recent_counts.index]
                
                latest_date = combined.index[-1]
                latest_day_sig = combined.loc[latest_date]
                
                details_list = []
                for stock in combined.columns:
                    is_buy = bool(latest_day_sig.get(stock, False))
                    details_list.append({
                        "symbol": stock,
                        "status": "🟢 BUY SIGNAL" if is_buy else "⚪ NO SIGNAL",
                        "tv_link": f"https://in.tradingview.com/chart/?symbol=NSE:{stock}"
                    })
                
                stock_details_map[sector] = details_list
                
                row = {
                    "Sector": sector,
                    "CompanyCount": len(tickers),
                    "Signals": [int((c / len(tickers)) * 100) for c in recent_counts.values]
                }
                matrix.append(row)

        CACHE = {
            "dates": dates_list,
            "matrix": matrix,
            "total_scanned": len(ALL_TICKERS),
            "stock_details": stock_details_map
        }
        print("Scanner cache refreshed successfully!")
    except Exception as e:
        print("Scanner refresh error:", e)

# Run initial load on startup
refresh_scanner()

def background_timer():
    while True:
        time.sleep(1800)  # Refresh every 30 mins
        refresh_scanner()

threading.Thread(target=background_timer, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/screener-data')
def screener_data():
    return jsonify({
        "dates": CACHE.get("dates", []),
        "matrix": CACHE.get("matrix", []),
        "total_scanned": CACHE.get("total_scanned", 0)
    })

@app.route('/details')
def details():
    sector_name = request.args.get('sector', '')
    details_map = CACHE.get("stock_details", {})
    stocks = details_map.get(sector_name, [])
    return render_template('details.html', sector=sector_name, stocks=stocks)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
