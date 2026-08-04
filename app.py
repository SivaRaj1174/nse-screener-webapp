from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf
import threading
import time

app = Flask(__name__)

# Global In-Memory Cache
CACHE = {
    "dates": [],
    "matrix": [],
    "total_scanned": 0,
    "last_updated": "Scanning in progress...",
    "stock_details": {}
}

def get_all_nse_symbols():
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df = pd.read_csv(url)
        df_eq = df[df[' SERIES'] == 'EQ']
        symbols = [f"{sym}.NS" for sym in df_eq['SYMBOL'].tolist()]
        return symbols
    except Exception:
        # Top High Volume NSE Stocks fallback
        return [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
            "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LTIM.NS", "KOTAKBANK.NS", "AXISBANK.NS",
            "LT.NS", "HCLTECH.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
            "ULTRACEMCO.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "TATASTEEL.NS", "JSWSTEEL.NS",
            "M&M.NS", "TITAN.NS", "ADANIENT.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "BPCL.NS"
        ]

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

def run_background_scanner():
    """ Runs continuously in background to keep data fresh without blocking web requests """
    global CACHE
    while True:
        try:
            symbols = get_all_nse_symbols()
            batch_size = 200
            all_signals = []
            
            # Batch downloading to avoid memory overload
            for i in range(0, min(len(symbols), 1200), batch_size):
                batch = symbols[i:i+batch_size]
                data = yf.download(batch, period="1mo", interval="1d", group_by='ticker', progress=False)
                
                for ticker in batch:
                    try:
                        clean_sym = ticker.replace(".NS", "")
                        df = data[ticker].dropna() if len(batch) > 1 else data.dropna()
                        if not df.empty and len(df) > 5:
                            sig = calculate_signals(df)
                            sig.name = clean_sym
                            all_signals.append(sig)
                    except Exception:
                        continue

            if all_signals:
                combined = pd.concat(all_signals, axis=1).fillna(False)
                daily_counts = combined.sum(axis=1)
                recent_counts = daily_counts.tail(5).iloc[::-1]
                dates_list = [d.strftime('%b %d') for d in recent_counts.index]
                
                latest_date = combined.index[-1]
                latest_signals = combined.loc[latest_date]
                
                stock_details = []
                for stock in combined.columns:
                    is_buy = bool(latest_signals.get(stock, False))
                    stock_details.append({
                        "symbol": stock,
                        "status": "🟢 BUY SIGNAL" if is_buy else "⚪ NO SIGNAL",
                        "tv_link": f"https://in.tradingview.com/chart/?symbol=NSE:{stock}"
                    })

                CACHE = {
                    "dates": dates_list,
                    "matrix": [{
                        "Sector": "ALL NSE EQUITIES",
                        "CompanyCount": len(all_signals),
                        "Signals": [int((c / len(all_signals)) * 100) for c in recent_counts.values]
                    }],
                    "total_scanned": len(symbols),
                    "last_updated": time.strftime("%H:%M:%S IST"),
                    "stock_details": stock_details
                }
        except Exception as e:
            print("Background scan error:", e)
            
        # Refresh background scan every 15 minutes
        time.sleep(900)

# Start Background Thread
scanner_thread = threading.Thread(target=run_background_scanner, daemon=True)
scanner_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/screener-data')
def screener_data():
    return jsonify(CACHE)

@app.route('/details')
def details():
    stocks = CACHE.get("stock_details", [])
    return render_template('details.html', sector="ALL NSE EQUITIES", stocks=stocks)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
