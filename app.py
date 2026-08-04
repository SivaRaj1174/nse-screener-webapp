from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf
import threading
import time

app = Flask(__name__)

# Standard Large List of NIFTY & Major NSE Stocks to avoid network fetch crashes
BASE_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", 
    "BHARTIARTL.NS", "LTIM.NS", "KOTAKBANK.NS", "AXISBANK.NS", "LT.NS", "HCLTECH.NS", "BAJFINANCE.NS", "MARUTI.NS", 
    "SUNPHARMA.NS", "TATAMOTORS.NS", "ULTRACEMCO.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "TATASTEEL.NS", "JSWSTEEL.NS", 
    "M&M.NS", "TITAN.NS", "ADANIENT.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "DLF.NS",
    "WIPRO.NS", "TECHM.NS", "PERSISTENT.NS", "COFORGE.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "LUPIN.NS",
    "HEROMOTOCO.NS", "EICHERMOT.NS", "TVSMOTOR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS", "DABUR.NS",
    "JINDALSTEL.NS", "SAIL.NS", "VEDL.NS", "BANKBARODA.NS", "PNB.NS", "AUBANK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS"
]

CACHE = {
    "dates": [],
    "matrix": [],
    "total_scanned": 0,
    "stock_details": []
}

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

def safe_batch_scan():
    global CACHE
    try:
        all_signals = []
        batch_size = 30
        
        for i in range(0, len(BASE_STOCKS), batch_size):
            batch = BASE_STOCKS[i:i+batch_size]
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
            time.sleep(0.2)

        if all_signals:
            combined = pd.concat(all_signals, axis=1).fillna(False)
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

            CACHE = {
                "dates": dates_list,
                "matrix": [{
                    "Sector": "NSE EQUITIES SCREENER",
                    "CompanyCount": len(all_signals),
                    "Signals": [int((c / len(all_signals)) * 100) for c in recent_counts.values]
                }],
                "total_scanned": len(all_signals),
                "stock_details": details_list
            }
            print("Safe Batch Scan Successful!")
    except Exception as e:
        print("Error in Safe Batch Scan:", e)

# Run once on server boot
safe_batch_scan()

def periodic_rescan():
    while True:
        time.sleep(1800)
        safe_batch_scan()

threading.Thread(target=periodic_rescan, daemon=True).start()

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
    stocks = CACHE.get("stock_details", [])
    return render_template('details.html', sector="NSE EQUITIES", stocks=stocks)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
