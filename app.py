from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf
import threading
import time

app = Flask(__name__)

CACHE = {
    "dates": [],
    "matrix": [],
    "total_scanned": 0,
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
        # Fallback list if NSE archive CSV fails
        return [f"STOCK{i}.NS" for i in range(500)]

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

def scan_all_nse_stocks_background():
    global CACHE
    while True:
        try:
            symbols = get_all_nse_symbols()
            batch_size = 100
            all_scanned_signals = []
            
            # Batching to bypass memory limit on free servers
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                try:
                    data = yf.download(batch, period="1mo", interval="1d", group_by='ticker', progress=False)
                    for ticker in batch:
                        try:
                            clean_sym = ticker.replace(".NS", "")
                            df = data[ticker].dropna() if len(batch) > 1 else data.dropna()
                            if not df.empty and len(df) > 5:
                                sig = calculate_signals(df)
                                sig.name = clean_sym
                                all_scanned_signals.append(sig)
                        except Exception:
                            continue
                except Exception:
                    continue
                time.sleep(0.1) # Small pause to keep CPU usage low

            if all_scanned_signals:
                combined = pd.concat(all_scanned_signals, axis=1).fillna(False)
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
                        "Sector": "ALL NSE EQUITIES",
                        "CompanyCount": len(all_scanned_signals),
                        "Signals": [int((c / len(all_scanned_signals)) * 100) for c in recent_counts.values]
                    }],
                    "total_scanned": len(symbols),
                    "stock_details": {"ALL NSE EQUITIES": details_list}
                }
                print(f"Successfully scanned {len(all_scanned_signals)} / {len(symbols)} stocks!")
        except Exception as e:
            print("Background Scanner Error:", e)
            
        time.sleep(1800) # Re-scan every 30 minutes

# Start Background Processing
bg_thread = threading.Thread(target=scan_all_nse_stocks_background, daemon=True)
bg_thread.start()

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
    sector_name = request.args.get('sector', 'ALL NSE EQUITIES')
    details_map = CACHE.get("stock_details", {})
    stocks = details_map.get(sector_name, [])
    return render_template('details.html', sector=sector_name, stocks=stocks)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
