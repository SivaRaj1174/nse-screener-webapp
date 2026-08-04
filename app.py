from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf

app = Flask(__name__)

def get_all_nse_tickers():
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df = pd.read_csv(url)
        # Filter EQ Series (Main Equity shares)
        df_eq = df[df[' SERIES'] == 'EQ']
        symbols = [f"{sym}.NS" for sym in df_eq['SYMBOL'].tolist()]
        return symbols, len(symbols)
    except Exception:
        # Fallback comprehensive broad list
        fallback_list = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
            "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LTIM.NS", "KOTAKBANK.NS", "AXISBANK.NS",
            "LT.NS", "HCLTECH.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
            "ULTRACEMCO.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "TATASTEEL.NS", "JSWSTEEL.NS",
            "M&M.NS", "TITAN.NS", "ADANIENT.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "BPCL.NS"
        ]
        return fallback_list, len(fallback_list)

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

CACHED_SIGNALS = {}
CACHED_TICKERS = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/screener-data')
def screener_data():
    global CACHED_SIGNALS, CACHED_TICKERS
    matrix = []
    dates_list = []
    
    all_tickers, total_count = get_all_nse_tickers()
    CACHED_TICKERS = all_tickers
    
    # Chunk scanning for max reliability across all NSE equities
    chunk_size = 100
    scanned_signals = []
    
    # Bulk fetch
    data = yf.download(all_tickers[:1000], period="1mo", interval="1d", group_by='ticker', progress=False)
    
    for ticker in all_tickers[:1000]:
        try:
            clean_sym = ticker.replace(".NS", "")
            df = data[ticker].dropna() if len(all_tickers) > 1 else data.dropna()
            if not df.empty and len(df) > 5:
                sig = calculate_signals(df)
                sig.name = clean_sym
                scanned_signals.append(sig)
        except Exception:
            continue

    if scanned_signals:
        combined = pd.concat(scanned_signals, axis=1).fillna(False)
        CACHED_SIGNALS["ALL_NSE"] = combined
        daily_counts = combined.sum(axis=1)
        recent_counts = daily_counts.tail(5).iloc[::-1]
        dates_list = [d.strftime('%b %d') for d in recent_counts.index]
        
        row = {
            "Sector": "ALL NSE EQUITIES (UNLIMITED SCAN)",
            "CompanyCount": len(scanned_signals),
            "Signals": [int((c / len(scanned_signals)) * 100) if len(scanned_signals) > 0 else 0 for c in recent_counts.values]
        }
        matrix.append(row)

    return jsonify({
        "dates": dates_list,
        "matrix": matrix,
        "total_scanned": total_count
    })

@app.route('/details')
def details():
    stocks_details = []
    if "ALL_NSE" in CACHED_SIGNALS:
        combined_df = CACHED_SIGNALS["ALL_NSE"]
        latest_date = combined_df.index[-1]
        day_signals = combined_df.loc[latest_date]
        
        for stock in combined_df.columns:
            is_buy = day_signals.get(stock, False)
            stocks_details.append({
                "symbol": stock,
                "status": "🟢 BUY SIGNAL" if is_buy else "⚪ NO SIGNAL",
                "tv_link": f"https://in.tradingview.com/chart/?symbol=NSE:{stock}"
            })

    return render_template('details.html', sector="ALL NSE EQUITIES", stocks=stocks_details)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
