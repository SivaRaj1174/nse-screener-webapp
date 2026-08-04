from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
import os
import threading
import time

app = Flask(__name__)

CACHE_FILE = "nse_history.csv"

SECTORS_MAP = {
    "NIFTY BANK": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "BANKBARODA", "PNB", "AUBANK", "FEDERALBNK", "IDFCFIRSTB", "CANBK"],
    "NIFTY IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "PERSISTENT", "COFORGE", "MPHASIS", "LTTS"],
    "NIFTY AUTO": ["TATAMOTORS", "MARUTI", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "TVSMOTOR", "BOSCHLTD", "BHARATFORG", "TIINDIA"],
    "NIFTY PHARMA": ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN", "AUROPHARMA", "TORNTPHARM", "ALKEM", "BIOCON", "GLENMARK"],
    "NIFTY FMCG": ["ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR", "GODREJCP", "MARICO", "COLPAL", "VBL"],
    "NIFTY METAL": ["TATASTEEL", "JINDALSTEL", "HINDALCO", "NMDC", "SAIL", "NATIONALUM", "VEDL", "JSWSTEEL", "APLAPOLLO", "HINDZINC"],
    "NIFTY ENERGY": ["RELIANCE", "NTPC", "POWERGRID", "ONGC", "BPCL", "IOC", "GAIL", "TATAPOWER", "ADANIGREEN", "COALINDIA"],
    "NIFTY REALTY": ["DLF", "LODHA", "GODREJPROP", "OBEROIRLTY", "PHOENIXLTD", "BRIGADE", "PRESTIGE", "SOBHA"]
}

CACHE = {
    "dates": [],
    "matrix": [],
    "total_scanned": 0,
    "stock_details": {}
}

cache_lock = threading.Lock()

def fetch_all_nse_symbols():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            df = pd.read_csv(io.StringIO(res.text))
            df_eq = df[df[' SERIES'] == 'EQ']
            return [f"{sym.strip()}.NS" for sym in df_eq['SYMBOL'].tolist()]
    except Exception:
        pass
    
    # Fallback list
    fallback = []
    for st_list in SECTORS_MAP.values():
        fallback.extend([f"{s}.NS" for s in st_list])
    return list(set(fallback))

def download_and_save_csv():
    """ Downloads historical data for 2000+ NSE stocks and saves locally """
    symbols = fetch_all_nse_symbols()
    batch_size = 50
    all_dfs = []

    print(f"Downloading historical data for {len(symbols)} stocks into local CSV...")
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        try:
            data = yf.download(batch, period="1mo", interval="1d", group_by='ticker', progress=False)
            for ticker in batch:
                clean_sym = ticker.replace(".NS", "").strip()
                try:
                    df = data[ticker].dropna() if len(batch) > 1 else data.dropna()
                    if not df.empty and len(df) > 5:
                        df = df[['Open', 'High', 'Low', 'Close']].copy()
                        df['Symbol'] = clean_sym
                        df['Date'] = df.index.strftime('%Y-%m-%d')
                        all_dfs.append(df)
                except Exception:
                    continue
        except Exception:
            continue
        time.sleep(0.05)

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df.to_csv(CACHE_FILE, index=False)
        print("CSV storage created successfully!")

def calculate_signals(df):
    try:
        df = df.copy()
        ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        ha_open = pd.Series(index=df.index, dtype=float)
        ha_open.iloc[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
        
        for i in range(1, len(df)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
            
        ha_high = np.maximum(df['High'], np.maximum(ha_open, ha_close))
        ha_low = np.minimum(df['Low'], np.minimum(ha_open, ha_close))
        
        ha_body = np.abs(ha_close - ha_open)
        nrml_body = np.abs(df['Close'] - df['Open'])
        
        buy_cond = (ha_body.shift(2) > ha_body.shift(1)) & (nrml_body.shift(2) < nrml_body.shift(1)) & (ha_open.shift(2) < ha_close.shift(2))
        sell_cond = (ha_body.shift(2) > ha_body.shift(1)) & (nrml_body.shift(2) < nrml_body.shift(1)) & (ha_open.shift(2) > ha_close.shift(2))
        
        saved_buy_price = pd.Series(np.where(buy_cond, ha_high.shift(1), np.nan), index=df.index).ffill()
        saved_sell_price = pd.Series(np.where(sell_cond, ha_low.shift(1), np.nan), index=df.index).ffill()
        
        buy_sig = (df['Close'] > saved_buy_price) & (~saved_buy_price.isna())
        sell_sig = (df['Close'] < saved_sell_price) & (~saved_sell_price.isna()) & (~buy_sig)
        
        return buy_sig.fillna(False), sell_sig.fillna(False)
    except Exception:
        empty = pd.Series([False]*len(df), index=df.index)
        return empty, empty

def process_csv_and_update_cache():
    global CACHE
    if not os.path.exists(CACHE_FILE):
        download_and_save_csv()
        
    if not os.path.exists(CACHE_FILE):
        return

    try:
        full_df = pd.read_csv(CACHE_FILE)
        sym_to_sector = {}
        for sec, st_list in SECTORS_MAP.items():
            for s in st_list:
                sym_to_sector[s] = sec

        scanned_records = {}

        for sym, group in full_df.groupby('Symbol'):
            group = group.sort_values('Date')
            group.set_index('Date', inplace=True)
            
            buy_sig, sell_sig = calculate_signals(group)
            sec = sym_to_sector.get(sym, "ALL OTHER NSE EQUITIES")
            
            if sec not in scanned_records:
                scanned_records[sec] = []
            
            scanned_records[sec].append({
                "symbol": sym,
                "buy_sig": buy_sig,
                "sell_sig": sell_sig
            })

        matrix = []
        dates_list = []
        stock_details_map = {}
        total_count = 0

        for sector, stock_list in scanned_records.items():
            if not stock_list:
                continue

            total_count += len(stock_list)
            ref_buy = stock_list[0]['buy_sig']
            recent_dates = ref_buy.tail(5).index[::-1]
            dates_list = [pd.to_datetime(d).strftime('%b %d') for d in recent_dates]

            buy_pcts = []
            sell_pcts = []

            for d in recent_dates:
                b_cnt = sum([1 for item in stock_list if item['buy_sig'].get(d, False)])
                s_cnt = sum([1 for item in stock_list if item['sell_sig'].get(d, False)])
                
                sec_total = len(stock_list)
                buy_pcts.append(int((b_cnt / sec_total) * 100))
                sell_pcts.append(int((s_cnt / sec_total) * 100))

            latest_d = recent_dates[0]
            details = []
            for item in stock_list:
                is_b = item['buy_sig'].get(latest_d, False)
                is_s = item['sell_sig'].get(latest_d, False)
                
                status = "⚪ NO SIGNAL"
                if is_b:
                    status = "🟢 BUY SIGNAL"
                elif is_s:
                    status = "🔴 SELL SIGNAL"

                details.append({
                    "symbol": item['symbol'],
                    "status": status,
                    "tv_link": f"https://in.tradingview.com/chart/?symbol=NSE:{item['symbol']}"
                })

            stock_details_map[sector] = details

            matrix.append({
                "Sector": sector,
                "CompanyCount": len(stock_list),
                "BuySignals": buy_pcts,
                "SellSignals": sell_pcts
            })

        with cache_lock:
            CACHE["dates"] = dates_list
            CACHE["matrix"] = matrix
            CACHE["total_scanned"] = total_count
            CACHE["stock_details"] = stock_details_map

    except Exception as e:
        print("CSV processing error:", e)

def bg_updater():
    while True:
        process_csv_and_update_cache()
        time.sleep(3600)  # Refresh every hour

threading.Thread(target=bg_updater, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/screener-data')
def screener_data():
    with cache_lock:
        return jsonify({
            "dates": CACHE.get("dates", []),
            "matrix": CACHE.get("matrix", []),
            "total_scanned": CACHE.get("total_scanned", 0)
        })

@app.route('/details')
def details():
    sector_name = request.args.get('sector', '')
    with cache_lock:
        details_map = CACHE.get("stock_details", {})
        stocks = details_map.get(sector_name, [])
    return render_template('details.html', sector=sector_name, stocks=stocks)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
