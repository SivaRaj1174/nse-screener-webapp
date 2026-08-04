from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf
import threading
import time

app = Flask(__name__)

SECTORS = {
    "NIFTY BANK": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "AUBANK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "CANBK.NS"],
    "NIFTY IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "LTTS.NS"],
    "NIFTY AUTO": ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "TVSMOTOR.NS", "BOSCHLTD.NS", "BHARATFORG.NS", "TIINDIA.NS"],
    "NIFTY PHARMA": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "LUPIN.NS", "AUROPHARMA.NS", "TORNTPHARM.NS", "ALKEM.NS", "BIOCON.NS", "GLENMARK.NS"],
    "NIFTY FMCG": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS", "DABUR.NS", "GODREJCP.NS", "MARICO.NS", "COLPAL.NS", "VBL.NS"],
    "NIFTY METAL": ["TATASTEEL.NS", "JINDALSTEL.NS", "HINDALCO.NS", "NMDC.NS", "SAIL.NS", "NATIONALUM.NS", "VEDL.NS", "JSWSTEEL.NS", "APLAPOLLO.NS", "HINDZINC.NS"],
    "NIFTY ENERGY": ["RELIANCE.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "TATAPOWER.NS", "ADANIGREEN.NS", "COALINDIA.NS"],
    "NIFTY REALTY": ["DLF.NS", "LODHA.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS", "BRIGADE.NS", "PRESTIGE.NS", "SOBHA.NS"]
}

CACHE = {
    "dates": [],
    "matrix": [],
    "current_scanned": 0,
    "total_to_scan": 0,
    "is_scanning": True,
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
        all_syms = []
        for syms in SECTORS.values():
            all_syms.extend(syms)
        return list(set(all_syms))

def calculate_buy_sell_signals(df):
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
        saved_sell_exit_price = pd.Series(np.where(sell_cond, ha_low.shift(1), np.nan), index=df.index).ffill()
        
        raw_buy = (df['Close'] > saved_buy_price) & (df['Close'] > saved_sell_exit_price)
        raw_sell = (df['Close'] < saved_sell_exit_price) | (df['Close'] < saved_buy_price)
        
        return raw_buy.fillna(False), raw_sell.fillna(False)
    except Exception:
        empty = pd.Series([False]*len(df), index=df.index)
        return empty, empty

def scan_all_stocks_background():
    global CACHE
    while True:
        try:
            all_symbols = get_all_nse_symbols()
            CACHE["total_to_scan"] = len(all_symbols)
            CACHE["current_scanned"] = 0
            CACHE["is_scanning"] = True
            
            batch_size = 50
            scanned_data = {}
            
            symbol_to_sector = {}
            for sector, t_list in SECTORS.items():
                for t in t_list:
                    symbol_to_sector[t] = sector

            for i in range(0, len(all_symbols), batch_size):
                batch = all_symbols[i:i+batch_size]
                try:
                    data = yf.download(batch, period="1mo", interval="1d", group_by='ticker', progress=False)
                    for ticker in batch:
                        try:
                            clean_sym = ticker.replace(".NS", "")
                            df = data[ticker].dropna() if len(batch) > 1 else data.dropna()
                            if not df.empty and len(df) > 5:
                                buy_sig, sell_sig = calculate_buy_sell_signals(df)
                                sec = symbol_to_sector.get(ticker, "OTHERS / ALL NSE EQUITIES")
                                if sec not in scanned_data:
                                    scanned_data[sec] = []
                                scanned_data[sec].append({
                                    "symbol": clean_sym,
                                    "buy_sig": buy_sig,
                                    "sell_sig": sell_sig
                                })
                        except Exception:
                            pass
                        CACHE["current_scanned"] += 1
                except Exception:
                    CACHE["current_scanned"] += len(batch)
                time.sleep(0.05)

            matrix = []
            dates_list = []
            stock_details_map = {}

            for sector, stock_list in scanned_data.items():
                if not stock_list:
                    continue
                
                ref_buy = stock_list[0]['buy_sig']
                recent_dates = ref_buy.tail(5).index[::-1]
                dates_list = [d.strftime('%b %d') for d in recent_dates]
                
                buy_percentages = []
                sell_percentages = []

                for d in recent_dates:
                    b_cnt = sum([1 for item in stock_list if item['buy_sig'].get(d, False)])
                    s_cnt = sum([1 for item in stock_list if item['sell_sig'].get(d, False)])
                    
                    buy_pct = int((b_cnt / len(stock_list)) * 100)
                    sell_pct = int((s_cnt / len(stock_list)) * 100)
                    
                    buy_percentages.append(buy_pct)
                    sell_percentages.append(sell_pct)

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
                    "BuySignals": buy_percentages,
                    "SellSignals": sell_percentages
                })

            CACHE["dates"] = dates_list
            CACHE["matrix"] = matrix
            CACHE["stock_details"] = stock_details_map
            CACHE["is_scanning"] = False
        except Exception as e:
            print("Scanner Error:", e)

        time.sleep(1800)

bg_thread = threading.Thread(target=scan_all_stocks_background, daemon=True)
bg_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/screener-data')
def screener_data():
    return jsonify({
        "dates": CACHE.get("dates", []),
        "matrix": CACHE.get("matrix", []),
        "current_scanned": CACHE.get("current_scanned", 0),
        "total_to_scan": CACHE.get("total_to_scan", 0),
        "is_scanning": CACHE.get("is_scanning", True)
    })

@app.route('/details')
def details():
    sector_name = request.args.get('sector', '')
    details_map = CACHE.get("stock_details", {})
    stocks = details_map.get(sector_name, [])
    return render_template('details.html', sector=sector_name, stocks=stocks)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
