from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/screener-data')
def screener_data():
    matrix = []
    dates_list = []
    total_scanned = 0

    for sector, tickers in SECTORS.items():
        try:
            # Batch fetch by sector (Lightweight)
            data = yf.download(tickers, period="1mo", interval="1d", group_by='ticker', progress=False)
            
            sector_buy = []
            sector_sell = []
            valid_tickers = []

            for ticker in tickers:
                try:
                    df = data[ticker].dropna() if len(tickers) > 1 else data.dropna()
                    if not df.empty and len(df) > 5:
                        b_sig, s_sig = calculate_buy_sell_signals(df)
                        clean_sym = ticker.replace(".NS", "")
                        b_sig.name = clean_sym
                        s_sig.name = clean_sym
                        sector_buy.append(b_sig)
                        sector_sell.append(s_sig)
                        valid_tickers.append(clean_sym)
                        total_scanned += 1
                except Exception:
                    continue

            if sector_buy and sector_sell:
                combined_buy = pd.concat(sector_buy, axis=1).fillna(False)
                combined_sell = pd.concat(sector_sell, axis=1).fillna(False)
                
                recent_dates = combined_buy.tail(5).index[::-1]
                dates_list = [d.strftime('%b %d') for d in recent_dates]

                buy_pcts = []
                sell_pcts = []

                for d in recent_dates:
                    b_cnt = int(combined_buy.loc[d].sum()) if d in combined_buy.index else 0
                    s_cnt = int(combined_sell.loc[d].sum()) if d in combined_sell.index else 0
                    
                    t_count = len(valid_tickers)
                    buy_pcts.append(int((b_cnt / t_count) * 100) if t_count > 0 else 0)
                    sell_pcts.append(int((s_cnt / t_count) * 100) if t_count > 0 else 0)

                matrix.append({
                    "Sector": sector,
                    "CompanyCount": len(valid_tickers),
                    "BuySignals": buy_pcts,
                    "SellSignals": sell_pcts
                })
        except Exception as e:
            print(f"Error scanning {sector}:", e)
            continue

    return jsonify({
        "dates": dates_list,
        "matrix": matrix,
        "total_scanned": total_scanned
    })

@app.route('/details')
def details():
    sector_name = request.args.get('sector', '')
    tickers = SECTORS.get(sector_name, [])
    stocks = []
    
    if tickers:
        try:
            data = yf.download(tickers, period="1mo", interval="1d", group_by='ticker', progress=False)
            for ticker in tickers:
                clean_sym = ticker.replace(".NS", "")
                try:
                    df = data[ticker].dropna() if len(tickers) > 1 else data.dropna()
                    if not df.empty and len(df) > 5:
                        b_sig, s_sig = calculate_buy_sell_signals(df)
                        latest_date = df.index[-1]
                        is_b = bool(b_sig.loc[latest_date])
                        is_s = bool(s_sig.loc[latest_date])
                        
                        status = "⚪ NO SIGNAL"
                        if is_b:
                            status = "🟢 BUY SIGNAL"
                        elif is_s:
                            status = "🔴 SELL SIGNAL"

                        stocks.append({
                            "symbol": clean_sym,
                            "status": status,
                            "tv_link": f"https://in.tradingview.com/chart/?symbol=NSE:{clean_sym}"
                        })
                except Exception:
                    stocks.append({"symbol": clean_sym, "status": "⚪ NO SIGNAL", "tv_link": f"https://in.tradingview.com/chart/?symbol=NSE:{clean_sym}"})
        except Exception:
            pass

    return render_template('details.html', sector=sector_name, stocks=stocks)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
