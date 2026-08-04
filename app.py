from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf

app = Flask(__name__)

# Complete NSE Sector Wise Mapping (Using .NS extension for Indian Stocks)
sector_dict = {
    "NIFTY BANK": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "AUBANK.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS", "CANBK.NS"],
    "NIFTY IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "LTTS.NS"],
    "NIFTY AUTO": ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "TVSMOTOR.NS", "BOSCHLTD.NS", "BHARATFORG.NS", "TIINDIA.NS"],
    "NIFTY PHARMA": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "LUPIN.NS", "AUBANK.NS", "TORNTPHARM.NS", "ALKEM.NS", "BIOCON.NS", "GLENMARK.NS"],
    "NIFTY FMCG": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS", "DABUR.NS", "GODREJCP.NS", "MARICO.NS", "COLPAL.NS", "VBL.NS"],
    "NIFTY METAL": ["TATASTEEL.NS", "JINDALSTEL.NS", "HINDALCO.NS", "NMDC.NS", "SAIL.NS", "NATIONALUM.NS", "VEDL.NS", "JSWSTEEL.NS", "APLAPOLLO.NS", "HINDZINC.NS"],
    "NIFTY ENERGY": ["RELIANCE.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "TATAPOWER.NS", "COALINDIA.NS"],
    "NIFTY REALTY": ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS", "BRIGADE.NS", "PRESTIGE.NS", "SOBHA.NS"]
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

CACHED_SIGNALS = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/screener-data')
def screener_data():
    global CACHED_SIGNALS
    matrix = []
    dates_list = []
    
    # Fast Bulk Download using yfinance
    all_tickers = [ticker for tickers in sector_dict.values() for ticker in tickers]
    data = yf.download(all_tickers, period="1mo", interval="1d", group_by='ticker', progress=False)
    
    for sector, tickers in sector_dict.items():
        all_signals = []
        for ticker in tickers:
            try:
                df = data[ticker].dropna()
                if not df.empty and len(df) > 5:
                    sig = calculate_signals(df)
                    sig.name = ticker.replace(".NS", "")
                    all_signals.append(sig)
            except Exception:
                continue
        
        if all_signals:
            combined = pd.concat(all_signals, axis=1).fillna(False)
            CACHED_SIGNALS[sector] = combined
            daily_counts = combined.sum(axis=1)
            recent_counts = daily_counts.tail(5).iloc[::-1]
            dates_list = [d.strftime('%b %d') for d in recent_counts.index]
            
            row = {
                "Sector": sector,
                "CompanyCount": len(tickers),
                "Signals": [int((c / len(tickers)) * 100) for c in recent_counts.values]
            }
            matrix.append(row)
            
    return jsonify({"dates": dates_list, "matrix": matrix})

@app.route('/details')
def details():
    sector_name = request.args.get('sector', '')
    stocks = sector_dict.get(sector_name, [])
    
    stock_details = []
    if sector_name in CACHED_SIGNALS:
        sector_df = CACHED_SIGNALS[sector_name]
        latest_date = sector_df.index[-1]
        day_signals = sector_df.loc[latest_date]
        
        for stock in stocks:
            clean_symbol = stock.replace(".NS", "")
            is_buy = day_signals.get(clean_symbol, False) if isinstance(day_signals, pd.Series) else False
            stock_details.append({
                "symbol": clean_symbol,
                "status": "🟢 BUY SIGNAL" if is_buy else "⚪ NO SIGNAL",
                "tv_link": f"https://in.tradingview.com/chart/?symbol=NSE:{clean_symbol}"
            })
    else:
        for stock in stocks:
            clean_symbol = stock.replace(".NS", "")
            stock_details.append({
                "symbol": clean_symbol,
                "status": "⚪ NO SIGNAL",
                "tv_link": f"https://in.tradingview.com/chart/?symbol=NSE:{clean_symbol}"
            })
            
    return render_template('details.html', sector=sector_name, stocks=stock_details)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
