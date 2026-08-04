from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from nselib import capital_market
from datetime import datetime, timedelta

app = Flask(__name__)

# Extended Comprehensive Sector List
sector_dict = {
    "NIFTY BANK": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "BANKBARODA", "PNB", "AUBANK", "IDFCFIRSTB", "FEDERALBNK", "CANBK"],
    "NIFTY IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "PERSISTENT", "COFORGE", "MPHASIS", "LTTS"],
    "NIFTY AUTO": ["TATAMOTORS", "MARUTI", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "TVSMOTOR", "BOSCHLTD", "BHARATFORG", "TIINDIA"],
    "NIFTY PHARMA": ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN", "AURPHARMA", "TORNTPHARM", "ALKEM", "BIOCON", "GLENMARK"],
    "NIFTY FMCG": ["ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR", "GODREJCP", "MARICO", "COLPAL", "VBL"],
    "NIFTY METAL": ["TATASTEEL", "JINDALSTEL", "HINDALCO", "NMDC", "SAIL", "NATIONALUM", "VEDL", "JSWSTEEL", "APLAPOLLO", "HINDZINC"],
    "NIFTY ENERGY": ["RELIANCE", "NTPC", "POWERGRID", "ONGC", "BPCL", "IOC", "GAIL", "TATAPOWER", "ADANIGREEN", "COALINDIA"],
    "NIFTY REALTY": ["DLF", "LODHA", "GODREJPROP", "OBEROIRLTY", "PHOENIXLTD", "BRIGADE", "PRESTIGE", "SOBHA"]
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
        ha_low = np.minimum(df['Low'], np.minimum(ha_open, ha_close))
        
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

def fetch_nse_data(symbol):
    try:
        end_date = datetime.now().strftime('%d-%m-%Y')
        start_date = (datetime.now() - timedelta(days=40)).strftime('%d-%m-%Y')
        df = capital_market.price_volume_and_deliverable_position_data(symbol=symbol, from_date=start_date, to_date=end_date)
        if df.empty: return pd.DataFrame()
        df = df[['Date', 'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice', 'TotalTradedQuantity']]
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True)
        return df[~df.index.duplicated(keep='first')]
    except Exception:
        return pd.DataFrame()

CACHED_SIGNALS = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/screener-data')
def screener_data():
    global CACHED_SIGNALS
    matrix = []
    dates_list = []
    
    for sector, tickers in sector_dict.items():
        all_signals = []
        for ticker in tickers:
            data = fetch_nse_data(ticker)
            if not data.empty and len(data) > 5:
                sig = calculate_signals(data)
                sig.name = ticker
                all_signals.append(sig)
        
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
            is_buy = day_signals.get(stock, False) if isinstance(day_signals, pd.Series) else False
            stock_details.append({
                "symbol": stock,
                "status": "🟢 BUY SIGNAL" if is_buy else "⚪ NO SIGNAL",
                "tv_link": f"https://in.tradingview.com/chart/?symbol=NSE:{stock}"
            })
    else:
        for stock in stocks:
            stock_details.append({
                "symbol": stock,
                "status": "⚪ NO SIGNAL",
                "tv_link": f"https://in.tradingview.com/chart/?symbol=NSE:{stock}"
            })
            
    return render_template('details.html', sector=sector_name, stocks=stock_details)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
