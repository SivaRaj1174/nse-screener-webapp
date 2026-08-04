from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# Sample Data for Testing
SECTOR_DATA = [
    {
        "Sector": "NIFTY BANK",
        "CompanyCount": 12,
        "Signals": [80, 50, 100, 30, 0]
    },
    {
        "Sector": "NIFTY IT",
        "CompanyCount": 10,
        "Signals": [30, 70, 40, 90, 60]
    },
    {
        "Sector": "NIFTY AUTO",
        "CompanyCount": 15,
        "Signals": [100, 80, 60, 40, 20]
    },
    {
        "Sector": "NIFTY PHARMA",
        "CompanyCount": 20,
        "Signals": [20, 10, 0, 50, 80]
    }
]

DATES = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/screener-data')
def screener_data():
    return jsonify({
        "dates": DATES,
        "matrix": SECTOR_DATA
    })

@app.route('/details')
def details():
    sector = request.args.get('sector', 'Unknown')
    sample_stocks = [
        {"symbol": f"{sector[:4]}_STOCK1", "status": "BUY SIGNAL", "tv_link": "https://www.tradingview.com/chart/"},
        {"symbol": f"{sector[:4]}_STOCK2", "status": "NO SIGNAL", "tv_link": "https://www.tradingview.com/chart/"},
        {"symbol": f"{sector[:4]}_STOCK3", "status": "BUY SIGNAL", "tv_link": "https://www.tradingview.com/chart/"}
    ]
    return render_template('details.html', sector=sector, stocks=sample_stocks)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
