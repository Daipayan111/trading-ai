import yfinance as yf
ticker = yf.Ticker('^NSEBANK')
price = ticker.history(period='1d')['Close'].iloc[-1]
expirations = ticker.options
print(f'Price: {price}')
print(f'Expirations: {expirations}')
if expirations:
    opt = ticker.option_chain(expirations[0])
    print(opt.calls.head())
