from fastapi import APIRouter, HTTPException
from analytics import fetch_stock_data, add_technical_indicators, generate_signals
import pandas as pd
import numpy as np

router = APIRouter()


@router.get("/target-sl-hitting/{symbol}")
async def get_target_sl_hitting_stats(symbol: str, period: str = "4y", interval: str = "1d"):
    """
    Computes 4-year Target 1, Target 2, Target 3 and Stop Loss hitting % 
    for any Indian stock or index (NIFTY, BANKNIFTY, SENSEX, Equities).
    """
    try:
        from analytics_target_sl_backtest import calculate_4y_target_sl_stats
        data = calculate_4y_target_sl_stats(symbol, period=period, interval=interval)
        return data
    except Exception as e:
        import traceback
        print("TARGET-SL BACKTEST ERROR:", traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{symbol}")
async def run_backtest(symbol: str, period: str = "2y", initial_capital: float = 100000.0):
    """Run backtest for a stock using generated signals"""
    try:
        df, used_symbol, ticker = fetch_stock_data(symbol, period=period, interval="1d")
        df = add_technical_indicators(df)
        signals = generate_signals(df)

        # Combine df and signals
        combined = df.copy()
        combined['Signal'] = signals['Signal']
        combined['Combined_Score'] = signals['Combined']

        # Backtest simulation
        capital = initial_capital
        position = 0  # number of shares
        entry_price = 0
        trades = []
        portfolio_values = []

        for i in range(1, len(combined)):
            row = combined.iloc[i]
            prev_row = combined.iloc[i - 1]
            price = float(row['Close'])
            date = combined.index[i].strftime("%Y-%m-%d")
            signal = row['Signal']

            # Entry: Buy signal and no current position
            if signal in ['BUY', 'STRONG BUY'] and position == 0:
                position = int(capital * 0.95 / price)  # Use 95% of capital
                entry_price = price
                cost = position * price
                capital -= cost
                trades.append({
                    "date": date,
                    "type": "BUY",
                    "price": round(price, 2),
                    "shares": position,
                    "value": round(cost, 2),
                    "capital_left": round(capital, 2)
                })

            # Exit: Sell signal and have position
            elif signal in ['SELL', 'STRONG SELL'] and position > 0:
                revenue = position * price
                profit = revenue - (position * entry_price)
                profit_pct = (profit / (position * entry_price)) * 100
                capital += revenue
                trades.append({
                    "date": date,
                    "type": "SELL",
                    "price": round(price, 2),
                    "shares": position,
                    "value": round(revenue, 2),
                    "profit": round(profit, 2),
                    "profit_pct": round(profit_pct, 2),
                    "capital": round(capital, 2)
                })
                position = 0
                entry_price = 0

            # Also exit on stop-loss (2 ATR below entry)
            elif position > 0:
                atr = float(row.get('ATR', price * 0.02))
                stop_loss = entry_price - (2 * atr)
                if price <= stop_loss:
                    revenue = position * price
                    profit = revenue - (position * entry_price)
                    profit_pct = (profit / (position * entry_price)) * 100
                    capital += revenue
                    trades.append({
                        "date": date,
                        "type": "STOP LOSS",
                        "price": round(price, 2),
                        "shares": position,
                        "value": round(revenue, 2),
                        "profit": round(profit, 2),
                        "profit_pct": round(profit_pct, 2),
                        "capital": round(capital, 2)
                    })
                    position = 0
                    entry_price = 0

            # Track portfolio value
            portfolio_value = capital + (position * price)
            portfolio_values.append({
                "date": date,
                "value": round(portfolio_value, 2)
            })

        # Close any open position at end
        if position > 0:
            final_price = float(combined.iloc[-1]['Close'])
            revenue = position * final_price
            capital += revenue

        final_capital = capital
        total_return = ((final_capital - initial_capital) / initial_capital) * 100

        # Calculate stats
        sell_trades = [t for t in trades if t['type'] in ['SELL', 'STOP LOSS']]
        winning_trades = [t for t in sell_trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('profit', 0) <= 0]

        win_rate = (len(winning_trades) / len(sell_trades) * 100) if sell_trades else 0
        avg_profit = np.mean([t['profit_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit_pct'] for t in losing_trades]) if losing_trades else 0

        # Max drawdown
        pv = [p['value'] for p in portfolio_values]
        max_val = initial_capital
        max_drawdown = 0
        for v in pv:
            if v > max_val:
                max_val = v
            dd = (max_val - v) / max_val * 100
            if dd > max_drawdown:
                max_drawdown = dd

        # Buy & Hold comparison
        buy_hold_return = ((float(combined.iloc[-1]['Close']) - float(combined.iloc[0]['Close'])) /
                           float(combined.iloc[0]['Close'])) * 100

        return {
            "symbol": used_symbol,
            "period": period,
            "initial_capital": initial_capital,
            "final_capital": round(final_capital, 2),
            "total_return_pct": round(total_return, 2),
            "buy_hold_return_pct": round(buy_hold_return, 2),
            "outperformance": round(total_return - buy_hold_return, 2),
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": round(win_rate, 2),
            "avg_profit_pct": round(avg_profit, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "profit_factor": round(abs(avg_profit / avg_loss), 2) if avg_loss != 0 else 0,
            "trades": trades[-50:],  # Last 50 trades
            "portfolio_values": portfolio_values[-252:]  # Last year
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
