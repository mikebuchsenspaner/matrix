def print_trade_log(log):
    for trade in log:
        print(f"Trade: {trade['action']} at {trade['timestamp']} for price {trade['price']}")