import os
from typing import Callable
import numpy as np
import pandas as pd
import vectorbt as vbt
from plotly.subplots import make_subplots
from experiment01_ensemble_uncertainty_hard_threshold.EnsembleWrapper import EnsembleWrapper

def benchmark_models(
        models:list[str],
        models_path:str,
        test_data_path:str,
        stocks_to_benchmark:list[str],
        output_path:str,
        scale_inputs: Callable,
        trading_strategy: Callable
):
    for model in models:
        model_out_path = os.path.join(output_path, model)
        os.makedirs(model_out_path, exist_ok=True)

        print(f"benchmarking model: {model}")
        portfolio = simulate_single_model(model, models_path, output_path, test_data_path, stocks_to_benchmark, scale_inputs, trading_strategy)
        visualize_multiple_ticker_benchmarks(portfolio, model_out_path)
        visualize_total_returns(portfolio, model_out_path)

def simulate_single_model(
        model_name:str,
        models_path:str,
        output_path:str,
        test_data_path:str,
        stocks_to_test:list[str],
        create_scaled_batches: Callable,
        trading_strategy: Callable
):
    costs = 0.002  # expected costs (0,2% ~ spread + slippage + transaction fees)
    sequence_size = 23 # number of previous timesteps to consider
    max_batch_size = 32
    prices_list, preds_list, std_list = [], [], []

    # load the ensemble
    model_dir = os.path.join(models_path, model_name)
    ensemble_paths = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(('.pt', '.pth'))]
    ensemble = EnsembleWrapper(ensemble_paths, model_name)

    for stock in stocks_to_test:
        print(f"processing stock: {stock}")
        # load the raw data of the stock
        test_samples = pd.read_csv(os.path.join(test_data_path, f"{stock}.csv"))
        test_samples['Datetime'] = pd.to_datetime(test_samples['Datetime'], utc=True)
        test_samples['Datetime'] = test_samples['Datetime'].dt.tz_convert('Europe/Berlin')
        test_samples.set_index('Datetime', inplace=True)

        stock_series = test_samples["Open"].rename(stock)
        prices_list.append(stock_series)

        # create and process the batch via the ensemble and store the predictions
        full_batch, target_dates = create_scaled_batches(test_samples, sequence_size)
        mean_preds_list, std_preds_list  = [], []
        total_batch_size = len(next(iter(full_batch.values())))

        # iterate over all batches in steps equal to the max_batch_size
        for start_idx in range(0, total_batch_size, max_batch_size):
            # full_batch is a dictionary containing the features as values
            end_idx = start_idx + max_batch_size
            chunk = {key: value[start_idx:end_idx, :, :] for key, value in full_batch.items()}
            # store the batch model results
            c_mean, c_std = ensemble.predict(chunk)
            mean_preds_list.extend(c_mean)
            std_preds_list.extend(c_std)

        mean_preds, std_preds = np.array(mean_preds_list), np.array(std_preds_list)

        df = pd.DataFrame({"mean": mean_preds, "std": std_preds})
        df.to_csv(os.path.join(output_path, model_name, f"{model_name}_{stock}_preds.csv"), index=False)
        # convert the resulting numpy arrays to series objects to later create a dataframe of it
        pred_series = pd.Series(mean_preds, index=target_dates, name=stock)
        std_series = pd.Series(std_preds, index=target_dates, name=stock)
        preds_list.append(pred_series)
        std_list.append(std_series)

    # store the series as dataframes where: cols = tickers, rows = value for given timestep
    prices = pd.concat(prices_list, axis=1)
    predictions = pd.concat(preds_list, axis=1)
    sigma_ens = pd.concat(std_list, axis=1)
    sigma_mkt = prices.pct_change().rolling(23).std() # market sigma rolling window (volatility)

    # for the first prices and sigmas there do not exist values, cut these
    prices = prices.iloc[sequence_size:]
    sigma_mkt = sigma_mkt.iloc[sequence_size:]

    target_weights = trading_strategy(sigma_mkt, sigma_ens, predictions, costs)

    # size_type='targetpercent' tells VBT, that the numbers should be interpreted as percent
    portfolio = vbt.Portfolio.from_orders(
        close=prices,
        size=target_weights,
        size_type='targetpercent',
        init_cash=2000,
        fees=costs,
        freq='2h'
    )
    return portfolio


def visualize_multiple_ticker_benchmarks(portfolio, output_path:str):
    tickers = portfolio.wrapper.columns

    fig_subplots = make_subplots(
        rows=len(tickers),
        cols=1,
        shared_xaxes=True,
        subplot_titles=tickers,
        vertical_spacing=0.01
    )

    for i, ticker in enumerate(tickers):
        sub_fig = portfolio[ticker].plot(subplots=['cum_returns'])

        for trace in sub_fig.data:
            trace.showlegend = (i == 0) # only show the legend in the first subplot
            fig_subplots.add_trace(trace, row=i + 1, col=1)

    fig_subplots.update_layout(
        height=350 * len(tickers),
        title_text="Benchmark Ticker View",
        showlegend=True
    )
    html_path = os.path.join(output_path, "ticker_benchmarks.html")
    fig_subplots.write_html(html_path)
    # fig_subplots.show()


def visualize_total_returns(portfolio, output_path:str):
    # summation of results from multiple tickers
    total_capital = portfolio.value().sum(axis=1)
    total_benchmark = portfolio.benchmark_value().sum(axis=1)
    total_return = total_capital.pct_change()
    print(total_return.vbt.returns(freq='2h').stats())

    # create dataframe with results
    total_portfolio_df = pd.DataFrame({
        'strategy (aggregated)': total_capital,
        'benchmark (aggregated)': total_benchmark
    })
    fig_aggregated = total_portfolio_df.vbt.plot(
        title="total portfolio: strategy vs. benchmark"
    )

#    rolling_sharpe = total_return.vbt.returns.rolling_sharpe_ratio(window=360)
    rolling_sharpe = total_return.vbt.returns(freq='2h').rolling_sharpe_ratio(window=360)
    fig_sharpe = rolling_sharpe.vbt.plot(title="Rolling Sharpe Ratio (aggregated over 360 periods)")
    # fig_aggregated.show()
    # fig_sharpe.show()

    trades = portfolio.trades.records_readable # contains every trade
    relevant_cols = ['Column', 'Entry Timestamp', 'Exit Timestamp', 'Direction', 'Size', 'PnL', 'Return']
    print(trades.sort_values('Return', ascending=True)[relevant_cols].head(10))

    html_path_agg = os.path.join(output_path, "total_portfolio_returns.html")
    html_path_sharpe = os.path.join(output_path, "rolling_sharpe.html")
    fig_aggregated.write_html(html_path_agg)
    fig_sharpe.write_html(html_path_sharpe)


# def trading_simulation():
#     np.random.seed(42)
#     costs = 0.002  # expected costs (0,05% für spread + slippage + transaction fees)
#
#     dates = pd.date_range("2025-01-01", periods=2000, freq="2h") # 2-Stunden-Intervalle über ca. 1 Jahr
#
#     tickers = [f"Ticker_{i}" for i in range(1, 11)]
#
#     # currently contains randomly generated values -> TODO: replace with actual vals
#     prices_data = np.cumprod(1 + np.random.normal(0, 0.002, (len(dates), len(tickers))), axis=0) * 100
#     prices = pd.DataFrame(prices_data, index=dates, columns=tickers)
#
#     predictions = pd.DataFrame(np.random.normal(0, 0.003, (len(dates), len(tickers))), index=dates, columns=tickers)
#     sigma_ens_sq = pd.DataFrame(np.random.uniform(0.000001, 0.00005, (len(dates), len(tickers))), index=dates, columns=tickers)
#     sigma_mkt_sq = prices.pct_change().rolling(20).var().bfill()
#
#     target_weights = apply_trading_strategy(sigma_mkt_sq, sigma_ens_sq, predictions, costs)
#
#     # size_type='targetpercent' sagt VBT, dass die Zahlen in "size" als % des Gesamtkapitals zu verstehen sind
#     portfolio = vbt.Portfolio.from_orders(
#         close=prices,
#         size=target_weights,
#         size_type='targetpercent',
#         init_cash=2000,
#         fees=costs,
#         freq='2h'
#     )
#     return portfolio

# portfolio = trading_simulation()
# visualize_multiple_ticker_benchmarks(portfolio)
# visualize_total_returns(portfolio)