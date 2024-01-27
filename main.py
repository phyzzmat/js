import argparse
from pathlib import Path
from types import SimpleNamespace
import matplotlib.pyplot as plt
import json
from speech_parsing import parse_speech
from tree import generate_tree
import speech_recognition as sr
from alsa_hack import noalsaerr
from utils import bootstrap, check_spread, get_side, get_spread, resolve_market
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import scipy


def play(config):

    all_rvs = []
    all_quotes = []
    all_results = []
    all_prefix_pnls = [config['trading_session']['initial_balance']]
    all_sides = []
    all_spreads = []
    init_rec = sr.Recognizer()

    for i in range(config['trading_session']['n_events']):
        expr = generate_tree(config)
        all_rvs.append(expr)
        print(f'{i}.', expr)

        spread = get_spread(config, expr)
        all_spreads.append(spread)
        print(
            f'Make a market with spread <= {spread}. Tap Enter when ready for 5 seconds of speech recognition.')
        quote = None

        while quote is None:
            input()
            quote = parse_speech(init_rec)
            if not check_spread(quote, spread):
                quote = None

        all_quotes.append(quote)
        side = get_side(expr, quote)
        all_sides.append(side)
        result, pnl_delta = resolve_market(expr, quote, side)
        all_prefix_pnls.append(all_prefix_pnls[-1] + float(pnl_delta))
        print(
            f'AI side: {side}. Market resolved to: {result}. You have earned {pnl_delta}')

    # -------------------- #
    #     Make summary     #
    # -------------------- #

    i = 0
    while Path(f'game_{i}.pdf').exists():
        i += 1
    pdf = PdfPages(f'game_{i}.pdf')
    simulations = [config['trading_session']['initial_balance']
                   ] * config['trading_session']['monte_carlo_simuls']
    for expr, quote, side, spread in zip(
        all_rvs, all_quotes, all_sides, all_spreads
    ):
        fig, ax = plt.subplots(3, figsize=(10, 10))
        boot = bootstrap(expr)
        ax[0].hist(boot, bins=50)
        ax[0].set_title(f'Random value pdf: {expr.__str__()}')

        taken_quote = quote.bid_price if side == 'bid' else quote.ask_price
        taken_amount = quote.bid_amount if side == 'bid' else quote.ask_amount
        non_taken_quote = quote.bid_price if side == 'ask' else quote.ask_price
        non_taken_amount = quote.bid_amount if side == 'ask' else quote.ask_amount

        pnls = []
        pnls_oppo = []
        for i in range(config['trading_session']['monte_carlo_simuls']):
            pnl = resolve_market(expr, quote, side)[1]
            pnl_oppo = resolve_market(
                expr, quote, 'bid' if side == 'ask' else 'ask')[1]
            simulations[i] += pnl
            pnls.append(pnl)
            pnls_oppo.append(pnl_oppo)

        ax[0].axvline(non_taken_quote, c='purple',
                      label=f'Your quote (not taken) = {non_taken_quote}')
        ax[0].text(non_taken_quote, 0.99, non_taken_amount, color='purple', ha='right', va='top', rotation=90,
                   transform=ax[0].get_xaxis_transform())
        ax[0].axvline(taken_quote, c='purple', linewidth=4,
                      label=f'Your quote (taken) = {taken_quote}')
        ax[0].text(taken_quote, 0.99, taken_amount, color='purple', ha='right', va='top', rotation=90,
                   transform=ax[0].get_xaxis_transform())
        ax[0].axvline(boot.mean() - spread / 2, c='green',
                      label=f'Possible optimal approach (bid) = {boot.mean() - spread / 2}')
        ax[0].axvline(boot.mean() + spread / 2, c='green',
                      label=f'Possible optimal approach (ask) = {boot.mean() + spread / 2}')
        ax[0].axvline(boot.mean(), c='yellow', label='Expected value')

        ax[0].axvspan(np.percentile(boot, 2.5), np.percentile(
            boot, 97.5), color='yellow', label='95 % CI', alpha=0.1)

        ax[0].legend()

        ax[1].hist(np.array(pnls), bins=50)
        ax[1].set_title('Your P&L distribution on this market')
        ax[1].legend()

        ax[2].hist(np.array(pnls_oppo), bins=50)
        ax[2].set_title(
            'Your P&L distribution on this market if AI took opposite quote')
        ax[2].legend()

        pdf.savefig(fig)

    simulations = np.array(simulations)
    fig, ax = plt.subplots(3, figsize=(10, 10))
    ax[0].set_title('P&L during simulation')
    ax[1].set_title(
        f"P&L distribution over {config['trading_session']['monte_carlo_simuls']} trading sessions")
    ax[2].set_title(
        f"P&L log1p-distribution over {config['trading_session']['monte_carlo_simuls']} trading sessions")
    ax[0].plot(all_prefix_pnls)
    ax[1].hist(simulations, bins=50)
    ax[2].hist(np.log1p(np.maximum(0, simulations)), bins=50)
    pdf.savefig(fig)

    pdf.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=False)
    args = parser.parse_args()
    config = json.loads(open(args.config).read())
    if args.seed:
        np.random.seed(args.seed)
    play(config)


if __name__ == "__main__":
    main()
