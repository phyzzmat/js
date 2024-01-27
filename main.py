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
        print(f'Make a market with spread <= {spread}. Tap Enter when ready for 5 seconds of speech recognition.')
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
        print(f'AI side: {side}. Market resolved to: {result}. You have earned {pnl_delta}')
    
    # -------------------- #
    #     Make summary     #
    # -------------------- #

    i = 0
    while Path(f'game_{i}.pdf').exists():
        i += 1
    pdf = PdfPages(f'game_{i}.pdf')
    simulations = [config['trading_session']['initial_balance']] * config['trading_session']['monte_carlo_simuls']
    for expr, quote, side, spread in zip(
        all_rvs, all_quotes, all_sides, all_spreads
    ):
        fig, ax = plt.subplots(figsize=(10, 10))
        boot = bootstrap(expr)
        ax.hist(boot, bins=50)
        ax.set_title(f'Random value pdf: {expr.__str__()}')

        taken_quote = quote.bid_price if side == 'bid' else quote.ask_price
        non_taken_quote = quote.bid_price if side == 'ask' else quote.ask_price
        
        for i in range(config['trading_session']['monte_carlo_simuls']):
            simulations[i] += resolve_market(expr, quote, side)[1]

        ax.axvline(non_taken_quote, c='purple', label='Your quote (not taken)')
        ax.axvline(taken_quote, c='purple', linewidth=4, label='Your quote (taken)')
        ax.axvline(boot.mean() - spread / 2, c='green', label=f'Possible optimal approach (bid) = {boot.mean() - spread / 2}')
        ax.axvline(boot.mean() + spread / 2, c='green', label=f'Possible optimal approach (ask) = {boot.mean() + spread / 2}')
        ax.axvline(boot.mean(), c='yellow', label='Expected value')
        
        ax.axvspan(np.percentile(boot, 2.5), np.percentile(boot, 97.5), color='yellow', label='95 % CI', alpha=0.1)

        ax.legend()

        pdf.savefig(fig)

    simulations = np.array(simulations)
    fig, ax = plt.subplots(3, figsize=(10, 10))
    ax[0].set_title('P&L during simulation')
    ax[1].set_title(f"P&L distribution over {config['trading_session']['monte_carlo_simuls']} trading sessions")
    ax[2].set_title(f"P&L log1p-distribution over {config['trading_session']['monte_carlo_simuls']} trading sessions")
    ax[0].plot(all_prefix_pnls)
    ax[1].hist(simulations)
    ax[2].hist(np.log1p(np.maximum(0, simulations)))
    pdf.savefig(fig)

    pdf.close()
    
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(open(args.config).read())
    play(config)


if __name__ == "__main__":
    main()
