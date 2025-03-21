# %reset -f
# https://atiselsts.github.io/pdfs/uniswap-v3-liquidity-math.pdf
import os
import sys
import pandas as pd
import numpy as np
import time
import pytz
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sklearn
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex, LinearSegmentedColormap, Normalize
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pprint import pprint
from typing import Optional, Union, List
from pprint import pprint
from urllib3 import HTTPResponse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import joblib
#parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
#sys.path.insert(0, parent_dir)
pd.set_option('display.max_columns', None)
pd.set_option('future.no_silent_downcasting', True)
pd.option_context('mode.use_inf_as_na', True)
load_dotenv()
import seaborn as sns
import itertools
from settings.plot import tailwind, _style, _style_white

folder='plots_position_managment_ai'
gmx_fee_entry = 0.001
gmx_fee_exit = 0.001
gmx_funding_rate = 0.0001

# =============================================================================
# Load Data
# =============================================================================
df_market = pd.read_csv(os.path.join('data', 'df_market.csv'))
df_gmx_rates = pd.read_csv(os.path.join('data', 'df_gmx_rates.csv'))
df_uni_positions = pd.read_csv(os.path.join('data', 'df_uni_positions.csv'))
df_uni_logs = pd.read_csv(os.path.join('data', 'df_uni_logs.csv'))
df_state_xtreamly = pd.read_csv(os.path.join('data', 'df_state_xtreamly.csv'))

df_market['_time'] = [datetime.fromisoformat(str(t)).replace(tzinfo=None) for t in df_market['_time']]
df_gmx_rates['_time'] = [datetime.fromisoformat(str(t)).replace(tzinfo=None) for t in df_gmx_rates['timestamp']]
df_uni_positions['_time'] = [datetime.fromisoformat(str(t)).replace(tzinfo=None) for t in df_uni_positions['_time']]
df_uni_logs['_time'] = [datetime.fromisoformat(str(t)).replace(tzinfo=None) for t in df_uni_logs['_time']]
df_state_xtreamly['_time'] = [datetime.fromisoformat(str(t)).replace(tzinfo=None) for t in df_state_xtreamly['_time']]

where_id = [True] + list(df_state_xtreamly[1:]['state'].values != df_state_xtreamly[:-1]['state'].values)
df_state_xtreamly.loc[:, 'signal'] = where_id
df_state_xtreamly.loc[where_id, 'state_id'] = df_state_xtreamly.loc[where_id, 'state']+'_'+df_state_xtreamly[where_id].index.values.astype(str)
df_state_xtreamly['state_id'] = df_state_xtreamly['state_id'].ffill()

with open(os.path.join('data', 'pools.json'), 'r') as file: data_pools = json.load(file)
df_pools = pd.DataFrame(data_pools)
df_pools['pool'] = df_pools['pool'].str.lower()
df_pools = df_pools[[('BTC' in t or 'ETH' in t) and 'USD' in t for t in df_pools['type']]]

# =============================================================================
# Filter Pool Positions
# =============================================================================
start_time = datetime.fromisoformat('2024-10-01').replace(tzinfo=None)
end_time = datetime.fromisoformat('2024-12-31').replace(tzinfo=None)
pool = '0x4e68ccd3e89f51c3074ca5072bbac773960dfa36'

def _pos_and_id(*arg):
    df_log = df_uni_logs.copy()
    df_log['id'] = df_log['position_id']
    df_log_fr = df_log.groupby('position_id').agg(
        {'_time': ['min', 'max'], 
         'owner': 'count',
         'price': 'first'
         }).reset_index(
            drop=False).rename(
                columns={
                    ('_time', 'min'): 'fr_time',
                    ('_time', 'max'): 'to_time',
                    'price': 'price_opn',
                    'owner': 'logs',
                    'position_id': 'id'
                })
    df_log_fr.columns = df_log_fr.columns.map('_'.join).str.strip('_')
     
    df_pos = df_uni_positions.copy()
    df_pos.pop('fr_time')#.pop('to_time')
    df_pos['to_time'] = [datetime.fromisoformat(t).replace(tzinfo=None) for t in df_pos['to_time']]
    df_pos = df_pos.merge(df_log_fr, on='id', how='left')
    df_pos['active_opn'] = \
        (df_pos['price_opn_first'] >= df_pos['price_lower']) & \
        (df_pos['price_opn_first'] <= df_pos['price_upper'])
    df_pos['concentration'] = (df_pos['price_upper']-df_pos['price_lower'])/df_pos['price_opn_first']
    df_pos['concentration'] = df_pos['concentration'].astype(float)
    df_pos['duration_min'] = (df_pos['time_max'] - df_pos['time_min']).dt.total_seconds()/60
    df_pos = df_pos[
        (df_pos['pool'].isin([pool])) & 
        (df_pos['active_opn']) &
        (df_pos['logs_count']==3) &
        (df_pos['duration_min']>=60) &
        (df_pos['concentration']<=.3) &
        (df_pos['time_min']>=start_time) &
        (df_pos['time_max']<=end_time)]
    pool_ids = df_pos['id'].unique()
    return df_pos, df_log, pool_ids

df_pos, df_log, pool_ids = _pos_and_id()

# =============================================================================
# Functions
# =============================================================================
def _get_L(pos_uni):
    p = pos_uni['p_opn']
    p_a = pos_uni['p_a']
    p_b = pos_uni['p_b']
    x = pos_uni['x_opn']
    y = pos_uni['y_opn']
    if p <= p_a:
        L = x*( (np.sqrt(p_a)*np.sqrt(p_b))/(np.sqrt(p_b)-np.sqrt(p_a)))
    elif p_a < p <= p_b:
        L = y/(np.sqrt(p)-np.sqrt(p_a))
    else:
        L = y/(np.sqrt(p_b)-np.sqrt(p_a))
    return L

def _upd_uni(p, t, pos_uni):
    upd_uni = pos_uni.copy()
    x = upd_uni['x_opn']
    y = upd_uni['y_opn']
    p_a = upd_uni['p_a']
    p_b = upd_uni['p_b']#
    L = upd_uni['L']

    p_prim = np.clip(p, p_a, p_b)
    x_prim = L * (np.sqrt(p_b)-np.sqrt(p_prim))/(np.sqrt(p_prim)*np.sqrt(p_b)) # if upd_uni['p_a']<p else 0.0
    y_prim = L * (np.sqrt(p_prim)-np.sqrt(p_a)) #if upd_uni['p_b']>p else 0.0
    upd_uni['p'] = p
    upd_uni['t'] = t
    upd_uni['buffer_l'] = max(0, np.round((p-p_a)/p,6))
    upd_uni['buffer_u'] = max(0, np.round((p_b-p)/p,6))
    upd_uni['active'] = p_a < p and p < p_b

    upd_uni['x'] = x_prim
    upd_uni['y'] = y_prim
    upd_uni['x_usd'] = upd_uni['x']*p
    upd_uni['y_usd'] = upd_uni['y']    
    upd_uni['v'] = upd_uni['x_usd']+upd_uni['y_usd']
    upd_uni['imp_loss'] = upd_uni['v']-upd_uni['inv_usd']
    return upd_uni

def _pos_uni(t, p, p_a, p_b, x_opn, y_opn):#, fees_usd):
    pos_uni = {
        't_opn': t,
        'p_opn': p,
        'p_a': p_a,
        'p_b': p_b,
        'p_mid': (p_a+p_b)/2,
        'x_opn': x_opn,
        'y_opn': y_opn,
        'x_prop': (x_opn*p)/(x_opn*p+y_opn),
        'inv_usd': x_opn*p+y_opn
        }  
    pos_uni['L'] = _get_L(pos_uni)
    pos_uni = _upd_uni(p, t, pos_uni)
    return pos_uni

def _df_uni(df_p, pos_uni):
    data_lp = [_upd_uni(r['open'], r['_time'], pos_uni.copy()) 
               for _,r in df_p.iterrows()]
    df_lp = pd.DataFrame(data_lp)

    where_a = np.array([True]+list(df_lp[:-1]['active'].values != df_lp[1:]['active'].values), dtype=bool)
    df_lp.loc[where_a, 'active_id'] = df_lp[where_a].index.values.astype(str)
    df_lp.loc[0, 'active_id'] = '0'
    df_lp['active_id'] = df_lp['active_id'].ffill()
    return df_lp

def _f_cum(df_uni, f_uni, inv_usd):
    a_time = df_uni['active'].cumsum()/(60*24)
    r = (f_uni)/inv_usd
    apy = (1 + r) ** (365 / a_time.iloc[-1]) - 1
    f_cum = ((1 + apy) ** (a_time / 365) - 1)*inv_usd
    df_uni.loc[df_uni['active'],'f_cum'] = f_cum
    df_uni['f_cum'] = df_uni['f_cum'].ffill()
    return df_uni

def _upd_gmx(p, t, df_p, pos_gmx):
    df = df_p[ (df_p['_time']>=pos_gmx['t_opn']) & (df_p['_time']<t)].copy().reset_index(drop=True)  
    
    upd_gmx = pos_gmx.copy()
    upd_gmx['t'] = t
    upd_gmx['p'] = p
    upd_gmx['p_ext'] = p * ((1-gmx_fee_exit) if upd_gmx['type'] == "long" else (1+gmx_fee_exit))
    upd_gmx['t_hours'] = int((t-upd_gmx['t_opn']).total_seconds() / 3600)
    upd_gmx['cost_rate'] = upd_gmx['s'] * upd_gmx['f_rate']*upd_gmx['t_hours'] # use df
    if df.shape[0]:
        upd_gmx['liq'] = (df['low'] <= upd_gmx['p_liq']) if upd_gmx['type'] == "long" else (df['high'] >= upd_gmx['p_liq']).any()
    else:
        upd_gmx['liq'] = False
    upd_gmx['p_diff'] = (upd_gmx['p_ext'] - upd_gmx['p_ent']) * (1 if upd_gmx['type'] == "long" else -1)

    upd_gmx['pnl'] = max(-upd_gmx['c'], upd_gmx['s']*(upd_gmx['p_diff']/pos_gmx['p_ent']) - upd_gmx['cost_rate'])
    upd_gmx['v'] = max(.0, upd_gmx['pnl'] + upd_gmx['c'])
    return upd_gmx

def _pos_gmx(t, p, df_p, perp_type, collateral, leverage):
    pos_gmx = {
        'type': perp_type,
        't_opn': t,
        'p_opn': p,
        'c': collateral,
        'l': leverage,
        'f_ent': gmx_fee_entry,
        'f_ext': gmx_fee_exit,
        'f_rate': gmx_funding_rate,
        } 
    pos_gmx['s'] = pos_gmx['c']*pos_gmx['l']
    pos_gmx['p_ent'] = pos_gmx['p_opn'] * (1+pos_gmx['f_ent'])
    pos_gmx['p_liq'] = pos_gmx['p_ent'] * ((1-1/pos_gmx['l']) if pos_gmx['type'] == "long" else (1+1/pos_gmx['l']))
    pos_gmx = _upd_gmx(pos_gmx['p_opn'], pos_gmx['t_opn'], df_p, pos_gmx.copy())
    return pos_gmx

def _plot_pos_hedge(*arg):
    sub_timeline = f"Original and Hedged Positions were Active in {'{:,.2f}%'.format(100*out['position_perc_active'])}% of Time."
    sub_return = f"Improvement in Return with Hedging: {'{:,.2f}%'.format(100*out['hedged_imprv_real'])}."
    sub_exp_imprv = f"Leverage: x{pos_gmx['l']}."
    sub_exp_imprv+= f" Hedging Expected Improve {'{:,.2f}%'.format(100*lim_imprv)} for price scenarios."
    sub_exp_imprv+= f" ({'{:,.2f}%'.format(100*conc_imprv)} within concentration levels)."
    
    nr = 5
    _style_white()
    fig, axes = plt.subplots(nrows=nr, figsize=(14, 5*nr))
    
    ax = axes[0]
    ax.set_title(f"Original Position Value", pad=35)
    ax.text(0.5, 1.06, "Without Hedging.", ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_ylabel(f"$ Value", labelpad=10)
    ax.fill_between(df_uni['t'], df_org['x_usd'], color=tailwind['purple-500'], alpha=0.8, label=f"ETH")   
    ax.fill_between(df_uni['t'], df_org['x_usd'], df_org['v'], df_uni['v'], color=tailwind['emerald-500'], alpha=0.8, label=f"USDT")
    ax.set_ylim((0,max(ax.get_yticks())))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))

    ax = axes[1]
    ax.set_title(f"Hedged Position Value", pad=35)
    ax.text(0.5, 1.06, "With Hedging on Perpetuals Short.", ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_ylabel(f"$ Value", labelpad=10)
    ax.fill_between(df_uni['t'], df_uni['x_usd'], color=tailwind['purple-500'], alpha=0.8, label=f"ETH")   
    ax.fill_between(df_uni['t'], df_uni['x_usd'], df_uni['v'], color=tailwind['emerald-500'], alpha=0.8, label=f"USDT")
    ax.fill_between(df_uni['t'], df_uni['v'], df_gmx['v']+df_uni['v'], color=tailwind['indigo-500'], alpha=0.99, label=f"Perpetual")
    ax.set_ylim((0,max(ax.get_yticks())))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))

    ax = axes[2]
    ax.set_title(f"Positions Return", pad=35)
    ax.text(0.5, 1.06, sub_return, ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_ylabel(f"% Return", labelpad=10)
    ax.plot(df_p['_time'], df_p['ret_org'], tailwind['stone-600'], alpha=0.9, linewidth=2, label="Original Position")  
    ax.plot(df_p['_time'], df_p['ret_hedge'], tailwind['indigo-600'], alpha=0.9, linewidth=2, label="Hedged Position")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.1f}%'.format(x*100)))
    
    ax = axes[3]
    ax.set_title(f"Hedged Position Timeline", pad=35)
    ax.text(0.5, 1.06, sub_timeline, ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_ylabel(f"$ Price", labelpad=10)
    x_min, x_max = df_p['_time'].min(), df_p['_time'].max()
    y_min, y_max = df_p['open'].min()/1.05, df_p['open'].max()*1.05
    ax.plot(df_p['_time'], df_p['open'], tailwind['stone-800'], alpha=0.9, linewidth=2, label="Price")
    ax.set_ylim((y_min, y_max))
# =============================================================================
#     ax.scatter(df_lp[df_lp['signal_hedge']]['t'], df_lp[df_lp['signal_hedge']]['p'], 
#                color=tailwind['indigo-300'], s=33, label="Hedge Signals", alpha=.6, zorder=4)
# =============================================================================
    ax.scatter(df_gmx[df_gmx['t'] == df_gmx['t_opn']]['t'], df_gmx[df_gmx['t'] == df_gmx['t_opn']]['p'], 
               color=tailwind['indigo-500'], s=88, label="Hedge Open", alpha=.99, zorder=4)
    for t_opn in df_gmx['t_opn'].unique():
        if t_opn==t_opn: 
            df_gmx_id = df_gmx[df_gmx['t_opn']==t_opn]
            ax.plot(df_gmx_id['t'], df_gmx_id['p_liq'], alpha=0.9, linewidth=3, color=tailwind['red-500'])
    ax.plot([x_min], [y_min], alpha=0.9, linewidth=2, color=tailwind['red-500'], label="Price Perp. Liquidation")
    for a in df_org['active_id'].unique():
        df_org_a = df_org[df_org['active_id']==a]
        pos_active = df_org_a['active'].iloc[0]
        a_color = tailwind['emerald-400'] if pos_active else tailwind['rose-400']
        ax.fill_between(df_org_a['t'],  df_org_a['p_a'],  df_org_a['p_b'], color=a_color, alpha=0.6)
    ax.fill_between([x_min],  y_min, y_min, color=tailwind['emerald-500'], alpha=0.7, label="LP Active")    
    ax.fill_between([x_min],  y_min, y_min, color=tailwind['rose-500'], alpha=0.7, label="LP Not active")    
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))

    ax = axes[4]
    ax.set_title(f"Positions Value for Price Scenarios", pad=35)
    ax.text(0.5, 1.06, sub_exp_imprv, ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_ylabel(f"$ Value", labelpad=10)
    ax.set_xlabel(f"$ Price Scenario", labelpad=10)
    ax.plot(p_values, df_org_v['v'], color=tailwind['stone-600'], alpha=0.9, linewidth=3, label=f"Original Position")
    ax.plot(p_values, df_uni_v['v'] + df_gmx_v['v'], color=tailwind['indigo-600'], alpha=0.99, linewidth=3, label=f"Hedged Position")
    y_min, y_max = max(.0, ax.get_yticks().min()), ax.get_yticks().max()
    ax.vlines(pos_gmx['p_opn'], y_min, y_max, color=tailwind['stone-900'], alpha=.9, label="Price Open")
    ax.vlines(pos_gmx['p_liq'], y_min, y_max, color=tailwind['red-500'], label="Price Perp. Liquidation")
    ax.fill_between([p_a,p_b], [y_min, y_min], [y_max, y_max], color=tailwind['emerald-400'], alpha=0.6, label="LP Active") 
    ax.set_xlim((p_values.min()),p_values.max())
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))

    for ax in axes:
        ax.set_yticks(ax.get_yticks())
        ax.set_xticks(ax.get_xticks())
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
        for spine in ax.spines.values(): spine.set_visible(False)
        legend = ax.legend(loc='lower right', fontsize=14)
        legend.get_frame().set_facecolor('white')  # Sets the background color to white
        legend.get_frame().set_edgecolor('black')  # Optional: Adds a border to the legend
        legend.get_frame().set_alpha(.6)  # Ensures no transparency (fully opaque)
    fig.tight_layout(rect=[0.01, 0.01, .99, .99])
    fig.subplots_adjust(hspace=.5)
    fig.savefig(os.path.join(folder, f'Position Hedge {position_id}'), dpi=200)
    plt.show()
    fig.clf()


# =============================================================================
# Simulate Hedge
# =============================================================================

# =============================================================================
# perp_l_min = 25
# perp_l_max = 25
# perp_time_min = 6
# perp_time_max = 12
# 
# data_hedge, data_perp = [], []
# perp_times_max = [12, 18, 24,48, 72, 96]
# for perp_time_max in perp_times_max:0
# =============================================================================

s = 'ETH'
perp_type = 'short'
perp_share = .05
perp_tp = .30
perp_state = {
    'lowvol': {
        'l_min': 40,
        'l_max': 40,
        't_hours_min': 1,
        't_hours_max': 12,
        },
    'midvol': {
        'l_min': 25,
        'l_max': 25,
        't_hours_min': 1,
        't_hours_max': 12,
        },
    'highvol': {
        'l_min': 12,
        'l_max': 12,
        't_hours_min': 1,
        't_hours_max': 12,
        },
    }
data_hedge, data_perps = [], []
for i, position_id in enumerate(pool_ids[:]):
    print(i, position_id)
    out = {'nr':i, 'id': position_id}
    pos = df_pos[df_pos['id'] == position_id].iloc[0].to_dict()
    pos_log = df_log[df_log['position_id'] == position_id].copy()
    pos_log.loc[:,'usd'] = (pos_log['amount1'] + pos_log['amount0']*pos_log['price']).values
    deposited = pos_log[pos_log['type']=='deposits'].iloc[0].to_dict()
    fees_usd = pos_log[pos_log['type']=='claimed-fees'].iloc[0]['usd']
    df_xtreamly = df_state_xtreamly[
        (df_state_xtreamly['symbol']==s) &
        (df_state_xtreamly['_time']>=pos_log['_time'].min()) &
        (df_state_xtreamly['_time']<=pos_log['_time'].max())
        ].copy().reset_index(drop=True)
    df_xtreamly = df_xtreamly[[c for c in df_xtreamly.columns if c not in ['symbol']]]
    df_p = df_market[
        (df_market['name']==s) &
        (df_market['_time']>=pos_log['_time'].min()) &
        (df_market['_time']<=pos_log['_time'].max())
        ].copy().reset_index(drop=True)
    df_p = df_p[['_time', 'open', 'low', 'high']].merge(df_xtreamly, on='_time', how='left')

    t = df_p['_time'].iloc[0]
    p = deposited['price'] #df_p['open'].iloc[0]
    p_a = pos['price_lower']
    p_b = pos['price_upper']
    x_opn = deposited['deposited_token0']
    y_opn = deposited['deposited_token1']

    pos_org = _pos_uni(t, p, p_a, p_b, x_opn, y_opn)
    pos_uni = _pos_uni(t, p, p_a, p_b, x_opn*(1-perp_share), y_opn*(1-perp_share))

    p_gmx = df_p['open'].iloc[0]
    collateral = np.abs(pos_org['v'])*(perp_share)
    leverage = int((1/(p_b/p-1)))-1
    perp = perp_state[df_p['state'].iloc[0]]
    leverage = min(perp['l_max'],max(perp['l_min'],leverage))     
    pos_gmx = _pos_gmx(t, p_gmx, df_p, perp_type, collateral, leverage)
    pos_gmx = {**pos_gmx, **perp}
    pos_gmx['state'] = df_p['state'].iloc[0]

    # Simulate
    data_uni, data_gmx = [pos_uni], [pos_gmx]
    for i,r in df_p.iloc[1:].iterrows():
        t = r['_time']
        p = r['open']
        pos_uni = data_uni[-1].copy()
        pos_gmx = data_gmx[-1].copy()
        perp = perp_state[df_p['state'].iloc[i]]

        if len(pos_gmx) > 0:
            new_gmx = _upd_gmx(p, t, df_p, pos_gmx)
            new_uni = _upd_uni(p, t, pos_uni)
            
            close = new_gmx['t_hours_max'] <= new_gmx['t_hours'] or new_gmx['liq']
            if close:
                new_uni = _pos_uni(t, p, p_a, p_b, 
                                   new_uni['x'] * (1+new_gmx['v']/new_uni['v']), 
                                   new_uni['y'] * (1+new_gmx['v']/new_uni['v']))
                new_gmx['v'] = 0
                new_gmx['cls'] = close

        if len(pos_gmx) == 0 or 'cls' in pos_gmx:
            new_uni = _upd_uni(p, t, pos_uni)
            hedge = (p >= new_uni['p_mid']/1.01) and (p <= new_uni['p_mid']*1.01) 
            if hedge:
                collateral = pos_uni['v']*perp_share
                leverage = int((1/(p_b/p-1)))-1
                leverage = min(perp['l_max'],max(perp['l_min'],leverage))    
                new_uni = _pos_uni(t, p, p_a, p_b, 
                                   new_uni['x'] * (1-collateral/new_uni['v']), 
                                   new_uni['y'] * (1-collateral/new_uni['v']))
                new_uni['hedge'] = hedge
                new_gmx = _pos_gmx(t, p, df_p, perp_type, collateral, leverage)
                new_gmx = {**new_gmx, **perp}
                new_gmx['state'] = df_p['state'].iloc[i]
            else: new_gmx = {}
        data_uni += [new_uni]
        data_gmx += [new_gmx]

    df_gmx = pd.DataFrame(data_gmx)
    df_gmx['v'] = df_gmx['v'].fillna(.0)
    df_gmx['liq'] = df_gmx['liq'].fillna(False)
    df_gmx['liq']*= 1
    df_uni = pd.DataFrame(data_uni)
    df_org = _df_uni(df_p, pos_org)

    fee_prop = df_uni[df_uni['active']]['v'].sum()/df_org[df_org['active']]['v'].sum()
    df_uni = _f_cum(df_uni, fees_usd*fee_prop, pos_org['inv_usd'])
    df_org = _f_cum(df_org, fees_usd, pos_org['inv_usd'])
    
    df_perps = df_gmx.groupby(['t_opn']).agg(
        collateral = ('c', 'mean'),
        pnl = ('pnl', 'last'),
        liq = ('liq', 'sum'),
        cost_rate = ('cost_rate', 'last'),
        state = ('state', 'first'),
        ).reset_index()
    df_perps['position_id'] = position_id
    
    df_p['ret_org'] = (df_org['v']+df_org['f_cum'])/df_org['inv_usd'].iloc[0]-1
    df_p['ret_org_f_cum'] = (df_org['f_cum'])/df_org['inv_usd'].iloc[0]
    df_p['ret_org_imp_loss'] = (df_org['v'])/df_org['inv_usd'].iloc[0]-1
    df_p['ret_hedge'] = (df_uni['v']+df_uni['f_cum']+df_gmx['v'])/df_org['inv_usd'].iloc[0]-1
    df_p['ret_hedge_f_cum'] = (df_uni['f_cum'])/df_org['inv_usd'].iloc[0]
    
    # Summary
    pos_uni = data_uni[0].copy()
    pos_gmx = data_gmx[0].copy()
    p_values = np.linspace(pos_uni['p']/1.4, pos_uni['p']*1.4, 1000)
    p_conc = (pos_uni['p_a']<=p_values) & (p_values<=pos_uni['p_b'])
    df_gmx_v = pd.DataFrame([_upd_gmx(p, pos_gmx['t_opn'], df_p, pos_gmx.copy()) for p in p_values])
    df_uni_v = pd.DataFrame([_upd_uni(p, pos_uni['t_opn'], pos_uni.copy()) for p in p_values])
    df_org_v = pd.DataFrame([_upd_uni(p, pos_uni['t_opn'], pos_org.copy()) for p in p_values]) 

    lim_avg_org = np.mean(df_org_v['v']).round(2)
    lim_avg_hedge = np.mean(df_uni_v['v']+df_gmx_v['v']).round(2)
    lim_imprv = (lim_avg_hedge-lim_avg_org)/pos_uni['v']
    conc_avg_org = np.mean(df_org_v[p_conc]['v']).round(2)
    conc_avg_hedge = np.mean(df_uni_v[p_conc]['v']+df_gmx_v[p_conc]['v']).round(2)
    conc_imprv = (conc_avg_hedge-conc_avg_org)/pos_uni['v'] 

    out = {
        'position_id': position_id,
        'position_time_min': pos_log['_time'].min(),
        'position_time_max': pos_log['_time'].max(),
        'position_usd_inv': df_org['inv_usd'].iloc[0],
        'position_usd_fee': df_org['f_cum'].iloc[-1],
        'position_usd_v': df_org['v'].iloc[-1],
        'position_perc_active': df_org['active'].sum()/df_org.shape[0],
        'position_perc_ret': df_p['ret_org'].iloc[-1],
        'hedged_usd_fee': df_uni['f_cum'].iloc[-1],
        'hedged_perc_ret': df_p['ret_hedge'].iloc[-1],
        'hedged_perc_impr': df_p['ret_hedge'].iloc[-1] - df_p['ret_org'].iloc[-1],
        'hedged_usd_v': df_uni['v'].iloc[-1] + df_gmx['v'].iloc[-1],
        'hedged_usd_perp_count': df_perps.shape[0],
        'hedged_usd_perp_liq': np.sum(df_perps['liq']),
        'hedged_usd_perp_collateral': df_perps['collateral'].sum(),
        'hedged_usd_perp_pnl': df_perps['pnl'].sum(),
        'hedged_usd_perp_ret': df_perps['pnl'].sum()/df_perps['collateral'].sum(),
        'hedged_usd_perp_cost_rate': df_perps['cost_rate'].iloc[-1],
        'hedged_imprv_exp': lim_imprv,
        'hedged_imprv_conc': conc_imprv,
        'hedged_imprv_real': df_p['ret_hedge'].iloc[-1]-df_p['ret_org'].iloc[-1],
        }
    data_hedge += [out]
    data_perps += [df_perps]
    #_plot_pos_hedge()
        
df_pos_hedge = pd.DataFrame(data_hedge)
df_pos_hedge['duration'] = (df_pos_hedge['position_time_max']-df_pos_hedge['position_time_min']).dt.total_seconds()/(24*3600) 
df_perps_all = pd.concat(data_perps)
df_perps_all['ret'] = df_perps_all['pnl']/df_perps_all['collateral']

df_pos_hedge['hedged_imprv_real'].mean()


# =============================================================================
# Plots
# =============================================================================
#for hedge_time_max in hedge_times_max:
    
df_pos_hedge = pd.DataFrame(data_hedge)
df_pos_hedge['duration'] = (df_pos_hedge['position_time_max']-df_pos_hedge['position_time_min']).dt.total_seconds()/(24*3600)


df_perps_all = pd.concat(data_perps)
df_perps_all['ret'] = df_perps_all['pnl']/df_perps_all['collateral']

_style_white()
fig, ax = plt.subplots(figsize=(16, 7))
df = df_perps_all.copy()
ax.set_title(f"Histogram of % Return on Hedging", pad=30)
ax.set_ylabel(f"% Share", labelpad=20)
ax.set_xlabel(f"% Improvement", labelpad=20)
bin_edges = np.linspace(df['ret'].min(), df['ret'].max(), 40)
bin_width = np.diff(bin_edges)[0]  # Calculate bin width
bar_width = bin_width / 3  # Divide bins equally among the three categories
for i, (state, color) in enumerate(zip(
    ['lowvol', 'midvol', 'highvol'], 
    [tailwind['teal-400'], tailwind['amber-400'], tailwind['pink-400']]
)):
    counts, _ = np.histogram(df[df['state'] == state]['ret'], bins=bin_edges)
    ax.bar(bin_edges[:-1] + i * bar_width, counts/sum(counts), width=bar_width, color=color, alpha=0.9, label=state.capitalize())
yticks = ax.get_yticks()
ax.vlines(0, 0, max(yticks), color=tailwind['stone-300'])
ax.set_yticks(ax.get_yticks())
ax.set_xticks(ax.get_xticks())
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.1f}%'.format(x*100)))
legend = ax.legend(loc='upper left')
legend.get_frame().set_alpha(0.9)
ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
for spine in ax.spines.values(): spine.set_visible(False)
fig.tight_layout(rect=[0.004, 0.004, .996, .996])
fig.savefig(os.path.join(folder, f'Histogram Return Perpetuals.png'), dpi=200)
fig.clf()

_style_white()
fig, ax = plt.subplots(figsize=(16, 7))
df = df_pos_hedge.copy()
cols = ['position_perc_ret', 'hedged_perc_ret']
ax.set_title(f"Histogram of % Performance for Original and Hedged Positions", pad=30)
ax.set_ylabel(f"# Count", labelpad=10)
ax.set_xlabel(f"% Return", labelpad=10)
bin_edges = np.linspace(df['position_perc_ret'].min(), df['position_perc_ret'].max(), 40)
bin_width = np.diff(bin_edges)[0]  # Calculate bin width
bar_width = bin_width / 2  # Divide bins equally among the three categories
for i, (c, color, lbl) in enumerate(zip(cols, [tailwind['stone-400'], tailwind['indigo-400']], ['Original','Hedged'])):
    counts, _ = np.histogram(df[c], bins=bin_edges)
    ax.bar(bin_edges[:-1] + i * bar_width, counts, width=bar_width, color=color, alpha=0.9, label=f"{lbl}")
yticks = ax.get_yticks()
ax.set_yticks(ax.get_yticks())
ax.set_xticks(ax.get_xticks())
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.1f}%'.format(x*100)))
legend = ax.legend(loc='upper left')
legend.get_frame().set_alpha(0.9)
ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
for spine in ax.spines.values(): spine.set_visible(False)
fig.tight_layout(rect=[0.004, 0.004, .996, .996])
fig.savefig(os.path.join(folder, f'Histogram Performance.png'), dpi=200)
fig.clf()
    
_style_white()
fig, ax = plt.subplots(figsize=(16, 7))
df = df_pos_hedge.copy()
ax.set_title(f"Histogram of % Improvement from Original to Hedged Positions", pad=30)
ax.set_ylabel(f"# Count", labelpad=10)
ax.set_xlabel(f"% Improvement", labelpad=10)
ax.hist(df['hedged_perc_impr'],bins=50, color=tailwind['teal-400'], alpha=0.9, label=f"Improvement on Hedging Position")
yticks = ax.get_yticks()
ax.set_yticks(ax.get_yticks())
ax.set_xticks(ax.get_xticks())
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.1f}%'.format(x*100)))
legend = ax.legend(loc='upper left')
legend.get_frame().set_alpha(0.9)
ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
for spine in ax.spines.values(): spine.set_visible(False)
fig.tight_layout(rect=[0.004, 0.004, .996, .996])
fig.savefig(os.path.join(folder, f'Histogram Improvement.png'), dpi=200)
fig.clf()
    
    
_style_white()
fig, ax = plt.subplots(figsize=(16, 7))
df = df_pos_hedge.copy()
ax.set_title(f"Histogram of Hedging Positions % Total Returns on Perpetuals", pad=30)
ax.set_ylabel(f"# Count", labelpad=10)
ax.set_xlabel(f"% Return Perpetuals", labelpad=10)
ax.hist(df['hedged_usd_perp_ret'],bins=40, color=tailwind['indigo-700'], alpha=0.9, label=f"LP Active Time")
yticks = ax.get_yticks()
ax.set_yticks(ax.get_yticks())
ax.set_xticks(ax.get_xticks())
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x*100)))
legend = ax.legend(loc='upper left')
legend.get_frame().set_alpha(0.9)
ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
for spine in ax.spines.values(): spine.set_visible(False)
fig.tight_layout(rect=[0.004, 0.004, .996, .996])
fig.savefig(os.path.join(folder, f'Histogram Return Perpetuals.png'), dpi=200)
fig.clf()

_style_white()
fig, ax = plt.subplots(figsize=(16, 7))
df = df_pos_hedge.copy()
ax.set_title(f"Histogram of % Active Time in LP Positions", pad=30)
ax.set_ylabel(f"# Count", labelpad=10)
ax.set_xlabel(f"% Active", labelpad=10)
ax.hist(df['position_perc_active'],bins=40, color=tailwind['teal-400'], alpha=0.9, label=f"% Active Time")
yticks = ax.get_yticks()
ax.set_yticks(ax.get_yticks())
ax.set_xticks(ax.get_xticks())
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.1f}%'.format(x*100)))
legend = ax.legend(loc='upper left')
legend.get_frame().set_alpha(0.9)
ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
for spine in ax.spines.values(): spine.set_visible(False)
fig.tight_layout(rect=[0.004, 0.004, .996, .996])
fig.savefig(os.path.join(folder, f'Histogram Active.png'), dpi=200)
fig.clf()
    
# Aggr Stats
df_pos_hedge = pd.DataFrame(data_hedge)
df_pos_hedge['duration'] = (df_pos_hedge['position_time_max']-df_pos_hedge['position_time_min']).dt.total_seconds()/(24*3600)
df_pos_hedge['pool'] = pool
df_hedge_agr = df_pos_hedge.groupby(['pool']).agg(
        position_perc_active=('position_perc_active', 'mean'),
        position_perc_ret=('position_perc_ret', 'mean'),
        hedged_perc_ret = ('hedged_perc_ret','mean'),
        hedged_usd_perp_count = ('hedged_usd_perp_count','mean'),
        hedged_usd_perp_liq = ('hedged_usd_perp_liq','mean'),
        ).reset_index()

df_perps_all = pd.concat(data_perps)
df_perps_all['ret'] = df_perps_all['pnl']/df_perps_all['collateral']
df_perp_agr = df_perps_all.groupby(['state']).agg(
        perp_ret=('ret', 'mean'),
        liq = ('liq','mean'),
        position_id=('position_id', 'nunique')
        ).reset_index()
cols_perp = {
    'state': 'Status',
    'position_id': '# Positions',
    'perp_ret': '% Performance',
    'liq': '% Liquidated',
    }
df_perp_agr = df_perp_agr[cols_perp.keys()].rename(columns=cols_perp)
for c in df_perp_agr.columns:
    if '%' in c:
        df_perp_agr[c]*=1
        df_perp_agr[c] = df_perp_agr[c].apply(lambda x: f"{x:.2%}")

fig, ax = plt.subplots(figsize=(8, 1))  # Adjust figure size
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df_perp_agr.values, 
                 colLabels=df_perp_agr.columns, 
                 cellLoc='center', 
                 loc='center')
ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
for spine in ax.spines.values(): spine.set_visible(False)
plt.savefig(os.path.join(folder, f'Perps Performance.png'), bbox_inches='tight', dpi=300)



# =============================================================================
# _style_white()
# fig, ax = plt.subplots(figsize=(16, 7))
# df = df_pos_hedge.copy()
# ax.set_title(f"Histogram of Hedging Positions % Total Returns on Perpetuals", pad=30)
# ax.set_ylabel(f"# Count", labelpad=10)
# ax.set_xlabel(f"% Active in Time", labelpad=10)
# ax.hist(df['hedged_usd_perp_ret'],bins=40, color=tailwind['indigo-700'], alpha=0.9, label=f"LP Active Time")
# yticks = ax.get_yticks()
# ax.set_yticks(ax.get_yticks())
# ax.set_xticks(ax.get_xticks())
# ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
# ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
# ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x*100)))
# legend = ax.legend(loc='upper left')
# legend.get_frame().set_alpha(0.9)
# ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
# for spine in ax.spines.values(): spine.set_visible(False)
# fig.tight_layout(rect=[0.004, 0.004, .996, .996])
# fig.savefig(os.path.join(folder, f'Histogram Liquidated Perpetuals.png'), dpi=200)
# fig.clf()
# =============================================================================


# position_id = '878458_0x902a7cebc98daa5a0e6de468052c75719c014797'
# position_id = '879576_0x5b393bd3c1d0d334b8bb9ae106edb4ec33801a3c'
# position_id = '851265_0x0d21716f645ce331cc4fadb9e621980acadf56dc'
# position_id = '873626_0x37c4d1abc89e1bb1ceaa9df3c17d4464fe95e21c'
# position_id = '840439_0x4942e2b839fc479c27c496b7758bab94ceb6b684'

# =============================================================================
# t = df_p['_time'].iloc[0]
# p = deposited['price'] #df_p['open'].iloc[0]
# pos_uni = _pos_uni(t, p, p_a, p_b, x_opn*(1-perp_share), y_opn*(1-perp_share))
# p_gmx = p#df_p['open'].iloc[0]
# collateral = np.abs(pos_org['v'])*(perp_share)
# leverage = int((1/(p_b/p-1)))-1
# leverage = min(perp_l_max,max(perp_l_min,leverage))     
# pos_gmx = _pos_gmx(t, p_gmx, df_p, perp_type, collateral, leverage)
# pos_gmx_min = _pos_gmx(t, (p+p_a)/2, df_p, perp_type, collateral, leverage)
# pos_gmx_max = _pos_gmx(t, p*1.01, df_p, perp_type, collateral, leverage)
# p_values = np.linspace(pos_uni['p']/1.4, pos_uni['p']*1.4, 1000)
# p_conc = (pos_uni['p_a']<=p_values) & (p_values<=pos_uni['p_b'])
# df_gmx_v = pd.DataFrame([_upd_gmx(p, t, df_p, pos_gmx.copy()) for p in p_values])
# df_gmx_min_v = pd.DataFrame([_upd_gmx(p, t, df_p, pos_gmx_min.copy()) for p in p_values])
# df_gmx_max_v = pd.DataFrame([_upd_gmx(p, t, df_p, pos_gmx_max.copy()) for p in p_values])
# df_uni_v = pd.DataFrame([_upd_uni(p, t, pos_uni.copy()) for p in p_values])
# df_org_v = pd.DataFrame([_upd_uni(p, t, pos_org.copy()) for p in p_values]) 
# 
# _style_white()
# fig, ax = plt.subplots(figsize=(16, 7))
# ax.set_title(f"Positions Value for Price Scenarios", pad=35)
# ax.text(0.5, 1.06, sub_exp_imprv, ha='center', va='center', fontsize=14, transform=ax.transAxes)
# ax.set_ylabel(f"$ Value", labelpad=10)
# ax.set_xlabel(f"$ Price Scenario", labelpad=10)
# ax.plot(p_values, df_org_v['v'], color=tailwind['stone-600'], alpha=0.9, linewidth=3, label=f"Original Position")
# ax.plot(p_values, df_uni_v['v'] + df_gmx_v['v'], color=tailwind['indigo-600'], alpha=0.99, linewidth=3, label=f"Hedged Position")
# ax.plot(p_values, df_uni_v['v'] + df_gmx_min_v['v'], color=tailwind['indigo-300'], alpha=0.99, linewidth=3, label=f"Hedged Position Min")
# ax.plot(p_values, df_uni_v['v'] + df_gmx_max_v['v'], color=tailwind['indigo-800'], alpha=0.99, linewidth=3, label=f"Hedged Position Max")
# y_min, y_max = max(.0, ax.get_yticks().min()), ax.get_yticks().max()
# ax.vlines(pos_gmx['p_opn'], y_min, y_max, color=tailwind['stone-900'], alpha=.9, label="Price Open")
# ax.vlines(pos_gmx['p_liq'], y_min, y_max, color=tailwind['red-500'], label="Price Perp. Liquidation")
# ax.fill_between([p_a,p_b], [y_min, y_min], [y_max, y_max], color=tailwind['emerald-400'], alpha=0.6, label="LP Active") 
# ax.set_xlim((p_values.min()),p_values.max())
# ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
# ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
# ax.set_yticks(ax.get_yticks())
# ax.set_xticks(ax.get_xticks())
# ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
# ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
# for spine in ax.spines.values(): spine.set_visible(False)
# legend = ax.legend(loc='lower right', fontsize=14)
# legend.get_frame().set_facecolor('white')  # Sets the background color to white
# legend.get_frame().set_edgecolor('black')  # Optional: Adds a border to the legend
# legend.get_frame().set_alpha(.6)  # Ensures no transparency (fully opaque)
# fig.tight_layout(rect=[0.01, 0.01, .99, .99])
# fig.savefig(os.path.join(folder, f'Position Value {position_id}'), dpi=200)
# plt.show()
# fig.clf()
# =============================================================================
