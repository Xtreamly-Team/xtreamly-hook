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
df_state_xtreamly = pd.read_csv(os.path.join('data', 'df_forecasts.csv'))

df_market['_time'] = [datetime.fromisoformat(str(t)).replace(tzinfo=None) for t in df_market['_time']]
df_gmx_rates['_time'] = [datetime.fromisoformat(str(t)).replace(tzinfo=None) for t in df_gmx_rates['timestamp']]
df_uni_positions['_time'] = [datetime.fromisoformat(str(t)).replace(tzinfo=None) for t in df_uni_positions['_time']]
df_uni_logs['_time'] = [datetime.fromisoformat(str(t)).replace(tzinfo=None) for t in df_uni_logs['_time']]
df_state_xtreamly['_time'] = [datetime.fromisoformat(str(t)).replace(tzinfo=None) for t in df_state_xtreamly['_time']]

where_id = [True] + list(df_state_xtreamly[1:]['state'].values != df_state_xtreamly[:-1]['state'].values)
df_state_xtreamly.loc[where_id, 'state_id'] = df_state_xtreamly.loc[where_id, 'state']+'_'+df_state_xtreamly[where_id].index.values.astype(str)
df_state_xtreamly['state_id'] = df_state_xtreamly['state_id'].ffill()

with open(os.path.join('data', 'pools.json'), 'r') as file: data_pools = json.load(file)
df_pools = pd.DataFrame(data_pools)
df_pools['pool'] = df_pools['pool'].str.lower()
df_pools = df_pools[[('BTC' in t or 'ETH' in t) and 'USD' in t for t in df_pools['type']]]
df_pools = df_pools[df_pools['type'] != 'USDC / ETH']

# =============================================================================
# Filter Pool Positions
# =============================================================================
start_time = datetime.fromisoformat('2024-12-01').replace(tzinfo=None)
end_time = datetime.fromisoformat('2025-03-25').replace(tzinfo=None)

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
    df_pos['pool'] = df_pos['pool'].str.lower()
    df_pos = df_pos[
        (df_pos['pool'].isin(df_pools['pool'].unique())) & 
        (df_pos['active_opn']) &
        (df_pos['logs_count']==3) &
        (df_pos['duration_min']>=60) &
        (df_pos['concentration']<=.3) &
        (df_pos['time_min']>=start_time) &
        (df_pos['time_max']<=end_time)]
    return df_pos, df_log

df_pos, df_log = _pos_and_id()

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
    data_lp = [pos_uni] + [_upd_uni(r['open'], r['_time'], pos_uni.copy()) 
                           for _,r in df_p.iloc[1:].iterrows()]
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
perp_type = 'short'
data_hedge, data_perps = [], []

# =============================================================================
# # For optimizing parameters
# for perp_share in [.04, .05]:0
# for perp_t_max in [32, 24,18, 12]:0
# =============================================================================
perp_share = .05
perp_t_max = 12

for pool in df_pools['pool'].unique(): 
    perp_state = {
        'lowvol': {
            'l_set': 30,
            't_max': perp_t_max,
            },
        'midvol': {
            'l_set': 25,
            't_max': perp_t_max,
            },
        'highvol': {
            'l_set': 20,
            't_max': perp_t_max,
            },
        }
    pool_info = df_pools[df_pools['pool']==pool]
    s = 'ETH' if 'ETH' in str(pool_info['type']) else 'BTC'
    pool_ids = df_pos[df_pos['pool']==pool]['id'].unique()
    
    print(pool_info['type'].iloc[0], pool_info['fee'].iloc[0], len(pool_ids))
    for position_idx, position_id in enumerate(pool_ids[:]):
        print(pool_info['type'].iloc[0], perp_share, perp_t_max, position_idx, position_id)
    
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
        df_xtreamly = df_xtreamly[[c for c in df_xtreamly.columns if c not in ['symbol', 'open', 'close']]]
        
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
    
        perp = perp_state[df_p['state'].iloc[0]]
        p_gmx = df_p['open'].iloc[0]
        collateral = np.abs(pos_org['v'])*(perp_share)
        leverage = perp['l_set']
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
                            
                close = ''
                close = 't_max' if new_gmx['t_max'] <= new_gmx['t_hours'] else close
                close = 'liq' if new_gmx['liq'] else close
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
                    leverage = perp['l_set']
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
        #df_gmx = df_gmx[df_gmx['t_opn']==df_gmx['t_opn']]
        df_gmx['v'] = df_gmx['v'].fillna(.0)
        if not 'cls' in df_gmx.columns: df_gmx['cls'] = '' 
        df_perps = df_gmx.groupby(['t_opn']).agg(
            state = ('state', 'first'),
            l = ('l', 'first'),
            collateral = ('c', 'mean'),
            pnl = ('pnl', 'last'),
            close = ('cls', 'last'),
            t_hours = ('t_hours', 'last'),
            cost_rate = ('cost_rate', 'last'),
            ).reset_index()
        df_perps['pool'] = pool
        df_perps['position_id'] = position_id
        df_perps['perp_share'] = perp_share
        df_perps['perp_t_max'] = perp_t_max
        
        df_uni = pd.DataFrame(data_uni)
        df_org = _df_uni(df_p, pos_org)
        fee_prop = df_uni[df_uni['active']]['v'].sum()/df_org[df_org['active']]['v'].sum()
        df_uni = _f_cum(df_uni, fees_usd*fee_prop, pos_org['inv_usd'])
        df_org = _f_cum(df_org, fees_usd, pos_org['inv_usd'])
    
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
            'perp_share': perp_share,
            'perp_t_max': perp_t_max,
            'pool': pool,
            'position_id': position_id,
            'position_t_min': pos_log['_time'].min(),
            'position_t_max': pos_log['_time'].max(),
            'position_usd_inv': df_org['inv_usd'].iloc[0],
            'position_usd_fee': df_org['f_cum'].iloc[-1],
            'position_usd_v': df_org['v'].iloc[-1],
            'position_perc_active': df_org['active'].sum()/df_org.shape[0],
            'position_perc_ret': df_p['ret_org'].iloc[-1],
            'hedged_usd_fee': df_uni['f_cum'].iloc[-1],
            'hedged_perc_ret': df_p['ret_hedge'].iloc[-1],
            'hedged_perc_impr': df_p['ret_hedge'].iloc[-1] - df_p['ret_org'].iloc[-1],
            'hedged_usd_v': df_uni['v'].iloc[-1] + df_gmx['v'].iloc[-1],
            'hedged_imprv_exp': lim_imprv,
            'hedged_imprv_conc': conc_imprv,
            'hedged_imprv_real': df_p['ret_hedge'].iloc[-1]-df_p['ret_org'].iloc[-1],
            'perp_count': df_perps.shape[0],
            'perp_liq': np.sum(df_perps['close']=='liq'),
            'perp_collateral': df_perps['collateral'].sum(),
            'perp_pnl': df_perps['pnl'].sum(),
            'perp_ret': df_perps['pnl'].sum()/df_perps['collateral'].sum(),
            'perp_cost_rate': df_perps['cost_rate'].iloc[-1],
            }
        data_hedge += [out]
        data_perps += [df_perps]
        if (df_perps['pnl'].sum()>0) and (position_idx < 10): _plot_pos_hedge()

# Data  
df_hedge_pos = pd.DataFrame(data_hedge)
df_hedge_pos['duration'] = (df_hedge_pos['position_t_max']-df_hedge_pos['position_t_min']).dt.total_seconds()/(24*3600)

df_perps_all = pd.concat(data_perps)
df_perps_all['ret'] = df_perps_all['pnl']/df_perps_all['collateral']
df_perps_all['liq'] = [1 if c=='liq' else 0 for c in df_perps_all['close']]
df_perps_all['t_max'] = [1 if c=='t_max' else 0 for c in df_perps_all['close']]
df_perps_all['lp_cls'] = [1 if c in ['', None] else 0 for c in df_perps_all['close']]
df_perps_all['x'] = 1

# =============================================================================
# Save
# =============================================================================
df_hedge_pos.to_csv(os.path.join(folder, f'df_hedge_pos.csv'), index=False)
df_perps_all.to_csv(os.path.join(folder, f'df_perps_all.csv'), index=False)

# =============================================================================
# # Aggr Stats
# =============================================================================
df_pools['ids'] = [df_hedge_pos[df_hedge_pos['pool']==pool]['position_id'].nunique() 
                    for pool in df_pools['pool'].unique()]
df_hedge_agr = df_hedge_pos.groupby([ 'pool']).agg(
        position_id = ('position_id','count'),
        position_perc_active=('position_perc_active', 'mean'),
        position_perc_ret=('position_perc_ret', 'mean'),
        position_usd_inv = ('position_usd_inv','sum'),
        hedged_perc_impr=('hedged_perc_impr', 'mean'),
        hedged_perc_ret = ('hedged_perc_ret','mean'),
        perp_ret = ('perp_ret','mean'),
        perp_count = ('perp_count','mean'),
        ).reset_index()
df_hedge_agr = df_hedge_agr.merge(
    df_pools[['pool','type','fee']], on='pool', how='left')
df_perp_agr = df_perps_all.groupby(['state']).agg(
        perp_ret=('ret', 'mean'),
        perp_pnl=('pnl', 'sum'),
        collateral=('collateral', 'sum'),
        position_id=('position_id', 'nunique'),
        lp_cls = ('lp_cls','mean'),
        liq = ('liq','mean'),
        t_max = ('t_max','mean'),
        ).reset_index()

df_perp_cum = df_perps_all.groupby(['x']).agg(
        perp_ret=('ret', 'mean'),
        perp_pnl=('pnl', 'sum'),
        collateral=('collateral', 'sum'),
        position_id=('position_id', 'nunique'),
        lp_cls = ('lp_cls','mean'),
        liq = ('liq','mean'),
        t_max = ('t_max','mean'),
        ).reset_index()

dict_perp_close = {
    '# Liquidated': df_perps_all['liq'].sum(),
    '# Expired': df_perps_all['t_max'].sum(),
    '# LP Closed': df_perps_all['lp_cls'].sum(),
    }

# =============================================================================
# Tables
# =============================================================================
color_menu = tailwind['stone-800']
def _tbl(*arg):
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, 
                     colLabels=df.columns, 
                     cellLoc='center', loc='center')
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(tailwind['stone-200'])
        cell.set_linewidth(1)
        cell.set_height(.2)
        if row == 0:
            cell.set_height(.25)
            cell.set_text_props(weight='semibold', color='white')
            cell.set_facecolor(color_menu)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    fig.tight_layout(rect=[0.004, 0.004, .996, .996])
    plt.show()
    return fig

fig, ax = plt.subplots(figsize=(7, 3.5))
color_menu = tailwind['pink-800']
df = df_hedge_agr.copy()
print("${:,.0f}".format(sum(df_hedge_agr['position_usd_inv']*df_hedge_agr['hedged_perc_impr'])))
print("${:,.0f}".format(df_hedge_agr['position_usd_inv'].sum()))
print("{:,.0f}".format(df_hedge_agr['position_id'].sum()))
df['fee'] /= 10000
cols_perp = {
    'type': 'Pool',
    'fee': '% Fee',
    'position_id': '# LP positions',
    'position_perc_active': '% Avg. Time\n Active',
    'position_usd_inv': '$ Total\n Deposits',    
    }
df = df[cols_perp.keys()].rename(columns=cols_perp)
for c in df.columns:
    if '%' in c:
        df[c] = df[c].apply(lambda x: f"{x:.2%}")
    if '$' in c:
        df[c] = df[c].apply(lambda x: f"${x:,.0f}")        
fig = _tbl()
fig.savefig(os.path.join(folder, f'TblPools.png'), bbox_inches='tight', dpi=300)

fig, ax = plt.subplots(figsize=(9, 3.5))
color_menu = tailwind['indigo-700']
df = df_hedge_agr.copy()
df['fee'] /= 10000
cols_perp = {
    'type': 'Pool',
    'fee': '% Fee',
    'position_id': '# LP positions',
    'position_perc_ret': '% Avg. Return\n LP position',
    'hedged_perc_impr': '% Avg. Improve\n with Hedging',    
    'perp_count': '# Avg. Perp. \n with Hedging',    
    }
df = df[cols_perp.keys()].rename(columns=cols_perp)
for c in df.columns:
    if '%' in c:
        df[c] = df[c].apply(lambda x: f"{x:.2%}")
    if '# Avg' in c:
        df[c] = df[c].apply(lambda x: f"{x:.2f}")
fig = _tbl()
fig.savefig(os.path.join(folder, f'TblHedge.png'), bbox_inches='tight', dpi=300)

fig, ax = plt.subplots(figsize=(9, 2))
color_menu = tailwind['indigo-600']
print("${:,.0f}".format(df_perp_agr['perp_pnl'].sum()+df_perp_agr['collateral'].sum()))
df = df_perp_agr.copy()
cols_perp = {
    'state': 'Volatility',
    'position_id': '# Positions',
    'perp_ret': '% Performance',
    'liq': '% Liquidated',
    't_max': '% Expired',
    'lp_cls': '% LP Closed',
    }
df = df[cols_perp.keys()].rename(columns=cols_perp)
for c in df.columns:
    if '%' in c:
        df[c] = df[c].apply(lambda x: f"{x:.2%}")
df['idx'] = df['Volatility'].replace(
    'highvol', 3).replace('midvol', 2).replace('lowvol', 1)
df = df.sort_values('idx')
df.pop('idx')
df['Volatility'] = df['Volatility'].replace(
    'highvol', 'High').replace(
        'lowvol', 'Low').replace('midvol', 'Medium')
fig = _tbl()
fig.savefig(os.path.join(folder, f'TblPerps.png'), bbox_inches='tight', dpi=300)

# =============================================================================
# Plots
# =============================================================================
for s in ['BTC', 'ETH']:       
    df = df_market[
        (df_market['name']==s) &
        (df_market['_time']>=start_time) &
        (df_market['_time']<=end_time)
        ].copy().reset_index(drop=True).rename(columns={'name':'symbol'})
    df = df.merge(df_state_xtreamly[['_time','symbol','state', 'state_id']], on=['_time','symbol'], how='left')
    
    _style_white()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(f"{s} Market State Forecast Timeline", pad=20)
    ax.set_ylabel(f"{s}USD Price", labelpad=20)
    #ax.plot(df['_time'], df['open'].values, color=tailwind['stone-900'], alpha=.9, linewidth=1, label=f"{s}USD Price")
    for state_id in df['state_id'].unique(): 
        df_s = df[df['state_id']==state_id]
        state = df_s['state'].iloc[0]
        if state == 'lowvol': c=tailwind['teal-400']
        if state == 'midvol': c=tailwind['amber-400']
        if state == 'highvol': c=tailwind['red-400']
        ax.plot(df_s['_time'], df_s['open'].values, color=c, alpha=.8, linewidth=2)
    ax.plot([df_s['_time'].iloc[0]], [df_s['open'].iloc[0]], color=tailwind['teal-400'], alpha=.8, linewidth=2, label='Low Volatility')
    ax.plot([df_s['_time'].iloc[0]], [df_s['open'].iloc[0]], color=tailwind['amber-400'], alpha=.8, linewidth=2, label='Medium Volatility')
    ax.plot([df_s['_time'].iloc[0]], [df_s['open'].iloc[0]], color=tailwind['red-400'], alpha=.8, linewidth=2, label='High Volatility')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
    ax.set_yticks(ax.get_yticks())
    ax.set_xticks(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
    legend = ax.legend(loc='upper right')
    legend.get_frame().set_alpha(0.9)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout(rect=[0.004, 0.004, .996, .996])
    fig.savefig(os.path.join(folder, f'TimelineForecastMarketState{s}.png'), dpi=200)
    plt.show()
    fig.clf()


_style_white()
fig, ax = plt.subplots(figsize=(10, 4))
df = df_hedge_pos.copy()
cols = ['position_perc_ret', 'hedged_perc_ret']
ax.set_title(f"LP Positions % Performance", pad=30)
ax.set_ylabel(f"# Count", labelpad=10)
ax.set_xlabel(f"% Return", labelpad=10)
bin_edges = np.linspace(df['position_perc_ret'].min(), df['position_perc_ret'].max(), 200)
bin_width = np.diff(bin_edges)[0]
bar_width = bin_width / 2
for i, (c, color, lbl) in enumerate(zip(cols, [tailwind['stone-400'], tailwind['indigo-700']], ['Original','Hedged'])):
    counts, _ = np.histogram(df[c], bins=bin_edges)
    ax.bar(bin_edges[:-1] + i * bar_width, counts, width=bar_width, color=color, alpha=0.8, label=f"{lbl} LP Positions")
ax.set_xlim(-.3, .3)
yticks = ax.get_yticks()
ax.set_yticks(ax.get_yticks())
ax.set_xticks(ax.get_xticks())
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.1f}%'.format(x*100)))
legend = ax.legend(loc='upper right')
legend.get_frame().set_alpha(0.9)
ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
for spine in ax.spines.values(): spine.set_visible(False)
fig.tight_layout(rect=[0.004, 0.004, .996, .996])
fig.savefig(os.path.join(folder, f'HistogramPerformance.png'), dpi=200)
fig.show()
fig.clf()
  
  
_style_white()
fig, ax = plt.subplots(figsize=(10, 4))
df = df_hedge_pos.copy()
ax.set_title(f"LP Positions % Improve with Hedging", pad=30)
ax.set_ylabel(f"# Count", labelpad=10)
ax.set_xlabel(f"% Improvement", labelpad=10)
ax.hist(df['hedged_perc_impr'],bins=120, color=tailwind['indigo-500'], alpha=0.8, label=f"Improvement on Hedging Position")
yticks = ax.get_yticks()
ax.set_yticks(ax.get_yticks())
ax.set_xticks(ax.get_xticks())
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}%'.format(x*100)))
ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
for spine in ax.spines.values(): spine.set_visible(False)
fig.tight_layout(rect=[0.004, 0.004, .996, .996])
fig.savefig(os.path.join(folder, f'HistogramImprovement.png'), dpi=200)
fig.clf()

    
_style_white()  
fig, ax = plt.subplots(figsize=(10, 4))
df = df_perps_all.copy()  
ax.set_title("Reasons for Closing Perpetuals", pad=30)
ax.set_ylabel("# Count", labelpad=10)
ax.bar(dict_perp_close.keys(), dict_perp_close.values(), width=.4, color=tailwind['indigo-600'], alpha=0.8)
yticks = ax.get_yticks()
ax.set_xlim(-.7,2.7)
ax.set_yticks(yticks)
ax.set_xticks(range(len(dict_perp_close)))  # Ensure x-ticks match bar positions
ax.set_yticklabels([f'{x:,.0f}' for x in yticks], rotation=0)  # Format y-tick labels directly
ax.set_xticklabels(dict_perp_close.keys(), rotation=0)  # Use dictionary keys as x-tick labels
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{x:,.0f}'))
ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
for spine in ax.spines.values(): spine.set_visible(False)
fig.tight_layout(rect=[0.004, 0.004, 0.996, 0.996])
fig.savefig(os.path.join(folder, 'BarClosePerpetuals.png'), dpi=200)
fig.clf()


_style_white()
fig, ax = plt.subplots(figsize=(10, 4))
df = df_perps_all.copy()
ax.set_title(f"% Return on Perpetuals", pad=30)
ax.set_ylabel(f"# Count", labelpad=10)
ax.set_xlabel(f"% Return", labelpad=10)
ax.hist(df['ret'],bins=120, color=tailwind['indigo-600'], alpha=0.8, label=f"% Returns Pepetuals")
ax.set_xlim(-1.,df['ret'].max())
yticks = ax.get_yticks()
ax.set_yticks(ax.get_yticks())
ax.set_xticks(ax.get_xticks())
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0%}'.format(x)))
ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
for spine in ax.spines.values(): spine.set_visible(False)
fig.tight_layout(rect=[0.004, 0.004, .996, .996])
fig.savefig(os.path.join(folder, f'HistogramReturnPerpetuals.png'), dpi=200)
fig.clf()

_style_white()
fig, ax = plt.subplots(figsize=(10, 4))
df = df_hedge_pos.copy()
ax.set_title(f"LP Positions % Active Time", pad=30)
ax.set_ylabel(f"# Count", labelpad=10)
ax.set_xlabel(f"% Active Time", labelpad=10)
ax.hist(df['position_perc_active'],bins=30, color=tailwind['pink-600'], alpha=0.9)#, label=f"% Active Time")
ax.set_xlim(0,1)
yticks = ax.get_yticks()
ax.set_yticks(ax.get_yticks())
ax.set_xticks(ax.get_xticks())
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}%'.format(x*100)))
ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
for spine in ax.spines.values(): spine.set_visible(False)
fig.tight_layout(rect=[0.004, 0.004, .996, .996])
fig.savefig(os.path.join(folder, f'HistogramActive.png'), dpi=200)
fig.clf()

# position_id = '878458_0x902a7cebc98daa5a0e6de468052c75719c014797'
# position_id = '879576_0x5b393bd3c1d0d334b8bb9ae106edb4ec33801a3c'
# position_id = '851265_0x0d21716f645ce331cc4fadb9e621980acadf56dc'
# position_id = '873626_0x37c4d1abc89e1bb1ceaa9df3c17d4464fe95e21c'
# position_id = '840439_0x4942e2b839fc479c27c496b7758bab94ceb6b684'