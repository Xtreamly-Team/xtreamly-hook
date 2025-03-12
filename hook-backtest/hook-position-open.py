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

with open(os.path.join('data', 'pools.json'), 'r') as file: data_pools = json.load(file)
df_pools = pd.DataFrame(data_pools)
df_pools['pool'] = df_pools['pool'].str.lower()
df_pools = df_pools[[('BTC' in t or 'ETH' in t) and 'USD' in t for t in df_pools['type']]]

# =============================================================================
# Pool Positions
# =============================================================================
start_time = datetime.fromisoformat('2024-10-01').replace(tzinfo=None)
end_time = datetime.fromisoformat('2024-12-31').replace(tzinfo=None)
pool = '0x4e68ccd3e89f51c3074ca5072bbac773960dfa36'
s = 'ETH'
pool_type = df_pools[df_pools['pool'] == pool].iloc[0]['type'].replace('/', '')
pool_fee = df_pools[df_pools['pool']==pool].iloc[0]['fee'] 

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

# =============================================================================
# Functions
# =============================================================================
def _get_L(pos_uni):
    p = pos_uni['p']
    p_a = pos_uni['p_a']
    p_b = pos_uni['p_b']
    x = pos_uni['x']
    y = pos_uni['y']
    if p <= p_a:
        L = x*( (np.sqrt(p_a)*np.sqrt(p_b))/(np.sqrt(p_b)-np.sqrt(p_a)))
    elif p_a < p <= p_b:
        L = y/(np.sqrt(p)-np.sqrt(p_a))
    else:
        L = y/(np.sqrt(p_b)-np.sqrt(p_a))
    return L

def _uni_upd(p, t, pos_uni):
    x = pos_uni['x']
    y = pos_uni['y']
    p_a = pos_uni['p_a']
    p_b = pos_uni['p_b']#
    L = pos_uni['L']

    p_prim = np.clip(p, p_a, p_b)
    x_prim = L * (np.sqrt(p_b)-np.sqrt(p_prim))/(np.sqrt(p_prim)*np.sqrt(p_b)) # if pos_uni['p_a']<p else 0.0
    y_prim = L * (np.sqrt(p_prim)-np.sqrt(p_a)) #if pos_uni['p_b']>p else 0.0
    pos_uni['p'] = p
    pos_uni['t'] = t
    pos_uni['buffer_l'] = max(0, np.round((p-p_a)/p,6))
    pos_uni['buffer_u'] = max(0, np.round((p_b-p)/p,6))
    pos_uni['active'] = p_a < p and p < p_b

    pos_uni['x_now'] = x_prim
    pos_uni['y_now'] = y_prim
    pos_uni['x_usd'] = pos_uni['x_now']*p
    pos_uni['y_usd'] = pos_uni['y_now']    
    pos_uni['v'] = pos_uni['x_usd']+pos_uni['y_usd']
    pos_uni['imp_loss'] = pos_uni['v']-pos_uni['inv_usd']
    return pos_uni

def _lp_uni(df_market, pos_uni):
    df = df_market.copy()
    data_lp = [pos_uni]
    for _,r in df.iterrows():
        t = r['_time']
        p = r['open']
        pos_uni = data_lp[-1].copy()
        data_lp += [_uni_upd(p, t, pos_uni)]
    df_lp = pd.DataFrame(data_lp)

    r_fees = df_lp.iloc[0]['f']/df_lp.iloc[0]['v']
    duraton_active = df_lp['active'].cumsum()/(60*24)
    apy = (1 + r_fees) ** (365 / duraton_active.iloc[-1]) - 1
    df_lp['f_cum'] = ((1 + apy) ** (duraton_active / 365) - 1)*df_lp.iloc[0]['v']

    data_active = [df_lp['t'].iloc[0]]
    a = df_lp['active'].iloc[0]
    for i_a, r_a in df_lp.iloc[1:].iterrows():
        if r_a['active'] != a:
            data_active+= [df_lp['t'].iloc[i_a]]
            a = r_a['active']
        else: data_active+= [np.nan]
    df_lp['active_id'] = data_active
    df_lp['active_id'] = df_lp['active_id'].ffill()
    return df_lp
    
hedge_share = .2
gmx_leverage_max = 10
gmx_leverage_min = 1
gmx_fee_entry = 0.001
gmx_fee_exit = 0.001
gmx_funding_rate = 0.0001
gmx_duration_max = 12
def _perp_gmx(df_market, gmx_type, gmx_collateral, gmx_leverage, gmx_duration_max):
    df = df_market.copy()

    gmx_size = gmx_collateral * gmx_leverage
    gmx_price_start = df_market['open'].iloc[0]
    gmx_price_end = df_market['close'].iloc[0]
    gmx_price_entry = gmx_price_start * (1 + gmx_fee_entry)
    gmx_price_liq = gmx_price_entry * ((1 - 1/gmx_leverage) if gmx_type == "long" else (1 + 1/gmx_leverage))
    df['price_liq'] = gmx_price_liq
    df.loc[:,'liq'] = (df['low'] <= gmx_price_liq) if gmx_type == "long" else (df['high'] >= gmx_price_liq)
    df.loc[:,'exit'] = df['close'] * ((1 - gmx_fee_exit) if gmx_type == "long" else (1 + gmx_fee_exit))
    
    df.loc[:,'cost_funding'] = 0.0
    df.loc[df['_time'].dt.minute == 0, 'cost_funding'] = gmx_size * gmx_funding_rate
    df.loc[:,'cost_funding'] = df['cost_funding'].fillna(0.0).cumsum()
    if gmx_type == "long":
        df.loc[:,'pnl'] = gmx_size*(df['exit'] - gmx_price_entry)/gmx_price_entry - df['cost_funding']
    else:
        df.loc[:,'pnl'] = gmx_size*(gmx_price_entry-df['exit'])/gmx_price_entry - df['cost_funding']
        
    if df['liq'].any():
        df.loc[(df[df['liq']]['liq'].index.min()):,'pnl'] = -gmx_collateral
        df['pnl'] = df['pnl'].clip(-gmx_collateral,df['pnl'].max())
 
    df['duration'] = (df['_time']-df['_time'].iloc[0]).dt.total_seconds()/(3600)
    df['lp_extra'] = False
    if df['duration'].max()>=gmx_duration_max:
        idx = df[df['duration']>=gmx_duration_max].index.min()
        df.loc[idx:,'pnl'] = df.loc[idx,'pnl'] 
        df.loc[idx:,'lp_extra'] = True
 
    df.loc[:,'collateral'] = gmx_collateral
    df.loc[:,'v'] = df['pnl']+gmx_collateral
    df.loc[:,'ret'] = df['pnl']/gmx_collateral
    df.loc[:,'ret_price'] = df['close']/gmx_price_start-1
    return df

def _plot_pos_hedge(position_id, i, df_lp_org, df_lp, gmx_collateral, df_perp, df_p):
    nr = 4
    _style_white()
    fig, axes = plt.subplots(nrows=nr, figsize=(14, 5*nr))
    
    ax = axes[0]
    ax.set_title(f"Original LP Value", pad=20)
    ax.fill_between(df_lp_org['t'], df_lp_org['x_usd'], df_lp_org['v'], color=tailwind['emerald-400'], alpha=0.9, label=f"LP USDT value")
    ax.fill_between(df_lp_org['t'], df_lp_org['x_usd'], color=tailwind['purple-400'], alpha=0.9, label=f"LP ETH value")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
    
    ax = axes[1]
    ax.set_title(f"Hedged LP Value", pad=20)
    ax.fill_between(df_lp['t'], df_lp['v'], df_lp['v']+gmx_collateral+df_perp['pnl'], color=tailwind['indigo-400'], 
                    alpha=0.9, label=f"Perp value")
    ax.fill_between(df_lp['t'], df_lp['x_usd'], df_lp['v'], color=tailwind['emerald-400'], alpha=0.9, label=f"LP USDT value")
    ax.fill_between(df_lp['t'], df_lp['x_usd'], color=tailwind['purple-400'], alpha=0.9, label=f"LP ETH value")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
    
    ax = axes[2]
    ax.set_title(f"LP Active (both hedged and not)", pad=20)
    ax.plot(df_p['_time'], df_p['open'], tailwind['stone-800'], alpha=0.9, label="Market Price")
    ax.plot(df_perp['_time'], df_perp['price_liq'], tailwind['purple-600'], alpha=0.9, label="Perp. Liquidation Price")
    for a in df_lp['active_id'].unique():
        df_lp_a = df_lp[df_lp['active_id']==a]
        pos_active = df_lp_a['active'].iloc[0]
        pos_color = tailwind['green-500'] if pos_active else tailwind['rose-500']
        ax.fill_between(df_lp_a['t'],  df_lp_a['p_a'],  df_lp_a['p_b'], color=pos_color, alpha=0.6)
    p_opn = df_p['open'].iloc[0]
    ax.fill_between([df_lp['t'].iloc[0]],  p_opn, p_opn, color=tailwind['green-500'], alpha=0.7, label="Active")    
    ax.fill_between([df_lp['t'].iloc[0]],  p_opn, p_opn, color=tailwind['rose-500'], alpha=0.7, label="Not active")    
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
    
    ax = axes[3]
    ax.set_title(f"Hedged LP Return", pad=20)
    ax.plot(df_p['_time'], df_p['ret_hedged'], tailwind['indigo-600'], alpha=0.9, label="Hedged %Return")
    ax.plot(df_p['_time'], df_p['ret_org'], tailwind['stone-600'], alpha=0.9, label="Original %Return")  
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.1f}%'.format(x*100)))
    
    for ax in axes:
        ax.set_yticks(ax.get_yticks())
        ax.set_xticks(ax.get_xticks())
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
        for spine in ax.spines.values(): spine.set_visible(False)
        legend = ax.legend(loc='upper left')
        legend.get_frame().set_facecolor('white')  # Sets the background color to white
        legend.get_frame().set_edgecolor('black')  # Optional: Adds a border to the legend
        legend.get_frame().set_alpha(1.0)  # Ensures no transparency (fully opaque)
        #legend.get_frame().set_alpha(0.9)

    fig.tight_layout(rect=[0.01, 0.01, .99, .99])
    fig.subplots_adjust(hspace=.5)

    fig.savefig(os.path.join('plots', f'Position Hedge {position_id}'), dpi=200)
    fig.clf()

# =============================================================================
# Run
# =============================================================================
data_pos_hedge = []
for i, position_id in enumerate(pool_ids[:]):
    print(gmx_duration_max, i, position_id)
    pos = df_pos[df_pos['id'] == position_id].iloc[0]
    pos_log = df_log[df_log['position_id'] == position_id]
    df_p = df_market[
        (df_market['name']==s) &
        (df_market['_time']>=pos_log['_time'].min()) &
        (df_market['_time']<=pos_log['_time'].max())
        ].copy().reset_index(drop=True)
    df_xtreamly = df_state_xtreamly[
        (df_state_xtreamly['symbol']==s) &
        (df_state_xtreamly['_time']>=pos_log['_time'].min()) &
        (df_state_xtreamly['_time']<=pos_log['_time'].max())
        ].copy().reset_index(drop=True)
    market_status = df_xtreamly.iloc[0]['state']
    
    deposited = pos_log[pos_log['type']=='deposits'].iloc[0]
    deposited_usd = deposited['deposited_token1'] + deposited['deposited_token0']*deposited['price']
    fees = pos_log[pos_log['type']=='claimed-fees'].iloc[0]
    fees_usd = fees['collected_fees_token1'] + fees['collected_fees_token0']*fees['price']
    withdrawals = pos_log[pos_log['type']=='withdrawals'].iloc[0]
    withdrawals_usd = withdrawals['withdrawn_token1'] + withdrawals['withdrawn_token0']*withdrawals['price']
    imp_loss = (withdrawals_usd-deposited_usd)/deposited_usd
    # print(i, imp_loss)

    pos_uni = {
        't_opn': pos_log['_time'].min(),
        'p_opn': deposited['price'],
        'p': deposited['price'],
        'p_a': pos['price_lower'],
        'p_b': pos['price_upper'],
        'f': fees_usd*(1-hedge_share),
        'x': deposited['deposited_token0']*(1-hedge_share), #eth - x - tkn_amount_0
        'y': deposited['deposited_token1']*(1-hedge_share), #usdt - y - tkn_amount_1
        }  
    pos_uni['inv_usd'] = pos_uni['x']*pos_uni['p'] + pos_uni['y']
    pos_uni['L'] = _get_L(pos_uni)
    pos_uni = _uni_upd(deposited['price'], pos_log['_time'].min(), pos_uni)
    df_lp = _lp_uni(df_p, pos_uni)

    pos_uni_org = pos_uni.copy()
    pos_uni_org['x'] = deposited['deposited_token0']
    pos_uni_org['y'] = deposited['deposited_token1']
    pos_uni_org['inv_usd'] = pos_uni_org['x']*pos_uni_org['p'] + pos_uni_org['y']
    pos_uni_org['L'] = _get_L(pos_uni_org)
    pos_uni_org = _uni_upd(deposited['price'], pos_log['_time'].min(), pos_uni_org)
    df_lp_org = _lp_uni(df_p, pos_uni_org)

    gmx_type = 'short'
    gmx_collateral = deposited_usd*hedge_share
    gmx_leverage = np.floor((1/(pos['price_upper']/deposited['price']-1)))-1
    gmx_leverage = min(gmx_leverage_max,max(gmx_leverage_min,gmx_leverage)) 
    df_perp = _perp_gmx(df_p, gmx_type, gmx_collateral, gmx_leverage, gmx_duration_max)
    
    df_p['ret_hedged'] = (df_perp['v']+df_lp['v']+df_lp['f_cum'])/deposited_usd-1
    df_p['ret_org'] = (df_lp_org['v']+df_lp_org['f_cum'])/deposited_usd-1
    # _plot_pos_hedge(position_id, i, df_lp_org, df_lp, gmx_collateral, df_perp, df_p)
    
    out = {
        'position_id': position_id,
        'min_time': pos_log['_time'].min(),
        'max_time': pos_log['_time'].max(),
        'hedge_share': hedge_share,
        'gmx_duration_max': gmx_duration_max,
        'gmx_leverage_max': gmx_leverage_max,
        'position_min_time': pos_log['_time'].min(),
        'position_max_time': pos_log['_time'].max(),
        'position_id': position_id,
        'position_deposited_usd': deposited_usd,
        'position_withdrawals_usd': withdrawals_usd, 
        'position_performance': df_p['ret_org'].iloc[-1],
        'hedged_deposited_usd': df_lp['v'].iloc[0]+df_perp['v'].iloc[0],
        'hedged_withdrawals_usd': df_lp['v'].iloc[-1]+df_lp['f_cum'].iloc[-1]+df_perp['v'].iloc[-1],
        'hedged_performance': df_p['ret_hedged'].iloc[-1],
        'hedge_improve': df_p['ret_hedged'].iloc[-1]-df_p['ret_org'].iloc[-1],
        'perp_collateral': df_perp['collateral'].iloc[0],
        'perp_pnl': df_perp['pnl'].iloc[-1],
        'perp_performance': df_perp['ret'].iloc[-1],
        'perp_cost_funding': df_perp['cost_funding'].iloc[-1],
        'lp_imp_loss': df_lp['imp_loss'].iloc[-1],
        'lp_fee_collected': df_lp['f_cum'].iloc[-1],
        'market_status': market_status,
        }
    data_pos_hedge += [out]
    
# =============================================================================
# Agr Positions
# =============================================================================
df_pos_hedge = pd.DataFrame(data_pos_hedge)    
df_pos_hedge['duration'] = (df_pos_hedge['max_time']-df_pos_hedge['min_time']).dt.total_seconds()/(24*3600)

_style_white()
fig, ax = plt.subplots(figsize=(16, 7))
df = df_pos_hedge.copy()
ax.set_title(f"Histogram of % Improvement on Hedging", pad=30)
ax.set_ylabel(f"% Share", labelpad=20)
ax.set_xlabel(f"% Improvement", labelpad=20)
bin_edges = np.linspace(df['hedge_improve'].min(), df['hedge_improve'].max(), 40)
bin_width = np.diff(bin_edges)[0]  # Calculate bin width
bar_width = bin_width / 3  # Divide bins equally among the three categories
for i, (status, color) in enumerate(zip(
    ['lowvol', 'midvol', 'highvol'], 
    [tailwind['teal-400'], tailwind['amber-400'], tailwind['pink-400']]
)):
    counts, _ = np.histogram(df[df['market_status'] == status]['hedge_improve'], bins=bin_edges)
    ax.bar(bin_edges[:-1] + i * bar_width, counts/sum(counts), width=bar_width, color=color, alpha=0.9, label=status.capitalize())
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
fig.savefig(os.path.join('plots', f'Histogram Improvement.png'), dpi=200)
fig.clf()

      
df_pos_hedge_agr = df_pos_hedge.groupby(['gmx_duration_max', 
                                     'market_status']).agg(
        hedged_performance=('hedged_performance', 'mean'),
        hedge_improve=('hedge_improve', 'mean'),
        position_id=('position_id', 'nunique')
        ).reset_index()
df_pos_hedge_agr = df_pos_hedge_agr[df_pos_hedge_agr['gmx_duration_max']==12]
cols_hedge = {
    'market_status': 'Status',
    'position_id': '# Positions',
    'hedged_performance': '% Performance with Hedge',
    'hedge_improve': '% Improvement with Hedge',
    }
df_pos_hedge_agr = df_pos_hedge_agr[cols_hedge.keys()].rename(columns=cols_hedge)
for c in df_pos_hedge_agr.columns:
    if '%' in c:
        df_pos_hedge_agr[c]*=1
        df_pos_hedge_agr[c] = df_pos_hedge_agr[c].apply(lambda x: f"{x:.2%}")

fig, ax = plt.subplots(figsize=(8, 1))  # Adjust figure size
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df_pos_hedge_agr.values, 
                 colLabels=df_pos_hedge_agr.columns, 
                 cellLoc='center', 
                 loc='center')
ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
for spine in ax.spines.values(): spine.set_visible(False)
plt.savefig(os.path.join('plots', f'Table Improvement.png'), bbox_inches='tight', dpi=300)

# position_id = '878458_0x902a7cebc98daa5a0e6de468052c75719c014797'
# position_id = '879576_0x5b393bd3c1d0d334b8bb9ae106edb4ec33801a3c'
# position_id = '851265_0x0d21716f645ce331cc4fadb9e621980acadf56dc'
# position_id = '873626_0x37c4d1abc89e1bb1ceaa9df3c17d4464fe95e21c'
# position_id = '840439_0x4942e2b839fc479c27c496b7758bab94ceb6b684'
