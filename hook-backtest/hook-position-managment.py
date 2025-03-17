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

folder='plots'
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

def _pos_uni(pos, pos_log, deposited):
    pos_uni = {
        't_opn': pos_log['_time'].min(),
        'p_opn': deposited['price'],
        #'p': deposited['price'],
        'p_a': pos['price_lower'],
        'p_b': pos['price_upper'],
        'x_opn': deposited['deposited_token0'], #eth - x - tkn_amount_0
        'y_opn': deposited['deposited_token1'], #usdt - y - tkn_amount_1
        }  
    pos_uni['inv_usd'] = pos_uni['x_opn']*pos_uni['p_opn'] + pos_uni['y_opn']
    pos_uni['L'] = _get_L(pos_uni)
    pos_uni = _upd_uni(pos_uni['p_opn'], pos_uni['t_opn'], pos_uni)
    return pos_uni

def _upd_gmx(p, t, df_p, pos_gmx):
    df = df_p[ (df_p['_time']>=pos_gmx['t_opn']) & (df_p['_time']<t)].copy().reset_index(drop=True)  
    
    upd_gmx = pos_gmx.copy()
    upd_gmx['t'] = t
    upd_gmx['p'] = p
    upd_gmx['p_ext'] = p * ((1-gmx_fee_exit) if upd_gmx['type'] == "long" else (1+gmx_fee_exit))
    upd_gmx['t_hours'] = int((t-upd_gmx['t_opn']).total_seconds() / 3600)
    upd_gmx['cost_rate'] = upd_gmx['s'] * upd_gmx['f_rate']*upd_gmx['t_hours'] # use df
    if df.shape[0]:
        upd_gmx['liq'] = (df['low'] <= upd_gmx['p_liq']) if gmx_type == "long" else (df['high'] >= upd_gmx['p_liq']).any()
    else:
        upd_gmx['liq'] = False
    upd_gmx['p_diff'] = (upd_gmx['p_ext'] - upd_gmx['p_ent']) * (1 if upd_gmx['type'] == "long" else -1)

    upd_gmx['pnl'] = max(-upd_gmx['c'], upd_gmx['s']*(upd_gmx['p_diff']/pos_gmx['p_ent']) - upd_gmx['cost_rate'])
    upd_gmx['v'] = max(.0, upd_gmx['pnl'] + upd_gmx['c'])
    return upd_gmx

def _pos_gmx(t, p, df_p, hedge_type, collateral, leverage):
    pos_gmx = {
        'type': hedge_type,
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

def _hedge_value(pos_org, pos_uni, pos_gmx, df_p):
    p_values = np.linspace(pos_uni['p']/1.4, pos_uni['p']*1.4, 1000)
    p_conc = (pos_uni['p_a']<=p_values) & (p_values<=pos_uni['p_b'])
    df_gmx_v = pd.DataFrame([_upd_gmx(p, pos_gmx['t_opn'], df_p, pos_gmx.copy()) for p in p_values])
    df_uni_v = pd.DataFrame([_upd_uni(p, pos_uni['t_opn'], pos_uni.copy()) for p in p_values])
    df_org_v = pd.DataFrame([_upd_uni(p, pos_uni['t_opn'], pos_org.copy()) for p in p_values])
    lim_avg_org = np.mean(df_org_v['v']).round(2)
    lim_avg_hedge = np.mean(df_uni_v['v']+df_gmx_v['v']).round(2)
    lim_imprv = lim_avg_hedge-lim_avg_org #'${:,.2f}'.format(v_avg_hedge-v_avg_org)
    
    conc_avg_org = np.mean(df_org_v[p_conc]['v']).round(2)
    conc_avg_hedge = np.mean(df_uni_v[p_conc]['v']+df_gmx_v[p_conc]['v']).round(2)
    conc_imprv = conc_avg_hedge-conc_avg_org
    return {
        'p_values': p_values,
        'df_gmx_v': df_gmx_v,
        'df_uni_v': df_uni_v,
        'df_org_v': df_org_v,
        'lim_avg_org': lim_avg_org,
        'lim_avg_hedge': lim_avg_hedge,
        'lim_imprv': lim_imprv,
        'conc_avg_org': conc_avg_org,
        'conc_avg_hedge': conc_avg_hedge,
        'conc_imprv': conc_imprv,
        }

def _plot_hedge_value(hedge_value, pos_gmx, pos_uni, position_id):
    p_values = hedge_value['p_values']
    df_gmx_v = hedge_value['df_gmx_v']
    df_uni_v = hedge_value['df_uni_v']
    df_org_v = hedge_value['df_org_v']
    imprv = '${:,.2f}'.format(hedge_value['lim_imprv'])
    p_fr = '${:,.0f}'.format(p_values.min())
    p_to = '${:,.0f}'.format(p_values.max())
    sub = f"Avg. Improvement on Value with Hedging: {imprv} for price scenarios ({p_fr} - {p_to}). Dynamic leverage: x{pos_gmx['l']}."
    print(sub)
    
    _style_white()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_title(f"$ Position Value for Different Price Scenarios", pad=40)
    ax.text(0.5, 1.05, sub, ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_ylabel(f"$ Value", labelpad=20)
    ax.set_xlabel(f"$ Price Scenario", labelpad=10)
    ax.plot(p_values, df_org_v['v'], color=tailwind['stone-800'], alpha=0.9, linewidth=3, label=f"Original Position Value")
    ax.plot(p_values, df_uni_v['v'] + df_gmx_v['v'], color=tailwind['teal-400'], alpha=0.99, linewidth=3, label=f"Hedged Position Value")
    y_min, y_max = max(.0, ax.get_yticks().min()), ax.get_yticks().max()
    ax.vlines(pos_uni['p_a'], y_min, y_max, color=tailwind['pink-500'], label="Price LP Lower")
    ax.vlines(pos_uni['p_b'], y_min, y_max, color=tailwind['pink-500'], label="Price LP Upper")
    ax.vlines(pos_gmx['p_liq'], y_min, y_max, color=tailwind['purple-500'], label="Price Perp. Liquidation")
    ax.vlines(pos_gmx['p_opn'], y_min, y_max, color=tailwind['stone-700'], alpha=.3, label="Price Open")
    #ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
    ax.set_xlim((p_values.min(), p_values.max()))
    ax.set_ylim((y_min, y_max))
    xticks = ax.get_xticks()
    xticks = xticks[(p_values.min() <= xticks) & (xticks <= p_values.max())]
    ax.set_xticks(xticks)
    ax.set_yticks(ax.get_yticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    legend = ax.legend(loc='upper left')
    legend.get_frame().set_alpha(0.5)
    ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout(rect=[0.004, 0.004, .996, .996])
    fig.savefig(os.path.join(folder, f'Position Value {position_id}'), dpi=200)
    plt.show()
    fig.clf()

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
# Simulate Hedge
# =============================================================================
hedge_type = 'short'
hedge_share = .1 
hedge_l_max = 12
hedge_l_min = 8
hedge_time_max = 1
hedge_time_min = 12

for i, position_id in enumerate(pool_ids[:5]):
    print(i, position_id)
    pos = df_pos[df_pos['id'] == position_id].iloc[0].to_dict()
    pos_log = df_log[df_log['position_id'] == position_id].copy()
    pos_log.loc[:,'usd'] = (pos_log['amount1'] + pos_log['amount0']*pos_log['price']).values

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
    df_p = df_p[['_time', 'open']].merge(df_xtreamly, on='_time', how='left')

    t = df_p['_time'].iloc[0]
    p = df_p['open'].iloc[0]
    
    fees_usd = pos_log[pos_log['type']=='claimed-fees'].iloc[0]['usd']
    deposited_org = pos_log[pos_log['type']=='deposits'].iloc[0].to_dict()
    pos_org = _pos_uni(pos, pos_log, deposited_org)

    deposited_hedge = deposited_org.copy()
    deposited_hedge['deposited_token0'] *= (1-hedge_share)
    deposited_hedge['deposited_token1'] *= (1-hedge_share)
    pos_uni = _pos_uni(pos, pos_log, deposited_hedge)

    collateral = np.abs(deposited_org['usd'])*(hedge_share)
    leverage = int((1/(pos['price_upper']/p-1)))-1
    leverage = min(hedge_l_max,max(hedge_l_min,leverage))     
    pos_gmx = _pos_gmx(t, p, df_p, hedge_type, collateral, leverage)
# =============================================================================
#     hedge_value = _hedge_value(pos_org, pos_uni, pos_gmx, df_p)
#     print(hedge_value['lim_imprv'], hedge_value['conc_imprv'])
#     _plot_hedge_value(hedge_value, pos_gmx, pos_uni, position_id)
# =============================================================================
     
    data_org = [pos_org]
    data_uni = [pos_uni]
    data_gmx = [pos_gmx]
    for i,r in df_p.iloc[1:111].iterrows():
        t = r['_time']
        p = r['open']
        
        
        
        
        
        
        
        
    


# =============================================================================
#     pos_uni = _uni_upd(deposited['price'], pos_log['_time'].min(), pos_uni)
#     
#     
#     
#     
#     
#     df_lp = _lp_uni(df_p, pos_uni)
# 
#     pos_uni_org = pos_uni.copy()
#     pos_uni_org['x'] = deposited['deposited_token0']
#     pos_uni_org['y'] = deposited['deposited_token1']
#     pos_uni_org['inv_usd'] = pos_uni_org['x']*pos_uni_org['p'] + pos_uni_org['y']
#     pos_uni_org['L'] = _get_L(pos_uni_org)
#     pos_uni_org = _uni_upd(deposited['price'], pos_log['_time'].min(), pos_uni_org)
#     df_lp_org = _lp_uni(df_p, pos_uni_org)
# 
#     gmx_type = 'short'
#     gmx_collateral = deposited_usd*hedge_share
#     gmx_leverage = int((1/(pos['price_upper']/deposited['price']-1)))-1
#     gmx_leverage = min(gmx_leverage_max,max(gmx_leverage_min,gmx_leverage)) 
#     df_perp = _perp_gmx(df_p, gmx_type, gmx_collateral, gmx_leverage, gmx_duration_max)
#     
#     df_p['ret_hedged'] = (df_perp['v']+df_lp['v']+df_lp['f_cum'])/deposited_usd-1
#     df_p['ret_org'] = (df_lp_org['v']+df_lp_org['f_cum'])/deposited_usd-1
#     _plot_pos_hedge(position_id, i, df_lp_org, df_lp, gmx_collateral, df_perp, df_p)
#     
#     out = {
#         'position_id': position_id,
#         'min_time': pos_log['_time'].min(),
#         'max_time': pos_log['_time'].max(),
#         'hedge_share': hedge_share,
#         'gmx_duration_max': gmx_duration_max,
#         'gmx_leverage_max': gmx_leverage_max,
#         'position_min_time': pos_log['_time'].min(),
#         'position_max_time': pos_log['_time'].max(),
#         'position_id': position_id,
#         'position_deposited_usd': deposited_usd,
#         'position_withdrawals_usd': withdrawals_usd, 
#         'position_performance': df_p['ret_org'].iloc[-1],
#         'hedged_deposited_usd': df_lp['v'].iloc[0]+df_perp['v'].iloc[0],
#         'hedged_withdrawals_usd': df_lp['v'].iloc[-1]+df_lp['f_cum'].iloc[-1]+df_perp['v'].iloc[-1],
#         'hedged_performance': df_p['ret_hedged'].iloc[-1],
#         'hedge_improve': df_p['ret_hedged'].iloc[-1]-df_p['ret_org'].iloc[-1],
#         'perp_collateral': df_perp['collateral'].iloc[0],
#         'perp_pnl': df_perp['pnl'].iloc[-1],
#         'perp_performance': df_perp['ret'].iloc[-1],
#         'perp_cost_funding': df_perp['cost_funding'].iloc[-1],
#         'lp_imp_loss': df_lp['imp_loss'].iloc[-1],
#         'lp_fee_collected': df_lp['f_cum'].iloc[-1],
#         'market_status': market_status,
#         }
#     data_pos_hedge += [out]
#     
# =============================================================================
# =============================================================================
# # =============================================================================
# # Agr Positions
# # =============================================================================
# df_pos_hedge = pd.DataFrame(data_pos_hedge)    
# df_pos_hedge['duration'] = (df_pos_hedge['max_time']-df_pos_hedge['min_time']).dt.total_seconds()/(24*3600)
# 
# _style_white()
# fig, ax = plt.subplots(figsize=(16, 7))
# df = df_pos_hedge.copy()
# ax.set_title(f"Histogram of % Improvement on Hedging", pad=30)
# ax.set_ylabel(f"% Share", labelpad=20)
# ax.set_xlabel(f"% Improvement", labelpad=20)
# bin_edges = np.linspace(df['hedge_improve'].min(), df['hedge_improve'].max(), 40)
# bin_width = np.diff(bin_edges)[0]  # Calculate bin width
# bar_width = bin_width / 3  # Divide bins equally among the three categories
# for i, (status, color) in enumerate(zip(
#     ['lowvol', 'midvol', 'highvol'], 
#     [tailwind['teal-400'], tailwind['amber-400'], tailwind['pink-400']]
# )):
#     counts, _ = np.histogram(df[df['market_status'] == status]['hedge_improve'], bins=bin_edges)
#     ax.bar(bin_edges[:-1] + i * bar_width, counts/sum(counts), width=bar_width, color=color, alpha=0.9, label=status.capitalize())
# yticks = ax.get_yticks()
# ax.vlines(0, 0, max(yticks), color=tailwind['stone-300'])
# ax.set_yticks(ax.get_yticks())
# ax.set_xticks(ax.get_xticks())
# ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
# ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.1f}%'.format(x*100)))
# legend = ax.legend(loc='upper left')
# legend.get_frame().set_alpha(0.9)
# ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
# for spine in ax.spines.values(): spine.set_visible(False)
# fig.tight_layout(rect=[0.004, 0.004, .996, .996])
# fig.savefig(os.path.join('plots', f'Histogram Improvement.png'), dpi=200)
# fig.clf()
# 
#       
# df_pos_hedge_agr = df_pos_hedge.groupby(['gmx_duration_max', 
#                                      'market_status']).agg(
#         hedged_performance=('hedged_performance', 'mean'),
#         hedge_improve=('hedge_improve', 'mean'),
#         position_id=('position_id', 'nunique')
#         ).reset_index()
# df_pos_hedge_agr = df_pos_hedge_agr[df_pos_hedge_agr['gmx_duration_max']==12]
# cols_hedge = {
#     'market_status': 'Status',
#     'position_id': '# Positions',
#     'hedged_performance': '% Performance with Hedge',
#     'hedge_improve': '% Improvement with Hedge',
#     }
# df_pos_hedge_agr = df_pos_hedge_agr[cols_hedge.keys()].rename(columns=cols_hedge)
# for c in df_pos_hedge_agr.columns:
#     if '%' in c:
#         df_pos_hedge_agr[c]*=1
#         df_pos_hedge_agr[c] = df_pos_hedge_agr[c].apply(lambda x: f"{x:.2%}")
# 
# fig, ax = plt.subplots(figsize=(8, 1))  # Adjust figure size
# ax.axis('tight')
# ax.axis('off')
# table = ax.table(cellText=df_pos_hedge_agr.values, 
#                  colLabels=df_pos_hedge_agr.columns, 
#                  cellLoc='center', 
#                  loc='center')
# ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
# for spine in ax.spines.values(): spine.set_visible(False)
# plt.savefig(os.path.join('plots', f'Table Improvement.png'), bbox_inches='tight', dpi=300)
# =============================================================================

# position_id = '878458_0x902a7cebc98daa5a0e6de468052c75719c014797'
# position_id = '879576_0x5b393bd3c1d0d334b8bb9ae106edb4ec33801a3c'
# position_id = '851265_0x0d21716f645ce331cc4fadb9e621980acadf56dc'
# position_id = '873626_0x37c4d1abc89e1bb1ceaa9df3c17d4464fe95e21c'
# position_id = '840439_0x4942e2b839fc479c27c496b7758bab94ceb6b684'
# =============================================================================
# def _perp_gmx(df_market, gmx_type, gmx_collateral, gmx_leverage, gmx_duration_max):
#     df = df_market.copy()
# 
#     gmx_size = gmx_collateral * gmx_leverage
#     gmx_price_start = df_market['open'].iloc[0]
#     gmx_price_end = df_market['close'].iloc[0]
#     gmx_price_entry = gmx_price_start * (1 + gmx_fee_entry)
#     gmx_price_liq = gmx_price_entry * ((1 - 1/gmx_leverage) if gmx_type == "long" else (1 + 1/gmx_leverage))
#     df['price_liq'] = gmx_price_liq
#     df.loc[:,'liq'] = (df['low'] <= gmx_price_liq) if gmx_type == "long" else (df['high'] >= gmx_price_liq)
#     df.loc[:,'exit'] = df['close'] * ((1 - gmx_fee_exit) if gmx_type == "long" else (1 + gmx_fee_exit))
#     
#     df.loc[:,'cost_funding'] = 0.0
#     df.loc[df['_time'].dt.minute == 0, 'cost_funding'] = gmx_size * gmx_funding_rate
#     df.loc[:,'cost_funding'] = df['cost_funding'].fillna(0.0).cumsum()
#     if gmx_type == "long":
#         df.loc[:,'pnl'] = gmx_size*(df['exit'] - gmx_price_entry)/gmx_price_entry - df['cost_funding']
#     else:
#         df.loc[:,'pnl'] = gmx_size*(gmx_price_entry-df['exit'])/gmx_price_entry - df['cost_funding']
#         
#     if df['liq'].any():
#         df.loc[(df[df['liq']]['liq'].index.min()):,'pnl'] = -gmx_collateral
#         df['pnl'] = df['pnl'].clip(-gmx_collateral,df['pnl'].max())
#  
#     df['duration'] = (df['_time']-df['_time'].iloc[0]).dt.total_seconds()/(3600)
#     df['lp_extra'] = False
#     if df['duration'].max()>=gmx_duration_max:
#         idx = df[df['duration']>=gmx_duration_max].index.min()
#         df.loc[idx:,'pnl'] = df.loc[idx,'pnl'] 
#         df.loc[idx:,'lp_extra'] = True
#  
#     df.loc[:,'collateral'] = gmx_collateral
#     df.loc[:,'v'] = df['pnl']+gmx_collateral
#     df.loc[:,'ret'] = df['pnl']/gmx_collateral
#     df.loc[:,'ret_price'] = df['close']/gmx_price_start-1
#     return df
# =============================================================================
# =============================================================================
# def _lp_uni(df_market, pos_uni):
#     df = df_market.copy()
#     data_lp = [pos_uni]
#     for _,r in df.iterrows():
#         t = r['_time']
#         p = r['open']
#         pos_uni = data_lp[-1].copy()
#         data_lp += [_uni_upd(p, t, pos_uni)]
#     df_lp = pd.DataFrame(data_lp)
# 
#     r_fees = df_lp.iloc[0]['f']/df_lp.iloc[0]['v']
#     duraton_active = df_lp['active'].cumsum()/(60*24)
#     apy = (1 + r_fees) ** (365 / duraton_active.iloc[-1]) - 1
#     df_lp['f_cum'] = ((1 + apy) ** (duraton_active / 365) - 1)*df_lp.iloc[0]['v']
# 
#     data_active = [df_lp['t'].iloc[0]]
#     a = df_lp['active'].iloc[0]
#     for i_a, r_a in df_lp.iloc[1:].iterrows():
#         if r_a['active'] != a:
#             data_active+= [df_lp['t'].iloc[i_a]]
#             a = r_a['active']
#         else: data_active+= [np.nan]
#     df_lp['active_id'] = data_active
#     df_lp['active_id'] = df_lp['active_id'].ffill()
#     return df_lp
# =============================================================================
