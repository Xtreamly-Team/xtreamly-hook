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

folder='plots_position_managment'
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

def _pos_uni(t, p, p_a, p_b, x_opn, y_opn):#, fees_usd):
    pos_uni = {
        't_opn': t,
        'p_opn': p,
        'p_a': p_a,
        'p_b': p_b,
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

def _plot_pos_hedge(*arg):
    pos_uni = data_uni[0]
    pos_gmx = data_gmx[0]    
    p_values = hedge_value['p_values']
    df_gmx_v = hedge_value['df_gmx_v']
    df_uni_v = hedge_value['df_uni_v']
    df_org_v = hedge_value['df_org_v']
    imprv = '${:,.2f}'.format(hedge_value['lim_imprv'])
    p_fr = '${:,.0f}'.format(p_values.min())
    p_to = '${:,.0f}'.format(p_values.max())
    sub = f"First Hedge: Expected Value Improve with Hedging: {imprv} for Price Scenarios: {p_fr} - {p_to}. Selected leverage: x{pos_gmx['l']}."
    a_share = (100*df_org['active'].sum()/df_org.shape[0]).round(2)
    r_impr = '{:,.2f}%'.format(100*((df_uni['v'].iloc[-1]+df_gmx['v'].iloc[-1])/df_org['v'].iloc[-1]-1))
    sub_timeline = f"Original and Hedged Positions were Active in {a_share}% of Time."
    sub_return = f"Improvement in Return for Hedged Position: {r_impr}."

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
    #ax.plot(df_p['_time'], df_p['ret_org_f_cum'], tailwind['teal-600'], alpha=0.5, linewidth=2, label="Original Position Fees")  
    #ax.plot(df_p['_time'], df_p['ret_org_imp_loss'], tailwind['pink-800'], alpha=0.9, linewidth=2, label="Original Position Imp. Loss")
    ax.plot(df_p['_time'], df_p['ret_hedge'], tailwind['indigo-600'], alpha=0.9, linewidth=2, label="Hedged Position")
    #ax.plot(df_p['_time'], df_p['ret_hedge_f_cum'], tailwind['teal-600'], alpha=0.9, linewidth=2, label="Hedged Position Fees")
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
#     for state_id in df_p['state_id'].unique():
#         if state_id.split('_')[0]=="lowvol": state_color = tailwind['teal-300']
#         elif state_id.split('_')[0]=="midvol": state_color = tailwind['yellow-300']
#         else: state_color = tailwind['orange-300']
#         df_p_s = df_p[df_p['state_id']==state_id]
#         state_time = [df_p_s['_time'].iloc[0],  df_p_s['_time'].iloc[-1]]
#         ax.fill_between(state_time, y_min, y_max, color=state_color, alpha=0.3)
# =============================================================================
    ax.scatter(df_lp[df_lp['signal_hedge']]['t'], df_lp[df_lp['signal_hedge']]['p'], 
               color=tailwind['indigo-300'], s=33, label="Hedge Signals", alpha=.6, zorder=4)
    ax.scatter(df_gmx[df_gmx['t'] == df_gmx['t_opn']]['t'], df_gmx[df_gmx['t'] == df_gmx['t_opn']]['p'], 
               color=tailwind['indigo-500'], s=88, label="Hedge Open", alpha=.99, zorder=4)
    for t_opn in df_gmx['t_opn'].unique():
        if t_opn==t_opn: 
            df_gmx_id = df_gmx[df_gmx['t_opn']==t_opn]
            ax.plot(df_gmx_id['t'], df_gmx_id['p_liq'], alpha=0.9, linewidth=3, color=tailwind['red-500'])
    ax.plot([x_min], [y_min], alpha=0.9, linewidth=2, color=tailwind['red-500'], label="Price Perp. Liquidation")
    for a in df_lp['active_id'].unique():
        df_lp_a = df_lp[df_lp['active_id']==a]
        pos_active = df_lp_a['active'].iloc[0]
        a_color = tailwind['emerald-400'] if pos_active else tailwind['rose-400']
        ax.fill_between(df_lp_a['t'],  df_lp_a['p_a'],  df_lp_a['p_b'], color=a_color, alpha=0.6)
# =============================================================================
#     ax.fill_between([x_min],  y_min, y_min, color=tailwind['teal-300'], alpha=0.7, label="Volatility Low")    
#     ax.fill_between([x_min],  y_min, y_min, color=tailwind['yellow-300'], alpha=0.7, label="Volatility Medium") 
#     ax.fill_between([x_min],  y_min, y_min, color=tailwind['orange-300'], alpha=0.7, label="Volatility High") 
# =============================================================================

    ax.fill_between([x_min],  y_min, y_min, color=tailwind['emerald-500'], alpha=0.7, label="LP Active")    
    ax.fill_between([x_min],  y_min, y_min, color=tailwind['rose-500'], alpha=0.7, label="LP Not active")    
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))

    ax = axes[4]
    ax.set_title(f"Positions Value for Price Scenarios", pad=35)
    ax.text(0.5, 1.06, sub, ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_ylabel(f"$ Value", labelpad=10)
    ax.set_xlabel(f"$ Price Scenario", labelpad=10)
    ax.plot(p_values, df_org_v['v'], color=tailwind['stone-600'], alpha=0.9, linewidth=3, label=f"Original Position")
    ax.plot(p_values, df_uni_v['v'] + df_gmx_v['v'], color=tailwind['indigo-600'], alpha=0.99, linewidth=3, label=f"Hedged Position")
    y_min, y_max = max(.0, ax.get_yticks().min()), ax.get_yticks().max()
    ax.vlines(pos_gmx['p_opn'], y_min, y_max, color=tailwind['stone-900'], alpha=.9, label="Price Open")
    ax.vlines(pos_gmx['p_liq'], y_min, y_max, color=tailwind['red-500'], label="Price Perp. Liquidation")
    ax.fill_between([p_a,p_b], [y_min, y_min], [y_max, y_max], color=tailwind['emerald-400'], alpha=0.6, label="LP Active") 
# =============================================================================
#     ax.fill_between([p_values.min(),p_a], [y_min, y_min], [y_max, y_max], color=tailwind['rose-400'], alpha=0.4, label="LP Not Active") 
#     ax.fill_between([p_b,p_values.max()], [y_min, y_min], [y_max, y_max], color=tailwind['rose-400'], alpha=0.4) 
#     
# =============================================================================
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
        legend = ax.legend(loc='lower right')
        legend.get_frame().set_facecolor('white')  # Sets the background color to white
        legend.get_frame().set_edgecolor('black')  # Optional: Adds a border to the legend
        legend.get_frame().set_alpha(.6)  # Ensures no transparency (fully opaque)
    fig.tight_layout(rect=[0.01, 0.01, .99, .99])
    fig.subplots_adjust(hspace=.5)
    fig.savefig(os.path.join(folder, f'Position Hedge {position_id}'), dpi=200)
    fig.clf()

# =============================================================================
# Simulate Hedge
# =============================================================================
hedge_type = 'short'
hedge_share = .1 
hedge_l_min = 8
hedge_l_max = 12
hedge_time_min = 1
hedge_time_max = 12

data_hedge, data_perp = [], []
hedge_times_max = [12, 18, 24,48, 72, 96]
for hedge_time_max in hedge_times_max:
    for i, position_id in enumerate(pool_ids[:]):
        print(hedge_time_max, i, position_id)
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
        df_org = _df_uni(df_p, pos_org)
        df_org['signal_hedge'] = \
            (df_org['p']<=df_org['p_opn']) & \
            (df_org['p']>=(df_org['p_a']+df_org['p_opn'])/2) & \
            (df_org['active'])
        df_org.loc[0,'signal_hedge'] = True
        df_lp = df_org
    
        pos_uni = _pos_uni(t, p, p_a, p_b, x_opn*(1-hedge_share), y_opn*(1-hedge_share))
    
        p_gmx = df_p['open'].iloc[0]
        collateral = np.abs(pos_org['v'])*(hedge_share)
        leverage = int((1/(p_b/p-1)))-1
        leverage = min(hedge_l_max,max(hedge_l_min,leverage))     
        pos_gmx = _pos_gmx(t, p_gmx, df_p, hedge_type, collateral, leverage)
        pos_gmx['state'] = df_p['state'].iloc[0]
        
        hedge_value = _hedge_value(pos_org, pos_uni, pos_gmx, df_p)
        hedge_imprv = {k: hedge_value[k] for k in ['lim_imprv', 'conc_imprv']} 
    
        # Simulate
        data_uni, data_gmx = [pos_uni], [pos_gmx]
        for i,r in df_p.iloc[1:].iterrows():
            t = r['_time']
            p = r['open']
            pos_uni = data_uni[-1].copy()
            pos_gmx = data_gmx[-1].copy()
            
            if len(pos_gmx) > 0:
                new_gmx = _upd_gmx(p, t, df_p, pos_gmx)
                new_uni = _upd_uni(p, t, pos_uni)
                if hedge_time_max <= new_gmx['t_hours'] or new_gmx['liq']:
                    new_uni = _pos_uni(t, p, p_a, p_b, 
                                       new_uni['x'] * (1+new_gmx['v']/new_uni['v']), 
                                       new_uni['y'] * (1+new_gmx['v']/new_uni['v']))
                    new_gmx['v'] = 0
                    new_gmx['cls'] = True
    
            if len(pos_gmx) == 0 or 'cls' in pos_gmx:
                new_uni = _upd_uni(p, t, pos_uni)
                if df_org['signal_hedge'].iloc[i]:
                    collateral = pos_uni['v']*hedge_share
                    leverage = int((1/(p_b/p-1)))-1
                    leverage = min(hedge_l_max,max(hedge_l_min,leverage))    
                    new_uni = _pos_uni(t, p, p_a, p_b, 
                                       new_uni['x'] * (1-collateral/new_uni['v']), 
                                       new_uni['y'] * (1-collateral/new_uni['v']))
                    new_gmx = _pos_gmx(t, p, df_p, hedge_type, collateral, leverage)
                    new_gmx['state'] = df_p['state'].iloc[i]
                else: new_gmx = {}
            data_uni += [new_uni]
            data_gmx += [new_gmx]
            
        df_gmx = pd.DataFrame(data_gmx)
        df_gmx['v'] = df_gmx['v'].fillna(.0)
        df_gmx['liq'] = df_gmx['liq'].fillna(False)
        df_gmx['liq']*= 1
        df_uni = pd.DataFrame(data_uni)
    
        # Return
        fee_prop = df_uni[df_uni['active']]['v'].sum()/df_org[df_org['active']]['v'].sum()
        f_uni = fees_usd*fee_prop
        f_org = fees_usd
        inv_usd = pos_org['inv_usd']
        df_uni = _f_cum(df_uni, f_uni, inv_usd)
        df_org = _f_cum(df_org, f_org, inv_usd)
        df_p['ret_org'] = (df_org['v']+df_org['f_cum'])/inv_usd-1
        df_p['ret_org_f_cum'] = (df_org['f_cum'])/inv_usd
        df_p['ret_org_imp_loss'] = (df_org['v'])/inv_usd-1
        df_p['ret_hedge'] = (df_uni['v']+df_uni['f_cum']+df_gmx['v'])/inv_usd-1
        df_p['ret_hedge_f_cum'] = (df_uni['f_cum'])/inv_usd
        
        if df_gmx['pnl'].sum()>0 and i<=30:
            _plot_pos_hedge()
        
        df_perps = df_gmx.groupby(['t_opn']).agg(
            collateral = ('c', 'mean'),
            pnl = ('pnl', 'last'),
            liq = ('liq', 'sum'),
            cost_rate = ('cost_rate', 'last'),
            state = ('state', 'first'),
            ).reset_index()
        df_perps['ret'] = df_perps['pnl']/df_perps['collateral']
        df_perps['position_id'] = position_id
        df_perps['hedge_time_max'] = hedge_time_max
        
        out = {
            'hedge_time_max': hedge_time_max,
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
            'hedged_usd_exp_imprv': hedge_imprv['lim_imprv'],
            'hedged_usd_perp_count': df_perps.shape[0],
            'hedged_usd_perp_liq': np.sum(df_perps['liq']),
            'hedged_usd_perp_collateral': df_perps['collateral'].sum(),
            'hedged_usd_perp_pnl': df_perps['pnl'].sum(),
            'hedged_usd_perp_ret': df_perps['pnl'].sum()/df_perps['collateral'].sum(),
            'hedged_usd_perp_cost_rate': df_perps['cost_rate'].iloc[-1],
            }
        data_hedge += [out]
        data_perp += [df_perps]
        
df_pos_hedge = pd.DataFrame(data_hedge)
df_pos_hedge['duration'] = (df_pos_hedge['position_time_max']-df_pos_hedge['position_time_min']).dt.total_seconds()/(24*3600) 
df_perps_all = pd.concat(data_perp)

# =============================================================================
# Plots
# =============================================================================
for hedge_time_max in hedge_times_max:
    
    df_pos_hedge = pd.DataFrame(data_hedge)
    df_pos_hedge['duration'] = (df_pos_hedge['position_time_max']-df_pos_hedge['position_time_min']).dt.total_seconds()/(24*3600)
    df_pos_hedge = df_pos_hedge[df_pos_hedge['hedge_time_max']==hedge_time_max]
    df_perps_all = pd.concat(data_perp)
    df_perps_all = df_perps_all[df_perps_all['hedge_time_max']==hedge_time_max]
    
    
    df_pos_hedge = pd.DataFrame(data_hedge)
    df_pos_hedge['duration'] = (df_pos_hedge['position_time_max']-df_pos_hedge['position_time_min']).dt.total_seconds()/(24*3600) 
    df_perps_all = pd.concat(data_perp)
    
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
    fig.savefig(os.path.join(folder, f'{hedge_time_max} Histogram Performance.png'), dpi=200)
    fig.clf()
    
    _style_white()
    fig, ax = plt.subplots(figsize=(16, 7))
    df = df_pos_hedge.copy()
    ax.set_title(f"Histogram of % Improvement from Original to Hedged Positions", pad=30)
    ax.set_ylabel(f"# Count", labelpad=10)
    ax.set_xlabel(f"% Improvement", labelpad=10)
    ax.hist(df['hedged_perc_impr'],bins=40, color=tailwind['teal-400'], alpha=0.9, label=f"Improvement on Hedging Position")
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
    fig.savefig(os.path.join(folder, f'{hedge_time_max} Histogram Improvement.png'), dpi=200)
    fig.clf()
    
    
    _style_white()
    fig, ax = plt.subplots(figsize=(16, 7))
    df = df_pos_hedge.copy()
    ax.set_title(f"Histogram of Hedging Positions % Total Returns on Perpetuals", pad=30)
    ax.set_ylabel(f"# Count", labelpad=10)
    ax.set_xlabel(f"% Active in Time", labelpad=10)
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
    fig.savefig(os.path.join(folder, f'{hedge_time_max} Histogram Return Perpetuals.png'), dpi=200)
    fig.clf()


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

# =============================================================================
# # Aggr Stats
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

