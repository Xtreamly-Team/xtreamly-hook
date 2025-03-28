from src.GMXPrices import GMXPrices
from src.Positions import Positions
from src.Volatility import get_volatility_state


def execute_policy(user_id: str):
    # @Pawel build policy in here.

    df = Positions().get_active_positions(user_id)
    gmx_price = GMXPrices().get_signed_price("ETH")
    vol_status = get_volatility_state("ETH")["classification"]  # midvol
    out = {
        'perp_close': False,
        'perp_open': False,
        }
    return out
# =============================================================================
# =============================================================================
# info_input = { # Amin - worth adding token name
#     'user_id': 2636221,
#     'token': 'ETH',
# }
# =============================================================================
# =============================================================================
# id: string;
# userId: string;
# tokenA: string;
# tokenB: string;
# amountA: number;
# amountB: number;
# lowerTick: number;
# upperTick: number;
# hedgeAmount: number;
# status: PositionStatus;
# uniswapPositionId?: string;
# gmxPositionId?: string;
# createdAt: Date;
# updatedAt: Date;
# =============================================================================

# # =============================================================================
# # INPUTS
# # =============================================================================
# from datetime import datetime, timedelta
# import pandas as pd
# time_now = datetime.utcnow()
# vol_status = 'lowvol' # to fetch from API

# pos = {}
# pos['id'] = 'string'
# pos['userId'] = 'string'
# pos['tokenA'] = 'ETH' #string
# pos['tokenB'] = 'USDT' #string
# pos['amountA'] = 0.0234 #number
# pos['amountB'] = 5345.21 # number
# pos['lowerTick'] = 79565 #number
# pos['upperTick'] = 81565 #number
# pos['hedgeAmount'] = 134.21 #number (presume in usdt)
# pos['status'] = True # PositionStatus;
# pos['uniswapPositionId?'] = 'uniswapPosition1' # string
# pos['gmxPositionId?'] = 'gmxPosition1' # string
# pos['createdAt'] = datetime.utcnow()-timedelta(hours=13) #(presume related to LP...)
# pos['updatedAt'] = datetime.utcnow()
# #add
# pos['gmxPositionStartedAt'] = datetime.utcnow()-timedelta(hours=13) #(for now at open but can be any other date)

# df = pd.DataFrame([pos]) # assume the output always provides 1-row tbl

# gmx_price = 3138.737087496404
# p = gmx_price # we get market price from GMX, right?

# # =============================================================================
# # POLICY
# # =============================================================================
# PERP_MAX_HOURS = 12
# PERP_LEVERAGE = {'lowvol': 30, 'midvol': 25, 'highvol': 20}
# if 'gmxPositionId' in pos: # check if perp currently opened (and not liquidated) to see if close
#     perp_close = False
#     perp_time = (time_now-pos['gmxPositionStartedAt']).total_seconds()/3600
    
#     perp_close = False
#     perp_close = True if perp_time >= PERP_MAX_HOURS else perp_close # to check to expire
#     out = {
#         'perp_close': perp_close,
#         'perp_open': False,
#         }
# else: # if not open, we may open perp
#     p_a = 1.0001**pos['lowerTick']
#     p_b = 1.0001**pos['upperTick']
#     p_mid = (p_a+p_b)/2

#     perp_open = False
#     perp_open = True if pos['createdAt'] == pos['updatedAt'] else perp_open # to check if we are just opening position
#     perp_open = True if (p >= p_mid/1.01) and (p <= p_mid*1.01) else perp_open # to check price condition
#     if perp_open:
#         leverage = PERP_LEVERAGE[vol_status]
#         collateral = pos['hedgeAmount'] 
#         out = {
#             'perp_close': False,
#             'perp_open': perp_open,
#             'direction': 'short',
#             'leverage': leverage,
#             'collateral': collateral, # (later to take from LP value)
#             }
#     else:
#         out = {
#             'perp_close': False,
#             'perp_open': perp_open,
#             }
        
# print(out) # this is a signal for Amin!
# # Panos shall we save a new record to DB??
# # =============================================================================
