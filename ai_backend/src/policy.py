from src.GMXPrices import GMXPrices
from src.Positions import Positions
from src.Volatility import get_volatility_state
from datetime import datetime
from decimal import Decimal

def execute_policy(user_id: str):
    gmx_positions = Positions().get_gmx_positions(user_id)
    gmx_positions = gmx_positions.sort_values("createdAt", ascending=False)
    lp_positions = Positions().get_active_lp_positions(user_id)
    lp_positions = lp_positions.sort_values("createdAt", ascending=False)

    gmx_price = GMXPrices().get_live_ticker("ETH")
    gmx_price = Decimal(gmx_price["maxPrice"]) / Decimal('1e12')

    vol_status = get_volatility_state("ETH")["classification"]  # midvol

    if len(lp_positions) == 0:
        return {}

    lp_pos = lp_positions.head(1).to_dict("records").pop()

    is_gmx_position_open = False
    if len(gmx_positions) > 0:
        pos = gmx_positions.head(1).to_dict("records").pop()
        is_gmx_position_open = pos["status"] == "active"
        is_first_active_position = len(gmx_positions) == 1

    time_now = datetime.utcnow()
    PERP_MAX_HOURS = 12
    PERP_LEVERAGE = {'lowvol': 30, 'midvol': 25, 'highvol': 20}
    if is_gmx_position_open: # check if perp currently opened (and not liquidated) to see if close
        perp_time = (time_now-pos['createdAt']).total_seconds() / 3600

        perp_close = False
        perp_close = True if perp_time >= PERP_MAX_HOURS else perp_close # to check to expire
        return {
            'perp_close': perp_close,
            'perp_open': False,
        }
    else: # if not open, we may open perp
        # p_a = 1.0001**pos['lowerTick']
        # p_b = 1.0001**pos['upperTick']
        # p_mid = (p_a+p_b)/2

        perp_open = False
        perp_open = True if is_first_active_position else perp_open # to check if we are just opening position
        # perp_open = True if (gmx_price >= p_mid/1.01) and (gmx_price <= p_mid*1.01) else perp_open # to check price condition
        if perp_open:
            leverage = PERP_LEVERAGE[vol_status]
            collateral_usdc = 0.05 * (lp_pos['amountA'] * gmx_price + lp_pos['amountB'])
            return {
                'perp_close': False,
                'perp_open': perp_open,
                'direction': 'short',
                'leverage': leverage,
                'collateral': collateral_usdc, # (later to take from LP value)
                }
        else:
            return {
                'perp_close': False,
                'perp_open': perp_open,
            }
