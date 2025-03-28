from src.GMXPrices import GMXPrices
from src.Positions import Positions
from src.Volatility import get_volatility_state


def execute_policy(user_id: str):
    # @Pawel build policy in here.

    df = Positions().get_active_positions(user_id)
    gmx_price = GMXPrices().get_signed_price("ETH")
    vol_status = get_volatility_state("ETH")["classification"]  # midvol
    return gmx_price
