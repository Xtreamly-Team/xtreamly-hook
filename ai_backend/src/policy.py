from src.GMXPrices import GMXPrices
from src.Positions import Positions


def execute_policy(user_id: str):
    # @Pawel build policy in here.

    df = Positions().get_active_positions(user_id)
    gmx_price = GMXPrices().get_signed_price("ETH")
    return gmx_price
