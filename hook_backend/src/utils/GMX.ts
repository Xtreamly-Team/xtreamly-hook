import {privateKeyToAccount} from "viem/accounts";
import {arbitrum} from "viem/chains";
import {createWalletClient, http} from "viem";
import {GmxSdk} from "index";
import {TradeAction} from "types/tradeHistory";

export class GMX {
  private sdk: GmxSdk;

  constructor(userPrivateKey: `0x${string}`) {
    const account = privateKeyToAccount(userPrivateKey);
    const rpcUrl = arbitrum.rpcUrls.default.http[0];

    const walletClient = createWalletClient({
      account,
      chain: arbitrum,
      transport: http(rpcUrl),
    });

    const subsquidUrl =
      'https://gmx.squids.live/gmx-synthetics-arbitrum:live/api/graphql';
    const subgraphUrl =
      'https://subgraph.satsuma-prod.com/3b2ced13c8d9/gmx/synthetics-arbitrum-stats/api';

    this.sdk = new GmxSdk({
      chainId: arbitrum.id,
      rpcUrl,
      oracleUrl: 'https://arbitrum-api.gmxinfra.io',
      walletClient,
      subsquidUrl,
      subgraphUrl,
    });
    this.sdk.setAccount(account.address);
  }

  getMarketInfo() {
    return this.sdk.markets.getMarketsInfo();
  }

  async getTrades(): Promise<TradeAction[]> {
    const { marketsInfoData, tokensData } = await this.getMarketInfo();
    return this.sdk.trades.getTradeHistory({
      marketsInfoData,
      tokensData,
      pageIndex: 0,
      pageSize: 10,
    });
  }
}
