import {PrivateKeyAccount, privateKeyToAccount} from 'viem/accounts';
import {arbitrum} from 'viem/chains';
import {createPublicClient, createWalletClient, formatUnits, http, parseEther} from 'viem';
import {GmxSdk} from 'index';
import {TradeAction} from 'types/tradeHistory';
import {Logger} from '@nestjs/common';
import {BATCH_CONFIGS} from '@gmx/configs/batch';
import {MarketsInfoResult} from 'modules/markets/types';
import {PositionsData, Position as GMXPosition} from 'types/positions';
import {convertToUsd} from 'utils/tokens';
import {Position, PositionStatus} from "@modules/position/entities/position.entity/position.entity";

const logger = new Logger('GMX');

const ETH_USDC_MARKET = '0x70d95587d40A2caf56bd97485aB3Eec10Bee6336';
const USDC_ADDRESS = '0xaf88d065e77c8cC2239327C5EDb3A432268e5831';

export class GMX {
  private sdk: GmxSdk;
  private walletClient: any;
  private publicClient: any;
  private marketInfo: MarketsInfoResult;
  private account: PrivateKeyAccount;

  constructor(userPrivateKey: `0x${string}`) {
    const account = privateKeyToAccount(userPrivateKey);
    this.account = account;
    const rpcUrl = arbitrum.rpcUrls.default.http[0];

    const transport = http(rpcUrl, {
      timeout: 60000,
      batch: {
        wait: 200,
        batchSize: 1000,
      },
      retryDelay: 1000,
      retryCount: 3,
    });

    const walletClient = createWalletClient({
      account,
      chain: arbitrum,
      transport,
    });
    this.walletClient = walletClient;

    const publicClient = createPublicClient({
      chain: arbitrum,
      transport,
      batch: BATCH_CONFIGS[arbitrum.id].client,
    });
    this.publicClient = publicClient;

    const subsquidUrl =
      'https://gmx.squids.live/gmx-synthetics-arbitrum:live/api/graphql';
    const subgraphUrl =
      'https://subgraph.satsuma-prod.com/3b2ced13c8d9/gmx/synthetics-arbitrum-stats/api';

    this.sdk = new GmxSdk({
      chainId: arbitrum.id,
      rpcUrl,
      oracleUrl: 'https://arbitrum-api.gmxinfra.io',
      walletClient,
      publicClient,
      subsquidUrl,
      subgraphUrl,
    });
    this.sdk.setAccount(account.address);
  }

  async init() {
    this.marketInfo = await this.getMarketInfo();
  }

  async getMarketInfo() {
    const marketInfo = await this.sdk.markets.getMarketsInfo();

    if (!marketInfo.marketsInfoData || !marketInfo.tokensData) {
      throw new Error('Unable to fetch Markets info...');
    }

    if (!(ETH_USDC_MARKET in marketInfo.marketsInfoData)) {
      throw new Error('Unable to fetch Markets info for ETH_USDC_MARKET...');
    }

    return marketInfo;
  }

  async getTrades(): Promise<TradeAction[]> {
    return this.sdk.trades.getTradeHistory({
      marketsInfoData: this.marketInfo.marketsInfoData,
      tokensData: this.marketInfo.tokensData,
      pageIndex: 0,
      pageSize: 10,
    });
  }

  async getPositions(): Promise<PositionsData | undefined> {
    return this.sdk.positions.getPositions({
      marketsData: this.marketInfo.marketsInfoData!,
      tokensData: this.marketInfo.tokensData!,
    }).then((res) => res.positionsData);
  }

  async getLastPosition(): Promise<GMXPosition> {
    const positions = await this.getPositions();

    const positionsArray = Object.values(positions || {});
    if (!positionsArray.length) {
      logger.warn('No positions found to close');
      throw new Error('No positions found to close');
    }

    // Get the first position (index 0) to close
    const position = positionsArray[0];
    logger.debug('Closing position:', position);
    return position;
  }

  getEthUSDCMarketInfo() {
    return this.marketInfo.marketsInfoData![ETH_USDC_MARKET];
  }

  getCollateralToken() {
    return this.marketInfo.tokensData![USDC_ADDRESS];
  }

  async short(amount: number, _leverage: number): Promise<Partial<Position>> {
    // 1. Set up basic parameters
    const COLLATERAL_AMOUNT_USDC = BigInt(amount) * 10n ** 6n;
    const leverage = BigInt(_leverage);

    const ALLOWED_SLIPPAGE_BPS = 50n;

    const ethUSDCMarketInfo = this.getEthUSDCMarketInfo();
    const collateralToken = this.getCollateralToken();

    // 2. Get current prices and calculate USD values
    const indexPrice = ethUSDCMarketInfo.indexToken.prices.maxPrice; // For shorts use maxPrice, for longs use minPrice
    const collateralPrice = collateralToken.prices.minPrice;

    // 3. Calculate position values
    const initialCollateralUsd = (COLLATERAL_AMOUNT_USDC * collateralPrice) / 10n**6n; // Convert to 30 decimals
    const sizeDeltaUsd = initialCollateralUsd * leverage;

    // 4. Calculate acceptable price with slippage
    const slippageMultiplier = 10000n - ALLOWED_SLIPPAGE_BPS; // Subtract for shorts, add for longs
    const acceptablePrice = (indexPrice * slippageMultiplier) / 10000n;

    await this.sdk.orders.createIncreaseOrder({
      // Market data from GMX SDK
      marketsInfoData: this.marketInfo.marketsInfoData!,
      tokensData: this.marketInfo.tokensData!,
      // Order type flags
      isLimit: false, // Market order = false, Limit order = true
      isLong: false, // Long = true, Short = false

      // Market address from marketsInfoData
      marketAddress: ethUSDCMarketInfo.marketTokenAddress,
      // Slippage tolerance in basis points (1 = 0.01%)   // e.g., ETH/USDC market address
      allowedSlippage: Number(ALLOWED_SLIPPAGE_BPS),
      // Collateral token information
      collateralToken, // Token object from tokensData
      collateralTokenAddress: collateralToken.address, // Address of collateral token (e.g., USDC)
      receiveTokenAddress: collateralToken.address, // Address of receive token (same as collateral token)

      // Token you're starting with (if different from collateral, will be swapped)
      fromToken: collateralToken,
      // Market information
      marketInfo: ethUSDCMarketInfo, // Market object from marketsInfoData
      // Index token information
      indexToken: ethUSDCMarketInfo.indexToken, // Token being traded (e.g., ETH for ETH/USDC)
      // Increase amounts
      increaseAmounts: {
        // Initial collateral amount in token decimals (e.g., 6 for USDC)
        initialCollateralAmount: COLLATERAL_AMOUNT_USDC,

        // Initial collateral converted to USD (30 decimals)
        // Calculate: initialCollateralAmount * collateralPrice * 10^(30-tokenDecimals)
        initialCollateralUsd,

        // Actual collateral amount after fees
        collateralDeltaAmount: COLLATERAL_AMOUNT_USDC,

        // Actual collateral in USD after fees (30 decimals)
        collateralDeltaUsd: initialCollateralUsd,

        // Amount of index token being traded (token decimals)
        // Calculate: sizeDeltaUsd / indexPrice
        indexTokenAmount: sizeDeltaUsd / indexPrice,

        // Position size in USD (30 decimals)
        // Calculate: collateralAmount * leverage * 10^30
        sizeDeltaUsd,

        // Position size in tokens (token decimals)
        // Calculate: sizeDeltaUsd / indexPrice
        sizeDeltaInTokens: sizeDeltaUsd / indexPrice,

        // Estimated leverage (BPS)
        // Calculate: leverage * 10000
        // Leverage * 10000 (2x = 20000)
        estimatedLeverage: leverage * 10000n,

        // Current index price (30 decimals)
        // Get from: marketInfo.indexToken.prices
        indexPrice,

        // Collateral token prices (30 decimals)
        initialCollateralPrice: collateralPrice,
        collateralPrice,

        // For market orders = 0
        triggerPrice: 0n,

        // Acceptable price (30 decimals)
        acceptablePrice,

        // Acceptable price delta (BPS)
        acceptablePriceDeltaBps: 0n,
        positionFeeUsd: 0n, // Will be calculated by GMX
        swapPathStats: undefined,
        uiFeeUsd: 0n,
        swapUiFeeUsd: 0n,
        feeDiscountUsd: 0n,
        borrowingFeeUsd: 0n,
        fundingFeeUsd: 0n,
        positionPriceImpactDeltaUsd: 0n,
        externalSwapQuote: undefined,
      },
    });

    const pos = await this.getLastPosition();
    
    console.log(pos);

    // Add this dynamic logger statement after the order creation
    logger.log(`
      ============================ ORDER CREATED ============================
      Type: Short Position (Increase Order)
      Market: ${ethUSDCMarketInfo.name} (${ethUSDCMarketInfo.indexToken.symbol}/USDC)
      Size: ${formatUnits(sizeDeltaUsd, 30)} USD
      Collateral: ${formatUnits(initialCollateralUsd / collateralPrice * 10n ** BigInt(collateralToken.decimals), collateralToken.decimals)} ${collateralToken.symbol}
      Leverage: ${leverage}x
      Entry Price: ${formatUnits(indexPrice, 30)} USD
      Acceptable Price: ${formatUnits(acceptablePrice, 30)} USD
      ===================================================================
    `);

    const hedgeAmount = Number(formatUnits(sizeDeltaUsd, 30));
    return {
      tokenA: ethUSDCMarketInfo.indexToken.symbol,
      tokenB: collateralToken.symbol,
      amountA: 0,
      amountB: 0,
      lowerTick: 0,
      upperTick: 0,
      hedgeAmount,
      status: PositionStatus.ACTIVE,
      gmxPositionId: 'NA',
      metadata: {
        gmx: {
          marketName: ethUSDCMarketInfo.name,
          indexToken: ethUSDCMarketInfo.indexToken.symbol,
          collateralToken: collateralToken.address,
          sizeDeltaUsd: hedgeAmount.toString(),
          // txHash,
          timestamp: new Date().toISOString(),
          originalWalletAddress: this.account.address,
          isHedge: true,
        },
      },
    };
  }

  async closeShort() {
    logger.log('Starting to close short position...');

    try {
      logger.log(`Attempting to close short position for address: ${this.account.address}`);

      // Get positions from the GMX SDK
      const position = await this.getLastPosition();

      // Log position details for verification
      const marketInfo = this.getEthUSDCMarketInfo();
      if (!marketInfo) {
        throw new Error('Market info not found for position');
      }

      logger.log(`Closing ${position.isLong ? 'Long' : 'Short'} position of ${formatUnits(position.sizeInUsd, 30)} USD ${marketInfo.name} @ETH/USDC`);

      // Get collateral token info
      const collateralToken = this.marketInfo.tokensData![position.collateralTokenAddress];
      if (!collateralToken) {
        throw new Error('Collateral token not found');
      }

      // Determine acceptable price for closing
      const ALLOWED_SLIPPAGE_BPS = 50n;  // 0.5%

      // Get price based on position type - IMPORTANT: reversed for closing!
      const prices = marketInfo.indexToken.prices;

      // When closing shorts, we want the MAX price (worst case)
      // When closing longs, we want the MIN price (worst case)
      const indexPrice = position.isLong ? prices.minPrice : prices.maxPrice;

      // Calculate acceptable price with slippage - IMPORTANT: reversed for closing!
      // For shorts: add slippage to accept higher price
      // For longs: subtract slippage to accept lower price
      const slippageMultiplier = position.isLong
        ? 10000n - ALLOWED_SLIPPAGE_BPS
        : 10000n + ALLOWED_SLIPPAGE_BPS;

      const acceptablePrice = (indexPrice * slippageMultiplier) / 10000n;

      logger.debug(`Position index price: ${formatUnits(indexPrice, 30)}`);
      logger.debug(`Acceptable price: ${formatUnits(acceptablePrice, 30)}`);

      // 6. Calculate execution fee (0.01 ETH)
      const executionFee = parseEther('0.01');

      // 7. Properly construct the decrease order
      logger.log('Creating decrease order to close position...');

      // Import the required enums from the GMX SDK
      const OrderType = {
        MarketSwap: 0,
        LimitSwap: 1,
        MarketIncrease: 2,
        LimitIncrease: 3,
        MarketDecrease: 4,
        LimitDecrease: 5,
        StopLossDecrease: 6,
        Liquidation: 7,
      };

      const DecreasePositionSwapType = {
        NoSwap: 0,
        SwapPnlTokenToCollateralToken: 1,
        SwapCollateralTokenToPnlToken: 2,
      };

      // Calculate collateral delta
      const collateralDeltaUsd = convertToUsd(
        position.collateralAmount,
        collateralToken.decimals,
        collateralToken.prices.minPrice,
      );

      // Wrap the final order creation in a try-catch block
      try {
        // Create the decrease order with correct parameters
        const result = await this.sdk.orders.createDecreaseOrder({
          marketsInfoData: this.marketInfo.marketsInfoData!,
          tokensData: this.marketInfo.tokensData!,
          marketInfo,
          isLong: position.isLong,
          allowedSlippage: Number(ALLOWED_SLIPPAGE_BPS),
          collateralToken,
          decreaseAmounts: {
            isFullClose: true,

            // Position size and collateral info - must match position exactly
            sizeDeltaUsd: position.sizeInUsd,
            sizeDeltaInTokens: position.sizeInTokens,
            collateralDeltaUsd: collateralDeltaUsd || 0n,
            collateralDeltaAmount: position.collateralAmount,

            // Price info
            indexPrice,
            collateralPrice: collateralToken.prices.minPrice,
            triggerPrice: 0n, // Market order
            acceptablePrice,

            // Required parameters - will be calculated by GMX
            acceptablePriceDeltaBps: 0n,
            recommendedAcceptablePriceDeltaBps: 0n,
            estimatedPnl: position.pnl || 0n,
            estimatedPnlPercentage: 0n,
            realizedPnl: 0n,
            realizedPnlPercentage: 0n,
            positionFeeUsd: 0n,
            uiFeeUsd: 0n,
            swapUiFeeUsd: 0n,
            feeDiscountUsd: 0n,
            borrowingFeeUsd: position.pendingBorrowingFeesUsd || 0n,
            fundingFeeUsd: 0n,
            swapProfitFeeUsd: 0n,
            positionPriceImpactDeltaUsd: 0n,
            priceImpactDiffUsd: 0n,
            payedRemainingCollateralAmount: 0n,
            payedOutputUsd: 0n,
            payedRemainingCollateralUsd: 0n,
            receiveTokenAmount: 0n,
            receiveUsd: 0n,

            // Enum values
            decreaseSwapType: DecreasePositionSwapType.NoSwap,
            triggerOrderType: OrderType.MarketDecrease,
          },
        });

        // Add formatted logger statement for position closure
        logger.log(`
        ============================= POSITION CLOSED ============================
        Market: ${marketInfo.name} (${marketInfo.indexToken.symbol}/USDC)
        Type: ${position.isLong ? 'Long' : 'Short'} Position
        Size: ${formatUnits(position.sizeInUsd, 30)} USD
        Collateral: ${formatUnits(position.collateralAmount, collateralToken.decimals)} ${collateralToken.symbol}
        Exit Price: ${formatUnits(indexPrice, 30)} USD
        Acceptable Price: ${formatUnits(acceptablePrice, 30)} USD
        PnL: ${formatUnits(position.pnl || 0n, 30)} USD
        ======================================================================
        `);

        logger.log('Position close order created successfully');

        const hedgeAmount = Number(formatUnits(position.sizeInUsd, 30));
        return {
          tokenA: marketInfo.indexToken.symbol,
          tokenB: collateralToken.symbol,
          amountA: 0,
          amountB: 0,
          lowerTick: 0,
          upperTick: 0,
          hedgeAmount,
          status: PositionStatus.CLOSED,
          gmxPositionId: 'NA',
          metadata: {
            gmx: {
              marketName: marketInfo.name,
              indexToken: marketInfo.indexToken.symbol,
              collateralToken: collateralToken.address,
              sizeDeltaUsd: hedgeAmount.toString(),
              // txHash,
              timestamp: new Date().toISOString(),
              originalWalletAddress: this.account.address,
              isHedge: true,
            },
          },
        };
      } catch (orderError) {
        logger.error(`Failed to create close order: ${orderError.message || orderError}`);
        logger.error('Failed to create decrease order:', orderError);
        throw new Error(`Position close failed: ${orderError.message || orderError}`);
      }
    } catch (error) {
      logger.error(`Position close failed: ${error.message || error}`);
      logger.error('Failed to close position:', error);
      throw new Error(`Position close failed: ${error.message || error}`);
    }
  }

  async approveTokenForRouter(tokenAddress: string, amount: bigint) {
    logger.log(`Approving ${formatUnits(amount, 18)} of token ${tokenAddress} for GMX router...`);

    // GMX Exchange Router address on Arbitrum
    const exchangeRouterAddress = '0x7452c558d45f8afC8c83dAe62C3f8A5BE19c71f6';

    // ERC20 approve function ABI
    const approveAbi = [
      {
        name: 'approve',
        type: 'function',
        inputs: [
          { name: 'spender', type: 'address' },
          { name: 'amount', type: 'uint256' },
        ],
        outputs: [{ type: 'bool' }],
        stateMutability: 'nonpayable',
      },
    ] as const;

    try {
      // Send the approval transaction
      const txHash = await this.walletClient.writeContract({
        address: tokenAddress as `0x${string}`,
        abi: approveAbi,
        functionName: 'approve',
        args: [exchangeRouterAddress as `0x${string}`, amount]
      });

      logger.log(`Approval transaction submitted: ${txHash}`);

      // Wait for the transaction to be mined
      const receipt = await this.publicClient.waitForTransactionReceipt({ hash: txHash });
      logger.log(`Approval confirmed in block ${receipt.blockNumber}`);

      return receipt;
    } catch (error) {
      logger.error(`Token approval failed: ${error.message || error}`);
      logger.error('Token approval failed:', error);
      throw error;
    }
  }
}
