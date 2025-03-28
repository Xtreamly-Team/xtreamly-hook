import {privateKeyToAccount} from "viem/accounts";
import {arbitrum} from "viem/chains";
import {createWalletClient, createPublicClient, http, formatUnits} from "viem";
import {GmxSdk} from "index";
import {TradeAction} from "types/tradeHistory";

import {Position} from "types/positions";
import {TokensData} from "types/tokens";
import { bigMath } from "@app/gmx_sdk/utils/bigmath";
import { getMarkPrice } from "@app/gmx_sdk/utils/prices";
import {
  convertToUsd,
  getIsEquivalentTokens,
} from "@app/gmx_sdk/utils/tokens";
import { USD_DECIMALS } from "@app/gmx_sdk/configs/factors";
import { sleep } from "@app/gmx_sdk/utils/common";
import { parseEther } from "viem";



const account = privateKeyToAccount(process.env.PRIVATE_KEY! as any);

const walletClient = createWalletClient({
  transport: http(undefined, {
    timeout: 60000,
    batch: {
      wait: 200,
      batchSize: 1000,
    },
    retryDelay: 1000,
    retryCount: 3,
  }),
  account: account,
  chain: arbitrum,
});


const publicClient = createPublicClient({
  chain: arbitrum,
  batch: {
    multicall: {
      wait: 200,
      batchSize: 1024 * 1024,
    },
  },
  transport: http(undefined, {
    timeout: 60000,
    batch: {
      wait: 200,
      batchSize: 1000,
    },
  }),
});

const sdk = new GmxSdk({
  account: account.address,
  chainId: arbitrum.id,
  rpcUrl: arbitrum.rpcUrls.default.http[0],
  oracleUrl: 'https://arbitrum-api.gmxinfra.io',
  walletClient,
  publicClient,
  subsquidUrl :'https://gmx.squids.live/gmx-synthetics-arbitrum:live/api/graphql',
  subgraphUrl: 'https://subgraph.satsuma-prod.com/3b2ced13c8d9/gmx/synthetics-arbitrum-stats/api'
});

export async function shortIt() {
  const userPrivateKey = process.env.PRIVATE_KEY as `0x${string}`;
  if (!userPrivateKey) {
    throw new Error('PRIVATE_KEY environment variable is not set');
  }
  const account = privateKeyToAccount(userPrivateKey);
  const { marketsInfoData, tokensData, pricesUpdatedAt } = await sdk.markets.getMarketsInfo();

  if (!marketsInfoData || !tokensData) return;

  const positions = await sdk.positions.getPositions({
    marketsData: marketsInfoData,
    tokensData: tokensData,
  });

  console.log(positions)

  // console.log(account.address)
  // console.log("marketsInfoData")
  // console.log(marketsInfoData)
  // console.log("########################")
  // console.log("########################")
  // console.log("tokensData")
  // console.log(tokensData)
  // console.log("########################")
  // console.log("########################")
  // console.log("pricesUpdatedAt")
  // console.log(pricesUpdatedAt)
  
  console.log("########################")
  console.log("########################")

  const FULL_BPS = 10000n; // 100%



  
  // Get market info for the position
  // let marketInfo = marketsInfoData[positionsArray[0].marketAddress];
  // console.log(marketInfo)
  // if (!marketInfo) {
  //   throw new Error('Market info not found for position');
  // }
  // const marketInfo = marketsInfo["0x47c031236e19d024b42f8AE6780E44A573170703"];
  const collateralTokenData = tokensData["0xaf88d065e77c8cc2239327c5edb3a432268e5831"];
  const collateralToken = "0xaf88d065e77c8cc2239327c5edb3a432268e5831";




  try {
    console.log("Attempting to create increase order...");

   const { marketsInfoData } = await sdk.markets.getMarketsInfo();

   // Add this debug log to verify market selection
   console.log("Available markets:", marketsInfoData ? Object.keys(marketsInfoData) : []);
   const ETH_USDC_MARKET = "0x70d95587d40A2caf56bd97485aB3Eec10Bee6336";
   let marketInfo = marketsInfoData ? marketsInfoData[ETH_USDC_MARKET] : undefined;

   if (!marketInfo) {
       throw new Error("ETH/USDC market not found");
   }

   // Verify it's the right market
   console.log("Market name:", marketInfo.name);
   console.log("Index token:", marketInfo.indexToken.symbol);

   const collateralToken = tokensData["0xaf88d065e77c8cC2239327C5EDb3A432268e5831"];
   
  //////////////////////////////////////////////////////////////////
  // APPROVAL LOGIC
  //////////////////////////////////////////////////////////////////

    // Function to approve tokens for the GMX exchange router
    async function approveTokenForRouter(tokenAddress: string, amount: bigint) {
      console.log(`Approving ${formatUnits(amount, 18)} of token ${tokenAddress} for GMX router...`);
      
      // GMX Exchange Router address on Arbitrum
      const exchangeRouterAddress = "0x7452c558d45f8afC8c83dAe62C3f8A5BE19c71f6";
      
      // ERC20 approve function ABI
      const approveAbi = [
        {
          name: "approve",
          type: "function",
          inputs: [
            { name: "spender", type: "address" },
            { name: "amount", type: "uint256" }
          ],
          outputs: [{ type: "bool" }],
          stateMutability: "nonpayable"
        }
      ] as const;
      
      try {
        // Send the approval transaction
        const txHash = await walletClient.writeContract({
          address: tokenAddress as `0x${string}`,
          abi: approveAbi,
          functionName: "approve",
          args: [exchangeRouterAddress as `0x${string}`, amount]
        });
        
        console.log(`Approval transaction submitted: ${txHash}`);
        
        // Wait for the transaction to be mined
        const receipt = await publicClient.waitForTransactionReceipt({ hash: txHash });
        console.log(`Approval confirmed in block ${receipt.blockNumber}`);
        
        return receipt;
      } catch (error) {
        console.error("Token approval failed:", error);
        throw error;
      }
    }
    
    // Approve the collateral token before creating the order
    // Using a large approval amount (max uint256) to avoid frequent approvals
    const maxApproval = 2n ** 256n - 1n;
    await approveTokenForRouter(collateralToken.address, maxApproval);
    
    // Alternatively, you can approve just the amount needed for this specific transaction:
    // await approveTokenForRouter(collateralToken, 3000000n);

    //////////////////////////////////////////////////////////////////
    // FORMAT POSITION
    //////////////////////////////////////////////////////////////////
    const formatPosition = (position: Position) => {
      const marketInfo = marketsInfoData?.[position.marketAddress];
      if (!marketInfo || !tokensData) return "";
      const marketName = marketInfo.name || "";
      const isLong = position.isLong ? "Long" : "Short";
  
      const sizeInUsd = formatUnits(position.sizeInUsd, USD_DECIMALS);
      const collateralTokenName =
        tokensData?.[position.collateralTokenAddress].name;

  
      const price = getMarkPrice({
        prices: marketInfo.indexToken.prices,
        isIncrease: false,
        isLong: position.isLong,
      });
      const formattedPrice = formatUnits(price, USD_DECIMALS);
  
      return `${marketName} ${isLong} ${sizeInUsd} ${collateralTokenName} ${formattedPrice}`;
    };
    //////////////////////////////////////////////////////////////////
    
    const positionsArray = Object.values(positions.positionsData || {});
    for (let index = 0; index < positionsArray.length; index++) {
      const position = positionsArray[index];
      console.log(index, formatPosition(position));
    }


  console.log(positionsArray)
    //////////////////////////////////////////////////////////////////

    // 1. Set up basic parameters
    const COLLATERAL_AMOUNT_USDC = 3n * 10n**6n; // 3 USDC (USDC has 6 decimals)
    const LEVERAGE = 4n; // 1x leverage
    const ALLOWED_SLIPPAGE_BPS = 50n; // 0.5%

    // 2. Get current prices and calculate USD values
    const indexPrice = marketInfo.indexToken.prices.maxPrice; // For shorts use maxPrice, for longs use minPrice
    const collateralPrice = collateralToken.prices.minPrice;

    // 3. Calculate position values
    const initialCollateralUsd = (COLLATERAL_AMOUNT_USDC * collateralPrice) / 10n**6n; // Convert to 30 decimals
    const sizeDeltaUsd = initialCollateralUsd * LEVERAGE;

    // 4. Calculate acceptable price with slippage
    const slippageMultiplier = 10000n - ALLOWED_SLIPPAGE_BPS; // Subtract for shorts, add for longs
    const acceptablePrice = (indexPrice * slippageMultiplier) / 10000n;

    // 5. Calculate execution fee (0.01 ETH or as needed)
    const executionFee = parseEther("0.01");

    // Now create the order with calculated values
    sdk.orders.createIncreaseOrder({
      // Market data from GMX SDK
      marketsInfoData: marketsInfoData!,
      tokensData,
      // Order type flags
      isLimit: false, // Market order = false, Limit order = true
      isLong: false,  // Long = true, Short = false
      
      // Market address from marketsInfoData
      marketAddress: marketInfo.marketTokenAddress,
      // Slippage tolerance in basis points (1 = 0.01%)   // e.g., ETH/USDC market address
      allowedSlippage: Number(ALLOWED_SLIPPAGE_BPS),
       // Collateral token information
      collateralToken,  // Token object from tokensData
      collateralTokenAddress: collateralToken.address,    // Address of collateral token (e.g., USDC)
      receiveTokenAddress: collateralToken.address,        // Address of receive token (same as collateral token)

      // Token you're starting with (if different from collateral, will be swapped)
      fromToken: collateralToken,
      // Market information
      marketInfo,  // Market object from marketsInfoData
      // Index token information
      indexToken: marketInfo.indexToken,  // Token being traded (e.g., ETH for ETH/USDC)
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
        estimatedLeverage: LEVERAGE * 10000n,
        
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
        externalSwapQuote: undefined
      },
    });

    console.log("Increase order created successfully");
  } catch (error) {
    console.error("Failed to create increase order:", error);
    throw new Error(`Order creation failed: ${error.message || error}`);
  }

}

export async function closeShort() {
  try {
    console.log("Attempting to close short position...");
    
    // 1. Get market info and tokens data
    const { marketsInfoData, tokensData } = await sdk.markets.getMarketsInfo();
    if (!marketsInfoData || !tokensData) {
      throw new Error("Failed to get markets info");
    }

    // 2. Get positions from the GMX SDK
    const positions = await sdk.positions.getPositions({
      marketsData: marketsInfoData,
      tokensData: tokensData,
    });

    const positionsArray = Object.values(positions.positionsData || {});
    if (!positionsArray.length) {
      throw new Error("No positions found to close");
    }

    // 3. Get the first position (index 0) to close
    const position = positionsArray[0];
    console.log("Closing position:", position);
    
    // Log position details for verification
    const marketInfo = marketsInfoData[position.marketAddress];
    if (!marketInfo) {
      throw new Error("Market info not found for position");
    }
    
    console.log(`Closing ${position.isLong ? "Long" : "Short"} position of ${formatUnits(position.sizeInUsd, 30)} USD on ${marketInfo.name}`);
    
    // 4. Get collateral token info
    const collateralToken = tokensData[position.collateralTokenAddress];
    if (!collateralToken) {
      throw new Error("Collateral token not found");
    }
    
    // 5. Determine acceptable price for closing
    const ALLOWED_SLIPPAGE_BPS = 50n; // 0.5%
    
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
    
    console.log(`Position index price: ${formatUnits(indexPrice, 30)}`);
    console.log(`Acceptable price: ${formatUnits(acceptablePrice, 30)}`);
    
    // 6. Calculate execution fee (0.01 ETH)
    const executionFee = parseEther("0.01");
    
    // 7. Properly construct the decrease order
    console.log("Creating decrease order to close position...");
    
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
      collateralToken.prices.minPrice
    );
    
    // Create the decrease order with correct parameters
    await sdk.orders.createDecreaseOrder({
      marketsInfoData,
      tokensData,
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
    
    console.log("Position close order created successfully");
    
  } catch (error) {
    console.error("Failed to close position:", error);
    throw new Error(`Position close failed: ${error.message || error}`);
  }
}

