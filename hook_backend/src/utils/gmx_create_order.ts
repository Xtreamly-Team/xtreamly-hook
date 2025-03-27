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


 



  // console.log(collateralTokenName)

  
  // const marketInfo = marketsInfoData[positionsArray[0].marketAddress];


  console.log("Creating stop loss order. Creating...");
  
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
    // await sdk.orders.createIncreaseOrder({
    //   marketsInfoData: marketsInfoData!,
    //   tokensData,
    //   isLimit: false,
    //   isLong: false,
    //   marketAddress: marketInfo.marketTokenAddress,
    //   allowedSlippage: 50,
    //   collateralToken: collateralTokenData,
    //   collateralTokenAddress: collateralToken,
    //   receiveTokenAddress: collateralToken,
    //   fromToken: tokensData["0x912CE59144191C1204E64559FE8253a0e49E6548"],
    //   marketInfo,
    //   indexToken: marketInfo.indexToken,
    //   increaseAmounts: {
    //     initialCollateralAmount: 3000000n,
    //     initialCollateralUsd: 2999578868393486100000000000000n,
    //     collateralDeltaAmount: 2997003n,
    //     collateralDeltaUsd: 2996582289103961007386100000000n,
    //     indexTokenAmount: 1919549334876037n,
    //     sizeDeltaUsd: 1679226208729489045987200000000n,
    //     sizeDeltaInTokens: 1919536061202302n,
    //     estimatedLeverage: 20000n,
    //     indexPrice: 3122169600000000000000000000000000n,
    //     initialCollateralPrice: 999859622797828700000000000000n,
    //     collateralPrice: 999859622797828700000000000000n,
    //     triggerPrice: 0n,
    //     acceptablePrice: 3122191190655414690893787784152819n,
    //     acceptablePriceDeltaBps: 0n,
    //     positionFeeUsd: 2996579289525092613900000000n,
    //     swapPathStats: undefined,
    //     uiFeeUsd: 0n,
    //     swapUiFeeUsd: 0n,
    //     feeDiscountUsd: 0n,
    //     borrowingFeeUsd: 0n,
    //     fundingFeeUsd: 0n,
    //     externalSwapQuote: undefined,
    //     positionPriceImpactDeltaUsd: 41444328240807630917223064n,
    //   },
    // });
   const { marketsInfoData } = await sdk.markets.getMarketsInfo();

   let marketInfo = marketsInfoData["0x47c031236e19d024b42f8AE6780E44A573170703"];
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

    sdk.orders.createIncreaseOrder({
      // Market data from GMX SDK
      marketsInfoData: marketsInfoData!,
      tokensData,
      
      // Order type flags
      isLimit: false,  // Market order = false, Limit order = true
      isLong: false,   // Long = true, Short = false
      
      // Market address from marketsInfoData
      marketAddress: marketInfo.marketTokenAddress,  // e.g., ETH/USDC market address
      
      // Slippage tolerance in basis points (1 = 0.01%)
      allowedSlippage: 50,  // 0.5% slippage allowed
      
      // Collateral token information
      collateralToken,  // Token object from tokensData
      collateralTokenAddress: collateralToken.address,  // Address of collateral token (e.g., USDC)
      receiveTokenAddress: collateralToken.address,     // Same as collateral for simple positions
      
      // Token you're starting with (if different from collateral, will be swapped)
      fromToken: tokensData["0x912CE59144191C1204E64559FE8253a0e49E6548"],
      
      // Market and index token info
      marketInfo,  // Market object from marketsInfoData
      indexToken: marketInfo.indexToken,  // Token being traded (e.g., ETH for ETH/USDC)
    
      increaseAmounts: {
        // Initial collateral amount in token decimals (e.g., 6 for USDC)
        initialCollateralAmount: 3000000n,  // 3 USDC = 3_000_000
        
        // Initial collateral converted to USD (30 decimals)
        // Calculate: initialCollateralAmount * collateralPrice * 10^(30-tokenDecimals)
        initialCollateralUsd: 2999578868393486100000000000000n,
        
        // Actual collateral amount after fees
        collateralDeltaAmount: 2997003n,
        
        // Actual collateral in USD after fees (30 decimals)
        collateralDeltaUsd: 2996582289103961007386100000000n,
        
        // Amount of index token being traded (token decimals)
        // Calculate: sizeDeltaUsd / indexPrice
        indexTokenAmount: 1919549334876037n,
        
        // Position size in USD (30 decimals)
        // Calculate: collateralAmount * leverage * 10^30
        sizeDeltaUsd: 5993158579050185227800000000000n,
        
        // Position size in tokens
        sizeDeltaInTokens: 1919536061202302n,
        
        // Leverage * 10000 (2x = 20000)
        estimatedLeverage: 20000n,
        
        // Current index price (30 decimals)
        // Get from: marketInfo.indexToken.prices
        indexPrice: 3122169600000000000000000000000000n,
        
        // Collateral token prices (30 decimals)
        initialCollateralPrice: 999859622797828700000000000000n,
        collateralPrice: 999859622797828700000000000000n,
        
        // For market orders = 0
        triggerPrice: 0n,
        
        // Maximum acceptable price for the trade
        // For longs: indexPrice * (1 + allowedSlippage)
        // For shorts: indexPrice * (1 - allowedSlippage)
        acceptablePrice: 3122191190655414690893787784152819n,
        
        // Additional required fields with default values
        acceptablePriceDeltaBps: 0n,
        positionFeeUsd: 2996579289525092613900000000n,
        swapPathStats: undefined,
        uiFeeUsd: 0n,
        swapUiFeeUsd: 0n,
        feeDiscountUsd: 0n,
        borrowingFeeUsd: 0n,
        fundingFeeUsd: 0n,
        positionPriceImpactDeltaUsd: 41444328240807630917223064n,
        externalSwapQuote: undefined  // Required field, can be undefined
      },
    });

    console.log("Increase order created successfully");
  } catch (error) {
    console.error("Failed to create increase order:", error);
    throw new Error(`Order creation failed: ${error.message || error}`);
  }

  // await sdk.orders.createDecreaseOrder({
  //   allowedSlippage: 50,
  //   collateralToken: collateralToken,
  //   marketInfo,
  //   marketsInfoData,
  //   decreaseAmounts: {
  //     isFullClose: true,
  //     sizeDeltaUsd: position.sizeInUsd,
  //     sizeDeltaInTokens: position.sizeInTokens,
  //     collateralDeltaUsd: estimatedCollateralUsd,
  //     collateralDeltaAmount: position.collateralAmount,
  //     indexPrice: 0n,
  //     collateralPrice: 0n,
  //     triggerPrice: stopLossPrice,
  //     acceptablePrice: stopLossPrice,
  //     acceptablePriceDeltaBps: 0n,
  //     recommendedAcceptablePriceDeltaBps: 0n,
  //     estimatedPnl: 0n,
  //     estimatedPnlPercentage: 0n,
  //     realizedPnl: 0n,
  //     realizedPnlPercentage: 0n,
  //     positionFeeUsd: 0n,
  //     uiFeeUsd: 0n,
  //     swapUiFeeUsd: 0n,
  //     feeDiscountUsd: 0n,
  //     borrowingFeeUsd: 0n,
  //     fundingFeeUsd: 0n,
  //     swapProfitFeeUsd: 0n,
  //     positionPriceImpactDeltaUsd: 0n,
  //     priceImpactDiffUsd: 0n,
  //     payedRemainingCollateralAmount: 0n,
  //     payedOutputUsd: 0n,
  //     payedRemainingCollateralUsd: 0n,
  //     receiveTokenAmount: 0n,
  //     receiveUsd: 0n,
  //     decreaseSwapType: DecreasePositionSwapType.NoSwap,
  //     triggerOrderType: OrderType.StopLossDecrease,
  //     // triggerThresholdType
  //   },
  //   isLong: position.isLong,
  //   tokensData,
  // });
}

