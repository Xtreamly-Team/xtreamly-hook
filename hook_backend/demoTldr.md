# Uniswap V4 Xtreamly Hook Demo: 

In this Demo we'll cover our submission E2E. 

## 1. Initial Setup
Our Uniswap v4 hook is deployed at: `0xEFfFf07530D1CA71C0d810Ec3FF6636458DD8540`
The hook automatically detects liquidity position changes and triggers our hedging service (Eventlistener WIP events emitted on hook fx afterAdd/RemoveLiquidity)

## 2. Register User
**Input:** User heads to our Front End, connects their wallet with our Webapp which triggers it:

- Call the endpoint /user/register with your wallet address
- System creates a user record and returns API credentials
- These credentials will be used for all subsequent API calls

**Output:** Backend generates new Wallet for User and ready to open GMX positions for the User

## 3. Uniswap Position Creation (Hook Trigger)
**Input:** User heads to uniV4, creates a pool with our hook. 

Create a new Uniswap v4 liquidity position with these parameters:
```json
HookAddress: 0xEFfFf07530D1CA71C0d810Ec3FF6636458DD8540
Token Pair: ETH/USDC
Amount A: 1 USD worth of ETH
Amount B: 1 USDC
Lower Tick: -887270.00000000  // or your custom tick
Upper Tick: 887270.00000000   // or your custom tick
```

The hook automatically detects this position creation (via our Eventlistener WIP)
`Position ID generated: 0x461a8302f046976f836add7c91eea3bc578816ba4d85ea537bb487070495005f`

**Output:** New ETH/USDC Pool is created, afterAddLiquidity emits event about position and sends it to our `hook_backend`

## 4. Database Entry Creation
**Input:** `hook_backend` receives info about our new LP Pool and populates our DB. 

System automatically creates a record in the positions table with:
- The token pair information (ETH/USDC)
- Position size information (amounts, ticks)
- Uniswap Position ID
- Initial status: "pending"

**Output:** Our DB has the latest information about the LP position we want to track/hedge...

## 5. AI Recommendation for Hedging
**Input:** Cron Job (5min interval) gets triggered sending active (positions we are tracking) to our `ai_backend` and then waiting for hedge params. 

- System calls our AI service with the position parameters
- AI analyzes the position and recommends:
Hedge x% of the ETH exposure (for the demo we open a hedge with $2 USD collateral, check whitepaper for correct hedge details)
- Use 4x leverage on GMX
- Direction: Short position (to hedge against ETH price decrease)

**Output:** `hook_backend` receives the params to open a gmx-v2 short/long on ETH/USDC using USDC as collateral with 4x leverage. 

## 6. GMX Short Position Creation
**Input:** Our `hook_backend` prepares our txn, approves the tokens in question and opens an order. 

Service calls GMX API to create a short position with:
- Market: ETH-USDC
- Short Size: 2 USD (with 4x leverage = 4 USD exposure)
- Collateral: USDC

Transaction is submitted and confirmed on-chain

Position details are logged in our system

Database entry is updated with GMX position ID and status changed to "active"

**Output:** We have now opened a short position on GMX and are tracking it in our DB. 

## 7. Monitoring Period

[Time passes - in a real scenario this could be a few hours or days]

System continuously monitors both Uniswap and GMX positions

Price fluctuations impact both positions in opposite ways, providing hedging effect

## 8. Close GMX Short Position

**Input:** Our `ai_backend` signals us to close our hedge. 

- System calls GMX API to close the short position
- Transaction is submitted and confirmed on-chain
- Database entry is updated with closed status and final P&L
- Position history is recorded for future analysis

**Output:** Our hedge is now closed and funding is back in the user's wallet. 

## 9. Final Result

- Complete hedge lifecycle demonstrated from creation to closing
- System maintains detailed (more to be added WIP) records of all transactions and position states 
- User can view the full history and performance metrics through the dashboard (xtreamlyUI WIP)
- Due to time constraints we didn't manage to build everything but we intend to continue our hook and deploy the missing components and parts.