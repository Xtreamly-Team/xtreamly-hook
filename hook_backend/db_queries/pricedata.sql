-- Get all raw price data for a specific token (e.g., ETH)
SELECT 
  "tokenSymbol",
  "priceUsd",
  timestamp,
  "chainId",
  source
FROM token_prices
WHERE "tokenSymbol" = 'ETH'
ORDER BY timestamp DESC
LIMIT 100;

-- Get the latest price for a specific token
SELECT 
  "tokenSymbol",
  "priceUsd",
  timestamp,
  "chainId",
  source
FROM token_prices
WHERE "tokenSymbol" = 'ETH'
ORDER BY timestamp DESC
LIMIT 1;

-- Get price data for a specific token within a date range
SELECT 
  "tokenSymbol",
  "priceUsd",
  timestamp,
  "chainId",
  source
FROM token_prices
WHERE "tokenSymbol" = 'ETH'
  AND timestamp BETWEEN '2023-07-01' AND '2023-07-31'
ORDER BY timestamp ASC;

