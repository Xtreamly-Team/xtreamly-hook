-- Compare latest prices of multiple tokens
SELECT 
  "tokenSymbol",
  "priceUsd",
  timestamp
FROM token_prices
WHERE "tokenSymbol" IN ('ETH', 'USDC')
  AND timestamp >= NOW() - INTERVAL '1 hour'
ORDER BY "tokenSymbol", timestamp DESC;

-- Compare price changes over the last 24 hours
WITH latest_prices AS (
  SELECT 
    "tokenSymbol",
    FIRST("priceUsd", timestamp) AS current_price,
    LAST("priceUsd", timestamp) AS price_24h_ago
  FROM token_prices
  WHERE timestamp >= NOW() - INTERVAL '24 hours'
  GROUP BY "tokenSymbol"
)
SELECT 
  "tokenSymbol",
  current_price,
  price_24h_ago,
  current_price - price_24h_ago AS absolute_change,
  ((current_price - price_24h_ago) / price_24h_ago) * 100 AS percentage_change
FROM latest_prices
ORDER BY percentage_change DESC;