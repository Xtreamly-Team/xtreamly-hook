-- Get hourly aggregated data for a specific token
SELECT 
  "tokenSymbol",
  bucket,
  avg_price,
  min_price,
  max_price,
  open_price,
  close_price,
  sample_count
FROM token_prices_hourly
WHERE "tokenSymbol" = 'ETH'
ORDER BY bucket DESC
LIMIT 24;

-- Get hourly data for a specific token within a date range
SELECT 
  "tokenSymbol",
  bucket,
  avg_price,
  min_price,
  max_price,
  open_price,
  close_price,
  sample_count
FROM token_prices_hourly
WHERE "tokenSymbol" = 'ETH'
  AND bucket BETWEEN '2023-07-01' AND '2023-07-31'
ORDER BY bucket ASC;

-- Get daily average prices for a specific token (using the hourly data)
SELECT 
  "tokenSymbol",
  date_trunc('day', bucket) AS day,
  AVG(avg_price) AS daily_avg_price,
  MIN(min_price) AS daily_min_price,
  MAX(max_price) AS daily_max_price,
  FIRST(open_price, bucket) AS daily_open_price,
  LAST(close_price, bucket) AS daily_close_price
FROM token_prices_hourly
WHERE "tokenSymbol" = 'ETH'
GROUP BY "tokenSymbol", date_trunc('day', bucket)
ORDER BY day DESC
LIMIT 30;