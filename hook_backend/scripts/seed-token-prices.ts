import { Client } from 'pg';
import { config } from 'dotenv';

// Load environment variables
config();

async function seedTokenPrices() {
  console.log('Starting to seed test data into token_prices table...');
  
  // Create a direct connection to the database
  const client = new Client({
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT || '5432'),
    user: process.env.DB_USERNAME || 'postgres',
    password: process.env.DB_PASSWORD || 'postgres',
    database: process.env.DB_NAME || 'xtr_trade_db',
  });

  try {
    // Connect directly to the database
    await client.connect();
    console.log('Connected directly to database');

    // Check if the token_prices table exists
    const tableCheck = await client.query(`
      SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'token_prices'
      );
    `);
    
    if (!tableCheck.rows[0].exists) {
      throw new Error('token_prices table does not exist. Please run the migration first.');
    }
    
    // Get the count of existing records
    const countResult = await client.query('SELECT COUNT(*) FROM token_prices;');
    const existingCount = parseInt(countResult.rows[0].count);
    
    console.log(`Found ${existingCount} existing records in token_prices table`);
    
    if (existingCount > 0) {
      console.log('Table already has data, skipping seeding');
      return;
    }
    
    // Generate test data for the past 7 days
    const now = new Date();
    const startDate = new Date(now);
    startDate.setDate(now.getDate() - 7);
    
    // Tokens to seed
    const tokens = [
      { symbol: 'ETH', address: '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1', chainId: 42161 },
      { symbol: 'USDC', address: '0xaf88d065e77c8cC2239327C5EDb3A432268e5831', chainId: 42161 },
    ];
    
    console.log(`Generating test data for ${tokens.length} tokens from ${startDate.toISOString()} to ${now.toISOString()}`);
    
    // Base prices (approximate)
    const basePrices = {
      'ETH': 3500,
      'USDC': 1,
    };
    
    // Generate data points every 10 minutes
    const interval = 10 * 60 * 1000; // 10 minutes in milliseconds
    let insertCount = 0;
    
    for (let timestamp = startDate.getTime(); timestamp <= now.getTime(); timestamp += interval) {
      for (const token of tokens) {
        // Add some random price movement (±2% for ETH, ±0.1% for USDC)
        const volatility = token.symbol === 'ETH' ? 0.02 : 0.001;
        const randomFactor = 1 + (Math.random() * volatility * 2 - volatility);
        const price = basePrices[token.symbol] * randomFactor;
        
        // Add some sine wave pattern for more realistic price movement
        const dayFactor = Math.sin((timestamp / (24 * 60 * 60 * 1000)) * Math.PI * 2) * 0.01;
        const adjustedPrice = price * (1 + dayFactor);
        
        // Insert the data point
        await client.query(`
          INSERT INTO token_prices (
            "tokenSymbol", 
            "tokenAddress", 
            "priceUsd", 
            timestamp, 
            source, 
            "chainId", 
            metadata
          ) VALUES (
            $1, $2, $3, $4, 'coingecko', $5, 
            '{"source": "test-data", "generated": true}'
          );
        `, [
          token.symbol,
          token.address,
          adjustedPrice,
          new Date(timestamp),
          token.chainId,
        ]);
        
        insertCount++;
        
        // Log progress every 100 inserts
        if (insertCount % 100 === 0) {
          console.log(`Inserted ${insertCount} data points...`);
        }
      }
    }
    
    console.log(`✅ Successfully inserted ${insertCount} data points into token_prices table`);
    
    // Refresh the materialized view if it exists
    const viewCheck = await client.query(`
      SELECT matviewname FROM pg_matviews WHERE matviewname = 'token_prices_hourly';
    `);
    
    if (viewCheck.rows.length > 0) {
      console.log('Refreshing token_prices_hourly materialized view...');
      await client.query('REFRESH MATERIALIZED VIEW token_prices_hourly;');
      console.log('✅ Successfully refreshed materialized view');
      
      // Get the count of records in the view
      const viewCountResult = await client.query('SELECT COUNT(*) FROM token_prices_hourly;');
      const viewCount = viewCountResult.rows[0].count;
      
      console.log(`The materialized view now contains ${viewCount} records`);
    }
    
  } catch (error) {
    console.error('Operation failed:', error);
    throw error;
  } finally {
    // Close the direct connection
    await client.end();
    console.log('Closed direct database connection');
  }
}

// Run the function
seedTokenPrices()
  .then(() => {
    console.log('Seed script completed successfully');
    process.exit(0);
  })
  .catch((error) => {
    console.error('Seed script failed:', error);
    process.exit(1);
  }); 