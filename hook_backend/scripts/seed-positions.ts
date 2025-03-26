import { Client } from 'pg';
import { config } from 'dotenv';

// Load environment variables
config();

async function seedPositions() {
  console.log('Starting to seed test data into positions table...');
  
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
    
    // Get the count of existing records
    const countResult = await client.query('SELECT COUNT(*) FROM positions;');
    const existingCount = parseInt(countResult.rows[0].count);
    
    console.log(`Found ${existingCount} existing records in positions table`);
    
    if (existingCount > 0) {
      console.log('Table already has data, skipping seeding');
      return;
    }
    
    // Tokens to seed
    const tokens = [
      { symbol: 'ETH', address: '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1', chainId: 42161 },
      { symbol: 'USDC', address: '0xaf88d065e77c8cC2239327C5EDb3A432268e5831', chainId: 42161 },
    ];
    
    // Base prices (approximate)
    const basePrices = {
      'ETH': 3500,
      'USDC': 1,
    };
    
    // Generate data points every 10 minutes
    const interval = 10 * 60 * 1000; // 10 minutes in milliseconds
    const total = 20;
    const now = new Date();
    const startTime = new Date(now.getTime() - total * interval);

    for (let i = 1; i <= total; i++) {
      const time = new Date(startTime.getTime() + i * 10 * 60 * 1000); // Add 10 minutes per loop

      const isUniswap = i === 1 || i === total

      const user = "e2e08280-a3f6-485f-b5a2-3972aa7cf38f"
      const amountA = basePrices[tokens[0].symbol] * Math.random();
      const amountB = basePrices[tokens[1].symbol] * Math.random();
      const lowerTick = basePrices[tokens[0].symbol] * Math.random();
      const upperTick = basePrices[tokens[0].symbol] * Math.random();
      const hedgeAmount = basePrices[tokens[0].symbol] * Math.random();
      const uniswapPositionId = isUniswap ? `uniswapPositionId_${i}` : null;
      const gmxPositionId = !isUniswap ? `gmxPositionId_${i}` : null;
      let status = i % 2 !== 0 ? 'active' : 'closed'
      if (!isUniswap) {
        status = i % 2 === 0 ? 'active' : 'closed'
      }

      const createdAt = new Date(time)

      const values = [
        user,
        tokens[0].address,
        tokens[1].address,
        amountA,
        amountB,
        lowerTick,
        upperTick,
        hedgeAmount,
        status,
        uniswapPositionId,
        gmxPositionId,
        createdAt,
        createdAt,
      ]
      // console.log(values);

      // Insert the data point
      await client.query(`
        INSERT INTO positions (
          "userId",
          "tokenA",
          "tokenB",
          "amountA",
          "amountB",
          "lowerTick",
          "upperTick",
          "hedgeAmount",
          "status",
          "uniswapPositionId",
          "gmxPositionId",
          "createdAt",
          "updatedAt",
          "metadata"
        ) VALUES (
          $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
          '{"source": "test-data", "generated": true}'
        );
      `, values);
    }
    

    console.log(`✅ Successfully inserted ${total} data points into positions table`);
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
seedPositions()
  .then(() => {
    console.log('Seed script completed successfully');
    process.exit(0);
  })
  .catch((error) => {
    console.error('Seed script failed:', error);
    process.exit(1);
  }); 