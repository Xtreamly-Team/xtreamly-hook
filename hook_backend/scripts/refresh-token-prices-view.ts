import { Client } from 'pg';
import { config } from 'dotenv';

// Load environment variables
config();

async function refreshMaterializedView() {
  console.log('Starting refresh of token_prices_hourly materialized view...');
  
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

    // Check if the view exists
    const viewCheck = await client.query(`
      SELECT matviewname FROM pg_matviews WHERE matviewname = 'token_prices_hourly';
    `);
    
    if (viewCheck.rows.length === 0) {
      throw new Error('token_prices_hourly materialized view does not exist');
    }
    
    console.log('Found token_prices_hourly materialized view, refreshing...');
    
    // Get the current time before refresh
    const beforeTime = new Date();
    
    // Refresh the materialized view
    await client.query('REFRESH MATERIALIZED VIEW token_prices_hourly;');
    
    // Get the current time after refresh
    const afterTime = new Date();
    
    // Calculate the time taken
    const timeTaken = (afterTime.getTime() - beforeTime.getTime()) / 1000;
    
    console.log(`✅ Successfully refreshed token_prices_hourly (took ${timeTaken.toFixed(2)} seconds)`);
    
    // Get the count of records in the view
    const countResult = await client.query('SELECT COUNT(*) FROM token_prices_hourly;');
    const recordCount = countResult.rows[0].count;
    
    console.log(`The view now contains ${recordCount} records`);
    
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
refreshMaterializedView()
  .then(() => {
    console.log('Refresh script completed successfully');
    process.exit(0);
  })
  .catch((error) => {
    console.error('Refresh script failed:', error);
    process.exit(1);
  }); 