import { DataSource } from 'typeorm';
import { databaseConfig } from '../../../src/config/database.config';
import { User } from '../../../src/modules/user/entities/user.entity/user.entity';
import { Position, PositionStatus } from '../../../src/modules/position/entities/position.entity/position.entity';
import { PositionHistory } from '../../../src/modules/position/entities/position-history.entity/position-history.entity';

async function queryData() {
  const dataSource = new DataSource({
    ...databaseConfig,
    logging: false, // Disable logging for cleaner output
  } as any);

  try {
    await dataSource.initialize();
    console.log('✅ Database connection successful');

    // Query users
    const users = await dataSource.getRepository(User).find();
    console.log(`\n📊 Users (${users.length}):`);
    users.forEach(user => {
      console.log(`- ${user.id}: ${user.walletAddress} (${user.email || 'No email'})`);
    });

    // Query positions
    const positions = await dataSource.getRepository(Position).find({
      relations: ['user'],
    });
    console.log(`\n📊 Positions (${positions.length}):`);
    positions.forEach(position => {
      console.log(`- ${position.id}: ${position.tokenA}/${position.tokenB} (${position.status})`);
      console.log(`  Amount: ${position.amountA} ${position.tokenA} / ${position.amountB} ${position.tokenB}`);
      console.log(`  User: ${position.user.walletAddress}`);
    });

    // Query position history with time-series aggregation
    console.log(`\n📊 Position History Aggregation:`);
    
    // First, let's check if the position_history table has any data
    const totalHistoryCount = await dataSource.getRepository(PositionHistory).count();
    console.log(`Total position history entries: ${totalHistoryCount}`);
    
    if (totalHistoryCount === 0) {
      console.log('No position history data found. Please run db:seed first.');
      return;
    }
    
    // For each position, get daily average values
    for (const position of positions) {
      if (position.status !== PositionStatus.ACTIVE) continue;
      
      // First, let's check if this position has any history entries
      const historyCount = await dataSource.getRepository(PositionHistory).count({
        where: { positionId: position.id }
      });
      
      console.log(`\n- Position ${position.id} (${position.tokenA}/${position.tokenB}): ${historyCount} entries`);
      
      if (historyCount === 0) {
        console.log('  No history data for this position');
        continue;
      }
      
      // Get a sample row to check column names
      const sampleRow = await dataSource.getRepository(PositionHistory).findOne({
        where: { positionId: position.id }
      });
      
      console.log('  Sample row:', sampleRow);
      
      // TimescaleDB time bucket aggregation
      try {
        const timeSeriesData = await dataSource.query(`
          SELECT 
            time_bucket('1 day', "timestamp") AS day,
            AVG("tokenAValue") AS avg_token_a_value,
            AVG("tokenBValue") AS avg_token_b_value,
            AVG("netValue") AS avg_net_value,
            COUNT(*) AS count
          FROM position_history
          WHERE "positionId" = $1
          GROUP BY day
          ORDER BY day DESC
          LIMIT 7
        `, [position.id]);
        
        console.log(`  Last 7 days of data:`);
        if (timeSeriesData.length === 0) {
          console.log('  No aggregated data available');
        } else {
          timeSeriesData.forEach((row: any) => {
            console.log(`  - ${row.day.toISOString().split('T')[0]}: Avg Net Value: ${parseFloat(row.avg_net_value).toFixed(2)}`);
          });
        }
      } catch (error) {
        console.error('  Error in time series query:', error.message);
        
        // Fallback to a simpler query without time_bucket
        console.log('  Trying a simpler query without time_bucket...');
        const simpleData = await dataSource.query(`
          SELECT 
            DATE("timestamp") AS day,
            AVG("tokenAValue") AS avg_token_a_value,
            AVG("tokenBValue") AS avg_token_b_value,
            AVG("netValue") AS avg_net_value,
            COUNT(*) AS count
          FROM position_history
          WHERE "positionId" = $1
          GROUP BY day
          ORDER BY day DESC
          LIMIT 7
        `, [position.id]);
        
        if (simpleData.length === 0) {
          console.log('  No data available with simpler query');
        } else {
          simpleData.forEach((row: any) => {
            console.log(`  - ${row.day.toISOString().split('T')[0]}: Avg Net Value: ${parseFloat(row.avg_net_value).toFixed(2)}`);
          });
        }
      }
    }

  } catch (error) {
    console.error('❌ Database query failed:', error);
  } finally {
    if (dataSource.isInitialized) {
      await dataSource.destroy();
    }
  }
}

queryData().catch(console.error); 