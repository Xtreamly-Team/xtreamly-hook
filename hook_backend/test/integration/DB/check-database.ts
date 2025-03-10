import { DataSource } from 'typeorm';
import { databaseConfig } from '../../../src/config/database.config';

async function checkDatabase() {
  const dataSource = new DataSource({
    ...databaseConfig,
    logging: true, // Enable logging to see queries
  } as any);

  try {
    await dataSource.initialize();
    console.log('✅ Database connection successful');

    // Check TimescaleDB extension
    const timescaledbCheck = await dataSource.query(
      "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')"
    );
    console.log('TimescaleDB extension installed:', timescaledbCheck[0].exists);

    // List all tables
    const tables = await dataSource.query(`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public'
      ORDER BY table_name;
    `);
    console.log('\n📋 Tables in database:');
    tables.forEach((table: any) => {
      console.log(`- ${table.table_name}`);
    });

    // Check if token_prices table exists
    const tokenPricesExists = tables.some((table: any) => table.table_name === 'token_prices');
    
    if (tokenPricesExists) {
      console.log('\n✅ token_prices table exists');
      
      // Check columns in token_prices
      const columns = await dataSource.query(`
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'token_prices'
        ORDER BY ordinal_position;
      `);
      
      console.log('\n📋 Columns in token_prices:');
      columns.forEach((column: any) => {
        console.log(`- ${column.column_name} (${column.data_type}, ${column.is_nullable === 'YES' ? 'nullable' : 'not nullable'})`);
      });
      
      // Check if it's a hypertable
      const hypertableCheck = await dataSource.query(`
        SELECT * FROM timescaledb_information.hypertables
        WHERE hypertable_name = 'token_prices';
      `);
      
      if (hypertableCheck.length > 0) {
        console.log('\n✅ token_prices is a TimescaleDB hypertable');
        console.log(`- Partitioning column: ${hypertableCheck[0].time_column_name}`);
        console.log(`- Chunk interval: ${hypertableCheck[0].chunk_time_interval}`);
      } else {
        console.log('\n❌ token_prices is NOT a TimescaleDB hypertable');
      }
      
      // Check materialized view
      const materializedViews = await dataSource.query(`
        SELECT matviewname 
        FROM pg_matviews
        WHERE schemaname = 'public';
      `);
      
      const hourlyViewExists = materializedViews.some((view: any) => view.matviewname === 'token_prices_hourly');
      
      if (hourlyViewExists) {
        console.log('\n✅ token_prices_hourly materialized view exists');
        
        // Check if it's a continuous aggregate
        const continuousAggregateCheck = await dataSource.query(`
          SELECT * FROM timescaledb_information.continuous_aggregates
          WHERE view_name = 'token_prices_hourly';
        `).catch(() => []);
        
        if (continuousAggregateCheck.length > 0) {
          console.log('✅ token_prices_hourly is a TimescaleDB continuous aggregate');
        } else {
          console.log('❌ token_prices_hourly is NOT a TimescaleDB continuous aggregate');
        }
        
        // Check columns in the materialized view
        const viewColumns = await dataSource.query(`
          SELECT column_name, data_type
          FROM information_schema.columns
          WHERE table_name = 'token_prices_hourly'
          ORDER BY ordinal_position;
        `);
        
        console.log('\n📋 Columns in token_prices_hourly:');
        viewColumns.forEach((column: any) => {
          console.log(`- ${column.column_name} (${column.data_type})`);
        });
      } else {
        console.log('\n❌ token_prices_hourly materialized view does NOT exist');
      }
      
      // Check policies
      try {
        const retentionPolicies = await dataSource.query(`
          SELECT * FROM timescaledb_information.policies
          WHERE hypertable_name = 'token_prices' AND policy_type = 'retention';
        `);
        
        if (retentionPolicies.length > 0) {
          console.log('\n✅ Retention policy exists for token_prices');
          console.log(`- Schedule interval: ${retentionPolicies[0].schedule_interval}`);
          console.log(`- Retention period: ${retentionPolicies[0].config.drop_after}`);
        } else {
          console.log('\n❓ No retention policy found for token_prices');
        }
        
        const refreshPolicies = await dataSource.query(`
          SELECT * FROM timescaledb_information.policies
          WHERE hypertable_name = 'token_prices_hourly' AND policy_type = 'refresh';
        `).catch(() => []);
        
        if (refreshPolicies.length > 0) {
          console.log('\n✅ Refresh policy exists for token_prices_hourly');
          console.log(`- Schedule interval: ${refreshPolicies[0].schedule_interval}`);
        } else {
          console.log('\n❓ No refresh policy found for token_prices_hourly');
        }
      } catch (error) {
        console.log('\n❓ Could not check policies - may be using an older TimescaleDB version');
      }
      
    } else {
      console.log('\n❌ token_prices table does NOT exist');
    }

  } catch (error) {
    console.error('❌ Database check failed:', error);
  } finally {
    if (dataSource.isInitialized) {
      await dataSource.destroy();
    }
  }
}

checkDatabase().catch(console.error); 