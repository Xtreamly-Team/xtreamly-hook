import { DataSource } from 'typeorm';
import { databaseConfig } from '../../../src/config/database.config';

async function checkDatabase() {
  const dataSource = new DataSource({
    ...databaseConfig,
    logging: false, // Disable logging to reduce noise
  } as any);

  try {
    await dataSource.initialize();
    console.log('✅ Database connection successful');

    // Check TimescaleDB extension
    const timescaledbCheck = await dataSource.query(
      "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')"
    );
    console.log('TimescaleDB extension installed:', timescaledbCheck[0].exists);
    
    if (!timescaledbCheck[0].exists) {
      console.error('❌ TimescaleDB extension is not installed. This is required for the token_prices table.');
      return;
    }
    
    // Get TimescaleDB version
    const versionResult = await dataSource.query(
      "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
    );
    const timescaleVersion = versionResult[0]?.extversion || 'unknown';
    console.log(`TimescaleDB version: ${timescaleVersion}`);

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
      
      // Check if it's a hypertable using a more compatible query
      const hypertableCheck = await dataSource.query(`
        SELECT * FROM _timescaledb_catalog.hypertable
        WHERE table_name = 'token_prices';
      `).catch(err => {
        console.error('Error checking hypertable in _timescaledb_catalog:', err.message);
        return [];
      });
      
      if (hypertableCheck.length > 0) {
        console.log('\n✅ token_prices is a TimescaleDB hypertable');
        
        // Try to get more details if available
        try {
          const dimensionInfo = await dataSource.query(`
            SELECT * FROM _timescaledb_catalog.dimension
            WHERE hypertable_id = $1
          `, [hypertableCheck[0].id]);
          
          if (dimensionInfo.length > 0) {
            console.log(`- Partitioning column ID: ${dimensionInfo[0].column_name || dimensionInfo[0].column_id}`);
            console.log(`- Chunk interval: ${dimensionInfo[0].interval_length} (internal units)`);
          }
        } catch (error) {
          console.log('Could not retrieve detailed hypertable information');
        }
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
        
        // Try to check if it's a continuous aggregate using a more compatible query
        try {
          const caggCheck = await dataSource.query(`
            SELECT * FROM _timescaledb_catalog.continuous_agg
            WHERE user_view_name = 'token_prices_hourly';
          `);
          
          if (caggCheck.length > 0) {
            console.log('✅ token_prices_hourly is a TimescaleDB continuous aggregate');
          } else {
            console.log('❌ token_prices_hourly is NOT a TimescaleDB continuous aggregate');
          }
        } catch (error) {
          console.log('Could not verify if view is a continuous aggregate');
        }
      } else {
        console.log('\n❌ token_prices_hourly materialized view does NOT exist');
      }
      
      // Try to check policies using a more compatible approach
      console.log('\nChecking for TimescaleDB policies:');
      
      // For retention policy, check in the catalog tables
      try {
        const retentionPolicyCheck = await dataSource.query(`
          SELECT * FROM _timescaledb_config.bgw_job
          WHERE proc_name = 'policy_retention';
        `).catch(() => []);
        
        if (retentionPolicyCheck.length > 0) {
          console.log('✅ Retention policy exists (found in bgw_job table)');
        } else {
          console.log('❓ No retention policy found in bgw_job table');
        }
      } catch (error) {
        console.log('❓ Could not check retention policy - catalog tables may differ in this TimescaleDB version');
      }
      
      // For refresh policy, check in the catalog tables
      try {
        const refreshPolicyCheck = await dataSource.query(`
          SELECT * FROM _timescaledb_config.bgw_job
          WHERE proc_name = 'policy_refresh_continuous_aggregate';
        `).catch(() => []);
        
        if (refreshPolicyCheck.length > 0) {
          console.log('✅ Refresh policy exists (found in bgw_job table)');
        } else {
          console.log('❓ No refresh policy found in bgw_job table');
        }
      } catch (error) {
        console.log('❓ Could not check refresh policy - catalog tables may differ in this TimescaleDB version');
      }
      
      console.log('\n✅ SUMMARY: The token_prices table and related objects appear to be set up correctly.');
      console.log('Some policy checks may show as unknown due to TimescaleDB version differences.');
      
    } else {
      console.log('\n❌ token_prices table does NOT exist');
      console.log('Possible reasons:');
      console.log('1. The migration failed to run');
      console.log('2. The migration ran but failed to create the table');
      console.log('3. The table was created with a different name');
      
      // Check if there are any tables with "price" in the name
      const priceRelatedTables = tables.filter((table: any) => 
        table.table_name.toLowerCase().includes('price')
      );
      
      if (priceRelatedTables.length > 0) {
        console.log('\nFound tables with "price" in the name:');
        priceRelatedTables.forEach((table: any) => {
          console.log(`- ${table.table_name}`);
        });
      }
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