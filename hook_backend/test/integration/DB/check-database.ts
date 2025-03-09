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
    `);
    console.log('\n📋 Existing tables:');
    tables.forEach((table: { table_name: string }) => {
      console.log(`- ${table.table_name}`);
    });

    // Check table structures
    for (const table of tables) {
      const columns = await dataSource.query(`
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = $1
      `, [table.table_name]);
      
      console.log(`\n📊 Structure of ${table.table_name}:`);
      columns.forEach((column: any) => {
        console.log(`- ${column.column_name}: ${column.data_type} (${column.is_nullable === 'YES' ? 'nullable' : 'not null'})`);
      });
    }

    // Check hypertables if an
    const hypertables = await dataSource.query(`
      SELECT "hypertable_name" as table_name
      FROM timescaledb_information.hypertables
    `).catch((err) => {
      console.log('Note: No hypertables found or TimescaleDB views not accessible');
      return [];
    });
    
    if (hypertables.length > 0) {
      console.log('\n🕒 Hypertables:');
      hypertables.forEach((ht: { table_name: string }) => {
        console.log(`- ${ht.table_name}`);
      });
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