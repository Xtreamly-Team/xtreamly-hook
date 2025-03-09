import { DataSource } from 'typeorm';
import { databaseConfig } from '../../../src/config/database.config';

async function checkColumns() {
  const dataSource = new DataSource({
    ...databaseConfig,
    logging: false,
  } as any);

  try {
    await dataSource.initialize();
    console.log('✅ Database connection successful');

    // Get column names from position_history table
    const columns = await dataSource.query(`
      SELECT column_name, data_type
      FROM information_schema.columns
      WHERE table_name = 'position_history'
      ORDER BY ordinal_position
    `);

    console.log('\n📊 Position History Table Columns:');
    columns.forEach((column: any) => {
      console.log(`- ${column.column_name} (${column.data_type})`);
    });

  } catch (error) {
    console.error('❌ Database query failed:', error);
  } finally {
    if (dataSource.isInitialized) {
      await dataSource.destroy();
    }
  }
}

checkColumns().catch(console.error); 