import { cleanEnv, str, port, json, bool } from 'envalid';
import { config } from 'dotenv';

// Load .env file first
config();

export const env = cleanEnv(process.env, {
  // App
  NODE_ENV: str({ 
    choices: ['development', 'test', 'production'],
    default: 'development',
    desc: 'Node environment'
  }),
  PORT: port({
    default: 3000,
    desc: 'Port to run the application on'
  }),
  FIREBASE_CREDENTIALS: json({
    desc: 'The firebase credentials.'
  }),
  API_KEY: str({
    desc: 'The allowed API KEY.'
  }),
  // Database
  DB_HOST: str({ 
    desc: 'PostgreSQL host',
    default: 'localhost'
  }),
  DB_PORT: port({ 
    default: 5432, 
    desc: 'PostgreSQL port' 
  }),
  DB_USERNAME: str({ 
    desc: 'PostgreSQL username',
    default: 'postgres'
  }),
  DB_PASSWORD: str({ 
    desc: 'PostgreSQL password'
  }),
  DB_NAME: str({ 
    desc: 'PostgreSQL database name'
  }),
}, {
  reporter: ({ errors }) => {
    if (Object.keys(errors).length > 0) {
      console.error('\n🚫 Environment validation failed:');
      for (const [key, error] of Object.entries(errors)) {
        console.error(`  • ${key}: ${error.message}`);
      }
      console.error('\nMake sure your .env file contains all required variables:\n');
      console.error('Required variables:');
      console.error('  DB_HOST - PostgreSQL host (e.g., localhost)');
      console.error('  DB_USERNAME - PostgreSQL username');
      console.error('  DB_PASSWORD - PostgreSQL password');
      console.error('  DB_NAME - PostgreSQL database name\n');
      console.error('Optional variables (with defaults):');
      console.error('  PORT - Application port (default: 3000)');
      console.error('  DB_PORT - PostgreSQL port (default: 5432)');
      console.error('  NODE_ENV - Environment (default: development)\n');
      process.exit(1);
    }
  }
}); 