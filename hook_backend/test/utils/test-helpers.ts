/**
 * Test Helper Utilities
 * 
 * This file contains helper functions for API testing.
 */

import axios from 'axios';
import * as dotenv from 'dotenv';
import * as path from 'path';

// Load test environment variables
dotenv.config({ path: path.resolve(__dirname, '../../.env.test') });

// API Configuration
export const API_URL = process.env.API_URL || 'http://localhost:3000';
export const TEST_WALLET = process.env.TEST_WALLET || '0x742d35Cc6634C0532925a3b844Bc454e4438f44e';

// HTTP Client
export const apiClient = axios.create({
  baseURL: API_URL,
  timeout: 5000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Helper function to log test results
export function logTestResult(testName: string, passed: boolean, data?: any) {
  if (passed) {
    console.log(`✅ ${testName} - PASSED`);
    if (data) {
      console.log(`   Result: ${typeof data === 'object' ? JSON.stringify(data, null, 2) : data}`);
    }
  } else {
    console.log(`❌ ${testName} - FAILED`);
    if (data) {
      console.log(`   Error: ${data}`);
    }
  }
  console.log(''); // Empty line for better readability
}

// Helper function to generate random tokens for testing
export function getRandomTokenPair() {
  const pairs = [
    { tokenA: 'ETH', tokenB: 'USDC' },
    { tokenA: 'WBTC', tokenB: 'USDC' },
    { tokenA: 'ETH', tokenB: 'WBTC' },
  ];
  return pairs[Math.floor(Math.random() * pairs.length)];
}

// Helper function to wait for a specified time
export function wait(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}