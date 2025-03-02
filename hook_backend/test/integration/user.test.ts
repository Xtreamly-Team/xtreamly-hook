/**
 * User Module Integration Tests
 * 
 * This script tests the user registration and retrieval functionality.
 */

import { apiClient, TEST_WALLET, logTestResult } from '../utils/test-helpers';

async function testUserModule() {
  console.log('🧪 TESTING USER MODULE 🧪');
  console.log('========================\n');
  
  try {
    // Test 1: Register a new user
    console.log('📝 Test: User Registration');
    const registerData = {
      walletAddress: TEST_WALLET,
      email: 'test@example.com',
    };
    
    const registerResponse = await apiClient.post('/users/register', registerData);
    
    logTestResult(
      'User Registration',
      registerResponse.status === 201 || registerResponse.status === 200,
      registerResponse.data
    );
    
    // Store the user ID for future tests
    const userId = registerResponse.data.id;
    
    // Test 2: Get user by wallet address
    console.log('🔍 Test: Get User by Wallet Address');
    const getUserResponse = await apiClient.get(`/users/${TEST_WALLET}`);
    
    logTestResult(
      'Get User by Wallet Address',
      getUserResponse.status === 200 && getUserResponse.data.walletAddress === TEST_WALLET,
      getUserResponse.data
    );
    
    return {
      success: true,
      userId,
    };
  } catch (error) {
    console.error('❌ Error in User Module Tests:', error.response?.data || error.message);
    return {
      success: false,
      error,
    };
  }
}

// If this file is run directly
if (require.main === module) {
  testUserModule()
    .then(() => console.log('✅ User module tests completed'))
    .catch(err => console.error('❌ User module tests failed:', err));
}

export { testUserModule };