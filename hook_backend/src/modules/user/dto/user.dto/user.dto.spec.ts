import { validate } from 'class-validator';
import { RegisterUserDto, UserResponseDto } from './user.dto';


  describe('RegisterUserDto', () => {
    it('should validate a valid ethereum address', async () => {
      const dto = new RegisterUserDto();
      dto.walletAddress = '0x742d35Cc6634C0532925a3b844Bc454e4438f44e';
      
      const errors = await validate(dto);
      expect(errors.length).toBe(0);
    });

    it('should fail with invalid ethereum address', async () => {
      const dto = new RegisterUserDto();
      dto.walletAddress = 'invalid-address';
      
      const errors = await validate(dto);
      expect(errors.length).toBeGreaterThan(0);
      expect(errors[0].constraints).toHaveProperty('isEthereumAddress');
    });

    it('should validate with optional email', async () => {
      const dto = new RegisterUserDto();
      dto.walletAddress = '0x742d35Cc6634C0532925a3b844Bc454e4438f44e';
      dto.email = 'test@example.com';
      
      const errors = await validate(dto);
      expect(errors.length).toBe(0);
    });

    it('should fail with invalid email format', async () => {
      const dto = new RegisterUserDto();
      dto.walletAddress = '0x742d35Cc6634C0532925a3b844Bc454e4438f44e';
      dto.email = 'invalid-email';
      
      const errors = await validate(dto);
      expect(errors.length).toBeGreaterThan(0);
      expect(errors[0].constraints).toHaveProperty('isEmail');
    });
  });

  describe('UserResponseDto', () => {
    it('should create a valid response DTO', () => {
      const response = new UserResponseDto();
      response.id = 'user-123';
      response.walletAddress = '0x742d35Cc6634C0532925a3b844Bc454e4438f44e';
      response.email = 'test@example.com';
      response.isActive = true;
      response.createdAt = new Date();

      expect(response).toHaveProperty('id');
      expect(response).toHaveProperty('walletAddress');
      expect(response).toHaveProperty('email');
      expect(response).toHaveProperty('isActive');
      expect(response).toHaveProperty('createdAt');
    });

    it('should allow response without optional email', () => {
      const response = new UserResponseDto();
      response.id = 'user-123';
      response.walletAddress = '0x742d35Cc6634C0532925a3b844Bc454e4438f44e';
      response.isActive = true;
      response.createdAt = new Date();

      expect(response.email).toBeUndefined();
      expect(response).toHaveProperty('id');
      expect(response).toHaveProperty('walletAddress');
      expect(response).toHaveProperty('isActive');
      expect(response).toHaveProperty('createdAt');
    });
  });

