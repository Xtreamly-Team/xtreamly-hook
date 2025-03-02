import { IsString, IsEmail, IsOptional, IsEthereumAddress } from 'class-validator';

export class RegisterUserDto {
  @IsEthereumAddress()
  walletAddress: string;

  @IsEmail()
  @IsOptional()
  email?: string;
}

export class UserResponseDto {
  id: string;
  walletAddress: string;
  email?: string;
  isActive: boolean;
  createdAt: Date;
}