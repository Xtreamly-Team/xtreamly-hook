import { IsString, IsEmail, IsOptional } from 'class-validator';

export class CreateUserDto {
  @IsString()
  walletAddress: string;

  @IsEmail()
  @IsOptional()
  email?: string;
}