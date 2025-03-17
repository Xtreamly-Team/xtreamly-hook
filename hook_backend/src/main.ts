import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { env } from './config/env.config';
import { ValidationPipe } from '@nestjs/common';
import {AuthGuard} from "@modules/auth/guards/auth.guard";

async function bootstrap() {
  // env is already validated by importing it
  const app = await NestFactory.create(AppModule, {
    snapshot: true,
  });
  app.useGlobalPipes(new ValidationPipe());
  const authGuard = app.get(AuthGuard);
  app.useGlobalGuards(authGuard);

  await app.listen(env.PORT || 3000);
}

// Catch any unhandled promise rejections
process.on('unhandledRejection', (error) => {
  console.error('Unhandled promise rejection:', error);
  process.exit(1);
});

bootstrap().then(() => console.log("Server is running on port: http://localhost:" + env.PORT));
