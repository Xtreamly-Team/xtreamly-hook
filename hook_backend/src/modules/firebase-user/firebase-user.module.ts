import { Module } from '@nestjs/common';
import { FirebaseUserService } from './services/firebase-user/firebase-user.service';
import {AuthGuard} from "@modules/auth/guards/auth.guard";

@Module({
  providers: [FirebaseUserService, AuthGuard],
  exports: [FirebaseUserService],
})
export class FirebaseUserModule {}