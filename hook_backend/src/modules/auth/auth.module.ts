import { Module, forwardRef } from '@nestjs/common';
import { FirebaseUserService } from '@modules/firebase-user/services/firebase-user/firebase-user.service';
import { UserService } from '@modules/user/services/user/user.service';
import {AuthGuard} from "@modules/auth/guards/auth.guard";
import {UserModule} from "@modules/user/user.module";

@Module({
  imports: [
      forwardRef(() => UserModule),
  ],
  providers: [FirebaseUserService, UserService, AuthGuard],
})
export class AuthModule {}