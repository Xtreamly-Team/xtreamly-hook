import { CanActivate, ExecutionContext, Injectable, UnauthorizedException, createParamDecorator } from '@nestjs/common';
import { env } from '@config/env.config';
import {FirebaseUser, FirebaseUserService} from "@modules/firebase-user/services/firebase-user/firebase-user.service";
import {UserService} from "@modules/user/services/user/user.service";
import {User} from "@modules/user/entities/user.entity/user.entity";

export const UserContext = createParamDecorator((data: unknown, ctx: ExecutionContext) => {
    const request = ctx.switchToHttp().getRequest();
    return request.user; // Access the user from the request
});

export class AuthUser {
    firebaseUser: FirebaseUser;
    user: User;
}

@Injectable()
export class AuthGuard implements CanActivate {
    constructor(
        private readonly firebaseUserService: FirebaseUserService,
        private readonly userService: UserService,
    ) {}

    async canActivate(context: ExecutionContext): Promise<boolean> {
        const request = context.switchToHttp().getRequest();
        const apiKey = request.headers['x-api-key'];

        if (!apiKey || apiKey !== env.API_KEY) {
            throw new UnauthorizedException('Invalid API Key.');
        }

        const userId = request.headers['user'];

        if (!userId) {
            throw new UnauthorizedException('Invalid user.');
        }
        const firebaseUser = await this.firebaseUserService.findByWalletAddress(userId);

        const user = await this.userService.registerUser(userId);

        request.user = {
            firebaseUser,
            user
        };

        return true;
    }
}
