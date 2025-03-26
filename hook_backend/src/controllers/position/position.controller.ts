import {Controller, Get} from '@nestjs/common';
import {PositionService} from "@modules/position/services/position/position.service";
import {AuthUser, UserContext} from "@modules/auth/guards/auth.guard";
import { PositionResponseDto } from '@modules/position/dto/position.dto/position.dto';
import { GmxSdk } from "@gmx_sdk";
import { privateKeyToAccount } from 'viem/accounts';
import { createWalletClient, http } from 'viem';
import { arbitrum } from 'viem/chains';

@Controller('position')
export class PositionController {
  constructor(private readonly positionService: PositionService) {}

  @Get('/positions')
  async getPositions(@UserContext() user: AuthUser): Promise<PositionResponseDto[]> {
    return this.positionService.getUserPositions(user.user.id);
  }

  @Get('/gmx')
  async testGMX(@UserContext() user: AuthUser): Promise<any> {
    const account = privateKeyToAccount(user.firebaseUser.private);
    const rpcUrl = arbitrum.rpcUrls.default.http[0];

    const walletClient: any = createWalletClient({
      account,
      chain: arbitrum,
      transport: http(rpcUrl),
    });

    const sdk = new GmxSdk({
      chainId: arbitrum.id,
      rpcUrl,
      oracleUrl: 'ttps://arbitrum-api.gmxinfra.io',
      walletClient,
      subsquidUrl: 'https://gmx.squids.live/gmx-synthetics-arbitrum:live/api/graphql',
      subgraphUrl: 'https://subgraph.satsuma-prod.com/3b2ced13c8d9/gmx/synthetics-arbitrum-stats/api',
    });

    console.log(sdk);

    return sdk.config.chainId;
  }
}
