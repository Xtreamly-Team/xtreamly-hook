import { Module } from '@nestjs/common';
import { QuoteComputeService } from './quote-compute/quote-compute.service';

@Module({
  providers: [QuoteComputeService]
})
export class QuoteComputeModule {}
