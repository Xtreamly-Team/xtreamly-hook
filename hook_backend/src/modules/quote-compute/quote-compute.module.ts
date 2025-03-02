import { Module } from '@nestjs/common';
import { QuoteComputeService } from './quote-compute/quote-compute.service';

@Module({
  providers: [QuoteComputeService],
  exports: [QuoteComputeService]
})
export class QuoteComputeModule {}