import { Injectable, Logger } from '@nestjs/common';
import { Cron, Interval } from '@nestjs/schedule';

@Injectable()
export class HeartbeatService {
  private readonly logger = new Logger(HeartbeatService.name);

  // // Run every 2 minutes
  // @Interval(120000)
  // handleHeartbeat() {
  //   this.logger.log('Backend online and ready');
  //   // You can add additional health checks here
  // }

  // For your future GMX query task (every 5 minutes)
  @Cron('*/1 * * * *')
  async handleGmxQuery() {
    this.logger.log('Monitoring GMX Positions...');
    try {
      // Your GMX query logic will go here
      // await this.gmxService.someMethod();
      
      this.logger.log('GMX query completed successfully');
    } catch (error) {
      this.logger.error(`GMX query failed: ${error.message}`);
    }
  }
} 