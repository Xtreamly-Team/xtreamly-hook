import { Injectable, Logger } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { PriceService } from './price.service';
import { DataSource } from 'typeorm';

@Injectable()
export class PriceUpdateService {
  private readonly logger = new Logger(PriceUpdateService.name);
  private isUpdating = false;
  private isRefreshing = false;

  constructor(
    private priceService: PriceService,
    private dataSource: DataSource,
  ) {}

  @Cron(CronExpression.EVERY_MINUTE)
  async updatePrices() {
    // Prevent concurrent updates
    if (this.isUpdating) {
      this.logger.debug('Price update already in progress, skipping...');
      return;
    }

    try {
      this.isUpdating = true;
      this.logger.debug('Starting scheduled price update');
      await this.priceService.updatePrices();
      this.logger.debug('Completed scheduled price update');
      
      // Refresh the materialized view immediately after updating prices
      await this.refreshMaterializedView();
    } catch (error) {
      this.logger.error(`Error in scheduled price update: ${error.message}`, error.stack);
    } finally {
      this.isUpdating = false;
    }
  }

  // This method is now called directly after price updates
  // but we also keep a separate scheduled refresh as a backup
  @Cron('*/5 * * * *') // Every 5 minutes as a backup
  async refreshMaterializedView() {
    // Prevent concurrent refreshes
    if (this.isRefreshing) {
      this.logger.debug('Materialized view refresh already in progress, skipping...');
      return;
    }

    try {
      this.isRefreshing = true;
      this.logger.debug('Starting materialized view refresh');
      
      // Check if the view exists
      const viewCheck = await this.dataSource.query(`
        SELECT matviewname FROM pg_matviews WHERE matviewname = 'token_prices_hourly';
      `);
      
      if (viewCheck.length === 0) {
        this.logger.warn('token_prices_hourly materialized view does not exist, skipping refresh');
        return;
      }
      
      // Get the current time before refresh
      const beforeTime = new Date();
      
      // Refresh the materialized view
      await this.dataSource.query('REFRESH MATERIALIZED VIEW token_prices_hourly;');
      
      // Get the current time after refresh
      const afterTime = new Date();
      
      // Calculate the time taken
      const timeTaken = (afterTime.getTime() - beforeTime.getTime()) / 1000;
      
      this.logger.debug(`Completed materialized view refresh (took ${timeTaken.toFixed(2)} seconds)`);
    } catch (error) {
      this.logger.error(`Error in materialized view refresh: ${error.message}`, error.stack);
    } finally {
      this.isRefreshing = false;
    }
  }
} 