# Database Commands
# ----------------


## SEEDING/WRITING
# seed-db: Populates the database with test data including users, positions,
#         and position history entries for testing and development
seed-db:
	pnpm db:seed

## QUERYING/READING

# check-db: Checks the database connection and schema
#          Lists all tables, their structures, and verifies TimescaleDB extension
check-db:
	pnpm db:check

# query-db: Displays current database contents including users, positions,
#          and aggregated position history data
query-db:
	pnpm db:query

# check-columns: Checks the columns in the database
check-columns:
	pnpm db:check-columns

## CLEANING
# clean-db: Drops all tables and clears the database
clean-db:
	pnpm db:clean

# reset-db: Drops all tables and runs migrations to recreate the database schema
#          Use this to start with a fresh database state
reset-db:
	pnpm db:reset

help:
	@echo "Database Commands:"
	@echo "  check-db    - Check database connection and display schema information"
	@echo "  db-clean    - Drop all tables and clear database"
	@echo "  db-sync     - Synchronize database schema"
	@echo "  db-reset    - Reset database (clean + run migrations)"
	@echo "  db-migrate  - Run pending migrations"
	@echo "  db-rollback - Revert last migration"