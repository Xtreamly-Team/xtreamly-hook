# Database Commands
# ----------------
# check-db: Checks the database connection and schema
#          Lists all tables, their structures, and verifies TimescaleDB extension
check-db:
	pnpm db:check

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