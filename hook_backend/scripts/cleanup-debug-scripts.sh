#!/bin/bash

# Cleanup script to remove unnecessary debug files

echo "Cleaning up unnecessary debug scripts..."

# List of files to remove
FILES_TO_REMOVE=(
  "scripts/create-token-prices-view.ts"
  "scripts/fix-token-prices-view.ts"
  "scripts/cleanup-token-prices-view.ts"
  "scripts/create-simple-view.ts"
  "test/integration/DB/check-columns.ts"
  "test/integration/DB/query-data.ts"
)

# Remove each file
for file in "${FILES_TO_REMOVE[@]}"; do
  if [ -f "$file" ]; then
    echo "Removing $file"
    rm "$file"
  else
    echo "File $file not found, skipping"
  fi
done

echo "Cleanup complete!"
echo "Remaining essential scripts:"
echo "- scripts/run-token-prices-migration.ts (db:create-token-prices)"
echo "- scripts/refresh-token-prices-view.ts (db:refresh-token-prices-view)"
echo "- scripts/seed-token-prices.ts (db:seed-token-prices)"
echo "- test/integration/DB/check-database.ts (db:check)"
echo "- test/integration/DB/seed-data.ts (db:seed)" 