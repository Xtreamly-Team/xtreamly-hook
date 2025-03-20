#!/bin/bash -e

# Set variables
PROJECT_ID="xtreamly-ai"
INSTANCE_NAME="xtreamly-hook"
DB_VERSION="POSTGRES_15"
REGION="us-central1"
TIER="db-custom-2-7680"
ROOT_PASSWORD="<PWD>"
DB_NAME="xtr_trade_db"
CLOUD_RUN_SERVICE_EMAIL="664616721985-compute@developer.gserviceaccount.com"

## Enable required services
echo "Enabling required services..."
gcloud services enable sqladmin.googleapis.com

# Extract CIDR blocks for us-central1
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$CLOUD_RUN_SERVICE_EMAIL" \
    --role="roles/cloudsql.client"

# Create Cloud SQL PostgreSQL instance
echo "Creating Cloud SQL PostgreSQL instance..."
gcloud sql instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --database-version=$DB_VERSION \
    --tier=$TIER \
    --region=$REGION \
    --root-password=$ROOT_PASSWORD

# Wait for the instance to be ready
echo "Waiting for the instance to be ready..."
sleep 60  # Adjust as needed

# Get the external IP of the instance
DB_IP=$(gcloud sql instances describe $INSTANCE_NAME --format="value(ipAddresses[0].ipAddress)")

# Output connection details
echo "Setup complete! Use the following connection string:"
echo "postgres://postgres:$ROOT_PASSWORD@$DB_IP:5432/$DB_NAME"