#!/bin/sh
echo "Backing up DB..."
cp /app/test.db /app/backup_$(date +%F).db
