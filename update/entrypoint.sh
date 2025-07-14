#!/bin/sh
echo "Updating containers..."
apk add --no-cache curl
curl -X POST http://localhost:8000/insert
