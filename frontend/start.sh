#!/bin/sh

CERT_PATH="/etc/letsencrypt/live/178.128.203.208.nip.io/fullchain.pem"

if [ -f "$CERT_PATH" ]; then
    echo "SSL certificates found, using HTTPS configuration"
    cp /etc/nginx/templates/nginx.conf /etc/nginx/conf.d/default.conf
else
    echo "SSL certificates not found, using HTTP configuration"
    cp /etc/nginx/templates/nginx.dev.conf /etc/nginx/conf.d/default.conf
fi

exec nginx -g 'daemon off;'
