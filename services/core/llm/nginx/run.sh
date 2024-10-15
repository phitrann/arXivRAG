#!/bin/sh

# Replace environment variables in the nginx configuration file
envsubst '$SERVER_HOST' < /etc/nginx/conf.d/nginx.conf.template > /etc/nginx/conf.d/default.conf

# Print the nginx configuration file
# cat /etc/nginx/conf.d/default.conf

# Run nginx
nginx -g 'daemon off;'
