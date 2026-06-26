#!/bin/bash
# Entrypoint script for personalizing developer environment

# Set up git user configuration
# These can be overridden by setting GIT_USER_NAME and GIT_USER_EMAIL environment variables
if [ ! -z "$GIT_USER_NAME" ]; then
    git config --global user.name "$GIT_USER_NAME"
fi

if [ ! -z "$GIT_USER_EMAIL" ]; then
    git config --global user.email "$GIT_USER_EMAIL"
fi

# Executar o comando indicado no docker run (por exemplo, bash)
exec "$@"
