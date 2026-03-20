#!/bin/bash

# Configuration des variables
REMOTE_USER_HOST="dce"
REMOTE_PATH="/usr/users/sdim/sdim_15/Documents/2025-2026-holotrack/holotrack_model/Results"
REMOTE_PATH_2="/usr/users/sdim/sdim_15/Documents/2025-2026-holotrack/Simulator/results/tmp/"
LOCAL_PATH="./Results_test"

echo "🚀 Début du transfert depuis $REMOTE_USER_HOST..."

rm -rf "$LOCAL_PATH/"*
scp -r "$REMOTE_USER_HOST:$REMOTE_PATH/*" "$LOCAL_PATH"
scp "$REMOTE_USER_HOST:$REMOTE_PATH_2/*" "$LOCAL_PATH"

# Vérification du succès
if [ $? -eq 0 ]; then
    echo "✅ Transfert terminé avec succès dans $LOCAL_PATH"
else
    echo "❌ Erreur lors du transfert"
    exit 1
fi
