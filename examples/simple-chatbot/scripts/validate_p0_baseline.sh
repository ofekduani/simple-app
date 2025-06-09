#!/bin/bash

LOGFILE="../server/bot-gemini.log"
ROOM_URL="https://simple-app.daily.co/zaqulY1av8fSYH2DCZWj"

if grep -q "Joined $ROOM_URL" "$LOGFILE"; then
  echo "PASS: Bot joined Daily room successfully."
  exit 0
else
  echo "FAIL: Bot did not join Daily room. Check $LOGFILE for errors."
  exit 1
fi 