#!/bin/bash
APP_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
"$APP_DIR/scripts/install-blackhole.sh" "$@"
EXIT_CODE=$?
printf '\n按回车关闭此窗口。'
read -r _
exit "$EXIT_CODE"
