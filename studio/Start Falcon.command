#!/bin/bash
APP_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
"$APP_DIR/scripts/launch.sh" "$@"
EXIT_CODE=$?
if [ "$EXIT_CODE" -ne 0 ] && [ "$EXIT_CODE" -ne 130 ]; then
    printf '\n启动未完成。请查看上面的错误信息，或阅读 README.md 的故障排查。\n按回车关闭此窗口。'
    read -r _
fi
exit "$EXIT_CODE"
