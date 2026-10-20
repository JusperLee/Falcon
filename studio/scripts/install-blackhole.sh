#!/bin/bash
# Explicit optional setup. Never called by the ordinary Falcon launcher.
set -euo pipefail

DRIVER_PATH="/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver"
BREW_BIN="$(command -v brew 2>/dev/null || true)"
if [ -z "$BREW_BIN" ]; then
    for candidate in /opt/homebrew/bin/brew /usr/local/bin/brew; do
        if [ -x "$candidate" ]; then
            BREW_BIN="$candidate"
            break
        fi
    done
fi

case "${1:-}" in
    --check)
        if [ -d "$DRIVER_PATH" ]; then
            printf 'BlackHole 2ch 驱动文件已存在：%s\n' "$DRIVER_PATH"
            printf '如果浏览器设备列表中仍未出现，请重启 macOS 后重试。\n'
        else
            printf '尚未找到 BlackHole 2ch 驱动。\n'
        fi
        if [ -n "$BREW_BIN" ]; then
            printf 'Homebrew：%s\n' "$BREW_BIN"
        else
            printf 'Homebrew：未找到；可使用官方安装包。\n'
        fi
        printf '官方说明：https://github.com/ExistentialAudio/BlackHole\n'
        exit 0
        ;;
    --help|-h)
        printf '用法：\n  ./scripts/install-blackhole.sh          安装驱动（可能需要管理员验证）\n  ./scripts/install-blackhole.sh --check  仅检查，不改动系统\n'
        exit 0
        ;;
    "") ;;
    *) printf '不支持的参数：%s\n' "$1" >&2; exit 2 ;;
esac

if [ "$(uname -s)" != "Darwin" ]; then
    printf 'BlackHole 会议音频输出仅适用于 macOS。\n' >&2
    exit 1
fi
if [ -d "$DRIVER_PATH" ]; then
    printf 'BlackHole 2ch 驱动已存在，无需重复安装。\n'
    printf '若尚未显示在音频设备中，请先重启 macOS；然后按 README 的会议接线步骤操作。\n'
    exit 0
fi
if [ -z "$BREW_BIN" ]; then
    printf '没有找到 Homebrew。请从官方页面下载 BlackHole 2ch 安装包：\nhttps://existential.audio/blackhole/\n\n安装完成后重启 macOS，再按 README 设置会议输出。\n' >&2
    exit 1
fi

cat <<'TEXT'

Falcon Studio · 会议音频设置

将通过 Homebrew 安装 BlackHole 2ch 虚拟音频驱动：
  brew install --cask blackhole-2ch

安装会修改系统音频驱动目录，可能要求输入 macOS 管理员密码。
密码由系统安装程序验证；终端输入密码时不会显示字符。
请先保存工作并退出正在使用音频的应用。安装完成后需要重启 macOS。
此脚本不会重启电脑，也不会改变系统默认麦克风或扬声器。

官方项目：https://github.com/ExistentialAudio/BlackHole
Homebrew：https://formulae.brew.sh/cask/blackhole-2ch
TEXT

HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_ANALYTICS=1 "$BREW_BIN" install --cask blackhole-2ch
printf '\nBlackHole 安装命令已完成。请保存工作并重启 macOS。\n'
printf '重启后：Falcon 会议输出选 BlackHole 2ch；会议麦克风选 BlackHole 2ch；系统和会议扬声器选耳机。\n'
