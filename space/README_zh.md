# Falcon Studio · Hugging Face Space

[English](README.md) | **中文**

运行在 ZeroGPU Space 上的 [Falcon Studio](https://github.com/JusperLee/Falcon/tree/main/studio) 界面。用摄像头和麦克风录一段，Falcon 会提取画面中嘴部对应的说话者的声音。

ZeroGPU 每次调用临时分配 GPU，并计入每日配额，不适合实时工作台每 0.5 秒调用一次的循环。所以这个 Space 先录制、再分离。实时分离、耳机监听和会议接入请在本地运行 Falcon Studio。

## 工作方式

1. 浏览器以 16 kHz 采集音频，并用 MediaPipe Face Landmarker 跟踪嘴部，与 Falcon Studio 相同。跟踪在浏览器内完成，视频画面不上传。
2. 每段录音最长 60 秒。停止后，音频和 96×96 灰度嘴部画面（各带采集时间）一起发送到 Space。开头和结尾没有嘴部画面的部分会被裁掉。
3. 服务端把整段切成实时工作台发送的窗口：2 秒音频和 50 个嘴部位置、0.5 秒步长、取每个窗口末端前 80 ms 的 0.5 秒、相邻 20 ms 交叉淡化、开头之前补零历史、结尾之后补零。每个位置取时间上最近的嘴部画面，与 Studio 一致。所有窗口在一次 GPU 调用内分批计算。嘴部画面相同时，结果等于没有跳过窗口的实时工作台；Studio 因画面过旧而暂停的地方，Space 沿用最近的画面。
4. 页面显示 GPU 耗时，可以试听、与原始混合对比，并导出两者的 WAV。

Space 启动时加载两个模型：[`JusperLee/Falcon`](https://huggingface.co/JusperLee/Falcon)（论文模型）和 [`JusperLee/Falcon-Big`](https://huggingface.co/JusperLee/Falcon-Big)。

## 隐私与配额

音频和嘴部画面只用于本次分离，不写入磁盘；结果只保存在你的浏览器中。从 Space 页面打开时，分离计入你自己的 ZeroGPU 配额。未登录访客的配额最少，可能被拒绝（"ZeroGPU runs limit"）；登录 Hugging Face 后配额更多、排队优先级更高。一次分离约占 0.2–2.5 秒 GPU 时间（8 秒 Small 到 60 秒 Big，实测），另加上传和排队时间。

## 开发

在 GitHub 仓库根目录、Python 3.12 下：

```bash
pip install -r space/requirements.txt gradio==6.29.1
python space/build.py --out /tmp/falcon-space    # 复制 studio/ 的运行时和网页文件，再加上 space/
cd /tmp/falcon-space && python app.py            # http://127.0.0.1:7860，默认 CPU，可用 FALCON_DEVICE 指定
python -m pytest space/tests                     # 在仓库根目录运行
python space/build.py --push JusperLee/Falcon-Studio
```

`space/` 只放与 Falcon Studio 不同的部分：`app.py`（Gradio server 模式、`@spaces.GPU`）、`clip.py`（窗口划分与请求校验）、`web/index.html` 和 `web/app.js`（先录制、再分离）以及 `web/i18n-space.js`。其余文件由 `build.py` 从 `studio/` 复制。
