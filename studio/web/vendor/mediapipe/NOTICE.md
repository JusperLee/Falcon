# Local face tracking assets

- Library: Google MediaPipe `@mediapipe/tasks-vision` 1.0.1 from the official npm registry (Apache-2.0; see LICENSE).
- Model: official Face Landmarker float16 model, version 1.
- Exact source URLs, npm integrity and file SHA-256 values are recorded in manifest.json.
- All assets are served from localhost; detection requires no CDN or cloud upload.
- Uses BlazeFace short-range face detection and FaceMesh-V2 (478 landmarks), VIDEO mode, a single face, CPU WebAssembly.
- Source: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
- The crop in this app tracks the mouth, with stable face-based scale. It is not a reconstruction of the unavailable original LRS3 landmark-alignment pipeline.
