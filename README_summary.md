# 视频目标检测 Agent 概要

基于 **FastAPI + YOLOE + ByteTrack + React** 的视频目标检测系统。用户上传视频并用自然语言描述检测目标，系统逐帧检测并通过 SSE 实时回传结果。

## 技术栈

- **后端**：FastAPI 异步框架 + SQLAlchemy 2.x + asyncpg
- **检测模型**：默认 YOLOE，可切换 YOLO-World、Grounding DINO、Florence-2，配合 SAHI 切片增强小目标召回
- **目标跟踪**：ByteTrack 提供持久 `track_id`
- **大模型 VLM**：MiniCPM-V 或任意 OpenAI 兼容 API，用于 Prompt 标准化与检测结果语义复核
- **持久化**：PostgreSQL 存储任务历史，DB 不可用时自动降级为内存模式
- **前端**：React 18 + Vite + Tailwind CSS

## 核心特性

- 每 N 帧全量检测 + 中间帧 Kalman 预测，吞吐提升 4~8 倍
- 检测得分与 VLM 得分按 0.4/0.6 加权融合，驱动 tentative→confirmed/rejected 状态机
- 支持自定义 YOLOE 模型训练，权重可直接接回后端
- 完整 REST API + SSE 实时推送 + ZIP 结果打包下载

部署环境：Linux + NVIDIA GPU（推荐 RTX 3090 / A100）。详细文档见 `README.md`。
