# 视频目标检测 Agent

基于 **FastAPI + Grounding DINO + ByteTrack + React** 的视频目标检测系统。  
用户上传视频，通过**自然语言**描述要检测的目标，系统逐帧检测并实时回传结果。

---

## 目录

- [技术方案](#技术方案)
- [项目目录结构](#项目目录结构)
- [各模块职责](#各模块职责)
- [快速启动](#快速启动)
- [迁移到新设备](#迁移到新设备)
- [大模型 (VLM) 配置](#大模型-vlm-配置)
- [任务历史持久化 (PostgreSQL)](#任务历史持久化-postgresql)
- [训练自定义 YOLOE 模型](#训练自定义-yoloe-模型)
- [API 接口说明](#api-接口说明)
- [示例请求与返回](#示例请求与返回)
- [检测可视化效果](#检测可视化效果)
- [扩展：多 GPU](#扩展多-gpu)
- [扩展：提升吞吐](#扩展提升吞吐)
- [扩展：接入 Redis / Celery](#扩展接入-redis--celery)

---

## 技术方案

| 层次 | 技术选型 | 说明 |
|------|---------|------|
| 后端框架 | **FastAPI** | 异步高性能，原生支持 SSE / WebSocket |
| 主检测模型 | **YOLOE**（Ultralytics） | 开放词汇 YOLO，速度快、显存低，**支持自定义训练**（见 [训练自定义 YOLOE 模型](#训练自定义-yoloe-模型)） |
| 备选模型 | **YOLO-World** / **Grounding DINO** / **Florence-2** | 同样支持开放词汇检测，可按需切换 |
| 小目标增强 | **SAHI** | 大图自动切片推理，显著提升小目标召回 |
| **大模型 (VLM)** | **MiniCPM-V**（默认）/ 任意 OpenAI 兼容模型 | **Prompt 标准化 + 检测结果语义复核**，支持本地 vLLM 与远端 API（见 [大模型 (VLM) 配置](#大模型-vlm-配置)） |
| 目标跟踪 | **ByteTrack** | 无需重检测的多目标跟踪，大幅降低 GPU 开销 |
| 实时推送 | **Server-Sent Events (SSE)** | 逐帧推送，前端实时刷新 |
| **任务持久化** | **PostgreSQL + SQLAlchemy 2.x (async / asyncpg)** | **历史任务、状态、进度落库；DB 不可用时降级为内存模式，不影响检测主路径**（见 [任务历史持久化](#任务历史持久化-postgresql)） |
| 前端 | **React 18 + Vite** | 快速构建、热更新；Tailwind CSS 样式 |
| 部署环境 | Linux + NVIDIA GPU (CUDA) | 推荐 RTX 3090 / A100 或更高 |

> 当前默认后端为 **YOLOE**（`yoloe-11l-seg.pt`），通过 Ultralytics 提供，可在 `backend/.env` 的 `DETECTION_MODEL` / `YOLO_WORLD_MODEL` 中切换。
> 大模型默认指向本地 vLLM 服务（`http://localhost:8010/v1`，MiniCPM-V），改 `VLM_API_BASE` / `VLM_MODEL_NAME` 即可切换到远端 API（如 OpenAI、Qwen-VL、智谱、DeepSeek 等）。

### 检测 + 跟踪流程（速度优先）

```
视频帧序列
    │
    ├── 每隔 N 帧（默认 N=5）→ YOLOE 全量检测（GPU，可选 SAHI 切片）
    │                           ↓
    │                        ByteTrack.update(detections)  ← 分配持久 track_id
    │                           ↓
    │            （可选）VLM 语义复核：对低/中置信度 track 把裁剪图发给大模型
    │                              判断是否真为目标，与检测得分融合
    │
    └── 其余帧 → ByteTrack.update(last_detections)  ← Kalman 预测（纯 CPU）
                    ↓
              画框 + 保存 + SSE 推送
```

此策略带来约 **4~8×** 的吞吐提升（取决于 N 值和模型大小）。

---

## 项目目录结构

```
sod/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── upload.py          # POST /api/upload
│   │   │   ├── detect.py          # POST /api/detect, GET /api/task, /stream, /download
│   │   │   └── history.py         # GET /api/tasks, GET /api/task/{id}/frames, DELETE /api/task/{id}, DELETE /api/tasks
│   │   ├── core/
│   │   │   ├── config.py          # Pydantic Settings（读 .env）
│   │   │   └── logging.py         # Loguru 日志配置
│   │   ├── db/
│   │   │   ├── session.py         # SQLAlchemy 异步引擎 + AsyncSessionLocal + init_db()
│   │   │   └── models.py          # ORM 模型：detection_tasks 任务历史表
│   │   ├── models/
│   │   │   └── schemas.py         # 所有 Pydantic 数据模型
│   │   ├── services/
│   │   │   ├── detector.py        # YOLOE / YOLO-World / Grounding DINO / Florence-2 检测器抽象
│   │   │   ├── tracker.py         # ByteTrack 封装
│   │   │   ├── visualizer.py      # 框 + 标签绘制、base64 编码
│   │   │   ├── pipeline.py        # 主视频处理流水线
│   │   │   ├── task_manager.py    # 任务注册表 + 异步队列
│   │   │   ├── vlm_service.py     # 大模型 (VLM) 客户端：OpenAI 兼容协议，检测结果语义复核
│   │   │   ├── prompt_normalizer.py # 调用大模型把中文 prompt 转成英文检测短语 + 颜色/类别元数据
│   │   │   ├── fusion_engine.py   # 检测器得分 × VLM 得分加权融合，决定 track 状态机
│   │   │   ├── color_filter.py    # 按 HSV 颜色规则给检测框降权
│   │   │   └── frame_quality.py   # 过滤过暗/过亮/全黑等低质量帧
│   │   ├── utils/
│   │   │   └── video_utils.py     # 视频元数据读取、时间戳格式化
│   │   └── main.py                # FastAPI 入口 + 路由注册
│   ├── YOLOWorld/                 # YOLOE / YOLO-World 后端 + SAHI 切片
│   │   ├── yolo_world_detector.py # YOLOE / YOLO-World 推理实现
│   │   ├── train_yoloe.py         # YOLOE 训练 CLI
│   │   ├── dataset.yaml.example   # 训练数据集模板
│   │   ├── TRAINING.md            # YOLOE 训练详细文档
│   │   └── README.md
│   ├── scripts/
│   │   └── init_postgres.sh       # PostgreSQL 一键初始化（角色 / 库 / 权限 / .env 写入）
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── VideoUploader.jsx  # 拖拽上传组件
│   │   │   ├── PromptInput.jsx    # 自然语言输入框 + 示例
│   │   │   ├── ProgressBar.jsx    # 进度条（上传/检测）
│   │   │   └── ResultViewer.jsx   # 实时帧展示 + 历史缩略图网格
│   │   ├── hooks/
│   │   │   └── useDetectionTask.js # 完整检测工作流 hook（上传→任务→SSE）
│   │   ├── services/
│   │   │   └── api.js             # Axios API 客户端
│   │   ├── App.jsx                # 根组件（左右双栏布局）
│   │   ├── main.jsx
│   │   └── index.css              # Tailwind + 自定义动画
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── example_pro.png                # 目标检测可视化效果参考图
└── Prompt.md                      # 原始需求文档
```

---

## 各模块职责

| 模块 | 职责 |
|------|------|
| `app/main.py` | 创建 FastAPI 实例，注册路由，启动/关闭钩子 |
| `app/core/config.py` | 统一配置（通过 `.env` 注入，支持多环境） |
| `app/api/upload.py` | 流式接收大视频文件，读取元数据，返回 `video_id` |
| `app/api/detect.py` | 创建检测任务、SSE 流、ZIP 下载 |
| `app/api/history.py` | 任务历史 CRUD：列表分页 / 帧文件清单 / 单任务删除 / 全量清空（含磁盘联动） |
| `app/db/session.py` | SQLAlchemy 异步引擎与会话工厂；`init_db()` 启动时自动建表 |
| `app/db/models.py` | ORM 模型 `TaskRecord` ↔ 表 `detection_tasks`（任务状态、进度、时间戳等） |
| `app/services/detector.py` | 封装 YOLOE / YOLO-World / Grounding DINO / Florence-2，统一接口 |
| `app/services/tracker.py` | 封装 ByteTrack，提供持久 `track_id` |
| `app/services/visualizer.py` | 绘制高对比度检测框、标签、时间戳 |
| `app/services/pipeline.py` | 主流水线：读帧→检测→跟踪→画框→推送→ZIP |
| `app/services/task_manager.py` | 任务注册表，SSE 异步队列，GPU 并发控制 |
| `app/services/vlm_service.py` | 大模型（VLM）客户端：把检测裁剪图 + 目标描述发给本地/远端 OpenAI 兼容 API，返回是否为目标的 JSON 判定 |
| `app/services/prompt_normalizer.py` | 调用大模型把中文自然语言转成英文检测短语，并抽取目标类型、颜色过滤等元信息 |
| `app/services/fusion_engine.py` | 把检测器分数与 VLM 分数加权融合（默认 0.4/0.6），驱动 track 的 tentative→confirmed→rejected 状态机 |
| `app/utils/video_utils.py` | `cv2` 读取视频信息、格式化时间戳 |
| `frontend/hooks/useDetectionTask.js` | 封装完整前端状态机（上传→任务→SSE→结果） |
| `frontend/components/ResultViewer.jsx` | 实时帧大图 + 历史缩略图网格，点击可查看任意帧 |

---

## 快速启动

### 1. 环境准备

```bash
# Python 3.10+，CUDA GPU
python -m venv venv
source venv/bin/activate
```

### 2. 安装后端依赖

```bash
cd backend
pip install -r requirements.txt
```

`requirements.txt` 已包含 **YOLOE 默认后端** 所需的 `ultralytics>=8.3.0` 与 `sahi>=0.11.18`。  
YOLOE 权重 `yoloe-11l-seg.pt` 与文本编码器 `mobileclip_blt.ts` **不随仓库分发（已被 `.gitignore` 忽略），且本项目强制离线、运行时不会自动联网下载**——需要手动下载并放到约定位置。完整步骤（含下载直链、放置路径、`.env` 改写）见 [迁移到新设备](#迁移到新设备) 一节。

如需 **Grounding DINO 或 Florence-2** 后端，再额外安装：

```bash
# Grounding DINO（二选一）
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
# 或：pip install groundingdino-py

# 下载 Grounding DINO 权重
mkdir -p models/groundingdino/weights models/groundingdino/config
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
     -O models/groundingdino/weights/groundingdino_swint_ogc.pth

# ByteTrack
pip install git+https://github.com/ifzhang/ByteTrack.git
# 或：pip install bytetracker
```

复制并编辑配置：

```bash
cp .env.example .env
# 检测模型默认 yolo_world（实际加载 yoloe-11l-seg.pt）
# 切换其他后端见下方“5. 切换检测模型”
```

### 3. 启动后端

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
# 访问 http://localhost:8000/docs 查看 Swagger 文档
```

### 4. 安装并启动前端

```bash
cd frontend
npm install
npm run dev
# 访问 http://localhost:5173
```

### 5. 切换检测模型

后端默认使用 **YOLOE**（通过 Ultralytics 加载 `yoloe-11l-seg.pt`）。可在 `backend/.env` 中切换：

#### YOLOE（默认，推荐）

```env
DETECTION_MODEL=yolo_world
YOLO_WORLD_MODEL=/home/user/.cache/ultralytics/hub/models/yoloe-11l-seg.pt
# 也可填模型文件名让 ultralytics 自行下载，如 yoloe-11l-seg.pt
```

可选 SAHI 切片参数（大图小目标场景显著提升召回）：

```env
SAHI_SLICE_HEIGHT=640
SAHI_SLICE_WIDTH=640
SAHI_OVERLAP_HEIGHT_RATIO=0.2
SAHI_OVERLAP_WIDTH_RATIO=0.2
```

#### YOLO-World（YOLOE 之外的另一个 Ultralytics 开放词汇模型）

```env
DETECTION_MODEL=yolo_world
YOLO_WORLD_MODEL=yolo11l-world.pt   # 或 yolo11m-world.pt / yolo11s-world.pt
```

#### Grounding DINO

```env
DETECTION_MODEL=grounding_dino
GDINO_CHECKPOINT_PATH=./models/groundingdino/weights/groundingdino_swint_ogc.pth
```

#### Florence-2

```env
DETECTION_MODEL=florence2
FLORENCE2_MODEL_ID=microsoft/Florence-2-large
```

首次运行 Florence-2 会从 HuggingFace 自动下载模型权重（约 1.5GB）。

---

## 迁移到新设备

> 这一节专门解决三件事：**(A) YOLOE 模型与额外文件从哪下载、放到哪里**；**(B) 数据集怎么配置、怎么搬**；**(C) PostgreSQL 怎么建库建表**。

**先理解一个前提**：本仓库 `.gitignore` 忽略了 `*.pt / *.pth / *.ts / weights/`，而 `datasets/ annotations/ runs/` 也没纳入版本库。换句话说，**`git clone` 只会带来代码，不会带来任何模型权重、数据集和训练产物**——这些都必须手动下载或拷贝。

### 0. clone 之后还缺什么（总览）

| 类别 | 随 `git clone` 过来? | 迁移动作 |
|---|---|---|
| 源码 / `requirements.txt` / `start*.sh` | ✅ 是 | 无 |
| `backend/.env` | ⚠️ 在版本库里，但含**旧机器的绝对路径 + 旧 DB 密码** | 按本节改写（A/C） |
| YOLOE 权重 `yoloe-11l-seg.pt` | ❌ 否 | 下载 / 拷贝（A） |
| 文本编码器 `mobileclip_blt.ts` | ❌ 否 | 下载 / 拷贝（A）——**缺它则文本提示检测直接报错** |
| Grounding DINO 权重（可选） | ❌ 否 | 仅 `DETECTION_MODEL=grounding_dino` 时需要（A） |
| 训练数据 `backend/datasets/`、`backend/annotations/` | ❌ 否 | 想保留历史标注就整目录拷贝（B） |
| 训练产物 `backend/runs/train/.../best.pt` | ❌ 否 | 想保留已训练模型就整目录拷贝（B） |
| PostgreSQL 数据 | ❌ 否（在数据库里，不在仓库） | 新机重新建库建表（C），或 `pg_dump` / `pg_restore` 连数据搬 |

### A. YOLOE 模型与额外文件：下载 + 放到哪里

本项目**强制离线模式**（`yolo_world_detector.py` / `image_detector.py` / `train_yoloe.py` 都设置了 `YOLO_OFFLINE=1`、`ULTRALYTICS_OFFLINE=1`），所以**运行时不会自动联网下载**，必须先把文件放到位。

| 文件 | 大小 | 目标路径（相对 `SOD/` 根） | 必需性 / 作用 |
|---|---|---|---|
| `yoloe-11l-seg.pt` | ~68 MB | `backend/models/yolo/yoloe-11l-seg.pt` | **必需**。视频 / 图片检测器与训练共用的基座权重 |
| `mobileclip_blt.ts` | ~572 MB | `backend/mobileclip_blt.ts` | **文本提示必需**。YOLOE 开放词表（`set_classes`）用它把类别文字编码成向量；缺失会抛 “需要 … mobileclip_blt.ts 文本编码器” |
| `groundingdino_swint_ogc.pth` | ~662 MB | `backend/models/groundingdino/weights/groundingdino_swint_ogc.pth` | 可选。只有切到 Grounding DINO 后端才需要 |
| `yolo26n.pt` | ~5 MB | （无） | **不需要**。历史遗留的自动下载文件，可忽略或删除 |

#### 下载方式一（推荐：在有网的机器上让 Ultralytics 自己拉）

```bash
cd backend
# 临时关掉离线开关，让 ultralytics 联网下载
YOLO_OFFLINE=0 ULTRALYTICS_OFFLINE=0 python - <<'PY'
from ultralytics import YOLOE
m = YOLOE("yoloe-11l-seg.pt")               # ← 下载 yoloe-11l-seg.pt 到当前目录
names = ["person", "car"]
m.set_classes(names, m.get_text_pe(names))  # ← 触发下载 mobileclip_blt.ts 到当前目录
print("downloaded ok")
PY

# 把基座权重归位；mobileclip_blt.ts 留在 backend/ 当前目录即可（见下方“放哪”说明）
mkdir -p models/yolo
mv -f yoloe-11l-seg.pt models/yolo/yoloe-11l-seg.pt
```

#### 下载方式二（手动直链：适合内网 / 离线分发）

```bash
cd backend
mkdir -p models/yolo
wget -O models/yolo/yoloe-11l-seg.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11l-seg.pt
wget -O mobileclip_blt.ts \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/mobileclip_blt.ts

# 可选：Grounding DINO 权重
mkdir -p models/groundingdino/weights
wget -O models/groundingdino/weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

> 换其它规格只需替换文件名：`yoloe-11s-seg.pt`（最小最快）/ `yoloe-11m-seg.pt`（均衡）/ `yoloe-11l-seg.pt`（当前用，精度最好）。

#### `mobileclip_blt.ts` 到底放哪（最容易踩的坑）

Ultralytics 按 `attempt_download_asset("mobileclip_blt.ts")` 加载文本编码器，路径是**相对“进程当前工作目录(CWD)”**解析的；而 `backend/start.sh` 会先 `cd backend` 再起 uvicorn，**CWD 就是 `backend/`**——所以把文件放在 **`backend/mobileclip_blt.ts`** 最稳妥。

⚠️ 仓库里 `backend/weights/mobileclip_blt.ts` 是一个指向 `/home/user/.config/Ultralytics/...`（旧机器用户名）的**软链接**，到新机器是**断链**。迁移后删掉或重建它，别让它挡住真实文件：

```bash
cd backend
rm -f weights/mobileclip_blt.ts                          # 删掉断掉的软链
# 如需在 weights/ 也保留一份，软链到真实文件即可：
# ln -sf "$(pwd)/mobileclip_blt.ts" weights/mobileclip_blt.ts
```

#### 改 `.env` 里的绝对路径

旧机器的绝对路径是 `/home/hmxh/workspace/sodv3/SOD/...`，新机器多半不同，必须改：

```env
# backend/.env —— 改成新机器上的真实绝对路径
DETECTION_MODEL=yolo_world      # 走 YOLOE 分支（项目里 yolo_world 后端实际加载的就是 YOLOE 权重）
DEVICE=cuda:0                   # 无 GPU 改 cpu
YOLO_WORLD_MODEL=/abs/path/to/SOD/backend/models/yolo/yoloe-11l-seg.pt
YOLOE_BASE_MODEL=/abs/path/to/SOD/backend/models/yolo/yoloe-11l-seg.pt
```

#### 放置自检

```bash
cd backend
python -c "from ultralytics import YOLOE; print('YOLOE import OK')"
ls -lh models/yolo/yoloe-11l-seg.pt mobileclip_blt.ts
```

### B. 数据集怎么配置

项目里有**两套**数据集体系，别混用：

#### B-1. 网页/接口的「类别 → 上传 → 标注 → 训练」工作流（应用自管，UUID 命名）

由后端自动维护，目录与 `.env` 一一对应：

```
backend/datasets/<category_id>/raw/<image_id>.jpg      # 上传的原图
backend/annotations/<category_id>/<image_id>.txt        # 可编辑的“工作标签”（YOLO 文本）
backend/datasets/<category_id>/yolo/<job_id>/           # 开始训练时自动“冻结”出的数据集
    images/{train,val}/...
    labels/{train,val}/...
    dataset.yaml                                         # 自动生成，names: {0: <类别名>}
```

```env
# backend/.env —— 一般保持相对路径即可（相对 backend/ 启动目录）
DATASETS_DIR=./datasets
ANNOTATIONS_DIR=./annotations
TRAIN_RUNS_DIR=./runs/train
```

- 这套是**单类别**：所有框统一写成 class `0`，`dataset.yaml` 的 `names` 用类别名；训练时还会自动把框转成分割多边形（因为基座是 `*-seg` 分割模型）。
- `train/val` 划分在“开始训练”时自动完成（`TRAIN_VAL_SPLIT` 默认 0.8）；当已标注图少于 `MIN_VAL_IMAGES`（默认 5）时，val 直接镜像 train，好让 Ultralytics 仍能算出 mAP。
- **迁移**：想保留已上传图片与标注，就把 `backend/datasets/`、`backend/annotations/` 整目录拷到新机器同样位置（它们**不在 git 里**）。

#### B-2. 命令行手工训练的数据集（标准 YOLO 布局，自己写 yaml）

完整说明见 [训练自定义 YOLOE 模型](#训练自定义-yoloe-模型) 与 `backend/YOLOWorld/TRAINING.md`，要点：

```
/your/dataset/
├── images/{train,val,test}/...   # 图片
└── labels/{train,val,test}/...   # 同名 .txt，每行: <class_id> <cx> <cy> <w> <h>（归一化 0–1）
```

```bash
cp backend/YOLOWorld/dataset.yaml.example backend/YOLOWorld/dataset.yaml
# 编辑 path(绝对路径) / train / val / names 后：
python backend/YOLOWorld/train_yoloe.py --data backend/YOLOWorld/dataset.yaml --epochs 100
```

#### 训练产物的迁移

已训练好的模型在 `backend/runs/train/<名字>/weights/best.pt`，**同样不在 git 里**。要在新机器继续用，二选一：整目录拷 `runs/` 过去，或只拷 `best.pt` 后在 `.env` 把 `YOLO_WORLD_MODEL` 指过去。

> ⚠️ **绝对路径警告**：数据库里 `yoloe_dataset_images.stored_path`、`yoloe_training_jobs.dataset_yaml`、`yoloe_trained_models.weights_path` 等列存的都是**绝对路径**。如果新机器的项目根路径与旧机器不同，这些历史记录会指向错误位置。最省事的做法是**把项目放到与旧机器相同的绝对路径**（即 `/home/<user>/workspace/sodv3/SOD`）；否则迁移后需用 SQL 批量改写这些列。

### C. PostgreSQL 怎么建库 / 建表

表结构由 SQLAlchemy 在后端启动时**自动创建**（`app/db/session.py` 的 `init_db()` → `Base.metadata.create_all`），**你不用手写任何建表 DDL**——但需要先把**数据库和角色**准备好。

**第 1 步：装好 PostgreSQL（≥13）并确认在运行**

```bash
sudo apt-get install -y postgresql       # Debian/Ubuntu 示例
pg_isready                               # 应输出 “... accepting connections”
```

**第 2 步：建角色 + 建库（推荐用一键脚本）**

```bash
bash backend/scripts/init_postgres.sh    # 需要 sudo（用 postgres 系统用户执行 DDL）
```

脚本做的事：建角色 `sod_app`（随机 24 位密码，或复用 `.env` 里已有的密码）→ 建库 `sod`（owner=sod_app）→ 授权 `public` schema → 用 TCP 密码登录验证一次 → 把 `DATABASE_URL` 写回 `backend/.env` 并 `chmod 600`。可用环境变量改名：

```bash
SOD_DB_NAME=mydb SOD_DB_USER=myrole bash backend/scripts/init_postgres.sh
```

> 不想用脚本时的手动等价做法：
> ```bash
> sudo -u postgres psql -c "CREATE ROLE sod_app LOGIN PASSWORD '改成你的密码';"
> sudo -u postgres psql -c "CREATE DATABASE sod OWNER sod_app;"
> sudo -u postgres psql -d sod -c "GRANT ALL ON SCHEMA public TO sod_app;"
> ```
> 再在 `backend/.env` 写（驱动后缀**必须**是 `+asyncpg`）：
> ```env
> DATABASE_URL=postgresql+asyncpg://sod_app:改成你的密码@localhost:5432/sod
> ```

**第 3 步：建表（启动后端即自动完成）**

```bash
cd backend && bash start.sh    # 或 uvicorn app.main:app --host 0.0.0.0 --port 8000
# 日志出现 “DB schema ensured” 即建表成功
```

启动时自动创建的 5 张表：

| 表名 | 用途 |
|---|---|
| `detection_tasks` | 视频检测任务历史（状态 / 进度 / 时间戳） |
| `yoloe_categories` | 训练类别（draft → annotating → ready → trained） |
| `yoloe_dataset_images` | 每个类别下上传的图片 + 标注状态 |
| `yoloe_training_jobs` | 每次训练任务（进度 / 指标 / best.pt 路径 / PID） |
| `yoloe_trained_models` | 已训练模型注册表（供前端模型列表选择） |

> ⚠️ `create_all` **只创建缺失的表，不会修改已存在的表**。新机器是空库 → 5 张表全建；若是迁移旧库且改过列定义，需用 Alembic 迁移（见 [表结构升级](#表结构升级)）。  
> 验证已建表：`sudo -u postgres psql -d sod -c '\dt'`。

**（可选）连数据一起迁**：想把旧机器的历史任务 / 训练记录也带过去：

```bash
# 旧机器导出
pg_dump -U sod_app -h localhost sod > sod_backup.sql
# 新机器（先用第 2 步建好空库 sod，再导入）
psql -U sod_app -h localhost -d sod < sod_backup.sql
```

（同样注意上面 B 节的“绝对路径警告”。）

### 迁移完成检查清单

- [ ] `git clone` 代码，`pip install -r backend/requirements.txt`（torch/torchvision 用 GPU 容器自带，不经 pip 装）
- [ ] `yoloe-11l-seg.pt` → `backend/models/yolo/`；`mobileclip_blt.ts` → `backend/`
- [ ] 删除 / 重建 `backend/weights/mobileclip_blt.ts` 断链
- [ ] 改 `backend/.env`：`YOLO_WORLD_MODEL` / `YOLOE_BASE_MODEL` 绝对路径、`DEVICE`、`DETECTION_MODEL=yolo_world`
- [ ] `bash backend/scripts/init_postgres.sh` 建库建角色（写好 `DATABASE_URL`）
- [ ] （可选）拷 `backend/datasets/ annotations/ runs/`，或 `pg_dump` 旧库连数据迁
- [ ] `bash backend/start.sh` → 看到 “DB schema ensured”，且检测能跑通
- [ ] 前端 `cd frontend && npm install && npm run dev`（默认 Vite 代理到 `localhost:8000`，本地开发无需额外配置）

---

## 大模型 (VLM) 配置

本项目集成了一个 **视觉-语言大模型（VLM）** 服务于两个用途：

1. **Prompt 标准化** — `app/services/prompt_normalizer.py`
   把用户的中文自然语言（如 `"菜地、水塘"`）转换成检测器友好的英文短语（如 `"vegetable field. water pond."`），并自动抽取目标类型（common/small/rare）、颜色过滤规则、视觉特征等元信息，用于后续的阈值调整与降权。

2. **检测结果语义复核** — `app/services/vlm_service.py` + `app/services/fusion_engine.py`
   对低/中置信度的 track，把检测裁剪图发送给 VLM，让大模型判断"图中目标是否就是用户想要的东西"，返回的得分会与检测器得分按 `VLM_WEIGHT` / `DINO_WEIGHT` 加权融合，驱动 track 的 `tentative → confirmed / rejected` 状态机。这一步显著降低小目标和罕见类的误检。

### 调用协议

两个模块统一使用 **OpenAI 兼容的 Chat Completions 协议**（`POST {VLM_API_BASE}/chat/completions`），因此**同一份代码既能调本地大模型，也能调远端云端 API**——只需要改 `.env` 中两个字段。

### 选项 A：本地大模型（默认，推荐）

默认指向本机 vLLM 启动的 **MiniCPM-V** 服务：

```env
# backend/.env
VLM_ENABLED=true
VLM_API_BASE=http://localhost:8010/v1
VLM_MODEL_NAME=MiniCPM-V-4_5
```

启动本地 vLLM 服务（示例，需提前下载好模型权重）：

```bash
# 安装 vLLM
pip install vllm

# 启动 OpenAI 兼容 server
python -m vllm.entrypoints.openai.api_server \
    --model openbmb/MiniCPM-V-2_6 \
    --served-model-name MiniCPM-V-4_5 \
    --port 8010 \
    --trust-remote-code \
    --dtype auto
```

也可用其它能提供 OpenAI 兼容接口的本地推理框架（Ollama、LM Studio、Xinference、SGLang 等），只要 `base_url` 和 `model name` 对得上即可。

### 选项 B：远端大模型 API

把 `VLM_API_BASE` 改成任意 OpenAI 兼容服务的根地址，把 `VLM_MODEL_NAME` 改成对应的模型名。常见示例（请用真实的 API key）：

| 提供商 | `VLM_API_BASE` | `VLM_MODEL_NAME` 示例 |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` / `gpt-4o-mini` |
| 阿里云通义 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-vl-max` / `qwen-vl-plus` |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-4v` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat`（仅纯文本 prompt 标准化场景） |
| 自建网关 | 你的网关地址 | 网关下注册的模型名 |

> 当前 `httpx.Client` 调用没有显式注入 `Authorization` header；如要走需要鉴权的远端 API，请在 `app/services/vlm_service.py` / `prompt_normalizer.py` 的 `httpx.Client(...)` 里加 `headers={"Authorization": f"Bearer {settings.VLM_API_KEY}"}`，并在 `app/core/config.py` 增加 `VLM_API_KEY` 字段。

### 关键配置项（`backend/.env` / `app/core/config.py`）

| 字段 | 默认值 | 说明 |
|---|---|---|
| `VLM_ENABLED` | `true` | 总开关。`false` 后流水线只跑检测器，不调用 VLM |
| `VLM_API_BASE` | `http://localhost:8010/v1` | OpenAI 兼容 endpoint 根地址 |
| `VLM_MODEL_NAME` | `MiniCPM-V-4_5` | `model` 字段填的模型名 |
| `VLM_MAX_CONCURRENT` | `1` | 同时向 VLM 发起的最大并发请求数 |
| `VLM_INTERVAL_TENTATIVE` | `3.0` | tentative track 的 VLM 复核最短间隔（秒） |
| `VLM_INTERVAL_CONFIRMED` | `8.0` | 已 confirmed track 的复核最短间隔（秒） |
| `VLM_SCORE_THRESHOLD` | `0.45` | VLM 输出 confidence 低于此值视为无效 |
| `VLM_WEIGHT` | `0.6` | 融合公式中 VLM 分数权重 |
| `DINO_WEIGHT` | `0.4` | 融合公式中检测器分数权重 |
| `VLM_CONFIRM_THRESHOLD` | `0.65` | VLM 高置信确认阈值 |
| `CONFIRM_THRESHOLD` | `0.65` | 融合得分 ≥ 此值则 track 转为 confirmed |
| `REJECT_THRESHOLD` | `0.35` | 融合得分 ≤ 此值则 track 转为 rejected |
| `CROP_PADDING_NORMAL` | `0.15` | 普通目标裁剪扩边比例 |
| `CROP_PADDING_SMALL` | `0.30` | 小目标裁剪扩边比例（面积占比 < `SMALL_OBJECT_AREA_RATIO`） |
| `SMALL_OBJECT_AREA_RATIO` | `0.01` | 小目标判定阈值（占画面比例） |

### 关闭 VLM（纯检测模式）

如果只想跑检测器、不调用大模型，把开关关掉即可：

```env
VLM_ENABLED=false
```

或在 API 请求层面单次关闭：

```json
POST /api/detect
{
  "video_id": "...",
  "prompt": "person . car",
  "enable_vlm": false
}
```

`POST /api/detect` 的 `enable_vlm` 字段会覆盖 `.env` 里的全局设置。

### 工作时序（简化）

```
检测器输出 box（low/mid confidence）
        │
        ▼
fusion_engine 判断该 track 是否需要复核
        │ 满足间隔条件
        ▼
vlm_service.crop_detection(frame, box)  ← 自适应扩边
        │
        ▼
POST {VLM_API_BASE}/chat/completions     ← 本地 or 远端
   payload: { image_b64, target_text }
        │
        ▼
解析返回 JSON: { is_target, matched_label, confidence, reason }
        │
        ▼
fusion_engine.apply_vlm_result(...)
   final_score = 0.4 * dino_score + 0.6 * vlm_score
   → confirmed / rejected / tentative
```

---

## 任务历史持久化 (PostgreSQL)

后端使用 **PostgreSQL** 持久化检测任务的元数据与状态，配合 SQLAlchemy 2.x 的 **async ORM** + `asyncpg` 驱动，全异步、不阻塞检测流水线。

### 设计要点

- **单表 `detection_tasks`**：一行 = 一个检测任务。表结构由 `app/db/models.py` 中的 `TaskRecord` 定义，启动时通过 `init_db()` 自动建表（MVP 不接 Alembic 迁移）。
- **懒连接 + 启动不崩**：引擎在 import 时创建，asyncpg 连接池在首次查询才建立——配置错误或数据库不可达**不会**阻止后端启动，错误在第一次查询时抛出并被持久层 `try/except` 捕获，**检测主路径不依赖 DB**。
- **DB 不可用时降级**：所有历史接口在 DB 失联时返回 `503 Service Unavailable`，前端可继续上传视频和跑检测，只是看不到历史；任务在内存中 `task_manager` 仍正常调度。
- **磁盘联动**：删除任务时同步清理 `RESULTS_DIR/{task_id}/`（帧 JPG + ZIP）；对 `uploads/{video_id}.*` 做引用计数，**仅当没有其它任务还在引用该视频时**才删上传文件。
- **活动任务保护**：处于 `pending / running / paused / packaging` 的任务无法被删除，必须先取消，避免误清理在跑任务。

### 表结构（`detection_tasks`）

| 列 | 类型 | 说明 |
|---|---|---|
| `task_id` | `VARCHAR(36)` PK | 任务 UUID |
| `video_id` | `VARCHAR(36)` | 关联上传视频 ID |
| `video_filename` | `VARCHAR(512)` | 原始文件名（可空） |
| `prompt` | `TEXT` | 用户的自然语言检测目标 |
| `status` | `VARCHAR(32)` *(indexed)* | `pending` / `running` / `paused` / `packaging` / `finished` / `failed` / `cancelled` |
| `progress` | `FLOAT` | 0.0 – 1.0 |
| `total_frames` / `processed_frames` | `INT` | 帧计数 |
| `error` | `TEXT` | 失败原因 |
| `zip_ready` | `BOOL` | ZIP 是否可下载 |
| `early_terminated` / `termination_reason` | `BOOL` / `TEXT` | 是否提前终止及原因 |
| `created_at` *(indexed)* / `updated_at` / `finished_at` | `TIMESTAMPTZ` | 时间戳（带时区） |

### 配置

`backend/.env`：

```env
# 默认值（仅供本地开发）；生产请用 init_postgres.sh 自动生成
DATABASE_URL=postgresql+asyncpg://sod_app:<password>@localhost:5432/sod

# 调试用：true 时打印每条 SQL
DATABASE_ECHO=false
```

> URL **必须**带 `+asyncpg` 驱动后缀，否则 SQLAlchemy 会回退到同步驱动。

### 一键初始化脚本

`backend/scripts/init_postgres.sh` 提供幂等的本地初始化：

```bash
sudo bash backend/scripts/init_postgres.sh
```

脚本做了什么：

1. 检查 `psql` / `openssl` / `python3` 可用且 PostgreSQL 在监听
2. 以本机 `postgres` OS 用户（peer auth）执行 DDL
3. 创建（或复用）角色 `sod_app`，密码：复用 `.env` 中已有 `DATABASE_URL` 的密码，否则随机生成 24 位
4. 创建（或确认）数据库 `sod`，owner 设为 `sod_app`
5. 授权 `public` schema
6. 通过 TCP + 密码登录验证一次（即应用真正走的路径）
7. 把 `DATABASE_URL` 写入或更新 `backend/.env`，并把 `.env` 权限收紧为 `600`
8. 输出脱敏后的连接串

可通过环境变量覆盖默认名：

```bash
SOD_DB_NAME=mydb SOD_DB_USER=myrole SOD_DB_HOST=127.0.0.1 \
    sudo bash backend/scripts/init_postgres.sh
```

### 历史相关 API

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/tasks?limit=100&offset=0&date=YYYY-MM-DD` | 分页列出历史任务（按 `created_at` 倒序）；可按日期过滤 |
| `GET` | `/api/task/{task_id}/frames` | 列出该任务保存的所有标注帧 JPG 文件名 |
| `DELETE` | `/api/task/{task_id}` | 删除单个任务：DB 行 + 结果目录 + ZIP + 上传视频（引用计数后） |
| `DELETE` | `/api/tasks` | **全量清空**：所有 DB 行 + `RESULTS_DIR` + `UPLOAD_DIR` |

> 后两个删除接口在有任务仍处于活动状态时会返回 `409 Conflict`。

### 表结构升级

当前 MVP 直接 `create_all`，没有迁移工具。Schema 演进时建议接入 **Alembic**：

```bash
pip install alembic
alembic init alembic
# 配置 sqlalchemy.url 指向 settings.DATABASE_URL
alembic revision --autogenerate -m "add new column"
alembic upgrade head
```

---

## 训练自定义 YOLOE 模型

支持基于自有数据集对 YOLOE 进行微调或从头训练。完整流程见 [`backend/YOLOWorld/TRAINING.md`](backend/YOLOWorld/TRAINING.md)，本节列出最关键的几步：

### 1. 准备数据集

YOLO 标准目录布局：

```
/your/dataset/
├── images/
│   ├── train/      # 训练集图片
│   ├── val/        # 验证集图片
│   └── test/       # 测试集图片（可选）
└── labels/
    ├── train/      # 训练集标注 (.txt，与图片同名)
    ├── val/        # 验证集标注
    └── test/       # 测试集标注（可选）
```

每个 `.txt` 一行一个目标，格式：`class_id cx cy w h`（中心坐标 + 宽高，均归一化到 0–1）。

### 2. 编写数据集 YAML

复制模板后改路径与类别：

```bash
cp backend/YOLOWorld/dataset.yaml.example backend/YOLOWorld/dataset.yaml
```

```yaml
path: /your/dataset          # 数据集根目录（绝对路径）
train: images/train
val: images/val
test: images/test            # 可选
names:
  0: tower
  1: insulator
  2: vegetation
```

### 3. 启动训练

```bash
# 单卡基础训练
python backend/YOLOWorld/train_yoloe.py \
    --data backend/YOLOWorld/dataset.yaml \
    --epochs 100 --imgsz 640 --batch 16 --device cuda:0

# 多卡训练
python backend/YOLOWorld/train_yoloe.py \
    --data backend/YOLOWorld/dataset.yaml \
    --device 0,1 --batch 32

# 断点续训
python backend/YOLOWorld/train_yoloe.py \
    --data backend/YOLOWorld/dataset.yaml --resume

# 冻结主干、轻量微调
python backend/YOLOWorld/train_yoloe.py \
    --data backend/YOLOWorld/dataset.yaml \
    --epochs 30 --freeze 10 --lr0 0.001
```

CLI 完整参数：`python backend/YOLOWorld/train_yoloe.py --help`

### 4. 训练产物 → 接回检测后端

训练结束后权重位于：

```
backend/runs/train/yoloe_exp/weights/
├── best.pt    # 验证集最优权重 ← 部署用这个
└── last.pt    # 最后一轮（用于续训）
```

把 `best.pt` 绝对路径写回 `backend/.env`：

```env
YOLO_WORLD_MODEL=/home/hmxh/workspace/sodv3/SOD/backend/runs/train/yoloe_exp/weights/best.pt
```

重启后端即可使用新模型，业务代码无需改动。

> 数据集结构细节、YOLO 标注格式、显存不足等常见问题，请参考 [`backend/YOLOWorld/TRAINING.md`](backend/YOLOWorld/TRAINING.md)。

---

## API 接口说明

### `POST /api/upload`

上传视频文件。

**请求**：`multipart/form-data`，字段 `file`

**响应**：
```json
{
  "video_id": "3f4a1b2c-...",
  "filename": "my_video.mp4",
  "size_bytes": 52428800,
  "duration_seconds": 30.5,
  "fps": 30.0,
  "total_frames": 915
}
```

---

### `POST /api/detect`

创建检测任务（立即返回，后台异步处理）。

**请求体**：
```json
{
  "video_id": "3f4a1b2c-...",
  "prompt": "帮我检测视频中的菜园",
  "detection_interval": 5
}
```

**响应** (`202 Accepted`)：
```json
{
  "task_id": "9a8b7c6d-...",
  "video_id": "3f4a1b2c-...",
  "prompt": "帮我检测视频中的菜园",
  "status": "pending"
}
```

---

### `GET /api/task/{task_id}`

查询任务状态（轮询备用接口）。

**响应**：
```json
{
  "task_id": "9a8b7c6d-...",
  "status": "running",
  "progress": 0.42,
  "total_frames": 915,
  "processed_frames": 385,
  "zip_ready": false,
  "results": [ ... ]
}
```

---

### `GET /api/stream/{task_id}`

Server-Sent Events 流，每处理完一帧推送一条消息。

**事件格式**：

```
data: {"event_type":"frame","task_id":"...","frame_result":{...},"progress":0.1,"processed_frames":10,"total_frames":100}

data: {"event_type":"done","task_id":"...","progress":1.0,"processed_frames":100,"total_frames":100}
```

---

### `GET /api/download/{task_id}`

下载 ZIP 结果包（任务状态为 `finished` 后可用）。

ZIP 内容：
```
results.zip
├── frame_000000_00-00-00-000.jpg
├── frame_000005_00-00-00-167.jpg
├── ...
├── results.json
└── results.csv
```

---

### `GET /api/tasks`

分页列出历史任务（按创建时间倒序）。

**查询参数**：

| 参数 | 类型 | 说明 |
|---|---|---|
| `limit` | int | 1–500，默认 100 |
| `offset` | int | 默认 0 |
| `date` | string | 可选，按 `YYYY-MM-DD` 过滤到某天（服务器本地时区） |

**响应**：

```json
[
  {
    "task_id": "9a8b7c6d-...",
    "video_id": "3f4a1b2c-...",
    "video_filename": "my_garden.mp4",
    "prompt": "帮我检测视频中的菜园",
    "status": "finished",
    "progress": 1.0,
    "processed_frames": 915,
    "total_frames": 915,
    "zip_ready": true,
    "early_terminated": false,
    "created_at": "2025-05-10T08:32:11+08:00",
    "finished_at": "2025-05-10T08:38:47+08:00"
  }
]
```

数据库不可达时返回 `503`。

---

### `GET /api/task/{task_id}/frames`

列出指定任务保存的所有标注帧文件名（升序）。前端用它配合 `/api/frame/{task_id}/{filename}` 拼接成可访问的图片 URL。

**响应**：

```json
["frame_000000_00-00-00-000.jpg", "frame_000005_00-00-00-167.jpg", "..."]
```

无对应结果目录时返回 `404`。

---

### `DELETE /api/task/{task_id}`

删除单个任务：DB 行 + `RESULTS_DIR/{task_id}/`（含 ZIP）+ 引用计数后的上传视频 + 内存中的 `task_manager` 记录。

- 任务处于 `pending / running / paused / packaging` 时返回 `409 Conflict`
- 成功返回 `204 No Content`

---

### `DELETE /api/tasks`

**全量清空**：所有 DB 行 + `RESULTS_DIR` 与 `UPLOAD_DIR` 下全部内容 + 内存全部状态。仍有活动任务时返回 `409`。

```bash
curl -X DELETE http://localhost:8000/api/tasks
```

---

## 示例请求与返回

### cURL 示例

```bash
# 1. 上传视频
VIDEO_ID=$(curl -s -X POST http://localhost:8000/api/upload \
  -F "file=@my_garden.mp4" | jq -r '.video_id')

# 2. 开始检测
TASK_ID=$(curl -s -X POST http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -d "{\"video_id\":\"$VIDEO_ID\",\"prompt\":\"帮我检测视频中的菜园\"}" \
  | jq -r '.task_id')

# 3. 订阅 SSE 流（实时查看结果）
curl -N http://localhost:8000/api/stream/$TASK_ID

# 4. 任务完成后下载 ZIP
curl -OJ http://localhost:8000/api/download/$TASK_ID
```

### FrameResult 示例

```json
{
  "frame_id": 125,
  "timestamp": "00:00:05.000",
  "timestamp_seconds": 5.0,
  "detections": [
    {
      "track_id": 3,
      "label": "临水菜园",
      "score": 0.847,
      "bbox": { "x1": 120.5, "y1": 80.2, "x2": 640.1, "y2": 350.7 }
    }
  ],
  "image_filename": "frame_000125_00-00-05-000.jpg",
  "image_b64": "/9j/4AAQSkZJRgAB..."
}
```

---

## 检测可视化效果

参考 `example_pro.png`，系统绘制效果：

- ✅ 高对比度矩形框（颜色随 `track_id` 轮换）
- ✅ 框上方：半透明标签背景 + 白色文字（标签 + 置信度）
- ✅ 右下角：时间戳水印（`HH:MM:SS.mmm`）
- ✅ 不同目标使用不同颜色，易于区分

---

## 扩展：多 GPU

当前设计：单 GPU，通过 `asyncio.Semaphore(MAX_CONCURRENT_TASKS)` 控制并发任务数。

### 方案 A：每 GPU 一个 Worker 进程

```bash
# GPU 0
DEVICE=cuda:0 uvicorn app.main:app --port 8000 &

# GPU 1
DEVICE=cuda:1 uvicorn app.main:app --port 8001 &

# Nginx 负载均衡
upstream backends {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
}
```

### 方案 B：Ray 任务调度

```python
import ray
ray.init()

@ray.remote(num_gpus=1)
def detect_on_gpu(video_path, prompt, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # ... run pipeline
```

### 方案 C：Celery + 多 GPU Worker

见下一节。

---

## 扩展：提升吞吐

| 优化手段 | 预期收益 |
|---------|---------|
| 增大 `DETECTION_INTERVAL` (5→10) | 推理帧数减半，速度提升约 2× |
| 使用 `Florence-2-base` 代替 `-large` | 推理速度提升约 1.5~2×，精度稍降 |
| 使用 TensorRT 量化 Grounding DINO | 推理速度提升约 2~3× |
| 使用 `video/io` 异步解码（NVDEC） | 解码不占 CPU |
| 批量推理（batch > 1） | 对检测帧批量 forward |
| 半精度（fp16）推理 | 速度提升约 1.5×，显存占用减半 |

---

## 扩展：接入 Redis / Celery

1. **安装依赖**

```bash
pip install celery[redis] redis
```

2. **配置**（`.env`）

```env
REDIS_URL=redis://localhost:6379/0
```

3. **定义 Celery 任务**

```python
# app/tasks/celery_app.py
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "video_detection",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

@celery_app.task(bind=True)
def run_detection_task(self, task_id, video_path, prompt):
    # 调用 pipeline._sync_pipeline(...)
    ...
```

4. **启动 Worker**

```bash
# GPU 0
DEVICE=cuda:0 celery -A app.tasks.celery_app worker --concurrency=1 -n worker0@%h

# GPU 1
DEVICE=cuda:1 celery -A app.tasks.celery_app worker --concurrency=1 -n worker1@%h
```

5. **修改 API 触发点**

在 `detect.py` 中将 `asyncio.create_task(...)` 替换为：

```python
run_detection_task.delay(task_id, str(video_path), prompt)
```

---

## 测试方法

### 后端单元测试

```bash
cd backend
pytest tests/ -v
```

### 手动端到端测试

```bash
# 准备一个测试视频
wget -O test.mp4 "https://your-test-video-url"

# 运行测试脚本
python -c "
import requests

# 上传
with open('test.mp4', 'rb') as f:
    r = requests.post('http://localhost:8000/api/upload', files={'file': f})
video_id = r.json()['video_id']
print('video_id:', video_id)

# 开始检测
r = requests.post('http://localhost:8000/api/detect', json={
    'video_id': video_id,
    'prompt': '帮我检测视频中的菜园',
})
task_id = r.json()['task_id']
print('task_id:', task_id)

# 轮询状态
import time
while True:
    r = requests.get(f'http://localhost:8000/api/task/{task_id}')
    d = r.json()
    print(f'status={d[\"status\"]} progress={d[\"progress\"]:.0%}')
    if d['status'] in ('finished', 'failed'):
        break
    time.sleep(2)
"
```

---

## 依赖版本要求

| 依赖 | 最低版本 | 用途 |
|------|---------|------|
| Python | 3.10 | — |
| PyTorch | 2.1.0 | 推理 / 训练 |
| CUDA | 11.8 | GPU 加速 |
| FastAPI | 0.111.0 | 后端框架 |
| **PostgreSQL** | **13+** | **任务历史数据库** |
| **SQLAlchemy** | **2.0** | **异步 ORM（async session + Mapped 列）** |
| **asyncpg** | **0.29** | **PostgreSQL 异步驱动** |
| **ultralytics** | **8.3.0** | **YOLOE / YOLO-World 推理与训练** |
| **sahi** | **0.11.18** | **小目标切片推理** |
| **httpx** | **0.27.0** | **调用 VLM（本地或远端 OpenAI 兼容 API）** |
| transformers | 4.40.0 | Grounding DINO / Florence-2（可选） |
| groundingdino-py | latest | Grounding DINO（可选） |

---

*构建于开源模型之上，感谢 Ultralytics（YOLOE / YOLO-World）、IDEA Research（Grounding DINO）、Microsoft（Florence-2）、OpenBMB（MiniCPM-V）、ByteTrack 与 SAHI 团队。*
