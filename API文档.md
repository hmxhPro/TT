# SOD 一键启动指南

`start_all.sh` 按依赖顺序启动整个 SOD 栈：vLLM（视觉模型）→ FastAPI 后端 → Vite 前端。脚本可放在任意位置（已通过绝对 `PROJECT_ROOT` 解耦），推荐放在 `~/start_all.sh` 实现登录后一行启动。

---

## 1. 启动的服务

| # | 服务      | 端口   | 健康检查路径           | 日志文件                  |
|---|-----------|--------|------------------------|---------------------------|
| 1 | vLLM      | `8010` | `/v1/models`           | `logs/vllm.log`           |
| 2 | 后端      | `8000` | `/openapi.json`        | `logs/backend.log`        |
| 3 | 前端 Vite | `5173`（Vite 自动让端口） | 启动日志中的 `http://localhost:…` | `logs/frontend.log` |

> 端口冲突说明：README 里 vLLM 默认用 `8000`，会和后端撞，本脚本将 vLLM 固定到 `8010`，与后端 `app/core/config.py` 中的 `VLM_API_BASE` 保持一致。

---

## 2. 前置依赖（只需准备一次）

脚本启动时会做存在性检查，缺哪项会立刻 `die` 报错。请确认以下都已就绪：

- **vLLM 虚拟环境**：`/home/user/Small_object_detection/local-deploy/.venv-minicpm/bin/activate`
- **vLLM 模型目录**：`/home/user/Small_object_detection/local-deploy/models/MiniCPM-V-4_5`
- **conda 入口脚本**：`/home/user/home/enter/etc/profile.d/conda.sh`
- **后端 conda 环境**：`sodv2`（包含 loguru / sqlalchemy / asyncpg 等）
- **后端启动脚本**：`backend/start.sh`
- **前端依赖**：`frontend/node_modules`（首次需要 `cd frontend && npm install`）
- **系统命令**：`curl`、`ss`（iproute2）

GPU 资源：vLLM 默认占用 `CUDA_VISIBLE_DEVICES=0,1` 两张卡，`gpu-memory-utilization=0.90`，`tensor-parallel-size=2`。启动前请确保两张 GPU 空闲。

---

## 3. 启动

```bash
bash ~/start_all.sh
```

启动流程：

1. **vLLM**：若 `:8010` 已经有可用 vLLM，则**复用**（Ctrl+C 时也不会停掉它）；否则在子 shell 中 `source` venv 并 `vllm serve`，最长等待 10 分钟（首次加载模型耗时 3–10 min）。每 30 秒打印一次最新日志行。
2. **后端**：在 `sodv2` conda 环境内执行 `backend/start.sh`，轮询 `/openapi.json`，30 秒超时。
3. **前端**：`npm run dev`，从日志中抓取 Vite 实际端口；30 秒内未抓到 URL 会提示去看日志。

启动完成会打印汇总 banner，列出三个服务地址和日志路径。

---

## 4. 查看日志

```bash
tail -f logs/vllm.log
tail -f logs/backend.log
tail -f logs/frontend.log
```

每次启动会**清空旧日志**（`: > log`），如需保留请提前转存。

---

## 5. 停止

在脚本所在终端按 **Ctrl+C** 即可。`cleanup` trap 会按"前端 → 后端 → vLLM"的反向顺序发送 `SIGTERM`，2 秒后对仍存活的进程发 `SIGKILL`。

特别注意：

- 若启动时 vLLM 是被**复用**的（`VLLM_REUSE=1`），Ctrl+C **不会**关闭 vLLM，需手动处理。
- 任一服务自行退出，脚本会感知到并触发 cleanup（`wait -n`）——即"一损俱损"语义。

---

## 6. 常见问题排错

| 现象 | 原因 / 排查 |
|------|-------------|
| `Port 8000 already in use (backend)` | 已有后端在跑；用 `ss -ltnp 'sport = :8000'` 查并 kill。 |
| `Port 8010 in use by something other than vLLM` | 8010 被非 vLLM 占用；释放端口或换 `VLLM_PORT`。 |
| `vLLM crashed` | 看 `logs/vllm.log` 末尾，常见为 CUDA OOM、模型路径错、tokenizer 不兼容。 |
| `vLLM did not become ready in 10 min` | 模型首次加载慢或卡死；看日志最后一行进度，或检查 GPU 是否被别的进程占用。 |
| `backend died during startup` | 多半是 conda env 未激活成功，或后端依赖缺失；看 `logs/backend.log`。 |
| `frontend died` | 多半是 `node_modules` 未装或 Vite 配置报错；看 `logs/frontend.log`。 |
| 前端 URL 拿不到 | Vite 端口可能从 5173 跳到 5174+；直接 `grep -oE 'http://localhost:[0-9]+' logs/frontend.log`。 |
| `conda env 'sodv2' not found` | 检查 `CONDA_SH` 路径是否正确，以及 `conda env list` 里有没有 `sodv2`。 |

---

## 7. 端口与路径速查

```text
PROJECT_ROOT  = /home/user/Small_object_detection/sodv3/SOD
BACKEND_DIR   = $PROJECT_ROOT/backend
FRONTEND_DIR  = $PROJECT_ROOT/frontend
LOG_DIR       = $PROJECT_ROOT/logs

VLLM_VENV         = /home/user/Small_object_detection/local-deploy/.venv-minicpm/bin/activate
VLLM_MODEL        = /home/user/Small_object_detection/local-deploy/models/MiniCPM-V-4_5
CONDA_SH          = /home/user/home/enter/etc/profile.d/conda.sh
BACKEND_CONDA_ENV = sodv2

VLLM_PORT     = 8010
BACKEND_PORT  = 8000
FRONTEND_PORT = 5173 (Vite 自动)
```

---

## 8. 其他说明

- 脚本使用 `set -euo pipefail` + `set -m`（job control），每个 `&` 子进程独立进程组，便于按组 kill。
- 不建议改造为 systemd 开机自启：vLLM 会持续吃满显存、Vite 是开发服务器、依赖路径在 systemd 最小环境下易踩坑。如需服务化，请先将前端 `npm run build` 出静态产物再托管。
