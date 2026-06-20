#!/usr/bin/env bash
# ============================================================================
# SOD 离线部署包构建脚本（在“有环境的源机”上运行，本例即开发机）
# ----------------------------------------------------------------------------
# 产出一个自包含的离线部署总包，可拷贝到无网络的目标服务器解包安装。
# 包含：conda 运行环境(conda-pack) + 应用代码 + 模型权重 + Ultralytics 字体 +
#       前端构建产物 + 部署脚本与文档。不含：上传视频、数据集、检测结果、训练产物等大体积用户数据。
#
# 用法：  bash deploy/package.sh
# 输出：  dist/sod-offline-bundle/            （展开的总包目录）
#         dist/sod-offline-bundle.tar          （单文件总包，便于 scp 传输）
# ============================================================================
set -euo pipefail

# ── 路径配置 ────────────────────────────────────────────────────────────────
SRC="/home/hmxh/workspace/sodv3/SOD"                 # 项目源目录
ENV_NAME="SOD"                                       # conda 环境名
CONDA="/home/hmxh/miniconda3/bin/conda-pack"         # conda-pack 可执行
ULTRA_CFG="$HOME/.config/Ultralytics"                # ultralytics 配置（字体）
OUT="/home/hmxh/workspace/sodv3/dist/sod-offline-bundle"
DEPLOY="$SRC/deploy"

say(){ echo -e "\n\033[1;36m▶ $*\033[0m"; }
ok(){  echo -e "  \033[32m✓\033[0m $*"; }

mkdir -p "$OUT"/{conda,app,ultralytics}

# ── 1. 打包 conda 运行环境（含 torch+cu130、ultralytics、所有依赖）──────────
say "1/5 conda-pack 运行环境 ($ENV_NAME)"
if [ -f "$OUT/conda/SOD_env.tar.gz" ]; then
  ok "已存在 conda/SOD_env.tar.gz（$(du -h "$OUT/conda/SOD_env.tar.gz"|cut -f1)），跳过重打包"
else
  "$CONDA" -n "$ENV_NAME" -o "$OUT/conda/SOD_env.tar.gz" --format tar.gz --n-threads -1 --compress-level 4
  ok "完成 → $(du -h "$OUT/conda/SOD_env.tar.gz"|cut -f1)"
fi

# ── 2. 打包应用代码 + 模型权重（排除用户数据 / 缓存 / 机密）────────────────
say "2/5 应用代码 + 模型权重"
tar -czf "$OUT/app/SOD-app.tar.gz" -C "$SRC" \
  --exclude='./backend/uploads'      --exclude='./backend/results' \
  --exclude='./backend/runs'         --exclude='./backend/datasets' \
  --exclude='./backend/annotations'  --exclude='./backend/logs' \
  --exclude='./uploads' --exclude='./results' --exclude='./logs' \
  --exclude='./mobileclip_blt.ts' \
  --exclude='./.git' --exclude='./.claude' --exclude='./backend/.claude' \
  --exclude='./backend/.pytest_cache' \
  --exclude='./deploy' \
  --exclude='*/__pycache__' --exclude='*.pyc' \
  --exclude='./frontend/node_modules/.cache' --exclude='./frontend/.vite' \
  .
ok "完成 → $(du -h "$OUT/app/SOD-app.tar.gz"|cut -f1)（含 backend/ frontend/ models/ mobileclip_blt.ts 与原始 .env，已排除用户数据）"

# ── 3. 打包 Ultralytics 字体 / 配置（离线标注必需）──────────────────────────
say "3/5 Ultralytics 字体与配置"
if [ -d "$ULTRA_CFG" ]; then
  tar -czf "$OUT/ultralytics/Ultralytics.tar.gz" -C "$ULTRA_CFG" \
    Arial.ttf Arial.Unicode.ttf settings.json persistent_cache.json 2>/dev/null \
    || tar -czf "$OUT/ultralytics/Ultralytics.tar.gz" -C "$ULTRA_CFG" .
  ok "完成 → $(du -h "$OUT/ultralytics/Ultralytics.tar.gz"|cut -f1)"
else
  echo "  ! 未找到 $ULTRA_CFG，跳过（目标机首次标注可能因缺字体失败）"
fi

# ── 4. 复制部署脚本与文档到总包根 ───────────────────────────────────────────
say "4/5 部署脚本与文档"
cp -f "$DEPLOY"/{install.sh,verify.sh,run_backend.sh,serve_frontend.py,env.template,nginx-sod.conf,sod-backend.service} "$OUT/"
[ -f "$DEPLOY/部署文档.md" ] && cp -f "$DEPLOY/部署文档.md" "$OUT/"
[ -f "$DEPLOY/快速部署.md" ] && cp -f "$DEPLOY/快速部署.md" "$OUT/"
[ -f "$DEPLOY/环境重建或复用.md" ] && cp -f "$DEPLOY/环境重建或复用.md" "$OUT/"
chmod +x "$OUT"/{install.sh,verify.sh,run_backend.sh,serve_frontend.py}
ok "已复制 install.sh / verify.sh / run_backend.sh / serve_frontend.py / 配置模板 / 部署文档.md"

# ── 5. 校验和 + 单文件总包 ──────────────────────────────────────────────────
say "5/5 生成校验和与单文件总包"
( cd "$OUT" && find conda app ultralytics -type f -exec sha256sum {} + > checksums.sha256 )
ok "校验和 → checksums.sha256"
( cd "$(dirname "$OUT")" && tar -cf sod-offline-bundle.tar "$(basename "$OUT")" )
ok "单文件总包 → $(dirname "$OUT")/sod-offline-bundle.tar（$(du -h "$(dirname "$OUT")/sod-offline-bundle.tar"|cut -f1)）"

echo ""
echo "════════════════════════════════════════════════════════════"
echo " ✅ 构建完成。总包内容："
( cd "$OUT" && du -ah --max-depth=2 . | sort -rh | head -20 )
echo ""
echo " 传输到目标机后：tar -xf sod-offline-bundle.tar && cd sod-offline-bundle && bash install.sh"
echo "════════════════════════════════════════════════════════════"
