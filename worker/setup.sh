#!/bin/bash
# Runs on every boot of a fresh worker (spot replacements start from zero). ~4-6 min to READY with a warm S3
# model cache. Env comes from /etc/media-worker.env (written by the launch template user data).
set -euxo pipefail
set -a; . /etc/media-worker.env; set +a
BOOT=/opt/media-bootstrap

TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
md() { curl -s -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/$1"; }
IID=$(md instance-id)
ITYPE=$(md instance-type)
LIFECYCLE=$(md instance-life-cycle)

status() {
  printf '{"instance_id":"%s","instance_type":"%s","lifecycle":"%s","phase":"%s","updated":"%s"}\n' \
    "$IID" "$ITYPE" "$LIFECYCLE" "$1" "$(date -u +%FT%TZ)" \
    | aws s3 cp - "s3://$MEDIA_BUCKET/status/workers/$IID.json" --content-type application/json || true
}
on_fail() {
  status "FAILED during setup (log: s3://$MEDIA_BUCKET/status/workers/$IID-setup.log)"
  aws s3 cp /var/log/media-setup.log "s3://$MEDIA_BUCKET/status/workers/$IID-setup.log" || true
  # Leave 30 min for a look over SSM, then stop paying for a broken box.
  sleep 1800
  aws autoscaling terminate-instance-in-auto-scaling-group --instance-id "$IID" --should-decrement-desired-capacity || true
}
trap on_fail ERR
status "booting"

# The Deep Learning AMI RAID0-mounts local NVMe at /opt/dlami/nvme: ~10x faster than EBS for 15-30 GB model loads.
for _ in $(seq 1 30); do mountpoint -q /opt/dlami/nvme && break; sleep 2; done
DATA=/opt/dlami/nvme/media
mountpoint -q /opt/dlami/nvme || DATA=/opt/media-data
mkdir -p "$DATA"/{input,output,hf-stage,work}
ln -sfn "$DATA" /opt/media

aws configure set default.s3.preferred_transfer_client crt
aws configure set default.s3.target_bandwidth 40Gb/s
command -v git >/dev/null || { apt-get update -y -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends git; }

status "installing python env and fetching models"
export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
UV=/root/.local/bin/uv
cd /opt/media
# MEDIA_COMFYUI_REF may be a branch, a tag or a commit sha, so fetch it rather than `clone --branch`.
if [ ! -d ComfyUI ]; then
  git init -q ComfyUI && git -C ComfyUI remote add origin https://github.com/comfyanonymous/ComfyUI.git
  git -C ComfyUI fetch -q --depth 1 origin "${MEDIA_COMFYUI_REF:-master}" && git -C ComfyUI checkout -q FETCH_HEAD
fi
git -C ComfyUI rev-parse HEAD > /opt/media/comfyui_commit.txt
$UV venv --python 3.12 /opt/media/venv
export VIRTUAL_ENV=/opt/media/venv
$UV pip install "huggingface_hub[hf_xet]>=0.34" boto3 imageio-ffmpeg

# Weights download while torch and the ComfyUI requirements install.
/opt/media/venv/bin/python $BOOT/fetch_models.py --backends "$MEDIA_BACKENDS" \
  --models /opt/media/ComfyUI/models --stage /opt/media/hf-stage > /var/log/media-models.log 2>&1 &
MODELS_PID=$!
# int8/fp8 kernels want CUDA 13 wheels (driver >= 580); older drivers fall back to cu128.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
CU=cu130; [ "$DRV" -lt 580 ] && CU=cu128
$UV pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CU"
$UV pip install -r ComfyUI/requirements.txt

# MiniMax's prompt-writing guides for the enhance_prompt knob (MiniMax H3 Community License, so fetched, not vendored).
if [[ ",$MEDIA_BACKENDS," == *",h3,"* ]]; then
  mkdir -p /opt/media/prompts
  for f in base-en.txt ref-en.txt; do
    curl -sfL "https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/${MEDIA_H3_GUIDE_REF:-main}/skills/h3-prompt-writing/references/$f" \
      -o "/opt/media/prompts/$f" || echo "warning: could not fetch H3 prompt guide $f"
  done
fi
# MiniMax-H3-Image-PE's expansion instructions (same reason: MiniMax-H3-Image Community License).
if [[ ",$MEDIA_BACKENDS," == *",h3-image,"* ]]; then
  mkdir -p /opt/media/prompts
  for f in t2i/system_prompt.txt t2i/task_prompt.txt edit/task_prompt.txt; do
    curl -sfL "https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE/resolve/${MEDIA_H3_IMAGE_PE_REF:-main}/prompt/$f" \
      -o "/opt/media/prompts/h3image_${f//\//_}" || echo "warning: could not fetch H3-Image PE prompt $f"
  done
fi

# H3 Image EasyFix (Apache-2.0): the Steering + Re-Noise Sampler nodes behind h3-image's tuning.easyfix. Pinned to the
# commit that was read and tested (2026-11-19); only its code and its one data table are kept.
if [[ ",$MEDIA_BACKENDS," == *",h3-image,"* ]]; then
  EF=/opt/media/ComfyUI/custom_nodes/ComfyUI-H3-Image-EasyFix
  if [ ! -d "$EF" ]; then
    git init -q "$EF" && git -C "$EF" remote add origin https://github.com/PedroMarinhoDev/ComfyUI-H3-Image-EasyFix.git
    git -C "$EF" fetch -q --depth 1 origin "${MEDIA_H3_IMAGE_EASYFIX_REF:-a73c53594cd8c362736b69c24814e948a2730fcf}"
    git -C "$EF" checkout -q FETCH_HEAD
    rm -rf "$EF/assets" "$EF/tests" "$EF/workflows"
  fi
fi

status "fetching models"
wait $MODELS_PID || { cat /var/log/media-models.log; false; }
cat /var/log/media-models.log

cat >/etc/systemd/system/comfyui.service <<EOF
[Unit]
Description=ComfyUI (open-media-mcp)
After=network-online.target

[Service]
WorkingDirectory=/opt/media/ComfyUI
ExecStart=/opt/media/venv/bin/python main.py --listen 127.0.0.1 --port 8188 --input-directory /opt/media/input --output-directory /opt/media/output --disable-metadata
Restart=always
RestartSec=5
Environment=HOME=/root

[Install]
WantedBy=multi-user.target
EOF

cat >/etc/systemd/system/media-worker.service <<EOF
[Unit]
Description=open-media-mcp job worker (SQS -> ComfyUI -> S3)
After=comfyui.service
Requires=comfyui.service

[Service]
EnvironmentFile=/etc/media-worker.env
Environment=HOME=/root
ExecStart=/opt/media/venv/bin/python $BOOT/worker.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Rendered files live in S3; keep the local output dir from filling the disk.
cat >/etc/cron.hourly/media-clean <<'EOF'
#!/bin/sh
find /opt/media/output /opt/media/input -type f -mmin +120 -delete
EOF
chmod +x /etc/cron.hourly/media-clean

systemctl daemon-reload
systemctl enable --now comfyui.service
status "starting comfyui"
for _ in $(seq 1 120); do curl -sf http://127.0.0.1:8188/system_stats >/dev/null && break; sleep 5; done
curl -sf http://127.0.0.1:8188/system_stats > /opt/media/system_stats.json
aws s3 cp /opt/media/system_stats.json "s3://$MEDIA_BUCKET/status/workers/$IID-system_stats.json" --only-show-errors || true
systemctl enable --now media-worker.service
aws s3 cp /var/log/media-setup.log "s3://$MEDIA_BUCKET/status/workers/$IID-setup.log" --only-show-errors || true
