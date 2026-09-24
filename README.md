# Ray Serve + PANNs：队列驱动的异步推理服务

这个项目参考 Ray Serve 的 async inference / video-indexing 设计，已经从“Serve 请求 -> Scheduler -> ObjectRef 轮询”改成了 **Producer + Task Consumer + Model Deployment** 三段式架构：

```text
HTTP client
    │ POST /v1/tasks（立即返回 task_id）
    ▼
PannsIngress（轻量 Producer）
    │ enqueue_task_sync
    ▼
Redis broker/result backend
    │ at-least-once、重试、失败队列
    ▼
PannsConsumer（@task_consumer，CPU）
    │ 音频读取/预处理 + encoder handle RPC
    ▼
PannsEncoder（标准 Serve Deployment，GPU/CPU）
    │ 常驻加载 Cnn14
    ▼
Redis result backend
    │ GET /v1/tasks/{task_id}
    ▼
HTTP client
```

## 为什么改成 Consumer Deployment

长推理不再占用 HTTP 连接，也不再需要自定义调度器把 ObjectRef 和任务状态放在内存里：

- `PannsIngress` 只校验参数、投递任务并返回 ID；
- `PannsConsumer` 使用 `@task_consumer` / `@task_handler` 从消息队列消费任务；
- Celery adapter 负责结果状态、至少一次投递、自动重试和 dead-letter queue；
- `PannsEncoder` 是独立的 Serve 模型副本，模型只加载一次；
- Consumer 对 Encoder 使用 Ray deployment handle，waveform 通过 Ray 对象存储/RPC 传递，不经过第二个 HTTP 服务；
- Consumer 可以按队列积压做横向扩展，Encoder 可以按 GPU 资源独立扩展。

自定义 `ray_panns/scheduler.py` 已移除；任务生命周期由 Redis/Celery adapter 管理，应用 graph 只包含 Producer、Consumer 和 Encoder。

## 本地运行

需要一个 Redis 作为 broker 和结果后端。最简单的方式：

```bash
docker run --rm --name ray-panns-redis -p 6379:6379 redis:7-alpine
```

然后在另一个终端：

```bash
pip install -e .
export PANNS_MOCK=1
export PANNS_DEVICE=cpu
export PANNS_BROKER_URL=redis://127.0.0.1:6379/0
export PANNS_BACKEND_URL=redis://127.0.0.1:6379/1
serve run main:app
```

提交任务和轮询结果：

```bash
curl -X POST http://127.0.0.1:8000/v1/tasks \
  -H 'content-type: application/json' \
  -d '{"audio_path":"/data/example.wav","top_k":5}'
# {"task_id":"...","status":"PENDING", ...}

curl http://127.0.0.1:8000/v1/tasks/<task_id>
# {"task_id":"...","status":"SUCCESS","result":{...}}
```

Mock 模式不读取 `audio_path`，只用于验证队列、Consumer、Serve handle 和结果轮询链路。真实推理需要：

```bash
export PANNS_MOCK=0
export PANNS_DEVICE=cuda
export PANNS_CHECKPOINT=/models/Cnn14_mAP=0.431.pth
```

真实环境不要允许客户端直接提交任意本机路径。应先上传到对象存储/共享文件系统，再传入受控 URI 或白名单路径，并增加文件大小、时长、格式和租户隔离校验。

## 配置

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `PANNS_BROKER_URL` | `redis://127.0.0.1:6379/0` | 任务消息 broker |
| `PANNS_BACKEND_URL` | 与 broker 相同 | 任务结果 backend |
| `PANNS_QUEUE_NAME` | `panns-inference` | 主消费队列 |
| `PANNS_MAX_RETRIES` | `3` | 失败自动重试次数 |
| `PANNS_FAILED_QUEUE` | `<queue>.failed` | 重试耗尽后的失败队列 |
| `PANNS_UNPROCESSABLE_QUEUE` | `<queue>.unprocessable` | 无法反序列化/找不到 handler 的队列 |
| `PANNS_CHECKPOINT` | 无 | Cnn14 checkpoint 路径 |
| `PANNS_DEVICE` | `cuda` | `cuda` 或 `cpu` |
| `PANNS_MOCK` | `0` | 本地 mock 开关 |

## Kubernetes / KubeRay

OrbStack 本地示例包含一个开发 Redis：

```bash
docker build -f Dockerfile.head -t ray-panns-head:k8s-v2 .
docker build -f Dockerfile.worker -t ray-panns-worker:k8s-v4 .
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/rayservice-orbstack.yaml
kubectl get rayservice ray-panns
kubectl port-forward svc/ray-panns-serve-svc 8001:8000
```

`k8s/rayservice-orbstack.yaml` 当前是 CPU mock 配置。生产 GPU 集群需把 Worker 镜像换成带 CUDA/PANNs 的镜像，设置 `PANNS_DEVICE=cuda`、挂载 checkpoint，并在 `PannsEncoder` 的 Serve 配置中声明 `num_gpus: 1`。Redis 也应替换成带持久化、认证、TLS 和备份的托管服务。

RayService 中的三个 deployment 职责是：

- `PannsIngress`：HTTP producer，1 个副本；
- `PannsConsumer`：队列消费者，默认 1 个副本，可扩到 4 个副本；
- `PannsEncoder`：模型副本，默认 1 个副本，单副本最多处理一个推理请求。

## 代码结构

```text
ray_panns/
├── config.py    # Celery/Redis TaskProcessorConfig 和 task name
├── service.py   # PannsIngress：提交任务、查询结果
├── workers.py   # PannsConsumer + PannsEncoder
└── panns.py     # CPU 音频读取、PANNs 预处理与模型适配
```

任务处理函数必须保持同步：只有 handler 返回后，Celery 才会确认消息；这样失败或 Worker 丢失时任务才能安全重投。业务写入外部存储时还要按业务 ID 做幂等，避免 at-least-once 重投造成重复副作用。
