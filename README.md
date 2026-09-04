# 构建你的第一个 Ray 应用：PANNs 音频标签服务

这是一个最小但完整的 Ray Serve + PANNs 示例：Serve 入口只负责 HTTP 校验和任务转发，Scheduler 负责任务状态与异步 object ref，GPU actor 常驻加载 PANNs Cnn14 模型并串行推理。

## 运行

```bash
pip install -e .
export PANNS_CHECKPOINT=/models/Cnn14_mAP=0.431.pth
serve run main:app
curl http://127.0.0.1:8000/healthz
curl -X POST http://127.0.0.1:8000/v1/tasks \
  -H 'content-type: application/json' \
  -d '{"audio_path":"/data/example.wav","top_k":5}'
curl http://127.0.0.1:8000/v1/tasks/<task_id>
```

生产环境应将音频先放入共享存储，并通过白名单路径或对象存储签名 URL 传入；不要让服务进程直接读取任意用户路径。PANNs checkpoint 也应通过镜像或制品仓库注入，而不是提交到 Git。

## 结构

`ray_panns/service.py` 是无状态 Serve 入口；`scheduler.py` 是单写者任务协调器；`workers.py` 以 `num_gpus=1` 声明唯一 GPU 消费者；`panns.py` 将 `panns-inference` 的 Cnn14 推理结果整理为稳定 JSON。模型依赖延迟导入，便于在无 GPU 的机器上做接口测试。

PANNs 的预训练模型来自 AudioSet。默认返回的是 AudioSet 类别索引；如需人类可读标签，可在 `panns.py` 中加载与 checkpoint 匹配的类别表，避免把不匹配的标签文件混用。

在没有 NVIDIA GPU 或 checkpoint 的 OrbStack 本地环境，可用 `PANNS_MOCK=1 PANNS_DEVICE=cpu` 验证 Ray Serve、Scheduler 和任务轮询链路。该模式返回固定的模拟标签，不代表真实 PANNs 推理；部署真实模型时去掉 `PANNS_MOCK` 并挂载 checkpoint。

## OrbStack Kubernetes / KubeRay

单容器 Docker 模式适合快速验证；集群模式使用 `k8s/rayservice-orbstack.yaml`：Head 使用基于官方 Ray 镜像的轻量入口镜像，Worker 使用 `Dockerfile.worker` 构建的业务镜像，KubeRay Operator 通过 `RayService` 管理 Serve 生命周期，不使用 `--working-dir`。业务 Replica 通过 `panns_worker` 自定义资源调度到 Worker，并使用 `proxy_location: EveryNode` 让 HTTP Proxy 只运行在有 Replica 的节点。该本地清单使用 CPU mock 模式；真实 GPU 集群需要替换 Worker 镜像、checkpoint、设备环境变量以及 GPU 资源声明。

首次部署：

```bash
docker build -f Dockerfile.head -t ray-panns-head:k8s-v1 .
docker build -f Dockerfile.worker -t ray-panns-worker:k8s-v3 .
kubectl apply -f k8s/rayservice-orbstack.yaml
```

发布业务代码时，重新构建业务镜像并更新 RayService；Head 不需要更新。只有 Ray 版本、PANNs 依赖、CUDA 或基础运行时变化时，才需要重新滚动 Worker。
