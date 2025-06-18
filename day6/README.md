# YOLOv8 与 YOLOv12 网络结构对比

## 1. 概述
- **YOLOv8**: Ultralytics 2023年发布的版本，在速度和精度之间取得了良好平衡
- **YOLOv12**: 2024年最新版本，主要针对小目标检测和实时性进行了优化

## 2. 网络架构对比

| 组件                | YOLOv8                          | YOLOv12                         |
|---------------------|---------------------------------|---------------------------------|
| **Backbone**        | CSPDarknet53 改进版             | Hybrid-Transformer (ViT+CNN)   |
| **Neck**            | PANet + SPPF                    | BiFPN + Adaptive Spatial Fusion|
| **Head**            | Decoupled Head (分类+回归分离)  | Dynamic Head (任务自适应)       |
| **激活函数**        | SiLU                            | Mish + SwiGLU                   |
| **下采样方式**      | Conv + MaxPool                  | Depthwise Separable Conv        |
| **特征融合**        | 3层FPN                          | 5层跨尺度融合                   |

## 3. 关键改进点

### YOLOv12 主要创新
1. **混合骨干网络**:
   - 前3层使用CNN提取低级特征
   - 后6层使用ViT结构捕获全局上下文

2. **动态标签分配**:
   ```python
   # YOLOv12的Task-aligned Assigner
   assign_metric = classification_score^α * iou_score^β
   α, β = learnable_parameters()
# YOLOv8 模型定义片段
class YOLOv8(nn.Module):
    def __init__(self):
        self.backbone = CSPDarknet()
        self.neck = PANet()
        self.head = Detect()

# YOLOv12 模型定义片段
class YOLOv12(nn.Module):
    def __init__(self):
        self.backbone = HybridBackbone()  # CNN+ViT混合
        self.neck = BiFPN()
        self.head = DynamicDetect()  # 动态参数头