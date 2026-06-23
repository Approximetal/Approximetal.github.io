# 从感知到行动：下一代人机交互系统的闭环架构

> 当前的交互系统大多仍是"被动的指令执行器"——用户说"打开灯"，它打开；用户沉默，它也沉默。而下一代人机交互系统需要完成一个根本性的范式转换：从逐条响应指令，到维持一个跨越时间、模态和交互轮次的连续理解过程。本文以感知-理解-执行为主线，梳理这一转换中浮现的核心技术问题、代表性解法与技术展望。

---

## 一、引言：当交互不再是"一问一答"

一个被反复引用的场景：用户刚结束一项紧张的任务，长舒一口气，身体后仰。系统没有等待用户开口，而是根据姿态变化和放松的表情，说了一句："辛苦了，终于能稍微缓一口气了。"

这个场景之所以被封为"理想交互"的缩影，不是因为它展示了多强的语言生成能力，而是因为它要求的是一整套迥异于当前系统的运行逻辑：系统不是在"回答一个问题"，而是在"追踪一个状态的连续变化"，并在恰当的时机做出恰当的反馈。这个过程中，**感知不是检测，而是预测；理解不是分类，而是状态追踪；执行不是合成，而是策略的外化**。

具体而言，一个具备上述能力的交互系统需要在三个层次上完成闭环：

- **感知层**：从多模态信号流（视觉、语音、姿态、轨迹）中实时识别当前事件，并预测短期未来事件——不仅回答"正在发生什么"，还要回答"接下来可能发生什么"。
- **理解与反馈层**：在感知输出的基础上管理对话状态、维护长期记忆、追踪情感变化，并据此决定交互策略——回答"这意味着什么"以及"我应该以什么角色和方式介入"。
- **执行层**：将高层的策略指令转化为具体的语言、语音、表情、动作和工具调用，并确保其安全与自然。

这三层之间的信息流动构成一个动态闭环：感知层输出结构化事件流，理解层基于事件更新状态并产出策略，执行层将策略转化为可感知的行动，而行动本身又成为下一轮感知的输入。

本文以该闭环为主线，梳理近年来相关研究的核心脉络，并结合 LLM Agent 社区沉淀的设计模式[1]，分析各层次的关键技术选择及其背后的方案权衡。文章不追求面面俱到的综述，而是以"这条系统链路究竟卡在哪里"为出发点，在每个节点上讨论有代表性的解法、它们的适用边界，以及不同技术路线之间的取舍逻辑。

---

## 二、感知层：从多模态信号到结构化事件

### 2.1 核心命题：感知的主输出是结构化事件，自然语言作为解释层

一个常见的直觉是把感知层的目标表述为"让机器理解场景"。但更务实的定义是：**感知层的主输出不应是自由自然语言，而应是可计算、可追溯、可校准的结构化事件流；自然语言更适合作为解释和证据摘要**。

这个事件流中的每一项都应该是可计算、可预测、可追溯的：

```json
{
  "speaker": "user_1",
  "addressee": "robot",
  "is_speaking": true,
  "gaze_target": "robot",
  "distance_to_robot": 1.6,
  "approach_intent_score": 0.82,
  "turn_shift_probability": 0.73,
  "audio_quality": "noisy",
  "visual_confidence": 0.91
}
```

生成这样的事件流，需要攻克至少三个互相关联的问题：在用户开口之前预判交互意图、在多人场景中锁定交互对象、在复杂声学环境中提取可用的多模态信号。而这三者共享同一个底层依赖——流式多模态信号的实时对齐与动态融合。

### 2.2 从轮次预测到接近意图预测

传统语音系统以 Voice Activity Detection (VAD) 为起点——检测到语音，就开始处理。但在自然交互中，这个时点已经太晚：在用户开口说话之前，交互早已开始——从视线转向系统、身体前倾、嘴唇微张，这些信号出现在语音之前数百毫秒到数秒。

交互意图预测实际包含两类相关但不同的任务。一类是**轮次预测（Turn-taking Prediction）**：预测未来短时间内谁会说话、是否发生轮次切换或 backchannel。另一类是**接近/参与意图预测（Approach/Engagement Intention）**：预测用户是否打算靠近系统、主动发起交互或请求帮助。

**轮次预测方面**，Voice Activity Projection (VAP) [2] 将问题建模为连续时间预测：不判断"当前是否在说话"，而是直接预测未来几百毫秒内的语音活动。这一范式的价值不在精度，而在于它把"反应式"系统变成了"预判式"系统——系统可以在语音出现之前预分配算力、调整麦克风指向、预热对话模型。VAP 的另一优势是其**自监督数据构造**：不需要人工标注，直接从原始语音活动中自动生成训练标签。MM-VAP [4] 将音频与面部表情、视线和头部姿态融合，在双方轮替预测中将 hold/shift 准确率从 79% 提升至 84%，其中面部表情贡献最大。

**接近意图预测方面**，服务机器人场景中的自监督方法 [3] 在交互发生前 3 秒以上即可实现 AUROC > 0.9 的预测效果，这对"用户是否需要帮助但还没开口"这类场景尤为关键。轨迹预测方法（PAR-D [5]）则从行人运动轨迹推断接近意图，适合公共空间场景。

在不同应用场景下，交互预测的实现方式各有侧重：

| 预测类型 | 代表工作 | 核心机制 | 适用场景 | 主要限制 |
|------|---------|---------|---------|---------|
| 轮替预测（音频自监督） | VAP [2] | 从语音活动自动生成标签，无需人工标注 | 数据稀缺的纯语音场景 | 无法感知视觉接近行为 |
| 轮替预测（多模态） | MM-VAP [4] | 视觉线索（表情、视线）结合音频联合建模轮替事件 | 社交/服务场景的双人对话 | 依赖同步多模态数据，遮挡时退化 |
| 接近意图预测 | 自监督交互意图 [3]、PAR-D [5] | 基于多模态行为或行人轨迹预测用户是否将靠近/交互 | 公共空间、服务机器人 | 需要空间定位或多模态采集基础设施 |

### 2.3 谁在说话，在对谁说：从说话人检测到交互对象确认

智能音箱在与朋友聊天时突然插嘴——这是典型的 addressee detection 失败。机器人需要回答两个连续问题：**谁在说话（Active Speaker Detection, ASD），以及这句话是不是对我说的（Addressee Detection）**。

ASD 方向在过去四年中形成了一条清晰的技术演进脉络：

| 方法 | 会议/年份 | mAP | 参数量 | 核心贡献 | 在演进中的位置 |
|------|----------|-----|--------|---------|-------------|
| TalkNet [6] | ACM MM 2021 | ~92% | ~15M | 首次将ASD建模为音视频同步问题，提出Cross-Attention + Long-term Self-Attention | 奠基工作，定义了ASD的核心架构范式 |
| LoCoNet [7] | CVPR 2024 | 95.2% | ~34M | 将时序建模拆分为长程同说话人(LIM)和短程多说话人交互(SIM)两个互补模块 | 精度路线的代表，验证了分而治之的时序建模策略 |
| LR-ASD [8] | IJCV 2025 | 94.5% | 0.84M | 仅用单候选输入+分离式2D/3D卷积+轻量GRU，参数量为LoCoNet的1/41 | 轻量化路线的代表，跨数据集泛化能力反而最优 |
| UniTalk [9] | -/2025 | — | — | 构建44.5小时真实场景数据集（覆盖噪声、多人重叠、低资源语言），复杂度远超 AVA | 揭示现有 benchmark 的覆盖盲区——主流模型在 AVA 上接近饱和，在 UniTalk 上仍有大量提升空间 |

这条演进脉络呈现两条分叉：**精度优先**（LoCoNet：更大模型，更高 mAP）和**轻量泛化优先**（LR-ASD：极小模型，跨场景更稳定）。实际部署中，后者的吸引力往往大于前者——端侧场景对推理延迟敏感，且训练数据永远无法覆盖全部部署环境，轻量模型的跨域泛化优势在此场景中比 0.7% 的精度差距更有价值。

一个更深层的问题是：**主流视觉 ASD 流程和常用 benchmark 在很大程度上依赖可见面部 ROI**。在遮挡、背向和出画场景中，纯视觉 ASD 的可靠性显著下降。因此 ASD 不能作为独立模块运行，而应与声源定位（SSL）和说话人日志（diarization）组成一个**视听空间三重确认机制**：面部可见时以 ASD 为主，被遮挡时以声源方向和声纹为辅。

此外，**addressee detection（判断话是不是对自己说的）是一个与 ASD 相关但独立的问题**。Addressee 的识别不仅依赖"谁在说话"，还依赖视线方向、韵律特征、语义称呼、空间站位和上一轮的轮次结构。Inoue 等人的 addressee recognition benchmark [64] 指出，多方对话中显式 addressee 标记仅出现在约 20% 的轮次中，GPT-4o 在该任务上的表现仅略高于随机水平——这说明 addressee 不是一个可以通过更大 ASR/ASD 模型自动解决的附带问题，而需要作为独立模块来建模。

### 2.4 远场嘈杂环境：从"听见"到"听清"

机器人面临的是复合声学退化——远场衰减、混响、背景噪声、多人重叠、说话人移动——这些因素极少单独出现。这一方向的核心问题不是"理解语义"，而是在语义处理之前**恢复语音信号的可用性**。

按问题类型梳理代表性解法：

| 问题类型 | 代表方法 | 机制 | 优势 | 边界条件 |
|----------|---------|------|------|---------|
| 远场单通道增强 | Whisper-FEST [10] | 单通道前端增强，无需并行近远场数据 | 即插即用，保护近场性能 | 性能上限受限于单通道信息量 |
| 多通道阵列增强 | LABNet [11] | 注意力机制动态加权各通道，支持任意阵列几何 | 系统姿态变化时无需重校准 | 需要多麦克风硬件 |
| 移动声源跟踪 | RTF引导波束形成[12] | 连续估计目标说话人的相对传递函数 | 适合说话人走动的场景 | 实时RTF估计的计算开销 |
| 音视频联合增强 | Cocktail-Party AVSR [13] | 视觉唇动信息辅助音频增强 | 极端噪声下WER从119%降至39.2% | 需要摄像头对准嘴部，遮挡时失效 |
| 高频补全 | Latent Bridge Models [14] | 生成式模型恢复远场丢失的高频成分 | 改善远场语音的自然度和可懂度 | 生成伪影对ASR的影响需验证 |

这些方法在实际部署中应被组织为**"信号处理前端 + 深度学习后端"的两级流水线**：波束形成先做低延迟的空间滤波（可解释、确定性），深度学习模型再做精细的非线性增强。同时，视觉辅助（AVSR）不应始终开启——当音频信噪比足够时，纯音频 ASR 更省算力且更稳定；仅当 SNR 低于阈值或检测到多人重叠时，才启动视觉分支。

仿真数据（如 Treble10 [15] 提供的 3100 个声学配置的宽带房间脉冲响应）在预训练阶段不可替代，但必须在其基础上用目标场景的真实数据进行微调——仿真与真实声学环境之间的 domain gap 在远场混响场景中尤为显著。

### 2.5 感知层的核心机制：流式多模态的动态对齐与融合

以上三个问题看似独立，但都依赖同一个底层机制：**系统如何在持续到达的异步多模态信号流中，实时对齐视觉帧、音频帧和文本 token，并根据环境动态决定每个模态的可信度**。

从神经科学借来的两个概念为这个问题提供了参考坐标系。Ernst 和 Banks [16] 提出的最大似然估计（MLE）模型表明，大脑根据各感官信号的可靠性（与方差成反比）动态调整权重——安静环境下听觉权重上升，嘈杂环境中视觉权重增加。McGurk 效应进一步说明这种融合发生在特征层面而非决策层面——当视觉唇形和听觉语音冲突时，融合后的感知是一个两者都不存在的中间音。

在多模态融合的算法层面，过去三年的进展同样呈现一条从"固定融合"到"动态门控"的演进线：

| 方法 | 会议/年份 | 核心机制 | 解决的问题 |
|------|----------|---------|-----------|
| QMF [17] | ICML 2023 | 理论证明融合权重应根据模态质量动态调整，以 uncertainty 作为质量代理 | 首次给出动态融合的理论基础 |
| URMF [18] | CVPR 2024 | 显式建模单模态的 aleatoric uncertainty，学习更稳健的单模态表示和融合表示 | 在低光照/遮挡场景中比 QMF 更稳定 |
| UDML [19] | CVPR 2026 Findings | 指出现有方法的 Double Penalty 缺陷——弱势模态因长期高 uncertainty 被持续降权 | 引入长期模态偏置修正项 |
| EMOE [20] | CVPR 2025 | 将每个模态视为 expert，通过 router network 做 sample-level 动态门控 | 从连续加权走向离散选择，减少低质量模态的干扰 |

这里涉及一个底层架构选择：**早期融合 vs 晚期融合**。早期融合在特征层面跨模态交互（能捕捉 McGurk 效应），但某一模态的噪声会污染全局；晚期融合各模态独立处理、互不干扰，但丢失了跨模态交互信息。对于实时交互系统，**混合融合 + uncertainty-based 动态加权**是务实的折中——底层做早期融合捕捉细粒度交互，上层用 uncertainty 做权重调节。当某个模态突然失效（麦克风被遮挡、光线变暗），系统自动降低该模态的贡献而非整体崩溃。从 LLM Agent 设计模式的视角来看，这本质上是一种 **Routing 模式**[1] 在感知层的实例化——根据每帧输入的模态质量，动态选择"哪个模态参与融合、以多大权重参与"。

最后，感知层的实时性不仅依赖算法效率，还涉及系统架构层面的设计选择：不同模态处理管线的延迟预算分配、异步事件队列的背压控制、低置信度场景下的回退策略、以及端侧轻量模型与云端大模型之间的协同推理。这些架构问题虽然不直接体现在论文的 benchmark 精度中，但对于一个需要在真实物理环境中稳定运行的交互系统而言，它们与算法精度同样关键。

---

## 三、理解与反馈层：从事件到策略的状态机

感知层回答"正在发生什么"，理解层回答"这意味着什么"以及"我该怎么做"。这一层的核心挑战不在于单次判断的准确率，而在于**在时间维度上维持一个连贯的交互状态表示**——包括情感状态的动态追踪、对话焦点的切换管理、跨会话的用户记忆维护，以及基于这些状态做出合适的交互策略选择。

LLM Agent 社区已经沉淀出一批可复用的设计模式[1]，覆盖从记忆管理、规划、反思到多智能体协作的完整链路。这些模式为构建交互系统的理解层提供了可操作的架构参考——不仅因为它们在实际系统中已被验证，更因为它们与交互系统面临的问题在结构上高度同构：都需要在不确定的输入上维持状态、在长时序上保持一致性、在多个备选策略中做出取舍。

### 3.1 核心命题：从情绪分类到情感状态追踪

把情绪理解简化为"开心/难过/愤怒"的分类，在交互系统中是不够的——不是因为分类不准，而是因为这种表示丢失了时间维度上的动态信息。同一句"压力小了很多"，如果上一轮用户处于高压状态，它的含义是"状态正在缓解"；如果上一轮用户本就平静，它的含义可能只是"例行汇报"。

因此，理解层的输出不应是孤立的情绪标签，而是一个包含历史轨迹和状态变化的**情感状态更新**：

```json
{
  "state_update": {
    "affective_social_state": {
      "previous_state": "high_stress",
      "current_state": "relief",
      "change": "stress_reduced",
      "cause": "urgent deadline met",
      "evidence": ["feeling much lighter now", "speech rate returned to normal"],
      "confidence": 0.78
    },
    "social_intent": "sharing_recent_event"
  },
  "feedback_policy": {
    "type": "empathetic_acknowledgement",
    "tone": "warm_light",
    "should_probe": false,
    "rationale": "user is sharing in a post-relief state; light acknowledgment is appropriate rather than probing further"
  }
}
```

这个表示将情感从静态标签变成了**状态机**：它不仅描述当前状态，还显式记录状态从哪里来、为什么变、证据是什么、以及这个变化对后续策略意味着什么。三个直接的好处：可解释（系统可以回溯判断依据）、可追溯（完整的状态变化轨迹）、可验证（可以通过后续用户反馈验证此前判断是否正确）。

### 3.2 情感追踪：从推理链到策略约束

将上述状态机运转起来，需要两个层次的能力：**从多模态信号中推理出情感状态**，以及**基于状态选择合适的交互策略**。

**情感推理**的路线在过去两年中演化出两种范式：

- **链式推理（Prompt Chaining 模式[1]）**：将推理过程显式分解为固定步骤。ESCoT [21] 在情感支持对话中定义了 identifying → understanding → regulating 三个推理阶段，每个阶段输出中间解释。SABER-LLM [22] 进一步强制模型先独立提取视觉证据、声学证据，再进行高层推理——这种结构化证据分解（SED）范式用流程约束换取了可解释性和抗幻觉能力，代价是推理链变长、延迟增加。

- **反思式推理（Reflection 模式[1]）**：允许模型在生成推理链后自我检查和修正。Emotion-LLaMA [23] 较早将 instruction tuning 引入多模态情感推理，输出的是"标签 + 解释"而非单纯标签。R1-Omni [24] 通过可验证奖励强化学习（RLVR）让模型显式分析各模态对情绪判断的贡献权重。EMO-R3 [25] 进一步引入反思机制——模型可以在生成推理链后发现矛盾并自我纠正。这一范式的优势在于能处理复杂情感的"反转"场景（如从疑惑到惊讶、从客气到不满），但训练过程本身的不稳定性更高。

两条路线不是互斥的：链式推理适合首次搭建（训练稳定、数据构造简单），反思式推理适合在基线稳定后提升可解释性和边缘案例的表现。

**情感与意图的联合理解**是另一个不可绕开的维度——同一个情感状态，不同的社交意图对应不同的交互策略。MC-EIU [26] 定义了 7 类情感-9 类意图的联合标注范式。IntentionESC [27] 以意图为中心构造推理链（情感状态 → 意图推断 → 策略选择），在情感支持对话中将策略选择准确率提升了可观的幅度。

**从理解到策略的最后一公里**，有一项重要发现来自 ACL 2024 杰出论文 [28]：LLM 在情感支持对话中天然存在策略偏好偏差——倾向于过度安慰、过度追问、过早给建议。这一发现的价值在于揭示了**策略选择不能完全委托给模型自由裁量**——需要有显式的约束机制引导策略分布。后续 STRIDE-ED [29] 通过策略感知的数据精炼和多目标强化学习，将策略选择作为一个显式推理步骤来训练，是近期较系统地把"理解—策略"显式建模的代表性工作。这本质上是一种 **Planning 模式**[1] 的实例化——先由策略规划器产出高层计划，再交由 LLM 生成符合策略约束的具体回复。

### 3.3 对话状态机与记忆管理：从单轮反应到跨会话的一致人格

如果 3.2 节讨论的是"一轮交互中如何做对反应"，本节讨论的是"如何在时间和交互轮次的累积中，让系统维持一个连贯的理解"。

**工作记忆：当前对话状态机**

每一轮交互都可以形式化为一个状态更新操作：`(state_t, observation_t) → state_update → state_{t+1}`。传统 Dialogue State Tracking (DST) 追踪的是任务型对话中的槽位填充（餐厅预订、航班查询），但在社交交互场景中，状态机需要扩展为四个通道的联合追踪：

- **对话状态**：当前话题、待回应的开放循环（open loops，如用户提到"下周有个重要面试"但尚未展开）、对话阶段
- **情感状态**：用户的情绪变化轨迹、情感强度和持续性
- **场景状态**：谁是活跃说话人、说话对象是谁、物理距离和社交情境
- **关系状态**：与用户的互动历史、信任程度、适宜的交互风格

FnCTOD [30] 将 DST 形式化为 function calling 问题——把每个 domain 的 schema 视为函数，状态更新视为函数调用——在 zero-shot 场景中超越了当时的 ChatGPT 基线。这一工作的启示不在于模型精度，而在于它揭示了**状态追踪的自然表示形式是函数调用**——这和 LLM Agent 设计模式中的 Tool Use 模式[1] 一脉相承：状态更新本身就可以被建模为一系列"更新某某字段为某某值"的工具调用。

**长期记忆：跨会话的用户理解**

长期记忆的核心不是存储容量，而是 **Memory Policy**——哪些信息值得写入？写入短期还是长期记忆？何时检索？何时更新？何时遗忘？

LLM Agent 社区的 Memory Management 模式[1] 为这一问题提供了系统性的实现基础：

| 系统 | 会议/年份 | 机制 | 核心贡献 | 适用场景 |
|------|----------|------|---------|---------|
| MemGPT/Letta [31] | -/2023 | 虚拟上下文管理：fast memory / archival memory / 类工具的记忆操作 | 定义了 agent memory 的存-管-取循环基本范式 | 通用 agent 记忆架构 |
| MemoryBank [32] | AAAI 2024 | 引入艾宾浩斯遗忘曲线模拟记忆衰减与强化 | 让记忆有了"自动过期"机制，适合长期陪伴 | 社交/陪伴机器人 |
| A-MEM [33] | NeurIPS 2025 | Zettelkasten 方法：新记忆触发旧记忆更新，形成演化的知识网络 | 将记忆从静态存储变为持续演化的结构 | 需要复杂知识关联的场景 |
| Mem0 [34] | ECAI 2025 | 产品级通用记忆层，token 消耗降低 90%，检索延迟降低 91% | 支持图结构记忆变体与持续偏好学习 | 快速集成和小规模部署 |

**部分实现做法**：
- **粒度**：LongMemEval [35] 的实验表明，**按对话轮次（round）存储比按会话（session）存储更利于后续检索**；显式时间戳索引对"上周三你说过……"这类时序查询至关重要。
- **写入策略**：采用渐进式写入——首次提及的信息进入短期记忆池，被多次引证或确证后提升至长期记忆。同时设置 TTL，定期清理低价值、低频检索项。这避免了两个极端：什么都不记（过于保守）和什么都要记（记忆膨胀）。
- **评估基准**：LoCoMo [36] 构造了跨 32 个 session、平均 600 轮的长程对话基准；PersonaMem-v2 [37] 在仅使用 1/16 输入 token 的条件下达到 55% 准确率（当时最优），证实了紧凑记忆蒸馏的有效性。

### 3.4 学会"何时保持沉默"：主动交互的策略学习与评估

如果说前两节解决的是"该怎么回应"，这一节要解决的问题更前置：**系统该不该主动出声**。

主动交互是一把双刃剑——做得好，系统显得"贴心""有眼力见"；做得不好，就是"打扰""过度热情""creepy"。KnowU-Bench [38] 对当前模型的核心发现直指问题本质：**模型默认倾向于行动，即使应该保持安静**。说"不"比说"是"更需要意志力——无论是人还是模型。

于是主动交互的策略学习面临一个悖论：训练数据中天然缺乏"不行动"的样本（因为"不行动"不会被记录），导致模型习得的策略分布向"多行动"偏移。Proactive Agent [39] 的工作为此提供了解决方案——同时标注系统的行动和用户的反应（accepted / rejected / ignored），将 rejection signal 纳入训练信号。其核心启示是：**主动交互的训练数据必须是轨迹式的**，包含"系统做了什么→用户怎么反应→这个反应是正向还是负向"的完整闭环，而不能只是 "user input → best response" 的静态对。

**适度的主动决策可以根据置信度和风险进行分层**：
- 高置信度 + 高价值 + 低风险 → 主动执行（如用户看起来迷路了，提供方向指引）
- 高置信度 + 高价值 + 高风险 → 主动确认（如用户可能需要帮助但涉及隐私，先询问"需要帮忙吗？"）
- 低置信度或高打扰风险 → 保持沉默，等待用户主动发起

这实际上是 **Human-in-the-Loop 模式**[1] 在交互策略层的体现——系统在不确定性边界处将决策权交还给用户，而不是强行替代用户做选择。从 HRI 传统看，"何时打断"本身就是一个独立建模问题。Interruptibility-aware robot 的研究表明，机器人如果能预测用户当前是否可被打断，会改善用户对其社交能力的感知 [65]。更近的 conversational robot 工作则将用户打断分为协作性同意、协作性帮助、澄清请求和破坏性打断，并在 LLM-powered social robot 中实时处理 93.69% 的用户打断 [66]。进一步地，non-intrusive assistance 将"主动帮助但不打断人"形式化为独立 HRI 范式，强调主动性不等于高频介入，而是在人的主任务流程中选择最小扰动的行动时机 [67]。

MMDuet2 [40] 通过多轮强化学习（GRPO）将"交互时机"转化为 RL 优化目标中的策略偏好，实现了回复内容与交互时机的端到端对齐。Plug-and-Play Policy Planner [41] 则走了另一条路——用一个轻量模型专门负责策略规划（何时说、以什么策略说），再将规划结果交给 LLM 生成具体回复——这种"先规划后生成"的分离式架构在情感支持和谈判场景中表现出比端到端生成更低的策略偏差。

**策略学习的评估**与传统 NLP 指标（BLEU / ROUGE / 准确率）存在根本性区别——一条"正确"的回复如果出现在错误的时机，可能是灾难性的。适用的评估维度应该是：

| 评估维度 | 含义 | 测量方式 |
|----------|------|---------|
| 时机适当性 (Timing) | 系统选择说话的时机是否合适 | 用户是否被打断、是否在沉默后主动发起 |
| 内容相关性 (Relevance) | 话语是否与当前状态和上下文相关 | 与用户后续回复的连贯性 |
| 打扰度 (Intrusiveness) | 系统的主动行为是否让用户感到不适 | 用户是否忽略/拒绝/主动终止对话 |
| 策略一致性 (Consistency) | 策略选择是否符合系统角色和关系状态 | 跨轮次的策略连贯性跟踪 |

### 3.5 反馈策略与工具调用：从决策到指令

理解层的最终输出不是自然语言文本，而是**高层语义指令**——一个描述"应该做什么"而不绑定具体实现的结构化计划。这种信息压缩是必要的：它将策略决策与执行细节解耦，使下游执行层可以独立迭代而不影响上游理解。

例如，用户说"咱们见两次了，来握个手吧"。理解层输出：

```json
{
  "policy_action": {
    "type": "embodied_action",
    "name": "handshake",
    "confidence": 0.98,
    "requires_safety_check": true
  },
  "verbal_response_policy": {
    "type": "positive_acceptance",
    "tone": "friendly"
  }
}
```

执行层随后负责将这个plan实例化为具体的话术、语音韵律、面部表情和机械臂运动轨迹。

**工具调用（Tool Use / Function Calling）**是理解层将策略转化为可执行指令的核心机制。从 LLM Agent 的 Tool Use 设计模式[1] 出发，工具调用面临的核心问题不是"模型能不能选对工具"，而是**什么时候不应该调用工具**。ToolSandbox [42] 的实验揭示了当前模型在工具使用中的系统性缺陷：当信息不足时，模型倾向于"编造参数强行调用"而非"先追问补齐信息"。SafeAgentBench [43] 的评测结果同样令人警醒——最佳基线在安全任务上成功率 69%，但对危险任务的拒绝率仅 5%，模型几乎"有求必应"。

分层架构在此处显现优势：**LLM 负责理解意图和选择工具，但工具的实际执行由确定性代码完成，且执行前经过三层校验**——格式验证（参数是否符合 schema）、权限检查（是否允许当前上下文调用该工具）、安全审查（是否涉及物理动作且未通过 safety gate）。

---

## 四、执行层：从策略指令到可感知的表达

理解层决定了"说什么和做什么"，执行层决定的是"说成什么样"和"怎么动"——即使策略完全正确，表达方式生硬或不合时宜，交互体验也会大打折扣。

### 4.1 核心命题：情感不是标签，是表达轨迹

执行层面对的核心矛盾是：理解层输出的情感状态是离散的结构化表示，但人类表达情感的方式是连续、多维且高度依赖语境的。同样一句话，在不同情境下可能表达截然相反的情感。电影《楚门的世界》中，主角的经典台词 "In case I don't see ya, good afternoon, good evening and good night." 在片头是例行公事的职业性寒暄——轻松、明亮、机械；在片尾则承载了告别虚假世界的决绝——沉重、克制、释然。相同的词句，完全不同的情感底色。将这种语境依赖的复合表达状态交给 TTS 引擎处理，需要一个比 "speak in a cheerful voice" 更精确的控制接口。

V-A-D（Valence–Arousal–Dominance，为避免与前文 Voice Activity Detection 的 VAD 混淆，本文对情感维度空间统一使用 V-A-D 表示）三维连续空间[44,45] 为这种复杂表达提供了数学描述。同一条台词在两种情境下的 V-A-D 坐标对比：

| 情境 | V-A-D 坐标 | 情感描述 |
|------|---------|---------|
| 片头：例行问候 | (+0.40, +0.30, +0.20) | 职业性寒暄，明亮但机械 |
| 片尾：告别与觉醒 | (−0.20, +0.50, +0.70) | 沉重而清醒的诀别，克制中带着决绝 |

这里的关键设计选择是：**离散标签与连续 V-A-D 坐标不应二选一**。离散标签适合高层分类（这是什么情感类型），V-A-D 适合强度和风格微调（这个情感有多强、偏向什么风格）。执行层接收的应为两者的混合——先用离散标签确定情感种类，再用 V-A-D 三维坐标控制表达的精细参数。

**标注策略的务实选择**：人工标注整句级离散标签和关键片段的表达意图（成本可控），用 forced alignment + 自动情感识别模型生成细粒度 V-A-D 伪标签，仅在明显错误的片段上做人工校准。句子级标注可以依靠 GoEmotions [46] 和 EmoBank [47] 等成熟数据集；影视级的连续情感轨迹标注可参照 VEATIC [48] 和 MovieGraphs [49] 的范式。

### 4.2 声音的"表演"：从 TTS 到可检索的表演技巧库

情感语音合成的技术路线在过去两年中形成了三个梯次的选择：

- **VAE 路线**（VITS / NaturalSpeech 系列）：推理速度快、社区成熟，适合实时对话系统。
- **扩散模型路线**（Grad-TTS、Diff-TTS）：生成质量最高，能捕捉语音的随机性细节，但推理需要多步去噪，延迟较高。
- **Flow Matching / Rectified Flow** 路线：在扩散模型的高质量生成与实时系统的低延迟需求之间寻求折中，是近年来生成式语音模型中的重要趋势。

在连续情感控制方面，EmoSphere-TTS [50] 将 V-A-D 空间映射为球面坐标，实现了情感风格和强度的连续可调。UDDETTS [51] 进一步统一了离散标签和维度空间，在 V-A-D 三个维度上实现了近似线性的控制精度——这是目前较系统地体现了 V-A-D 连续控制与 LLM-based TTS 结合的代表性工作。

**Technique-RAG：一个更长远的演进方向**

当前的情感 TTS 基本停留在"整句风格"的控制粒度上。但真实的人类表达——无论是说话还是唱歌——由大量可辨识的技巧片段构成：停顿、重音、反讽语调、包袱节奏、气声、假声、颤音、滑音。

将这些"表演参考"显式地建成一个可检索、可组合、可迁移的技巧库——即 **Technique-RAG**——是比粗粒度 style prompt 更有技术深度的发展方向。从 LLM Agent 的 RAG 设计模式[1] 出发，Technique-RAG 的完整架构可以参照 Planner → Retriever → Generator → Verifier 的四阶段流水线：Planner 将情感执行计划拆解为技巧需求序列，Retriever 从技巧库中检索匹配的参考片段，Generator 将目标声纹与技巧融合生成语音，Verifier 检查生成的韵律和频谱是否与目标技巧一致。

在歌声领域，TCSinger 2 [52] 已实现 6 种唱法技巧（Breathy、Falsetto、Mixed Voice 等）的零样本迁移；AutoStyle-TTS [53] 则将 RAG 引入语音风格匹配，根据输入文本和上下文场景自动检索最合适的整体风格提示——这两项工作分别解决了"技巧迁移"和"风格检索"的子问题，但距离完整的 Technique-RAG 仍有显著距离。

### 4.3 执行的安全基座：工具执行与具身动作的安全门控

当策略指令涉及外部工具调用或物理动作时，安全不是加分项，是不可协商的底线。

VLA（Vision-Language-Action）端到端模型如 OpenVLA [54] 在跨本体泛化上的表现令人印象深刻——29 项任务上超越 RT-2-X（55B）16.5% 的绝对成功率。但 SafeAgentBench [43] 揭示的 5% 危险任务拒绝率表明，端到端模型在追求"能做什么"时严重忽视了"不该做什么"。这与 Execution 层的安全要求存在根本性冲突——你需要一个能够且**必然**拒绝危险指令的机制，而端到端模型的输出本质上是概率性的。

一个可行的两层架构：上层由 VLA 或 LLM 做高层语义理解和任务规划（输出"向前移动 0.5 米同时伸手"这类自然语言级的动作计划），下层由一个独立的安全模块审查每个动作计划——碰撞风险、关节限制、紧急停止条件——并在通过后将指令转发给确定性控制器（PID/MPC）执行具体轨迹。AEROS [55] 提出的 policy-separated runtime ——执行约束和安全策略独立于能力逻辑进行强制——与这一思路完全对应。这本质上是 **Exception Handling 模式**[1] 在具身执行层的实例化：系统不是假设所有输入都是安全的，而是主动检查每个输出是否违反安全边界。

---

## 五、从闭环能力到可部署系统：治理、评估与修复

前几章沿着感知-理解-执行的闭环梳理了各层次的技术进展。但这些讨论隐含了一个共同的前提：系统是在"实验室条件"下运行的。当这样一个持续观察、推理、记忆并主动行动的系统进入真实部署环境时，一系列横跨三层的问题会同时浮现——隐私边界如何定义、记忆如何受控、闭环表现如何评估、交互破裂后如何修复。这些问题不是某一层的附属问题，而是决定系统能否从原型走向部署的全局约束。

### 5.1 隐私与同意：从权限开关到上下文完整性

一个持续采集视觉、语音、姿态数据并维护长期记忆的交互系统，其隐私问题不能简化为"是否获得了摄像头/麦克风权限"。Nissenbaum 提出的 Contextual Integrity 理论 [56] 将隐私定义为信息流是否符合具体社会场景的规范——同样的数据采集行为，在家庭场景中可能是侵犯性的，在公共场所可能是可接受的，在工作场景中可能需要第三方授权。这一框架尤其适用于 embodied interaction：系统的感知、记忆和主动介入行为在不同场景（家庭、办公室、公共空间）中对应着不同的信息流规范。

在社交机器人领域，CONFIDANT [57] 将这一思想落地为一个实际的 privacy controller——它不依赖静态权限列表，而是根据关系、话题、情感等对话元数据，动态判断信息的敏感性。后续工作进一步将 Contextual Integrity 操作化为 LLM-based 助手的决策约束 [58]。但另一条研究线索揭示了现实困境：即使是最先进的 LLM/VLM，在不经专门训练的情况下，对隐私相关情境的识别能力也远不可靠 [59]。因此，LLM 可以参与隐私推理，但不应成为隐私决策的唯一执行者；隐私判断应由可审计的 policy layer 约束，并保留用户覆写权。

### 5.2 记忆治理：可写入、可更正、可遗忘、可防泄漏

3.3 节讨论了长期记忆的"构建"——MemGPT、MemoryBank、A-MEM、Mem0 等系统分别解决了记忆的存储、检索和演化问题。但从部署视角看，**记忆的"受控性"是一个对称且同等重要的问题**。

首先，agent 的长期记忆本身就是攻击面。ACL 2025 的一项研究 [60] 首次提出 Memory Extraction Attack（MEXTRA），证明攻击者可以通过精心设计的黑盒 prompt 从 agent 记忆中提取用户的敏感信息——这意味着记忆系统不能假设"只有善意用户会访问"。MemPrivacy [61] 提出了一种隐私保护方案：在端侧识别敏感信息片段（span），云端只处理脱敏后的占位符，从而在不牺牲推理能力的前提下控制信息暴露面。更一般地，近期综述 [62] 将长期记忆安全提炼为一个独立的安全子领域，强调其"可写入、可跨会话传播"的特性使其与传统静态数据安全存在根本区别。

其次，**记忆需要"遗忘"机制，而遗忘不等于删除缓存**。Memora [63] 提出了 Forgetting-Aware Memory Accuracy 指标，专门惩罚模型继续使用已经失效的过时记忆——例如用户改变了偏好、纠正了之前的推断、或某项信息的时效性已过。有效的遗忘需要同时解决技术层面的存储清理和策略层面的"何时该遗忘"判断，两者缺一不可。

综合来看，可部署的记忆系统应满足四个硬约束：**可写、可更正、可遗忘、可防越权访问**。前两者是 3.3 节已讨论的能力问题，后两者是本节补充的治理问题。

### 5.3 闭环评估：从模块精度到交互后果

3.4 节从交互层面讨论了时机适当性、内容相关性、打扰度和策略一致性四个评估维度。但当我们将视野从单次交互扩展到持续部署时，还需要一组系统级的闭环指标。

端到端延迟（从传感器输入到执行输出的总时间）决定了交互的自然度上限。人类对话中的轮次间隔通常处在数百毫秒量级，因此实时语音/具身系统往往需要用流式 ASR、提前预测和 filler/backchannel 将可感知等待控制在较低范围内；具体阈值则随场景、任务和表达方式变化。误触发率衡量系统在不该介入时介入的频率，沉默正确率则衡量系统在该保持沉默时确实沉默的比例——KnowU-Bench [38] 的实验表明，当前模型在后者上的表现远低于前者，因为训练数据中天然缺乏"不行动"的正样本。长期信任变化跟踪多 session 交互后用户对系统的信任度曲线——连续误触发会快速消耗初期建立的信任，而信任一旦耗尽，恢复比建立更难。

更重要的是，**单靠模块级的准确率无法捕捉交互后果的严重性**。ToolSandbox [42] 的实验揭示，即便工具选择的表面准确率较高，在 state dependency、canonicalization 和 insufficient information 等具状态依赖场景下，系统仍会犯下游影响巨大的错误。SafeAgentBench [43] 的报告更为警醒：最佳基线在安全任务上的成功率为 69%，但对危险任务的拒绝率仅 5%。在具身协作场景中，PARTNR [70] 的 100,000 个自然语言任务基准显示，SOTA LLM 在协调、任务追踪和错误恢复上仍存在明显局限；Habitat 3.0 [71] 则强调具身交互评估必须纳入 human-in-the-loop 的社会维度。这些基准共同指向一个结论：**闭环评估需要在系统层面度量交互后果，而不能仅用模块级 benchmark 的精度数字来替代**。

### 5.4 交互修复：系统出错后如何优雅恢复

有了安全门控（4.3 节）和闭环评估（5.3 节），剩下的问题是：当系统确实出错了——无论是抓取失败、打断用户、误解意图还是给出不适当回复——它是否具备检测和修复的能力？

交互修复闭环可以概括为四个阶段的循环：**检测**（detect failure/rupture）→ **确认**（acknowledge/clarify/apologize）→ **重新规划**（replan）→ **安全执行**（safe re-execution）。其中，"检测"阶段的信号来源远比硬件故障诊断更丰富——用户表情的尴尬、语气的困惑、重复指令、主动纠正机器人的行为，这些都是交互破裂（social breakdown）的可观测信号。ERR@HRI 2024 Challenge [68] 提供了多模态 HRI failure 数据集，标注了 robot mistakes、user awkwardness 和 interaction ruptures。ERR@HRI 2.0 [69] 进一步把错误检测拆成系统视角与用户视角两个子任务：前者检测机器人行为是否偏离设计预期，后者检测用户是否正在尝试纠正机器人行为。

从架构角度看，交互修复不是执行层的专属问题。修复循环需要感知层检测异常信号（用户困惑的表情、后退的身体姿态），理解层判断是否需要以及如何修复（道歉、澄清、还是沉默），执行层安全地重试或降级——这是一个横跨三层的协作流程。交互修复的关键不是生成一句道歉，而是把失败本身写回系统状态：感知层记录异常信号，理解层更新对用户意图和当前关系状态的判断，执行层基于新的约束重新规划动作。CHAIC [72] 进一步要求 agent 推断人类协作者的意图和约束后再做合作规划，这为修复循环中的"replan"阶段提供了更贴近真实人类协作的范式。

---

## 六、总结

回顾全文的讨论，构建下一代人机交互系统的技术挑战可以归纳为四个方向的突破：

**从检测到预测**。感知层的核心范式转换是将"检测当前事件"升级为"预测未来事件"（VAP、交互意图预测、轮替预判），让系统从"跟上用户"变为"等用户"。

**从单轮到状态**。理解层的核心进路是将交互建模为跨时间的状态追踪问题——工作记忆维护当前对话的动态结构（话题、情绪、开放循环），长期记忆维护跨会话的用户理解（偏好、关系、个性），策略选择基于完整的状态而非孤立的当前输入。

**从策略到表达**。执行层的核心挑战是将离散的策略标签转化为连续的多维表达——情感不是整句的固定标签而是高度语境依赖的 V-A-D 坐标，语音表达不是"用开心的语气说"而是从技巧库中检索、组合、迁移具体的表演参考。

**从能力到治理**。当系统具备持续感知、长期记忆和主动执行能力之后，核心问题不再只是"能不能做"，还包括"是否应该做、是否被允许做、出错后能否被发现和修复"。因此，隐私同意、记忆治理、闭环评估和交互修复应被视为与感知、理解、执行同等重要的部署层约束，而非附属问题。

这四个方向与 LLM Agent 设计模式及更广义的系统治理机制之间存在结构对应：前三类主要对应 Agent 工程实践中的 Routing、Planning、Memory Management、Tool Use 等模式[1]，而"可治理"则进一步引入 Contextual Integrity、记忆治理和交互修复等部署层机制[56,60,68,69]。

| 交互系统关键词 | 核心问题 | 对应的架构 / 治理机制 |
|--------------|---------|---------------------|
| 预测性 (Predictive) | 模态质量的动态评估与选择 | Routing、Parallelization |
| 状态性 (Stateful) | 跨时间的记忆维护与策略规划 | Memory Management、Planning、Reflection |
| 可执行与安全 (Executable & Safe) | 策略到行动的安全转换 | Tool Use、Human-in-the-Loop、Exception Handling |
| 可治理 (Governable) | 隐私边界、记忆受控、评估闭环、失败恢复 | Contextual Integrity、Memory Governance、Repair Loops |

具身智能的交互，最终考验的是系统在物理世界和社会规则双重约束下，进行实时感知、推理、决策和行动的整合能力。将 Agent 设计模式的架构方法与交互系统的领域知识结合，可能是缩短从"能回答问题"到"能真正交互"这段距离的最现实路径。

---

## 参考文献

### 综述与设计模式
[1] Antonio Gullí. Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems. Springer, 2025. — 系统总结 Prompt Chaining、Routing、Parallelization、Reflection、Tool Use、Planning、Memory Management、Human-in-the-Loop、RAG、Exception Handling 等 agentic design patterns。中文双语版为社区翻译项目，覆盖 21 个核心模式。

### 交互意图预测与轮替建模
[2] Ekstedt & Skantze. Voice Activity Projection: Self-supervised Learning of Turn-taking Events. INTERSPEECH 2022. — VAP 奠基工作，将轮替建模统一为连续时间预测问题，首次实现自监督训练。

[3] Self-Supervised Prediction of the Intention to Interact with a Service Robot. arXiv 2023. — 在咖啡机场景中实现交互发生前 3 秒 AUROC > 0.9。

[4] MM-VAP: Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction. ACL Findings 2025. — 将视觉线索融入 VAP，性能从 79% 提升至 84%。

[5] PAR-D: People Approaching Robots Database. ICMI 2024. — 行人接近意图预测基准数据集。

### 说话人检测与交互对象确认
[6] Tao et al. TalkNet: Active Speaker Detection. ACM MM 2021. — ASD 的奠基工作，首次将音视频同步建模引入说话人检测。

[7] Wang et al. LoCoNet: Long-Short Context Network for ASD. CVPR 2024. — 将时序建模拆分为长程同说话人和短程多说话人交互两个互补模块。

[8] Liao et al. LR-ASD: Lightweight and Robust Network for ASD. IJCV 2025. — 仅 0.84M 参数，跨数据集泛化优于 LoCoNet，端侧部署性价比最高。

[9] UniTalk: A Real-World Benchmark for ASD. 2025. — 构建覆盖噪声、多人重叠和低资源语言的真实场景数据集，揭示传统 benchmark 的覆盖盲区。

[64] Inoue et al. An LLM Benchmark for Addressee Recognition in Multi-modal Multi-party Dialogue. IWSDS 2025. — 多方对话中显式 addressee 仅出现在约 20% 轮次，GPT-4o 表现仅略高于随机。

### 远场语音与视听融合
[10] Whisper-FEST: Single-Channel Far-Field Enhanced Speech-to-Text without Parallel Data. Samsung Research, 2026. — 面向 Whisper S2T 的单通道远场语音前端增强方案，无需并行近远场数据。

[11] LABNet: Lightweight Attentive Beamforming Network. arXiv 2025. — 支持任意临时麦克风阵列的实时语音增强。

[12] Interpretable Binaural Deep Beamforming Guided by Time-Varying RTF. 2025. — 通过跟踪移动声源的相对传递函数实现动态波束形成。

[13] Cocktail-Party Audio-Visual Speech Recognition. Interspeech 2025. — 极端噪声下 WER 从 119% 降至 39.2%。

[14] Audio Super-Resolution with Latent Bridge Models. 2025. — 生成式模型恢复远场语音的高频成分。

[15] Treble10: High-Quality Dataset for Far-Field Speech. 2025. — 3100 个声学配置的宽带房间脉冲响应。

### 多模态动态融合
[16] Ernst & Banks. Humans Integrate Visual and Haptic Information in a Statistically Optimal Fashion. Nature 2002. — MLE 多感官整合模型的理论基础。

[17] Zhang et al. QMF: Quality-aware Multimodal Fusion. ICML 2023. — 理论证明融合权重应根据模态质量动态调整，以 uncertainty 作为质量代理。

[18] Gao et al. Embracing Unimodal Aleatoric Uncertainty for Robust Multimodal Fusion. CVPR 2024. — 显式量化单模态 aleatoric uncertainty，并用其学习更稳健的单模态和融合表示。

[19] Wei et al. Unbiased Dynamic Multimodal Fusion. CVPR 2026 Findings. — 解决弱势模态在动态融合中的 Double Penalty 问题。

[20] EMOE: Modality-Specific Enhanced Dynamic Emotion Experts. CVPR 2025. — 从连续加权走向 Router-based 动态选择。

### 情感识别与推理
[21] ESCoT: Towards Interpretable Emotional Support Dialogue Systems. ACL 2024. — 将情感支持分解为 identifying → understanding → regulating 三阶段链式推理。

[22] Zhao et al. Integrating Fine-Grained Audio-Visual Evidence for Robust Multimodal Emotion Reasoning (SABER-LLM). arXiv 2026. — 提出 structured evidence decomposition 范式，将多模态情感分析从静态分类推进到证据驱动的生成式推理。

[23] Cheng et al. Emotion-LLaMA: Multimodal Emotion Recognition with Instruction Tuning. NeurIPS 2024. — 较早将 instruction tuning 引入多模态情感推理的代表性工作。

[24] R1-Omni: Explainable Omni-Multimodal Emotion Recognition with RL. 2025. — 通过 RLVR 让模型显式分析各模态对情绪判断的贡献。

[25] EMO-R3: Reflective Reinforcement Learning for Emotional Reasoning. 2026. — 引入反思式自我纠正机制。

[26] MC-EIU: Emotion and Intent Joint Understanding in Multimodal Conversation. NeurIPS 2024. — 7 类情感-9 类意图联合标注范式。

[27] IntentionESC: Intention-Centered Emotional Support. ACL 2025. — 以意图为中心的情感支持推理链。

### 策略选择与约束
[28] Can Large Language Models be Good Emotional Supporter? ACL 2024 Outstanding Paper. — 系统揭示 LLM 在情感支持中的策略偏好偏差。

[29] STRIDE-ED: Strategy-Grounded Stepwise Reasoning for Empathetic Dialogue. arXiv 2026. — 通过策略感知多目标 RL 对齐模型行为与策略约束。

### 对话状态机与长期记忆
[30] FnCTOD: LLMs as Zero-shot Dialogue State Tracker through Function Calling. ACL 2024. — 将 DST 形式化为 function calling。

[31] Packer et al. MemGPT/Letta: Towards LLMs as Operating Systems. 2023. — 虚拟上下文管理，定义 agent memory 基本范式。

[32] MemoryBank: Enhancing LLMs with Long-Term Memory. AAAI 2024. — 引入遗忘曲线机制。

[33] A-MEM: Agentic Memory for LLM Agents. NeurIPS 2025. — Zettelkasten 式演化记忆网络。

[34] Chhikara et al. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. ECAI 2025. — 产品级通用记忆层。

[35] LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. ICLR 2025. — 覆盖五类核心记忆能力的基准测试。

[36] LoCoMo: Evaluating Very Long-Term Conversational Memory of LLM Agents. ACL 2024. — 跨 32 session 长程对话记忆基准。

[37] PersonaMem-v2: Learning Implicit User Personas via Agentic Memory. 2025. — 1/16 token 达当时最优的记忆蒸馏。

### 主动交互
[38] KnowU-Bench: Towards Interactive, Proactive, and Personalized Mobile Agent Evaluation. 2026. — 核心发现：当前模型默认倾向于行动，即使应保持沉默。

[39] Proactive Agent: Shifting LLM Agents to Active Assistance. ICLR 2025. — 将用户 rejection signal 纳入主动策略训练。

[40] MMDuet2: Enhancing Proactive Interaction with Multi-Turn RL. ICLR 2026. — 将交互时机纳入 RL 优化目标。

[41] Plug-and-Play Policy Planner for LLM-Powered Dialogue Agents. ICLR 2024. — 策略规划与回复生成解耦的分离式架构。

### 主动交互、打断与非侵入式辅助
[65] Banerjee et al. Effects of Interruptibility-Aware Robot Behavior. 2018. — 证明 interruptibility-aware robot 改善用户对机器人社交能力的感知。

[66] Cao et al. Interruption Handling for Conversational Robots. arXiv 2025. — 实时检测用户打断并按其意图分类管理，实验中处理了 93.69% 的用户打断。

[67] Assistance Without Interruption: A Benchmark for Non-Intrusive Human-Robot Assistance. arXiv 2026. — 将 non-intrusive assistance 形式化为独立 HRI 范式。

### 工具调用与安全
[42] ToolSandbox: Stateful, Conversational Evaluation for LLM Tool Use. NAACL 2025. — Apple 开源的状态化工具调用基准。

[43] SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents. 2024. — 最佳基线对危险任务仅 5% 拒绝率。

### 情感理论与情感数据集
[44] Russell. A Circumplex Model of Affect. 1980. — 情感环形模型，Valence-Arousal 二维空间的理论基础。

[45] Mehrabian. The PAD Model. 1974-1996. — 在 V-A 基础上加入 Dominance 维度。

[46] GoEmotions: A Dataset of Fine-Grained Emotions. ACL 2020. — 58K 条 27 类细粒度情感标注。

[47] EmoBank: Dimensional Emotion Analysis. EACL 2017. — 10K 句子的 V-A-D 三维标注。

[48] VEATIC: Video-based Emotion and Affect Tracking in Context. 2023. — 124 段视频的逐帧连续 V-A 标注。

[49] MovieGraphs: Towards Understanding Human-Centric Situations from Videos. CVPR 2018. — 51 部电影的角色情感-情境-关系图标注。

### 情感语音合成与技巧迁移
[50] EmoSphere-TTS: Emotional Style and Intensity Modeling via Spherical Emotion Vector. Interspeech 2024. — 将 V-A-D 空间映射为球面坐标实现连续情感控制。

[51] UDDETTS: Unifying Discrete and Dimensional Emotions for Controllable TTS. 2025. — 统一离散情感标签与维度情感空间（原文采用 ADV 即 Arousal–Dominance–Valence 顺序，本文统一记作 V-A-D），实现近似线性的连续情感控制。

[52] TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis. ACL Findings 2025. — 6 种唱法技巧的零样本迁移。

[53] AutoStyle-TTS: RAG-based Automatic Style Matching TTS. ICME 2025. — 将 RAG 引入语音风格匹配的代表性工作。

### 具身动作与 VLA
[54] OpenVLA: An Open-Source Vision-Language-Action Model. 2024. — 7B 参数开源 VLA，29 项跨本体任务超越 RT-2-X 16.5%。

[55] AEROS: A Single-Agent Operating Architecture with Embodied Capability Modules. 2026. — Policy-separated runtime，支持动态重规划和安全门控。

### 隐私与记忆治理
[56] Nissenbaum. Privacy in Context: Technology, Policy, and the Integrity of Social Life. 2010. — Contextual Integrity 理论的奠基著作，将隐私定义为信息流是否符合社会场景规范。

[57] Tang et al. CONFIDANT: A Privacy Controller for Social Robots. HRI 2022. — 基于关系、话题、情感等对话元数据动态判断信息敏感性的社交机器人隐私控制器。

[58] Operationalizing Contextual Integrity in Privacy-Conscious Assistants. arXiv 2024. — 将 CI 理论操作化为 LLM 助手的决策约束。

[59] Benchmarking LLM Privacy Recognition for Social Robot Decision Making. arXiv 2025. — 用 Contextual Integrity 构造家庭机器人隐私场景，发现 out-of-the-box LLM 与人类隐私判断一致性较低。

[60] Wang et al. Unveiling Privacy Risks in LLM Agent Memory. ACL 2025. — 提出 Memory Extraction Attack，证明 agent 长期记忆可被黑盒 prompt 攻击抽取敏感信息。

[61] MemPrivacy: Privacy-Preserving Personalized Memory Management for Edge-Cloud Agents. arXiv 2026. — 端侧识别敏感 span、云端仅处理占位符的隐私保护记忆方案。

[62] A Survey on Long-Term Memory Security in LLM Agents: Attacks, Defenses, and Governance Across the Memory Lifecycle. arXiv 2026. — 将 agent 长期记忆安全提炼为独立子领域，覆盖 write/store/retrieve/forget 等 lifecycle 阶段。

[63] Uddin et al. From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents. arXiv 2026. — 提出 Memora 和 FAMA，惩罚使用已失效记忆。

### 闭环评估与交互修复
[68] ERR@HRI 2024 Challenge: Multimodal Detection of Errors and Failures in Human-Robot Interactions. arXiv 2024. — 多模态 HRI failure 数据集，标注 robot mistakes、user awkwardness、interaction ruptures。

[69] ERR@HRI 2.0 Challenge: Multimodal Detection of Errors and Failures in Human-Robot Conversations. arXiv 2025. — 16 小时 dyadic human-robot 对话，区分 system perspective 与 user perspective 两个子挑战。

[70] PARTNR: A Benchmark for Planning and Reasoning in Embodied Multi-agent Tasks. arXiv 2024. — 100,000 个自然语言人机协作家务任务基准，SOTA LLM 在协调和错误恢复上仍有限。

[71] Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots. ICLR 2024. — 强调具身交互评估需纳入 human-in-the-loop 的社会维度。

[72] Constrained Human-AI Cooperation: An Inclusive Embodied Social Intelligence Challenge (CHAIC). arXiv 2024. — 要求 agent 推断人的意图和约束后再做合作规划。

