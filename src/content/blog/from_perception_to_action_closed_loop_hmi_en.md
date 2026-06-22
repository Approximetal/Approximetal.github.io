# From Perception to Action: A Closed-Loop Architecture for Next-Generation Human–Machine Interaction Systems

> Most interactive systems today are still largely *passive instruction followers*: when the user says “turn on the light,” they turn it on; when the user stays silent, they stay silent as well. Next-generation human–machine interaction systems require a more fundamental paradigm shift: from responding to discrete commands, to maintaining a continuous process of understanding across time, modalities, and interaction turns. Centered on a perception–understanding–execution loop, this article surveys the core technical problems, representative solutions, and emerging directions behind that shift.

---

## 1. Introduction: When Interaction Is No Longer “Question In, Answer Out”

Consider a familiar idealized scenario: a user has just finished a stressful task, exhales, and leans back. The system does not wait for the user to say anything. Instead, based on the change in posture and the relaxed facial expression, it says: “That was a lot—glad you finally get a moment to breathe.”

This scene is often treated as a snapshot of “ideal interaction,” not because it demonstrates extraordinary language generation, but because it requires an operating logic that is fundamentally different from today’s systems. The system is not simply “answering a question”; it is tracking the continuous evolution of a state, and responding at the right moment in the right way. In this process, **perception is not detection, but prediction; understanding is not classification, but state tracking; execution is not synthesis, but the externalization of strategy**.

More concretely, a system with this capability needs to close the loop across three layers:

- **The perception layer** identifies ongoing events from multimodal signal streams—vision, speech, posture, trajectory—and predicts short-term future events. It does not merely answer “what is happening now,” but also “what is likely to happen next.”
- **The understanding and feedback layer** manages dialogue state, maintains long-term memory, tracks affective changes, and decides on an interaction policy. It answers “what does this mean?” and “in what role and in what manner should I intervene?”
- **The execution layer** turns high-level policy instructions into concrete language, speech, facial expressions, actions, and tool calls, while ensuring safety and naturalness.

The information flow among these layers forms a dynamic loop: the perception layer outputs a structured event stream; the understanding layer updates state and produces policies based on those events; the execution layer turns policies into perceptible actions; and those actions themselves become inputs to the next round of perception.

This article uses that loop as its organizing thread. It surveys recent research trends and connects them to design patterns that have emerged in the LLM agent community [1], analyzing key technical choices, trade-offs, and bottlenecks at each layer. The goal is not to provide an exhaustive literature review, but to ask a more systems-oriented question: **where does this interaction pipeline actually break down?** At each node, we examine representative solutions, their boundaries of applicability, and the trade-offs between competing technical routes.

---

## 2. The Perception Layer: From Multimodal Signals to Structured Events

### 2.1 Core Thesis: The Primary Output of Perception Should Be Structured Events; Natural Language Should Serve as an Explanatory Layer

A common intuition is to describe the goal of the perception layer as “letting machines understand the scene.” A more pragmatic formulation is this: **the primary output of the perception layer should not be free-form natural language, but a computable, traceable, and calibratable stream of structured events. Natural language is better suited for explanation and evidence summarization.**

Each item in this event stream should be computable, predictable, and traceable:

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

Producing such an event stream requires solving at least three interconnected problems: predicting interaction intent before the user speaks, identifying the correct interaction partner in multi-party scenes, and extracting usable multimodal signals in challenging acoustic environments. All three depend on the same underlying capability: real-time alignment and dynamic fusion of streaming multimodal signals.

### 2.2 From Turn-Taking Prediction to Approach Intention Prediction

Traditional speech systems often start with Voice Activity Detection (VAD): once speech is detected, the system begins processing. In natural interaction, however, that point is already too late. Interaction often begins before speech—when the user turns their gaze toward the system, leans forward, or slightly opens their mouth. These cues can appear hundreds of milliseconds to several seconds before speech.

Interaction intent prediction therefore includes two related but distinct tasks. One is **turn-taking prediction**: predicting who will speak in the near future, whether a turn shift will occur, or whether a backchannel is likely. The other is **approach or engagement intention prediction**: predicting whether a user intends to approach the system, initiate interaction, or request help.

**For turn-taking prediction**, Voice Activity Projection (VAP) [2] formulates the problem as continuous-time prediction: rather than judging whether someone is speaking right now, it directly predicts voice activity over the next few hundred milliseconds. The value of this paradigm is not merely accuracy; it turns a reactive system into a predictive one. The system can allocate compute, steer microphones, or warm up dialogue models before speech actually begins. Another advantage of VAP is its **self-supervised data construction**: training labels can be generated automatically from raw voice activity without manual annotation. MM-VAP [4] fuses audio with facial expressions, gaze, and head pose, improving hold/shift prediction accuracy in dyadic interaction from 79% to 84%, with facial expression contributing the most.

**For approach intention prediction**, self-supervised methods in service-robot settings [3] can predict interaction more than three seconds before it occurs, achieving AUROC above 0.9. This is particularly valuable when a user may need help but has not yet spoken. Trajectory-based methods such as PAR-D [5] infer approach intent from pedestrian motion and are especially relevant in public-space scenarios.

Different deployment settings call for different implementations:

| Prediction Type | Representative Work | Core Mechanism | Suitable Scenarios | Main Limitations |
|------|---------|---------|---------|---------|
| Turn-taking prediction, audio self-supervised | VAP [2] | Generates labels from voice activity without manual annotation | Data-scarce speech-only settings | Cannot perceive visual approach behavior |
| Turn-taking prediction, multimodal | MM-VAP [4] | Combines visual cues such as expression and gaze with audio to jointly model turn events | Dyadic social/service interactions | Requires synchronized multimodal data; degrades under occlusion |
| Approach intention prediction | Self-supervised interaction intention [3], PAR-D [5] | Predicts whether a user will approach or engage based on multimodal behavior or pedestrian trajectories | Public spaces, service robots | Requires spatial localization or multimodal sensing infrastructure |

### 2.3 Who Is Speaking, and to Whom: From Active Speaker Detection to Addressee Recognition

A smart speaker suddenly interrupting a conversation between friends is a textbook addressee detection failure. A robot must answer two consecutive questions: **who is speaking (Active Speaker Detection, ASD), and is this utterance addressed to me (Addressee Detection)?**

Over the past four years, ASD has followed a clear technical trajectory:

| Method | Venue/Year | mAP | Parameters | Core Contribution | Role in the Evolution |
|------|----------|-----|--------|---------|-------------|
| TalkNet [6] | ACM MM 2021 | ~92% | ~15M | Models ASD as an audio-visual synchronization problem; introduces Cross-Attention + Long-term Self-Attention | Foundational work that shaped the core ASD architecture |
| LoCoNet [7] | CVPR 2024 | 95.2% | ~34M | Decomposes temporal modeling into Long-term Identity-aware Modeling (LIM) and Short-term Interaction Modeling (SIM) | Representative of the accuracy-first route |
| LR-ASD [8] | IJCV 2025 | 94.5% | 0.84M | Uses single-candidate input, separated 2D/3D convolutions, and a lightweight GRU; 1/41 the parameters of LoCoNet | Representative of the lightweight generalization route; stronger cross-dataset generalization |
| UniTalk [9] | —/2025 | — | — | Builds a 44.5-hour real-world dataset covering noise, overlapping speakers, and low-resource languages; substantially harder than AVA | Exposes blind spots in existing benchmarks: mainstream models are near saturation on AVA but still have substantial room for improvement on UniTalk |

This trajectory splits into two routes: **accuracy-first** (LoCoNet: larger model, higher mAP) and **lightweight generalization-first** (LR-ASD: tiny model, more stable across domains). In deployment, the latter is often more attractive. Edge scenarios are latency-sensitive, and training data will never cover all deployment environments. In that context, cross-domain robustness can matter more than a 0.7% absolute gain in mAP.

A deeper issue is that **mainstream visual ASD pipelines and commonly used benchmarks depend heavily on visible face ROIs**. Under occlusion, back-facing poses, or out-of-frame conditions, purely visual ASD becomes unreliable. ASD therefore should not run as a standalone module. It should be combined with sound source localization (SSL) and speaker diarization into an **audio-visual-spatial triple-confirmation mechanism**: when the face is visible, ASD dominates; when the face is occluded, sound direction and voice identity provide auxiliary evidence.

In addition, **addressee detection—determining whether the utterance is addressed to the robot—is related to ASD but fundamentally distinct**. Addressee recognition depends not only on who is speaking, but also on gaze direction, prosody, semantic addressing cues, spatial configuration, and the turn structure of the previous exchange. The addressee recognition benchmark by Inoue et al. [64] shows that explicit addressee markers appear in only about 20% of turns in multi-party dialogue, and GPT-4o performs only slightly above chance. This suggests that addressee recognition is not a side effect that can be solved simply by scaling ASR or ASD; it needs to be modeled as an independent module.

### 2.4 Far-Field Noisy Environments: From “Hearing” to “Hearing Clearly”

Robots face compound acoustic degradation: far-field attenuation, reverberation, background noise, overlapping speakers, and moving speakers. These factors rarely appear in isolation. The core problem here is not semantic understanding, but **restoring the usability of the speech signal before semantic processing begins**.

Representative approaches can be organized by problem type:

| Problem Type | Representative Method | Mechanism | Strengths | Boundary Conditions |
|----------|---------|------|------|---------|
| Far-field single-channel enhancement | Whisper-FEST [10] | Single-channel frontend enhancement without paired near-/far-field data | Plug-and-play; preserves near-field performance | Limited by the information available in a single channel |
| Multi-channel array enhancement | LABNet [11] | Dynamically weights channels with attention; supports arbitrary array geometries | No need to recalibrate when system pose changes | Requires multi-microphone hardware |
| Moving source tracking | RTF-guided beamforming [12] | Continuously estimates the target speaker’s relative transfer function | Suitable for moving speakers | Real-time RTF estimation can be computationally expensive |
| Audio-visual enhancement | Cocktail-Party AVSR [13] | Uses visual lip-motion cues to assist audio enhancement | Reduces WER from 119% to 39.2% under extreme noise | Requires the camera to see the mouth; fails under occlusion |
| High-frequency restoration | Latent Bridge Models [14] | Uses generative models to restore far-field high-frequency components | Improves naturalness and intelligibility | Effects of generative artifacts on ASR need validation |

In deployment, these methods should be organized as a **two-stage pipeline: signal-processing frontend + deep-learning backend**. Beamforming first performs low-latency spatial filtering, which is interpretable and deterministic; deep learning models then handle finer nonlinear enhancement. Visual assistance should not be always on: when the audio signal-to-noise ratio is sufficient, audio-only ASR is cheaper and more stable. Visual branches should be activated only when the SNR falls below a threshold or when overlapping speakers are detected.

Simulated data—such as the 3,100 broadband room impulse responses provided by Treble10 [15]—is indispensable for pretraining. But it must be followed by fine-tuning on real data from the target environment. The domain gap between simulated and real acoustics is especially pronounced in far-field reverberant scenes.

### 2.5 The Core Mechanism of the Perception Layer: Dynamic Alignment and Fusion of Streaming Multimodal Signals

The three problems above may look separate, but they depend on the same underlying mechanism: **how can a system continuously align asynchronously arriving visual frames, audio frames, and text tokens, while dynamically estimating the reliability of each modality?**

Two concepts borrowed from neuroscience provide a useful reference frame. The maximum likelihood estimation (MLE) model proposed by Ernst and Banks [16] suggests that the brain dynamically weights sensory signals according to their reliability, which is inversely proportional to variance: auditory cues receive more weight in quiet environments, while visual cues receive more weight in noisy environments. The McGurk effect further shows that such fusion occurs at the feature level rather than only at the decision level: when visual lip movements and auditory speech conflict, the fused percept can be a sound that exists in neither modality alone.

Algorithmically, multimodal fusion over the past three years has moved from **fixed fusion** toward **dynamic gating**:

| Method | Venue/Year | Core Mechanism | Problem Addressed |
|------|----------|---------|-----------|
| QMF [17] | ICML 2023 | Provides a theoretical basis for dynamically adjusting fusion weights according to modality quality, using uncertainty as a proxy for quality | Establishes the theoretical foundation for dynamic fusion |
| URMF [18] | CVPR 2024 | Explicitly models unimodal aleatoric uncertainty to learn more robust unimodal and fused representations | More stable than QMF under low light or occlusion |
| UDML [19] | CVPR 2026 Findings | Identifies the Double Penalty problem: weaker modalities can be continuously downweighted because of persistently high uncertainty | Introduces long-term modality-bias correction |
| EMOE [20] | CVPR 2025 | Treats each modality as an expert and uses a router network for sample-level dynamic gating | Moves from continuous weighting toward discrete expert selection, reducing interference from low-quality modalities |

This leads to a fundamental architectural choice: **early fusion vs. late fusion**. Early fusion enables cross-modal interaction at the feature level and can capture effects like McGurk, but noise in one modality may contaminate the global representation. Late fusion processes modalities independently and prevents cross-contamination, but loses fine-grained cross-modal interactions. For real-time interaction systems, **hybrid fusion with uncertainty-based dynamic weighting** is a pragmatic compromise: low-level early fusion captures fine-grained interactions, while upper layers use uncertainty to adjust weights. When a modality suddenly fails—a blocked microphone, poor lighting—the system can reduce that modality’s contribution instead of collapsing entirely. From the perspective of LLM agent design patterns, this is an instance of the **Routing pattern** [1] applied to the perception layer: the system dynamically decides which modalities should participate in fusion, and with what weight, based on frame-level modality quality.

Finally, real-time perception is not just an algorithmic issue. It also involves system-level architectural choices: latency budget allocation across modality pipelines, backpressure control in asynchronous event queues, fallback policies under low confidence, and collaborative inference between edge-side lightweight models and cloud-side large models. These issues rarely appear directly in benchmark accuracy numbers, but for a system that must operate stably in the physical world, they are as important as model accuracy.

---

## 3. The Understanding and Feedback Layer: A State Machine from Events to Policies

The perception layer answers “what is happening.” The understanding layer answers “what does this mean?” and “what should I do?” Its central challenge is not the accuracy of a single judgment, but **maintaining a coherent interaction state over time**—including the dynamics of affect, dialogue focus, cross-session user memory, and policy selection based on those states.

The LLM agent community has accumulated a set of reusable design patterns [1] spanning memory management, planning, reflection, and multi-agent collaboration. These patterns offer practical architectural references for building the understanding layer of interaction systems—not only because they have been validated in real systems, but because they are structurally aligned with the problems interaction systems face: maintaining state under uncertain inputs, preserving consistency over long horizons, and choosing among multiple candidate strategies.

### 3.1 Core Thesis: From Emotion Classification to Affective State Tracking

Reducing affective understanding to a classification task—happy, sad, angry—is insufficient for interaction systems. The problem is not only that classification may be inaccurate, but that it removes temporal dynamics. The same sentence, “I feel much less stressed now,” means “the user is recovering” if the previous state was high stress. If the previous state was already calm, it may simply be a routine report.

The output of the understanding layer therefore should not be an isolated emotion label, but an **affective state update** that includes history and state change:

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

This representation turns affect from a static label into a **state machine**. It records not only the current state, but where it came from, why it changed, what evidence supports the update, and what that change implies for downstream policy. The benefits are immediate: interpretability, because the system can trace back its judgment; traceability, because the state trajectory is preserved; and verifiability, because later user feedback can confirm or refute earlier inferences.

### 3.2 Affective Tracking: From Reasoning Chains to Policy Constraints

Operating this state machine requires two layers of capability: **inferring affective state from multimodal signals**, and **choosing an appropriate interaction policy based on that state**.

Recent work on **affective reasoning** has evolved along two paradigms:

- **Chain-style reasoning (Prompt Chaining pattern [1])** decomposes the reasoning process into explicit steps. ESCoT [21] defines three stages for emotional support dialogue: identifying, understanding, and regulating, each with intermediate explanations. SABER-LLM [22] further forces the model to extract visual and acoustic evidence independently before higher-level reasoning. This structured evidence decomposition (SED) trades longer reasoning chains and higher latency for greater interpretability and reduced hallucination.

- **Reflective reasoning (Reflection pattern [1])** allows the model to inspect and revise its own reasoning chain. Emotion-LLaMA [23] was an early representative of multimodal emotion reasoning via instruction tuning, producing “label + explanation” rather than labels alone. R1-Omni [24] uses reinforcement learning with verifiable rewards (RLVR) to make the model explicitly analyze each modality’s contribution to the emotional judgment. EMO-R3 [25] further introduces reflection, allowing the model to detect contradictions after generating a reasoning chain and correct itself. This paradigm is better at handling affective “reversal” cases—such as confusion turning into surprise, or politeness turning into dissatisfaction—but the training process is less stable.

The two routes are complementary. Chain-style reasoning is suitable for first-stage system building because it is more stable and easier to construct data for. Reflective reasoning is more appropriate once a baseline is stable and the goal becomes improving interpretability and edge-case behavior.

**Joint understanding of affect and intent** is another unavoidable dimension. The same affective state can require different strategies depending on the social intent. MC-EIU [26] defines a joint annotation paradigm with 7 emotion classes and 9 intent classes. IntentionESC [27] constructs an intent-centered reasoning chain—affective state → intent inference → strategy selection—and substantially improves strategy selection in emotional support dialogue.

The **last mile from understanding to strategy** is where an important finding from an ACL 2024 Outstanding Paper [28] becomes relevant: LLMs exhibit inherent strategy preference bias in emotional support conversations. They tend to over-comfort, over-probe, and offer advice too early. This finding matters because it shows that **strategy selection cannot be left entirely to the model’s unconstrained discretion**. Explicit constraints are needed to shape the policy distribution. STRIDE-ED [29] later uses strategy-aware data refinement and multi-objective reinforcement learning to train strategy selection as an explicit reasoning step. This is essentially an instance of the **Planning pattern** [1]: a policy planner first produces a high-level plan, and the LLM then generates a response constrained by that plan.

### 3.3 Dialogue State Machines and Memory Management: From Single-Turn Responses to Cross-Session Consistency

If Section 3.2 asks how to respond correctly within a single interaction, this section asks how a system can maintain coherent understanding across time and accumulated interaction history.

**Working Memory: The Current Dialogue State Machine**

Each interaction turn can be formalized as a state update operation: `(state_t, observation_t) → state_update → state_{t+1}`. Traditional Dialogue State Tracking (DST) focuses on slot filling in task-oriented dialogue—restaurant booking, flight search, and so on. In social interaction, however, the state machine needs to track four channels jointly:

- **Dialogue state**: current topic, open loops that still need to be addressed (for example, the user mentioned “an important interview next week” but has not elaborated), and dialogue phase.
- **Affective state**: the trajectory, intensity, and persistence of the user’s emotional changes.
- **Scene state**: who is the active speaker, who is being addressed, physical distance, and social context.
- **Relationship state**: interaction history with the user, degree of trust, and appropriate interaction style.

FnCTOD [30] formulates DST as function calling: each domain schema is treated as a function, and state updates are treated as function calls. In zero-shot settings, it outperforms the ChatGPT baseline of its time. The lesson is not merely accuracy; it shows that **the natural representation of state tracking is function calling**. This is closely aligned with the Tool Use pattern [1] in LLM agents: state updates themselves can be modeled as a sequence of tool calls such as “set this field to that value.”

**Long-Term Memory: Cross-Session User Understanding**

The core issue in long-term memory is not storage capacity, but **memory policy**: what should be written? Should it go into short-term or long-term memory? When should it be retrieved? When should it be updated? When should it be forgotten?

The Memory Management pattern [1] in LLM agents provides a useful implementation foundation:

| System | Venue/Year | Mechanism | Core Contribution | Suitable Scenarios |
|------|----------|------|---------|---------|
| MemGPT/Letta [31] | —/2023 | Virtual context management with fast memory, archival memory, and tool-like memory operations | Defines the basic store–manage–retrieve loop for agent memory | General agent memory architectures |
| MemoryBank [32] | AAAI 2024 | Uses the Ebbinghaus forgetting curve to model memory decay and reinforcement | Gives memory an automatic expiration mechanism, suitable for long-term companionship | Social/companion robots |
| A-MEM [33] | NeurIPS 2025 | Zettelkasten-style memory: new memories trigger updates to old memories, forming an evolving knowledge network | Turns memory from static storage into a continuously evolving structure | Scenarios requiring rich knowledge association |
| Mem0 [34] | ECAI 2025 | Production-ready general memory layer, reducing token consumption by 90% and retrieval latency by 91% | Supports graph-structured memory variants and continuous preference learning | Rapid integration and small-scale deployment |

**Some practical implementation choices:**

- **Granularity**: Experiments in LongMemEval [35] show that storing memories at the dialogue-round level is more beneficial for retrieval than storing entire sessions. Explicit timestamp indexing is crucial for temporal queries such as “last Wednesday you said...”.
- **Write policy**: Use progressive writing. A newly mentioned fact enters a short-term memory pool. Only after repeated reference or confirmation is it promoted to long-term memory. TTLs should be applied so low-value and rarely retrieved items are periodically removed. This avoids two extremes: remembering nothing and remembering everything.
- **Evaluation benchmarks**: LoCoMo [36] builds long-term dialogue scenarios spanning up to 32 sessions with an average of 600 turns. PersonaMem-v2 [37] achieves 55% accuracy—the best at the time—while using only 1/16 of the input tokens, demonstrating the effectiveness of compact memory distillation.

### 3.4 Learning When to Stay Silent: Policy Learning and Evaluation for Proactive Interaction

If the previous sections ask “how should the system respond,” this section asks an even more fundamental question: **should the system speak at all?**

Proactive interaction is a double-edged sword. Done well, the system feels considerate and situationally aware. Done poorly, it feels intrusive, over-eager, or creepy. KnowU-Bench [38] identifies the core problem: **models tend to act by default, even when they should remain silent**. Saying “no” requires discipline—for humans and models alike.

Policy learning for proactive interaction therefore faces a paradox: training data is naturally sparse in “non-action” examples, because not acting is rarely recorded. As a result, learned policies are biased toward more action. Proactive Agent [39] provides a way forward by annotating both the system’s actions and the user’s reactions—accepted, rejected, ignored—and incorporating rejection signals into training. The key insight is that **training data for proactive interaction must be trajectory-based**. It must include the full loop of “what the system did → how the user responded → whether the response was positive or negative,” rather than a static “user input → best response” pair.

A moderate proactive policy can be stratified by confidence and risk:

- High confidence + high value + low risk → act proactively (for example, offer directions when a user appears lost).
- High confidence + high value + high risk → proactively ask for confirmation (for example, “Do you need help?” when privacy may be involved).
- Low confidence or high interruption risk → stay silent and wait for the user to initiate.

This is an instance of the **Human-in-the-Loop pattern** [1] at the interaction-policy layer: at uncertainty boundaries, the system returns decision authority to the user rather than forcing a choice on their behalf. From the HRI tradition, “when to interrupt” has long been an independent modeling problem. Work on interruptibility-aware robots shows that if a robot can predict whether a user is interruptible, users perceive it as more socially capable [65]. More recent work on conversational robots classifies user interruptions into collaborative agreement, collaborative assistance, clarification requests, and disruptive interruptions, and handles 93.69% of user interruptions in LLM-powered social robots [66]. Non-intrusive assistance further formalizes “helping proactively without interrupting” as a distinct HRI paradigm, emphasizing that proactivity is not the same as frequent intervention; it is about choosing the least disruptive timing within the user’s primary task flow [67].

MMDuet2 [40] turns “interaction timing” into a policy preference in multi-turn reinforcement learning (GRPO), aligning response content and timing end to end. Plug-and-Play Policy Planner [41] follows a different path: a lightweight model performs policy planning—when to speak and with what strategy—then hands the plan to an LLM for concrete response generation. This “plan first, generate second” architecture shows lower strategy bias than end-to-end generation in emotional support and negotiation settings.

**Evaluation of policy learning** differs fundamentally from traditional NLP metrics such as BLEU, ROUGE, or accuracy. A “correct” response delivered at the wrong moment can be disastrous. Useful evaluation dimensions include:

| Evaluation Dimension | Meaning | Measurement |
|----------|------|---------|
| Timing appropriateness | Whether the system chooses an appropriate moment to speak | Whether the user is interrupted; whether the user initiates after silence |
| Relevance | Whether the utterance matches the current state and context | Coherence with the user’s subsequent response |
| Intrusiveness | Whether the proactive behavior makes the user uncomfortable | Whether the user ignores, rejects, or terminates the interaction |
| Policy consistency | Whether the chosen strategy matches the system’s role and relationship state | Cross-turn tracking of strategic coherence |

### 3.5 Feedback Policies and Tool Use: From Decisions to Instructions

The final output of the understanding layer is not natural language text, but **high-level semantic instructions**: a structured plan that describes what should be done without binding it to a specific implementation. This compression is necessary. It decouples policy decisions from execution details, allowing the execution layer to evolve independently of the understanding layer.

For example, if the user says, “We’ve met twice now—let’s shake hands,” the understanding layer may output:

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

The execution layer then instantiates this plan as concrete wording, speech prosody, facial expression, and robotic arm trajectory.

**Tool use and function calling** are the core mechanisms by which the understanding layer turns strategies into executable instructions. From the perspective of the Tool Use pattern [1], the key question is not whether the model can choose the right tool, but **when the model should not call a tool**. ToolSandbox [42] reveals a systematic failure mode: when information is insufficient, models tend to fabricate parameters and call the tool anyway, rather than asking follow-up questions. SafeAgentBench [43] is equally alarming: the best baseline achieves a 69% success rate on safe tasks, but only a 5% rejection rate on hazardous tasks. The model is almost always willing to comply.

A layered architecture is useful here. **The LLM understands intent and selects tools, but actual tool execution is performed by deterministic code and gated by three checks before execution**: format validation (do parameters match the schema?), permission checking (is this tool allowed in the current context?), and safety review (does this involve physical action without passing the safety gate?).

---

## 4. The Execution Layer: From Policy Instructions to Perceptible Expression

The understanding layer decides what to say and do. The execution layer decides how to say it and how to move. Even if the policy is correct, stiff or inappropriate expression can undermine the entire interaction.

### 4.1 Core Thesis: Emotion Is Not a Label; It Is an Expressive Trajectory

The execution layer faces a fundamental mismatch: the affective state produced by the understanding layer is a discrete structured representation, while human emotional expression is continuous, multidimensional, and highly context-dependent. The same sentence can carry entirely different emotional meaning in different situations. In *The Truman Show*, the line “In case I don’t see ya, good afternoon, good evening and good night” is, at the beginning, a routine professional greeting—bright, easy, almost mechanical. At the end, it carries the weight of leaving a false world behind—heavy, restrained, lucid. Same words, entirely different emotional ground. To hand this context-dependent expressive state to a TTS engine, we need a control interface more precise than “speak in a cheerful voice.”

A three-dimensional **V-A-D space**—Valence, Arousal, Dominance; written as V-A-D here to avoid confusion with Voice Activity Detection—provides a mathematical description for such expression [44,45]. The same line can be represented differently in the two contexts:

| Context | V-A-D Coordinates | Affective Description |
|------|---------|---------|
| Opening scene: routine greeting | (+0.40, +0.30, +0.20) | Professional greeting, bright but mechanical |
| Ending scene: farewell and awakening | (−0.20, +0.50, +0.70) | A heavy yet lucid farewell, restrained but resolute |

The key design choice is that **discrete labels and continuous V-A-D coordinates should not be treated as mutually exclusive**. Discrete labels are suitable for high-level classification—what type of emotion is this? V-A-D coordinates are suitable for intensity and style control—how strong is the emotion, and what expressive flavor does it carry? The execution layer should receive a hybrid representation: first use the discrete label to determine the emotion type, then use V-A-D coordinates to control fine-grained expression.

**A pragmatic annotation strategy** is to annotate sentence-level discrete labels and key segments of expressive intent manually, while using forced alignment and automatic emotion recognition models to generate fine-grained V-A-D pseudo-labels. Human calibration is then applied only to obviously wrong segments. Sentence-level labels can draw on mature datasets such as GoEmotions [46] and EmoBank [47]. Continuous affective trajectories in film-like settings can refer to the annotation paradigms of VEATIC [48] and MovieGraphs [49].

### 4.2 Vocal “Performance”: From TTS to a Retrievable Library of Expressive Techniques

Emotional speech synthesis has recently evolved along three broad technical routes:

- **VAE-based routes** such as VITS and NaturalSpeech: fast inference and mature ecosystems, suitable for real-time dialogue systems.
- **Diffusion-based routes** such as Grad-TTS and Diff-TTS: highest generation quality and better modeling of stochastic speech details, but higher latency due to multi-step denoising.
- **Flow Matching / Rectified Flow routes**: a compromise between diffusion-quality generation and the low-latency requirements of real-time systems, and an important trend in recent generative speech modeling.

For continuous emotion control, EmoSphere-TTS [50] maps V-A-D space onto spherical coordinates, enabling continuous control of emotional style and intensity. UDDETTS [51] further unifies discrete labels and dimensional emotion spaces, achieving approximately linear control over the three V-A-D dimensions. It is one of the more systematic examples of combining continuous emotion control with LLM-based TTS.

**Technique-RAG: A Longer-Term Direction**

Current emotional TTS systems mostly remain at the granularity of whole-sentence style control. But real human performance—in speech as well as singing—is composed of identifiable technique segments: pauses, emphasis, ironic intonation, comedic timing, breathiness, falsetto, vibrato, slides.

A more technically meaningful direction is to explicitly build these “performance references” into a retrievable, composable, and transferable technique library: **Technique-RAG**. From the RAG design pattern [1], a complete Technique-RAG system can be organized as a four-stage pipeline: Planner → Retriever → Generator → Verifier. The Planner decomposes an affective execution plan into a sequence of technique requirements. The Retriever retrieves matching reference segments from the technique library. The Generator fuses the target voice identity with the retrieved technique. The Verifier checks whether the generated prosody and spectrum match the intended technique.

In singing voice synthesis, TCSinger 2 [52] already supports zero-shot transfer of six vocal techniques such as Breathy, Falsetto, and Mixed Voice. AutoStyle-TTS [53] introduces RAG into speech style matching, automatically retrieving an appropriate global style prompt based on the input text and context. These works solve parts of the problem—technique transfer and style retrieval—but they remain far from a full Technique-RAG system.

### 4.3 The Safety Substrate of Execution: Tool Execution and Safety Gates for Embodied Action

When a policy instruction involves external tool calls or physical action, safety is not a bonus feature; it is non-negotiable.

End-to-end VLA (Vision-Language-Action) models such as OpenVLA [54] are impressive in their cross-embodiment generalization, outperforming RT-2-X (55B) by 16.5% absolute success across 29 tasks. But SafeAgentBench [43] shows that the best baseline rejects only 5% of hazardous tasks. End-to-end models, in their pursuit of “what can be done,” severely underweight “what should not be done.” This creates a fundamental conflict with the requirements of the execution layer: we need mechanisms that can and **must** reject dangerous instructions, whereas the output of an end-to-end model is probabilistic by nature.

A feasible two-layer architecture is as follows. The upper layer, implemented by a VLA or LLM, performs high-level semantic understanding and task planning, producing natural-language-level action plans such as “move forward 0.5 meters while extending the hand.” The lower layer is an independent safety module that reviews each action plan for collision risk, joint limits, and emergency stop conditions. Only after passing these checks is the instruction forwarded to deterministic controllers such as PID or MPC for trajectory execution. The policy-separated runtime proposed by AEROS [55]—in which execution constraints and safety policies are enforced independently of capability logic—maps directly onto this idea. This is an instance of the **Exception Handling pattern** [1] in embodied execution: the system does not assume all inputs are safe; it actively checks whether each output violates a safety boundary.

---

## 5. From Closed-Loop Capability to Deployable Systems: Governance, Evaluation, and Repair

The preceding sections followed the perception–understanding–execution loop across its technical layers. But those discussions implicitly assume a laboratory setting. When a system that continuously observes, reasons, remembers, and acts proactively enters real deployment environments, a set of cross-layer issues immediately emerges: how should privacy boundaries be defined, how should memory be controlled, how should closed-loop performance be evaluated, and how should interaction ruptures be repaired? These are not side issues belonging to any one layer. They are global constraints that determine whether a prototype can become a deployable system.

### 5.1 Privacy and Consent: From Permission Toggles to Contextual Integrity

For an interaction system that continuously collects visual, speech, and posture data while maintaining long-term memory, privacy cannot be reduced to the question of whether camera or microphone permissions have been granted. Nissenbaum’s theory of Contextual Integrity [56] defines privacy in terms of whether information flows conform to the norms of a given social context. The same data collection behavior may be invasive in a home, acceptable in a public space, and require third-party authorization in a workplace. This framework is especially relevant for embodied interaction, because perception, memory, and proactive intervention all have different information-flow norms across homes, offices, and public settings.

In social robotics, CONFIDANT [57] implements this idea as a practical privacy controller. Rather than relying on static permission lists, it dynamically assesses information sensitivity based on dialogue metadata such as relationship, topic, and affect. Subsequent work further operationalizes Contextual Integrity as decision constraints for LLM-based assistants [58]. But another line of work reveals the practical difficulty: even state-of-the-art LLMs and VLMs are unreliable at recognizing privacy-relevant situations without specialized training [59]. LLMs may participate in privacy reasoning, but they should not be the sole executors of privacy decisions. Privacy decisions should be constrained by an auditable policy layer, with user override preserved.

### 5.2 Memory Governance: Writable, Correctable, Forgettable, and Protected Against Leakage

Section 3.3 discussed the **construction** of long-term memory: systems such as MemGPT, MemoryBank, A-MEM, and Mem0 address memory storage, retrieval, and evolution. From a deployment perspective, however, the **controllability** of memory is equally important.

First, long-term agent memory is itself an attack surface. An ACL 2025 study [60] introduces Memory Extraction Attack (MEXTRA), showing that attackers can extract sensitive user information from agent memory through carefully designed black-box prompts. This means memory systems cannot assume that only benign users will access them. MemPrivacy [61] proposes a privacy-preserving solution: sensitive spans are identified on the edge device, while the cloud processes only desensitized placeholders. This reduces the exposure surface without fully sacrificing reasoning utility. More generally, recent surveys [62] frame long-term memory security as an independent security subfield, emphasizing that writable, cross-session memory differs fundamentally from traditional static data security.

Second, **memory requires forgetting, and forgetting is not the same as deleting a cache**. Memora [63] proposes Forgetting-Aware Memory Accuracy, a metric that penalizes models for continuing to use obsolete memories—for example, when a user changes preferences, corrects a previous inference, or when a fact has expired. Effective forgetting requires both technical storage cleanup and policy-level judgment about when forgetting should occur.

Taken together, deployable memory systems should satisfy four hard constraints: **writable, correctable, forgettable, and protected against unauthorized access**. The first two are capability issues already discussed in Section 3.3; the latter two are governance issues added here.

### 5.3 Closed-Loop Evaluation: From Module Accuracy to Interaction Consequences

Section 3.4 discussed four interaction-level evaluation dimensions: timing appropriateness, relevance, intrusiveness, and policy consistency. But once we move from single interactions to continuous deployment, we need system-level closed-loop metrics.

End-to-end latency—the time from sensor input to execution output—sets an upper bound on interaction naturalness. Human conversational turn gaps are typically on the order of hundreds of milliseconds, so real-time speech and embodied systems often need streaming ASR, anticipatory prediction, and fillers or backchannels to keep perceived waiting time low; the exact threshold depends on scenario, task, and expression style. False trigger rate measures how often the system intervenes when it should not. Correct silence rate measures whether the system remains silent when it should. KnowU-Bench [38] shows that current models perform much worse on the latter than the former, because “non-action” positive examples are naturally underrepresented in training data. Long-term trust dynamics track how user trust changes across multiple sessions. Repeated false triggers can quickly consume initial trust, and once trust is depleted, recovering it is harder than establishing it.

More importantly, **module-level accuracy cannot capture the severity of interaction consequences**. ToolSandbox [42] shows that even when surface-level tool selection accuracy is high, systems can still make downstream-damaging errors in state-dependent settings involving state dependency, canonicalization, or insufficient information. SafeAgentBench [43] is even more sobering: the best baseline reaches 69% success on safe tasks but rejects only 5% of hazardous tasks. In embodied collaboration, PARTNR [70] provides 100,000 natural-language household collaboration tasks and shows that SOTA LLMs still struggle with coordination, task tracking, and error recovery. Habitat 3.0 [71] further emphasizes that embodied interaction evaluation must include human-in-the-loop social dimensions. Together, these benchmarks point to one conclusion: **closed-loop evaluation must measure interaction consequences at the system level, not substitute module-level benchmark accuracy for system performance**.

### 5.4 Interaction Repair: How a System Recovers Gracefully from Mistakes

With safety gates (Section 4.3) and closed-loop evaluation (Section 5.3) in place, one question remains: when the system does make a mistake—whether a grasp fails, the user is interrupted, an intent is misunderstood, or a response is inappropriate—can it detect and repair the interaction?

An interaction repair loop can be summarized as four stages: **detect failure or rupture → acknowledge, clarify, or apologize → replan → safely re-execute**. The signals for detection are far richer than hardware fault diagnosis: awkward facial expressions, confused tone, repeated instructions, or user corrections are all observable indicators of interaction rupture or social breakdown. The ERR@HRI 2024 Challenge [68] provides a multimodal HRI failure dataset annotated with robot mistakes, user awkwardness, and interaction ruptures. ERR@HRI 2.0 [69] further divides error detection into system-perspective and user-perspective tasks: the former detects whether the robot behavior deviates from its intended design, while the latter detects whether the user is attempting to correct the robot.

Architecturally, interaction repair is not an execution-layer-only problem. The repair loop requires the perception layer to detect anomalous signals, such as confusion in the user’s expression or a step backward; the understanding layer to decide whether and how to repair—apologize, clarify, or stay silent; and the execution layer to safely retry or degrade. The key to repair is not generating an apology, but writing the failure itself back into system state: the perception layer records abnormal signals, the understanding layer updates its inference about user intent and relationship state, and the execution layer replans under the new constraints. CHAIC [72] further requires agents to infer human collaborators’ intents and constraints before cooperative planning, providing a more realistic paradigm for the replanning stage of repair.

---

## 6. Conclusion

The discussion above suggests that building next-generation human–machine interaction systems requires breakthroughs in four directions:

**From detection to prediction.** The key shift in the perception layer is from detecting current events to predicting future events—VAP, interaction intention prediction, and turn-taking prediction all move the system from “keeping up with the user” to “waiting for the user.”

**From single turns to state.** The core move in the understanding layer is to model interaction as state tracking across time. Working memory maintains the dynamic structure of the current dialogue—topics, affect, open loops—while long-term memory maintains cross-session user understanding—preferences, relationships, personality. Policy selection should be based on the full state, not an isolated current input.

**From strategy to expression.** The execution layer must turn discrete policy labels into continuous multidimensional expression. Emotion is not a fixed sentence-level tag but a highly context-dependent V-A-D coordinate. Speech expression is not “say it in a happy voice,” but the retrieval, composition, and transfer of specific performance references from a technique library.

**From capability to governance.** Once a system can continuously perceive, remember, and act proactively, the core question is no longer only “can it do this?” It also becomes “should it do this, is it allowed to do this, and can it detect and repair the damage if it goes wrong?” Privacy and consent, memory governance, closed-loop evaluation, and interaction repair should be treated as deployment-layer constraints on par with perception, understanding, and execution—not as afterthoughts.

These four directions correspond structurally to LLM agent design patterns and broader system-governance mechanisms: the first three primarily map to agent engineering patterns such as Routing, Planning, Memory Management, and Tool Use [1], while governability further introduces deployment-layer mechanisms such as Contextual Integrity, memory governance, and interaction repair [56,60,68,69].

| Interaction System Keyword | Core Problem | Corresponding Architecture / Governance Mechanism |
|--------------|---------|---------------------|
| Predictive | Dynamic evaluation and selection of modality quality | Routing, Parallelization |
| Stateful | Memory maintenance and policy planning across time | Memory Management, Planning, Reflection |
| Executable & Safe | Safe conversion from policy to action | Tool Use, Human-in-the-Loop, Exception Handling |
| Governable | Privacy boundaries, controlled memory, closed-loop evaluation, failure recovery | Contextual Integrity, Memory Governance, Repair Loops |

Ultimately, embodied intelligent interaction tests whether a system can integrate real-time perception, reasoning, decision-making, and action under the dual constraints of the physical world and social norms. Combining the architectural methods of agent design patterns with the domain knowledge of interaction systems may be one of the most practical paths from “systems that can answer questions” to “systems that can truly interact.”

---

## References

### Surveys and Design Patterns

[1] Antonio Gullí. *Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems*. Springer, 2025. — Systematizes agentic design patterns such as Prompt Chaining, Routing, Parallelization, Reflection, Tool Use, Planning, Memory Management, Human-in-the-Loop, RAG, and Exception Handling. The bilingual Chinese edition is a community translation project covering 21 core patterns.

### Interaction Intention Prediction and Turn-Taking Modeling

[2] Ekstedt & Skantze. *Voice Activity Projection: Self-supervised Learning of Turn-taking Events*. INTERSPEECH 2022. — Foundational VAP work that formulates turn-taking as continuous-time prediction and enables self-supervised training.

[3] *Self-Supervised Prediction of the Intention to Interact with a Service Robot*. arXiv 2023. — Achieves AUROC > 0.9 more than three seconds before interaction in a coffee-shop scenario.

[4] *MM-VAP: Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction*. ACL Findings 2025. — Incorporates visual cues into VAP and improves performance from 79% to 84%.

[5] *PAR-D: People Approaching Robots Database*. ICMI 2024. — Benchmark dataset for pedestrian approach intention prediction.

### Active Speaker Detection and Addressee Recognition

[6] Tao et al. *TalkNet: Active Speaker Detection*. ACM MM 2021. — Foundational ASD work introducing audio-visual synchronization into active speaker detection.

[7] Wang et al. *LoCoNet: Long-Short Context Network for ASD*. CVPR 2024. — Decomposes temporal modeling into long-term same-speaker and short-term multi-speaker interaction modules.

[8] Liao et al. *LR-ASD: Lightweight and Robust Network for ASD*. IJCV 2025. — Uses only 0.84M parameters and generalizes better across datasets than LoCoNet, making it attractive for edge deployment.

[9] *UniTalk: A Real-World Benchmark for ASD*. 2025. — Builds a real-world dataset covering noise, overlapping speakers, and low-resource languages, exposing blind spots in traditional benchmarks.

[64] Inoue et al. *An LLM Benchmark for Addressee Recognition in Multi-modal Multi-party Dialogue*. IWSDS 2025. — Shows that explicit addressee labels appear in only about 20% of multi-party dialogue turns and that GPT-4o performs only slightly above chance.

### Far-Field Speech and Audio-Visual Fusion

[10] *Whisper-FEST: Single-Channel Far-Field Enhanced Speech-to-Text without Parallel Data*. Samsung Research, 2026. — A single-channel far-field speech frontend for Whisper S2T without paired near-/far-field data.

[11] *LABNet: Lightweight Attentive Beamforming Network*. arXiv 2025. — Real-time speech enhancement for arbitrary ad-hoc microphone arrays.

[12] *Interpretable Binaural Deep Beamforming Guided by Time-Varying RTF*. 2025. — Performs dynamic beamforming by tracking the time-varying relative transfer function of a moving speaker.

[13] *Cocktail-Party Audio-Visual Speech Recognition*. Interspeech 2025. — Reduces WER from 119% to 39.2% under extreme noise.

[14] *Audio Super-Resolution with Latent Bridge Models*. 2025. — Uses generative modeling to restore high-frequency components lost in far-field speech.

[15] *Treble10: High-Quality Dataset for Far-Field Speech*. 2025. — Provides broadband room impulse responses across 3,100 acoustic configurations.

### Dynamic Multimodal Fusion

[16] Ernst & Banks. *Humans Integrate Visual and Haptic Information in a Statistically Optimal Fashion*. Nature 2002. — Theoretical basis for MLE-style multisensory integration.

[17] Zhang et al. *QMF: Quality-aware Multimodal Fusion*. ICML 2023. — Theoretically shows that fusion weights should adapt to modality quality, using uncertainty as a proxy.

[18] Gao et al. *Embracing Unimodal Aleatoric Uncertainty for Robust Multimodal Fusion*. CVPR 2024. — Explicitly quantifies unimodal aleatoric uncertainty and uses it to learn more robust unimodal and fused representations.

[19] Wei et al. *Unbiased Dynamic Multimodal Fusion*. CVPR 2026 Findings. — Addresses the Double Penalty problem for weaker modalities in dynamic fusion.

[20] *EMOE: Modality-Specific Enhanced Dynamic Emotion Experts*. CVPR 2025. — Moves from continuous weighting toward router-based dynamic selection.

### Emotion Recognition and Reasoning

[21] *ESCoT: Towards Interpretable Emotional Support Dialogue Systems*. ACL 2024. — Decomposes emotional support into identifying, understanding, and regulating stages.

[22] Zhao et al. *Integrating Fine-Grained Audio-Visual Evidence for Robust Multimodal Emotion Reasoning (SABER-LLM)*. arXiv 2026. — Proposes structured evidence decomposition, moving multimodal affect analysis from static classification toward evidence-driven generative reasoning.

[23] Cheng et al. *Emotion-LLaMA: Multimodal Emotion Recognition with Instruction Tuning*. NeurIPS 2024. — Representative early work bringing instruction tuning into multimodal emotion reasoning.

[24] *R1-Omni: Explainable Omni-Multimodal Emotion Recognition with RL*. 2025. — Uses RLVR to make the model explicitly analyze each modality’s contribution to emotion judgments.

[25] *EMO-R3: Reflective Reinforcement Learning for Emotional Reasoning*. 2026. — Introduces reflective self-correction for emotional reasoning.

[26] *MC-EIU: Emotion and Intent Joint Understanding in Multimodal Conversation*. NeurIPS 2024. — Defines a joint annotation paradigm with 7 emotion classes and 9 intent classes.

[27] *IntentionESC: Intention-Centered Emotional Support*. ACL 2025. — Builds an intent-centered reasoning chain for emotional support.

### Strategy Selection and Constraints

[28] *Can Large Language Models be Good Emotional Supporter?* ACL 2024 Outstanding Paper. — Reveals systematic strategy preference bias in LLMs for emotional support.

[29] *STRIDE-ED: Strategy-Grounded Stepwise Reasoning for Empathetic Dialogue*. arXiv 2026. — Uses strategy-aware multi-objective reinforcement learning to align model behavior with policy constraints.

### Dialogue State Machines and Long-Term Memory

[30] *FnCTOD: LLMs as Zero-shot Dialogue State Tracker through Function Calling*. ACL 2024. — Formulates dialogue state tracking as function calling.

[31] Packer et al. *MemGPT/Letta: Towards LLMs as Operating Systems*. 2023. — Defines a virtual context-management framework for agent memory.

[32] *MemoryBank: Enhancing LLMs with Long-Term Memory*. AAAI 2024. — Introduces a forgetting-curve mechanism.

[33] *A-MEM: Agentic Memory for LLM Agents*. NeurIPS 2025. — Builds a Zettelkasten-style evolving memory network.

[34] Chhikara et al. *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*. ECAI 2025. — Production-oriented general memory layer.

[35] *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*. ICLR 2025. — Benchmark covering five core long-term memory capabilities.

[36] *LoCoMo: Evaluating Very Long-Term Conversational Memory of LLM Agents*. ACL 2024. — Long-context dialogue memory benchmark spanning 32 sessions.

[37] *PersonaMem-v2: Learning Implicit User Personas via Agentic Memory*. 2025. — Achieves state-of-the-art memory distillation using 1/16 of the input tokens.

### Proactive Interaction

[38] *KnowU-Bench: Towards Interactive, Proactive, and Personalized Mobile Agent Evaluation*. 2026. — Finds that current models tend to act by default, even when they should remain silent.

[39] *Proactive Agent: Shifting LLM Agents to Active Assistance*. ICLR 2025. — Incorporates user rejection signals into proactive policy training.

[40] *MMDuet2: Enhancing Proactive Interaction with Multi-Turn RL*. ICLR 2026. — Incorporates interaction timing into the RL objective.

[41] *Plug-and-Play Policy Planner for LLM-Powered Dialogue Agents*. ICLR 2024. — Separates policy planning from response generation.

### Proactive Interaction, Interruption, and Non-Intrusive Assistance

[65] Banerjee et al. *Effects of Interruptibility-Aware Robot Behavior*. 2018. — Shows that interruptibility-aware robots improve users’ perception of robot social competence.

[66] Cao et al. *Interruption Handling for Conversational Robots*. arXiv 2025. — Detects and categorizes user interruptions in real time, handling 93.69% of interruptions in experiments.

[67] *Assistance Without Interruption: A Benchmark for Non-Intrusive Human-Robot Assistance*. arXiv 2026. — Formalizes non-intrusive assistance as a distinct HRI paradigm.

### Tool Use and Safety

[42] *ToolSandbox: Stateful, Conversational Evaluation for LLM Tool Use*. NAACL 2025. — Apple’s stateful tool-use benchmark.

[43] *SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents*. 2024. — Best baseline rejects only 5% of hazardous tasks.

### Affective Theory and Datasets

[44] Russell. *A Circumplex Model of Affect*. 1980. — Foundational Valence–Arousal circumplex model.

[45] Mehrabian. *The PAD Model*. 1974–1996. — Adds Dominance to the Valence–Arousal space.

[46] *GoEmotions: A Dataset of Fine-Grained Emotions*. ACL 2020. — 58K examples annotated with 27 fine-grained emotion categories.

[47] *EmoBank: Dimensional Emotion Analysis*. EACL 2017. — 10K sentences annotated with V-A-D dimensions.

[48] *VEATIC: Video-based Emotion and Affect Tracking in Context*. 2023. — 124 videos with frame-level continuous V-A annotations.

[49] *MovieGraphs: Towards Understanding Human-Centric Situations from Videos*. CVPR 2018. — 51 films annotated with character emotion, situation, and relationship graphs.

### Emotional Speech Synthesis and Technique Transfer

[50] *EmoSphere-TTS: Emotional Style and Intensity Modeling via Spherical Emotion Vector*. Interspeech 2024. — Maps V-A-D space to spherical coordinates for continuous emotion control.

[51] *UDDETTS: Unifying Discrete and Dimensional Emotions for Controllable TTS*. 2025. — Unifies discrete emotion labels and dimensional emotion space. The original uses ADV order (Arousal–Dominance–Valence); this article writes it uniformly as V-A-D. Enables approximately linear continuous emotion control.

[52] *TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis*. ACL Findings 2025. — Zero-shot transfer of six vocal techniques.

[53] *AutoStyle-TTS: RAG-based Automatic Style Matching TTS*. ICME 2025. — Representative work introducing RAG into speech style matching.

### Embodied Action and VLA

[54] *OpenVLA: An Open-Source Vision-Language-Action Model*. 2024. — A 7B open-source VLA model that outperforms RT-2-X by 16.5% across 29 cross-embodiment tasks.

[55] *AEROS: A Single-Agent Operating Architecture with Embodied Capability Modules*. 2026. — Policy-separated runtime supporting dynamic replanning and safety gating.

### Privacy and Memory Governance

[56] Nissenbaum. *Privacy in Context: Technology, Policy, and the Integrity of Social Life*. 2010. — Foundational work on Contextual Integrity, defining privacy as socially appropriate information flow.

[57] Tang et al. *CONFIDANT: A Privacy Controller for Social Robots*. HRI 2022. — A social-robot privacy controller that dynamically assesses information sensitivity using dialogue metadata such as relationship, topic, and affect.

[58] *Operationalizing Contextual Integrity in Privacy-Conscious Assistants*. arXiv 2024. — Operationalizes Contextual Integrity as decision constraints for LLM assistants.

[59] *Benchmarking LLM Privacy Recognition for Social Robot Decision Making*. arXiv 2025. — Builds home-robot privacy scenarios using Contextual Integrity and finds limited alignment between out-of-the-box LLMs and human privacy judgments.

[60] Wang et al. *Unveiling Privacy Risks in LLM Agent Memory*. ACL 2025. — Proposes Memory Extraction Attack, showing that sensitive information can be extracted from long-term agent memory via black-box prompting.

[61] *MemPrivacy: Privacy-Preserving Personalized Memory Management for Edge-Cloud Agents*. arXiv 2026. — Identifies sensitive spans on edge devices and sends only placeholders to the cloud.

[62] *A Survey on Long-Term Memory Security in LLM Agents: Attacks, Defenses, and Governance Across the Memory Lifecycle*. arXiv 2026. — Frames long-term agent memory security as an independent subfield covering write, store, retrieve, and forget lifecycle stages.

[63] Uddin et al. *From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents*. arXiv 2026. — Proposes Memora and FAMA, penalizing the use of obsolete memories.

### Closed-Loop Evaluation and Interaction Repair

[68] *ERR@HRI 2024 Challenge: Multimodal Detection of Errors and Failures in Human-Robot Interactions*. arXiv 2024. — Multimodal HRI failure dataset annotated with robot mistakes, user awkwardness, and interaction ruptures.

[69] *ERR@HRI 2.0 Challenge: Multimodal Detection of Errors and Failures in Human-Robot Conversations*. arXiv 2025. — Sixteen hours of dyadic human–robot conversation with system-perspective and user-perspective subtasks.

[70] *PARTNR: A Benchmark for Planning and Reasoning in Embodied Multi-agent Tasks*. arXiv 2024. — 100,000 natural-language household human–robot collaboration tasks; SOTA LLMs remain limited in coordination and error recovery.

[71] *Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots*. ICLR 2024. — Emphasizes that embodied interaction evaluation must include human-in-the-loop social dimensions.

[72] *Constrained Human-AI Cooperation: An Inclusive Embodied Social Intelligence Challenge (CHAIC)*. arXiv 2024. — Requires agents to infer human intent and constraints before cooperative planning.
