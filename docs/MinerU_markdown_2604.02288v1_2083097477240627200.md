# Unifying Group-Relative and Self-Distillation Policy Optimization via Sample Routing

Gengsheng Li <sup>1,2,∗</sup>, Tianyu Yang <sup>1,2,∗</sup>, Junfeng Fang <sup>3</sup>, Mingyang Song <sup>4</sup>, Mao Zheng <sup>4</sup>, Haiyun Guo <sup>1,2</sup>, Dan Zhang <sup>3</sup>, Jinqiao Wang <sup>1,2,5</sup>, Tat-Seng Chua 3 

<sup>1</sup>Foundation Model Research Center, Institute of Automation, Chinese Academy of Sciences 

<sup>2</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences 

<sup>3</sup>National University of Singapore 

<sup>4</sup>Tencent 

<sup>5</sup>Wuhan AI Research 

<sup>∗</sup>Equal contribution 

Correspondence: haiyun.guo@nlpr.ia.ac.cn, zhangdan25@nus.edu.sg 

## Abstract

Reinforcement learning with verifiable rewards (RLVR) has become a standard paradigm for post-training large language models. While Group Relative Policy Optimization (GRPO) is widely adopted, its coarse credit assignment uniformly penalizes failed rollouts, lacking the token-level focus needed to efficiently address specific deviations. Self-Distillation Policy Optimization (SDPO) addresses this by providing denser, more targeted logit-level supervision that facilitates rapid early improvement, yet it frequently collapses during prolonged training. We trace this latestage instability to two intrinsic flaws: self-distillation on already-correct samples introduces optimization ambiguity, and the self-teacher’s signal reliability progressively degrades. To resolve these issues, we propose Sample-Routed Policy Optimization (SRPO), a unified on-policy framework that routes correct samples to GRPO’s reward-aligned reinforcement and failed samples to SDPO’s targeted logit-level correction. SRPO further incorporates an entropy-aware dynamic weighting mechanism to suppress high-entropy, unreliable distillation targets while emphasizing confident ones. Evaluated across five benchmarks and two model scales, SRPO achieves both the rapid early improvement of SDPO and the long-horizon stability of GRPO. It consistently surpasses the peak performance of both baselines, raising the five-benchmark average on Qwen3-8B by 3.4% over GRPO and 6.3% over SDPO, while simultaneously yielding moderate response lengths and lowering per-step compute cost by up to 17.2%. 

## 1 Introduction

Post-training large language models through reinforcement learning with verifiable rewards (RLVR) has emerged as a standard approach for improving reasoning and problem-solving capabilities (Jaech et al., 2024; Guo et al., 2025; Team et al., 2025; Yang et al., 2025). Among RLVR methods, Group Relative Policy Optimization (GRPO; Shao et al., 2024) is widely adopted for its simplicity and stability. GRPO estimates advantages by normalizing outcome rewards across a group of rollouts, producing a single scalar advantage that is applied uniformly to every token in a rollout. For successful rollouts, this uniform assignment is gen erally appropriate, as most intermediate steps support the correct outcome. Conversely, for failed rollouts, this coarse token credit assignment distributes a uniform penalty across the entire sequence. Consequently, the policy update lacks the focus needed to address specific deviations, which ultimately diminishes sample efficiency and slows convergence (Khandoga et al., 2026; Kumar et al., 2026; Parthasarathi et al., 2025). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-28/ced84c0c-2995-40f4-9aa7-cd7c8c72d160/726d4c79db7e120e818d9eb845635e3090ab83d173213180cb125de6179610a8.jpg)



Figure 1: Training dynamics and diagnostic analysis on Chemistry with Qwen3-8B. (a) SDPO improves faster than GRPO in early training, but is later overtaken and collapses, whereas SRPO achieves both rapid initial improvement and stable long-horizon optimization. (b) Restricting SDPO updates to incorrect samples retains most of its overall benefit, whereas applying SDPO only to correct samples degrades performance and destabilizes training, supporting the necessity of sample routing. (c) The self-teacher’s token-level entropy rises during training, indicating that the distillation signal becomes increasingly dominated by uncertain predictions. Curves show a 5-step rolling mean and shaded bands denote ±1 std.


To overcome this sparsity in credit assignment, recent work has turned to on-policy distillation (Agarwal et al., 2024; Lu & Lab, 2025) and self-distillation (Hübotter et al., 2026; Zhao et al., 2026; Ye et al., 2026; Song et al., 2026), which provide dense logit-level guidance for more precise optimization. Self-distillation removes the need for an external teacher by conditioning the model on privileged context (e.g., the correct solution) to supervise its own generated trajectories. A prominent example, Self-Distillation Policy Optimization (SDPO; Hübotter et al., 2026), often achieves much faster early convergence in complex domains such as scientific reasoning and agentic tool use. However, as shown in Figure 1(a), this early advantage is not sustained: under prolonged training, SDPO is consistently surpassed by GRPO and often suffers catastrophic collapse. While recent work Kim et al. (2026) attributes similar instability in math domains to the suppression of epistemic verbalization, we provide a complementary diagnosis from the perspective of the distillation signal and attribute this instability to two intrinsic causes within the self-distillation mechanism. 

First, self-distillation on already-correct samples introduces optimization ambiguity. In SDPO, the self-teacher is conditioned on a successful sibling rollout to provide dense, logit-level targets. While this is effective for correcting failed samples, it can be counterproductive for already correct ones: forcing a successful rollout to match a different successful sibling imposes arbitrary logit-level preferences between reward-equivalent reasoning paths. Figure 1(b) supports this view: restricting SDPO updates to failed samples retains most of its benefit, whereas applying it only to correct samples degrades performance and accelerates collapse. 

Second, the quality of the self-teacher’s distillation signal degrades as training progresses. As the gap between the self-teacher and student narrows during training (Hübotter et al., 2026), the distillation signal becomes less informative, while the self-teacher’s token-level entropy rises (Figure 1(c)), indicating increasingly uncertain predictions. This degradation in informativeness and reliability contributes directly to the late-stage instability of SDPO. 

These observations suggest that GRPO and SDPO have complementary optimization properties. For correct samples, the sequence-level credit assignment of GRPO is usually sufficient, and its Monte Carlo advantages robustly anchor the policy update toward expected reward maximization (Zhang et al., 2024; Hübotter et al., 2026). But for failed samples with localized reasoning errors, dense logit-level correction of SDPO is more effective and avoids the ambiguity above when restricted to failed trajectories. Based on this insight, we introduce Sample-Routed Policy Optimization (SRPO), a unified on-policy framework that routes correct samples to a GRPO branch and failed samples with available teacher information to an SDPO branch. To mitigate late-stage signal degradation, we further equip the SDPO branch with an entropy-aware dynamic weighting mechanism that downweights uncertain distillation targets and emphasizes reliable corrections. This design enables rapid correction early in training while increasingly relying on reward-aligned reinforcement as more rollouts become correct, thereby stabilizing late-stage optimization. 

Evaluated across five benchmarks following the protocol of Hübotter et al. (2026) and two Qwen3 model scales (Yang et al., 2025), SRPO consistently achieves the highest peak performance. Specifically, it raises the five-benchmark average on Qwen3-8B to $7 7 . 4 \% \dot { ( } + 3 . 4 $ over GRPO, +6.3 over SDPO) and on Qwen3-4B to 74.2% (+4.5 over GRPO, +7.5 over SDPO). Furthermore, SRPO maintains a moderate response length, avoiding both the verbosity of GRPO and the excessive brevity of pure SDPO, a phenomenon recently linked to degraded epistemic reasoning (Kim et al., 2026). It also reduces per-step compute cost by up to 17.2% over long training horizons. Our contributions are threefold: 

• We identify two intrinsic causes of late-stage instability in SDPO: self-distillation on already-correct samples introduces optimization ambiguity, and the quality of the self-teacher’s distillation signal progressively degrades. 

• We propose SRPO, a unified framework that bridges group-relative and selfdistillation policy optimization by routing each sample to the optimization signal best suited to its learning status, augmented by entropy-aware dynamic weighting to suppress unreliable distillation targets and emphasize reliable ones. 

• We demonstrate across five benchmarks and two model scales that SRPO improves early training efficiency, long-horizon stability, and peak accuracy, while simultaneously yielding moderate response lengths and lower per-step compute time. 

## 2 Preliminaries

We review the two optimization paradigms unified by SRPO. Throughout, let x denote a prompt, $\{ y _ { i } \} _ { i = 1 } ^ { G }$ a group of G on-policy rollouts sampled from the current policy $\pi _ { \theta } ,$ and $\{ r _ { i } \} _ { i = 1 } ^ { G }$ the corresponding scalar rewards. 

## 2.1 Group Relative Policy Optimization

GRPO is a policy-gradient method for post-training with verifiable rewards that eliminates the need for a learned critic. For each prompt x, the policy generates a group of G rollouts and obtains a scalar reward for each. The advantage of rollout i is estimated by normalizing its reward relative to the group: 

$$
A _ {i} ^ {\mathrm{GRPO}} = \frac {r _ {i} - \bar {r}}{\sigma_ {r} + \epsilon},
$$

where r¯ and $\sigma _ { r }$ are the mean and standard deviation of $\{ r _ { i } \} _ { i = 1 } ^ { G }$ . The policy is updated via a clipped surrogate objective: 

$$
\mathcal {L} _ {\mathrm{GRPO}} (\theta) = \mathbb {E} \left[ \min \left(\rho_ {i, t} (\theta) A _ {i} ^ {\mathrm{GRPO}}, \operatorname{clip} \left(\rho_ {i, t} (\theta), 1 - \varepsilon , 1 + \varepsilon\right) A _ {i} ^ {\mathrm{GRPO}}\right) \right],
$$

where $\rho _ { i , t } ( \theta ) = \pi _ { \theta } ( y _ { i , t } \mid x , y _ { i , < t } ) / \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i , t } \mid x , y _ { i , < t } )$ is the importance-sampling ratio at token position t of rollout i. Because $A _ { i } ^ { \mathrm { \tiny \mathrm { \mathrm { G R P O } } } }$ is a sequence-level quantity assigned uniformly to every token in a rollout, GRPO delivers reward-aligned yet coarse-grained credit assignment: it reliably reinforces or suppresses entire rollouts, but cannot identify which individual tokens are responsible for the outcome. 

## 2.2 Self-Distillation Policy Optimization

SDPO augments the reward signal with dense logit-level supervision derived from selfdistillation. Rather than relying solely on scalar rewards, it constructs a feedback-conditioned self-teacher from the same model. The student distribution is $\pi _ { \theta } ( \cdot \mid x )$ , while the self-teacher distribution is $\pi _ { \theta } ( \cdot \mid x , f )$ , where f denotes auxiliary information obtained during the rollout process (e.g., a successful sibling rollout from the same group or environment feedback such as execution traces). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-28/ced84c0c-2995-40f4-9aa7-cd7c8c72d160/0d5ba987db6a9a59f4b76298815287a4194f1087753c386b47e2cfc970b51395.jpg)



Figure 2: Overview of SRPO. Given a prompt $x ,$ the policy $\pi _ { \theta }$ generates a group of on-policy rollouts. $\mathrm { A }$ correctness check routes each rollout to one of two branches: correct samples are sent to the GRPO branch (top), where group-relative advantages provide a reward-aligned policy update; incorrect samples with available teacher information are sent to the SDPO branch (bottom), where a feedback-conditioned self-teacher produces logit-level distillation targets via $\mathrm { K L } ( \dot { P } \parallel$ stopgrad(Q)) for dense corrective supervision.


Given a rollout $y _ { i } ,$ SDPO trains the student to match the self-teacher’s distribution along the original trajectory by minimizing a logit-level divergence. Using KL divergence as an illustration: 

$$
\mathcal {L} _ {\mathrm{SDPO}} (\theta) = \sum_ {t} \mathrm{KL} \left(\pi_ {\theta} (\cdot | x, y _ {i, <   t}) \| \text {stopgrad} [ \pi_ {\theta} (\cdot | x, f, y _ {i, <   t}) ]\right),
$$

where the specific divergence may also be instantiated as the reverse KL or Jensen–Shannon divergence, and the self-teacher parameters are maintained as an exponential moving average (EMA) of the student (Hübotter et al., 2026). 

The self-teacher does not generate a new trajectory; it re-scores the student’s own rollout under the enriched context $\left( x , f \right)$ , so the entire procedure remains on-policy while providing dense logit-level guidance on the model’s own rollouts. 

The two methods differ fundamentally in their supervision signals. GRPO is reward-driven: its advantage is derived from outcome rewards via group normalization, producing updates that are directly aligned with expected return but uniformly distributed across tokens. SDPO is teacher-driven: its advantage is induced by the discrepancy between the self-teacher and student distributions, yielding dense logit-level guidance whose quality depends on the self-teacher. The complementarity between coarse, reward-aligned updates and dense, teacher-dependent guidance motivates SRPO, which routes each sample to the supervision signal best suited to its learning needs. 

## 3 The SRPO

SRPO is a unified on-policy framework that routes each rollout to the supervision signal best suited to its learning status. Correct rollouts are optimized with GRPO for reward-aligned re inforcement; incorrect rollouts with available teacher information are optimized with SDPO for dense logit-level correction. An entropy-aware dynamic weighting mechanism further modulates token-level contributions on the SDPO branch, suppressing unreliable distillation targets while emphasizing confident ones. Figure 2 illustrates the overall framework. 

## 3.1 Sample-Level Routing

For each rollout $y _ { i } ,$ we define two binary indicators: a correctness flag $c _ { i } = \mathbf { 1 }$ [y<sub>i</sub> is correct] and a teacher-availability flag $m _ { i } = \mathbf { 1 }$ [teacher information is available for $y _ { i } ]$ . The routing mask is then 

$$
z _ {i} ^ {\mathrm{SDPO}} = (1 - c _ {i}) m _ {i}, \qquad z _ {i} ^ {\mathrm{GRPO}} = 1 - z _ {i} ^ {\mathrm{SDPO}}.
$$

That is, only incorrect rollouts with available teacher information are routed to the SDPO branch; all remaining rollouts are optimized with GRPO. 

This routing does not alter the underlying policy-gradient structure, because both branches update the same policy on the same on-policy trajectories, with only the form of the advantage estimator differing. For GRPO, the gradient takes the standard policy-gradient form 

$$
\nabla_ {\theta} \mathcal {L} _ {\mathrm{GRPO}} = - \mathbb {E} \left[ \sum_ {t} \nabla_ {\theta} \log \pi_ {\theta} (y _ {t} \mid x, y _ {<   t}) \cdot A _ {i} ^ {\mathrm{GRPO}} \right],
$$

where the sequence-level advantage $A _ { i } ^ { \mathrm { G R P O } }$ is shared across all tokens in rollout i. For SDPO, prior work (Hübotter et al., 2026) shows that distillation gradient admits an analogous form 

$$
- \nabla_ {\theta} \mathcal {L} _ {\mathrm{SDPO}} = \mathbb {E} \left[ \sum_ {t} \sum_ {v \in \mathcal {V}} \nabla_ {\theta} \log \pi_ {\theta} (v \mid x, y _ {<   t}) \cdot A _ {t} ^ {\mathrm{SDPO}} (v) \right],
$$

where the logit-level advantage $A _ { t } ^ { \mathrm { S D P O } } ( v )$ is induced by the discrepancy between the selfteacher and student distributions. The two methods can thus be viewed as advantage estimators at different granularities (reward-derived and sequence-level versus teacherderived and logit-level), and sample routing simply selects the more appropriate estimator for each sample. 

## 3.2 Dynamic-Weighted SDPO

Even within the SDPO branch, teacher supervision is not equally reliable across tokens: low-entropy predictions typically provide clear corrective signals, whereas high-entropy predictions are more likely to introduce noise. We therefore introduce entropy-aware dynamic weighting, which reweights the SDPO loss at the token level according to teacher entropy. For brevity, we refer to this variant as Dynamic-Weighted SDPO (DW-SDPO) throughout this section. 

Let $q _ { i , t } ( v ) = \pi _ { \theta } ( v \mid x , f _ { i } , y _ { i , < t } )$ denote the self-teacher distribution at position t of rollout $i ,$ and let 

$$
H _ {i, t} = - \sum_ {v \in \mathcal {V}} q _ {i, t} (v) \log q _ {i, t} (v)
$$

be its entropy. We define the unnormalized weight $\tilde { w } _ { i , t } = \exp ( - \beta H _ { i , t } )$ , where $\beta > 0$ controls sensitivity to entropy differences, and normalize over all valid SDPO tokens to preserve the overall loss scale: 

$$
w _ {i, t} = \frac {\tilde {w} _ {i , t}}{\frac {1}{| \Omega_ {\mathrm{sdpo}} |} \sum_ {(j , s) \in \Omega_ {\mathrm{sdpo}}} \tilde {w} _ {j , s}},
$$

where $\Omega _ { \mathrm { { s d p o } } }$ is the set of valid tokens routed to the SDPO branch. The weighted token loss is then $\ell _ { i , t } ^ { \mathrm { D W - S D P O } } = w _ { i , t } \ell _ { i , t } ^ { \mathrm { S D P O } }$ , where $\ell _ { i , t } ^ { \mathrm { S D P O } }$ is the base SDPO token loss. This reweighting does not alter the functional form of SDPO; it only modulates each token’s contribution according to teacher confidence, emphasizing reliable corrections while suppressing uncertain ones. 

## 3.3 Training Objective

Let $\ell _ { i , t } ^ { \mathrm { G R P O } }$ denote the token-level GRPO loss (the sequence-level advantage distributed over valid response tokens) and $\ell _ { i , t } ^ { \mathrm { D W - S D P O } }$ the weighted SDPO loss defined above. The combined objective is 

$$
\mathcal {L} _ {\mathrm{final}} = \frac {\sum_ {i , t} z _ {i} ^ {\mathrm{GRPO}} \ell_ {i , t} ^ {\mathrm{GRPO}} + \sum_ {i , t} z _ {i} ^ {\mathrm{SDPO}} \ell_ {i , t} ^ {\mathrm{DW-SDPO}}}{\sum_ {i , t} z _ {i} ^ {\mathrm{GRPO}} + \sum_ {i , t} z _ {i} ^ {\mathrm{SDPO}}},
$$

Algorithm 1 Sample-Routed Policy Optimization (SRPO)

Require: Policy $\pi_{\theta}$ ; dataset of prompts D; rollout number G; environment for reward and feedback

1: repeat

2: Sample a prompt x from D

3: Sample rollouts $\{y_{i}\}_{i=1}^{G} \sim \pi_{\theta}(\cdot \mid x)$ 4: Evaluate $\{y_{i}\}_{i=1}^{G}$ in the environment to obtain rewards $\{r_{i}\}_{i=1}^{G}$ 5: Construct teacher information $\{f_{i}\}_{i=1}^{G}$ from successful sibling rollouts and/or environment feedback

6: for i = 1 to G do

7: if $y_{i}$ is incorrect and teacher information is available then

8: Compute teacher distribution $q_{i,t}(v) = \pi_{\theta}(v \mid x, f_{i}, y_{i, <t})$ 9: Compute weighted SDPO loss $\ell_{i,t}^{DW-SDPO}$ 10: else

11: Compute GRPO loss $\ell_{i,t}^{GRPO}$ 12: end if

13: end for

14: Aggregate routed losses over valid response tokens

15: Update $\theta$ by gradient descent

16: until converged 

where all summations over t are restricted to valid response tokens. The denominator normalizes by the total number of routed tokens, so each branch contributes in proportion to the tokens it covers. This avoids introducing an additional mixing hyperparameter and naturally adapts to the evolving sample composition: early in training, when failures are frequent, more tokens flow through the SDPO branch, giving dense correction a larger effective weight; as the policy improves and more rollouts succeed, the GRPO branch dominates, anchoring the update to the reward objective. 

Algorithm 1 summarizes the full training procedure. 

## 4 Experiments

## 4.1 Experimental Setup

Model We use instruct-tuned base models from the Qwen3 family (Yang et al., 2025) at two scales: Qwen3-4B and Qwen3-8B. This setting allows us to examine whether the behavior of SRPO is consistent across model sizes. Unless otherwise noted, analyses other than the main performance comparison are conducted at the 8B scale. 

Datasets We follow the evaluation setup of SDPO and consider five benchmarks: Chemistry, Physics, Biology, Materials, and Tool Use. The first four are science question-answering tasks built from the reasoning subsets of SciKnowEval (Feng et al., 2024) and target undergraduatelevel scientific reasoning in different domains. Tool Use evaluates whether the model can map a user request and a tool specification to the correct tool call, using ToolAlpaca (Tang et al., 2023). Following SDPO, we perform a train-test split on each benchmark to evaluate in-domain generalization. 

Baselines We compare against two baselines: (1) GRPO, a strengthened implementation of GRPO (Shao et al., 2024) following recent best practices (Olmo et al., 2025; Khatri et al., 2025), including asymmetric clipping (Yu et al., 2025), unbiased advantage normalization (Liu et al., 2025), and off-policy correction for distributed inference (Yao et al., 2025); and (2) SDPO, which replaces reward-only supervision with self-distillation from a feedbackconditioned self-teacher and provides a finer-grained but potentially biased training signal. In our experiments, SDPO uses successful sibling rollouts within the same group as teacher information for failed samples. 


Table 1: Main results on five benchmarks at three training budgets. Each entry reports the highest achieved avg@16 accuracy (%) within the corresponding wall-clock budget. The last three columns report the mean over the five benchmarks. Within each model scale, the best result in each column is in bold and the second-best is underlined.


<table><tr><td rowspan="2"></td><td colspan="3">Chemistry</td><td colspan="3">Physics</td><td colspan="3">Biology</td><td colspan="3">Materials</td><td colspan="3">Tool Use</td><td colspan="3">Average</td></tr><tr><td>1h</td><td>5h</td><td>10h</td><td>1h</td><td>5h</td><td>10h</td><td>1h</td><td>5h</td><td>10h</td><td>1h</td><td>5h</td><td>10h</td><td>1h</td><td>5h</td><td>10h</td><td>1h</td><td>5h</td><td>10h</td></tr><tr><td>Qwen3-8B</td><td colspan="3">41.1</td><td colspan="3">58.7</td><td colspan="3">30.5</td><td colspan="3">59.3</td><td colspan="3">57.9</td><td colspan="3">49.5</td></tr><tr><td>+ GRPO</td><td>62.1</td><td>75.9</td><td>78.9</td><td>61.0</td><td>72.3</td><td>73.6</td><td>46.9</td><td>68.1</td><td>70.6</td><td>74.7</td><td>77.6</td><td>77.8</td><td>64.3</td><td>68.5</td><td>69.0</td><td>61.8</td><td>72.5</td><td>74.0</td></tr><tr><td>+ SDPO</td><td>71.6</td><td>80.6</td><td>80.6</td><td>67.6</td><td>74.0</td><td>74.0</td><td>52.1</td><td>58.5</td><td>58.5</td><td>68.1</td><td>76.6</td><td>76.6</td><td>64.8</td><td>65.7</td><td>65.7</td><td>64.8</td><td>71.1</td><td>71.1</td></tr><tr><td>+ SRPO</td><td>69.2</td><td>81.8</td><td>83.0</td><td>69.5</td><td>77.1</td><td>78.4</td><td>55.8</td><td>68.3</td><td>72.8</td><td>74.9</td><td>79.2</td><td>81.5</td><td>65.2</td><td>71.2</td><td>71.2</td><td>66.9</td><td>75.5</td><td>77.4</td></tr><tr><td>Qwen3-4B</td><td colspan="3">43.6</td><td colspan="3">59.8</td><td colspan="3">30.8</td><td colspan="3">61.2</td><td colspan="3">58.8</td><td colspan="3">50.8</td></tr><tr><td>+ GRPO</td><td>64.1</td><td>76.9</td><td>78.3</td><td>64.8</td><td>71.9</td><td>71.9</td><td>39.1</td><td>51.6</td><td>55.5</td><td>78.0</td><td>78.9</td><td>80.1</td><td>62.9</td><td>62.9</td><td>62.9</td><td>61.8</td><td>68.4</td><td>69.7</td></tr><tr><td>+ SDPO</td><td>70.0</td><td>77.3</td><td>77.3</td><td>65.4</td><td>66.7</td><td>66.7</td><td>54.0</td><td>54.0</td><td>54.0</td><td>74.3</td><td>74.3</td><td>74.3</td><td>61.1</td><td>61.1</td><td>61.1</td><td>65.0</td><td>66.7</td><td>66.7</td></tr><tr><td>+ SRPO</td><td>68.8</td><td>81.0</td><td>82.7</td><td>69.2</td><td>74.0</td><td>74.0</td><td>53.8</td><td>58.6</td><td>65.8</td><td>75.7</td><td>79.1</td><td>81.3</td><td>61.4</td><td>63.1</td><td>67.0</td><td>65.8</td><td>71.2</td><td>74.2</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-28/ced84c0c-2995-40f4-9aa7-cd7c8c72d160/abeadd63a5a15db35ff263854a06d9d577022d5e52544d3b176c891a825fee83.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-28/ced84c0c-2995-40f4-9aa7-cd7c8c72d160/c30ae1319929849a61a2a16808feabff6cf8f32f331019cd0a94e7be53fea6b0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-28/ced84c0c-2995-40f4-9aa7-cd7c8c72d160/d7d9151f65b9e04f23495317867564e54f934af6870d07e6e56b3061850dfb81.jpg)



Figure 3: Training curves on three representative benchmarks for Qwen3-8B. We plot avg@16 against wall-clock training time on (a) Chemistry, (b) Biology, and (c) Tool Use. These curves complement Table 1, which reports the highest achieved result within each training budget. All curves show a 5-step rolling mean and shaded bands denote ±1 std.


Implementation Details For both GRPO and SDPO, we adopt the training setup and hyperparameters from the original SDPO paper, where each method’s configuration was selected via grid search over learning rates and mini-batch sizes to maximize the validation accuracy (Hübotter et al., 2026). Both methods use a training batch size of 32 and sample 8 rollouts per prompt; the main differences are the mini-batch size and learning rate: GRPO uses a mini-batch size of 8 and a learning rate of $1 \times 1 0 ^ { - 6 } ,$ , whereas SDPO uses 32 with $1 \times 1 0 ^ { - 5 }$ . For SRPO, we keep the training batch size, mini-batch size, and rollout number the same as in SDPO, set the learning rate to $5 \times 1 0 ^ { - 6 }$ to balance the reward-driven and self-distillation signals within a single objective, and use a dynamic-weighting temperature $\beta$ with default value 1. All experiments are conducted on 8 NVIDIA H20 GPUs. 

## 4.2 Main Results

SRPO achieves early efficiency, long-horizon stability, and a higher performance ceiling. Table 1 reports the highest avg@16 achieved within each wall-clock budget, following the reporting protocol of SDPO.<sup>1</sup> On Qwen3-8B, SRPO improves the 10h average from 71.1 (SDPO) and 74.0 (GRPO) to 77.4; on Qwen3-4B, the corresponding improvement is from 66.7 and 69.7 to 74.2. Across both scales, SDPO saturates early, as evidenced by its identical 5h and 10h averages, while GRPO improves more steadily before eventually plateauing. SRPO largely avoids both issues, matching the early training efficiency of SDPO while maintaining steady improvement over longer horizons and ultimately exceeding the peak performance of both baselines. Notably, at 10h on Qwen3-8B, SRPO improves over GRPO by +4.1 on Chemistry, +4.8 on Physics, +2.2 on Biology, +3.7 on Materials, and +2.2 on Tool Use. We attribute this to entropy-aware dynamic weighting on the SDPO branch: even when the self-teacher becomes noisier in later training, reweighting by teacher confidence preserves useful logit-level guidance while suppressing uncertain targets, enabling SRPO to continue improving beyond the point where pure GRPO plateaus. 


Table 2: Ablation results on Qwen3-8B, reported as avg@16 accuracy (%) across five benchmarks. The first block isolates the mixing strategy, and the second isolates the additional effect of dynamic weighting on top of sample routing. Colored deltas are measured relative to the reference row within each block.


<table><tr><td>Ablation Target</td><td>Variant</td><td>1h</td><td>5h</td><td>10h</td></tr><tr><td rowspan="2">Mixing Strategy</td><td>SRPO w/o dynamic weighting</td><td>66.5</td><td>74.8</td><td>75.6</td></tr><tr><td>Advantage Mix</td><td>67.2 +0.7</td><td>72.3 -2.5</td><td>72.3 -3.3</td></tr><tr><td rowspan="2">Dynamic Weighting</td><td>SRPO</td><td>66.9</td><td>75.5</td><td>77.4</td></tr><tr><td>SRPO w/o dynamic weighting</td><td>66.5 -0.4</td><td>74.8 -0.7</td><td>75.6 -1.8</td></tr></table>

To complement the tabular summary, Figure 3 plots representative learning curves on Qwen3-8B, which reveal two recurring patterns. 

Pattern 1: When self-distillation is effective, SRPO extends the advantage. In Chemistry, SDPO leads at 1h (71.6 vs. 69.2 for SRPO), but SRPO overtakes it by 5h and reaches 83.0 at 10h, exceeding both SDPO (80.6) and GRPO (78.9). As Figure 3(a) shows, SRPO tracks SDPO’s rapid early rise while avoiding its subsequent collapse. Biology follows a similar trajectory: SRPO achieves the best 1h result (55.8), and the gap widens as SDPO stalls at 58.5 while SRPO climbs to 72.8 at 10h (Figure 3(b)). 

Pattern 2: When self-distillation is ineffective, SRPO remains stable. As Figure 3(c) shows, SDPO degrades substantially over time on Tool Use, whereas SRPO remains stable and tracks or exceeds GRPO throughout (65.2, 71.2, 71.2 vs. 64.3, 68.5, 69.0 for GRPO). Both patterns reflect the effectiveness of the sample-routing design: when self-distillation is useful, SRPO exploits it to accelerate learning; when it is not, the GRPO branch anchors optimization to the reward objective and prevents drift. 

## 4.3 Ablation Study

Sample routing is more robust than advantage-level mixing over long horizons. To isolate the mixing strategy, we first compare SRPO w/o dynamic weighting against an Advantage Mix control that combines GRPO and SDPO at the advantage level: 

$$
A _ {i, t} ^ {\mathrm{Mix}} (v) = \lambda A _ {i, t} ^ {\mathrm{GRPO}} (v) + (1 - \lambda) A _ {i, t} ^ {\mathrm{SDPO}} (v), \qquad \lambda \in [ 0, 1 ],
$$

where the GRPO term is reward-derived and the SDPO term is feedback-derived. We set λ = 0.9 to keep the two advantages on a comparable scale, consistent with the mixing ratio used in SDPO (Hübotter et al., 2026), and keep all other hyperparameters unchanged. Advantage Mix is slightly better at 1h (+0.7), but falls behind by 2.5 points at 5h and 3.3 points at 10h, with no further gain after 5h. 

This pattern matches the changing roles of the two signals over training. Early on, when self-distillation remains high quality, mixing dense SDPO guidance with reward-aligned GRPO updates can help. Later, as the SDPO signal becomes less reliable, advantage-level mixing instead propagates this noise into the learning process, harming stability. By contrast, sample routing confines SDPO to failed samples and leaves correct samples under GRPO, reducing interference and yielding stronger long-term performance. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-28/ced84c0c-2995-40f4-9aa7-cd7c8c72d160/77abf7d1480cc5b1b7384a142ee7b03a43aca2a82541ac17b0a0d093fb057d6a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-28/ced84c0c-2995-40f4-9aa7-cd7c8c72d160/369dd52eccc1eed5c5d7618fb80d3a7ea1bac3d2537e5742ae7228284ce59475.jpg)



Figure 4: Response length and per-step compute time for Qwen3-8B. (a) Response length on Chemistry: GRPO remains consistently long, SDPO drops rapidly, and SRPO stays moderate. Curves show a 5-step rolling mean and shaded bands denote ±1 std. (b) Average seconds per training step, averaged across five benchmarks and measured over the 1h, 5h, and 10h windows. SRPO incurs a modest overhead relative to GRPO in the early stage of training, but becomes faster than both GRPO and SDPO over longer training horizons.


Dynamic weighting provides an additional late-stage gain on top of sample routing. We then compare SRPO against SRPO w/o dynamic weighting to isolate the effect of entropy-aware weighting. Adding dynamic weighting improves the average result by 0.4 at 1h, 0.7 at 5h, and 1.8 at 10h. The widening gain suggests that this component matters most when the selfteacher becomes less reliable and noisier. This is consistent with the role of entropy-aware weighting: it emphasizes high-confidence dense corrections while suppressing uncertain targets, further stabilizing the SDPO branch in later training. 

Together, these ablations suggest that SRPO’s gains come from two complementary components: sample routing provides the stronger mixing strategy and the main source of long-horizon robustness, while dynamic weighting adds further late-stage improvement by improving the reliability of the SDPO branch. 

## 4.4 Response Length and Compute Time

SRPO yields moderate response lengths between GRPO and SDPO. Figure 4(a) shows response length during training of Qwen3-8B on Chemistry. The three methods exhibit different trends: GRPO produces the longest responses, SDPO the shortest, and SRPO settles between the two. The verbosity of GRPO inflates inference cost, while the excessive brevity of SDPO has been linked to degraded reasoning due to the suppression of epistemic verbalization (Kim et al., 2026). SRPO’s moderate response length suggests a balance between the two, potentially mitigating both issues. 

SRPO achieves the lowest per-step compute time over long training horizons. Figure 4(b) reports the average seconds per training step of Qwen3-8B, averaged over the five benchmarks. At 1h, SRPO incurs a 17.4% overhead relative to GRPO (83.4s vs. 71.0s per step), while being lower than SDPO (83.4s vs. 85.9s). As training proceeds, the cost advantage shifts in favor of SRPO. At 5h, it is 4.9% faster than GRPO and 6.7% faster than SDPO (78.3s vs. 82.4s and 83.9s). At 10h, the advantage widens further, reaching 17.2% over GRPO and 9.4% over SDPO (75.8s vs. 91.5s and 83.7s). 

These results are consistent with the design of SRPO. Early in training, failed samples are more frequent, so the SDPO branch is activated more often and the additional self-teacher log-probs computation is more visible. Later in training, the fraction of failed samples decreases, reducing the self-teacher overhead. At the same time, SRPO produces shorter responses than GRPO, further lowering inference cost. Taken together, SRPO improves not only training efficiency and stability, but also computational efficiency in terms of response length and per-step compute time. 

## 5 Conclusion

We revisit the trade-off between reward-driven reinforcement and self-distillation in LLM post-training and propose SRPO, a unified on-policy framework that routes successful samples to GRPO for reward-aligned reinforcement and failed samples with teacher information to SDPO for dense logit-level correction, together with entropy-aware dynamic weighting to suppress unreliable self-distillation signals and emphasize confident ones. Across five benchmarks and two model scales, SRPO consistently outperforms both pure GRPO and SDPO, demonstrating that sample-level routing can preserve the early efficiency of selfdistillation while maintaining the long-horizon stability of reward-driven reinforcement. Moreover, SRPO yields moderate response lengths and lower per-step compute time over long training horizons. An important direction for future work is to extend this framework to environments with richer feedback, so that self-distillation branch can better leverage environment information. 

## Ethics Statement

This work studies post-training optimization methods for large language models and does not introduce new capabilities targeted at harmful applications. However, improving reasoning quality may still increase dual-use risks (e.g., more effective generation of misleading or unsafe content). We therefore recommend deployment only under standard safety controls, including content moderation, policy-based filtering, and rate limiting. 

Our experiments use publicly available benchmark datasets (SciKnowEval and ToolAlpacastyle tool-use tasks) and automatic verifiable rewards. We do not collect personal data, do not involve human subjects, and do not perform user profiling. The training objective does not use private annotations or sensitive metadata. 

From an environmental perspective, SRPO is trained on GPU clusters and thus incurs non-trivial energy use. At the same time, our results show lower per-step compute time over long horizons compared with strong baselines, which may partially reduce the total compute required to reach a target performance level. We plan to release implementation details to support transparent evaluation and responsible reproduction. 

## References



Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In The twelfth international conference on learning representations, 2024. 





Thomas Kleine Buening, Jonas Hübotter, Barna Pásztor, Idan Shenfeld, Giorgia Ramponi, and Andreas Krause. Aligning language models from user interactions. arXiv preprint arXiv:2603.12273, 2026. 





Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Yuchen Zhang, Jiacheng Chen, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025. 





Kehua Feng, Xinyi Shen, Weijie Wang, Xiang Zhuang, Yuqi Tang, Qiang Zhang, and Keyan Ding. Sciknoweval: Evaluating multi-level scientific knowledge of large language models. arXiv preprint arXiv:2406.09098, 2024. 





Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. arXiv preprint arXiv:2306.08543, 2023. 





Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 





Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 





Jonas Hübotter, Leander Diaz-Bone, Ido Hakimi, Andreas Krause, and Moritz Hardt. Learn ing on the job: Test-time curricula for targeted reinforcement learning. arXiv preprint arXiv:2510.04786, 2025. 





Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al. Reinforcement learning via self-distillation. arXiv preprint arXiv:2601.20802, 2026. 





Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024. 





Mykola Khandoga, Rui Yuan, and Vinay Kumar Sankarapu. Beyond uniform credit: Causal credit assignment for policy optimization. arXiv preprint arXiv:2602.09331, 2026. 





Devvrit Khatri, Lovish Madaan, Rishabh Tiwari, Rachit Bansal, Sai Surya Duvvuri, Manzil Zaheer, Inderjit S Dhillon, David Brandfonbrener, and Rishabh Agarwal. The art of scaling reinforcement learning compute for llms. arXiv preprint arXiv:2510.13786, 2025. 





Jeonghye Kim, Xufang Luo, Minbeom Kim, Sangmook Lee, Dohyung Kim, Jiwon Jeon, Dongsheng Li, and Yuqing Yang. Why does self-distillation (sometimes) degrade the reasoning capability of llms? arXiv preprint arXiv:2603.24472, 2026. 





Yoon Kim and Alexander M Rush. Sequence-level knowledge distillation. In Proceedings of the 2016 conference on empirical methods in natural language processing, pp. 1317–1327, 2016. 





Abhijit Kumar, Natalya Kumar, and Shikhar Gupta. Execution-grounded credit assignment for grpo in code generation. In The 1st Workshop on Scaling Post-training for LLMs, 2026. 





Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pp. 611–626, 2023. 





Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The twelfth international conference on learning representations, 2023. 





Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id=5PAF7PAY2Y. 





Nicholas Lourie, Michael Y Hu, and Kyunghyun Cho. Scaling laws are unreliable for downstream tasks: A reality check. arXiv preprint arXiv:2507.00885, 2025. 





Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026. https://thinkingmachines.ai/blog/onpolicy-distillation. 





Ian R. McKenzie, Alexander Lyzhov, Michael Martin Pieler, Alicia Parrish, Aaron Mueller, Ameya Prabhu, Euan McLean, Xudong Shen, Joe Cavanagh, Andrew George Gritsevskiy, Derik Kauffman, Aaron T. Kirtland, Zhengping Zhou, Yuhui Zhang, Sicong Huang, Daniel Wurgaft, Max Weiss, Alexis Ross, Gabriel Recchia, Alisa Liu, Jiacheng Liu, Tom Tseng, Tomasz Korbak, Najoung Kim, Samuel R. Bowman, and Ethan Perez. Inverse scaling: When bigger isn’t better. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=DwgRm72GQF. Featured Certification. 





Purbesh Mitra and Sennur Ulukus. Semantic soft bootstrapping: Long context reasoning in llms without reinforcement learning. arXiv preprint arXiv:2512.05105, 2025. 





Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, et al. Olmo 3. arXiv preprint arXiv:2512.13961, 2025. 





Prasanna Parthasarathi, Mathieu Reymond, Boxing Chen, Yufei Cui, and Sarath Chandar. Grpo-λ: Credit assignment improves llm reasoning. arXiv preprint arXiv:2510.00194, 2025. 





Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019. 





John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 





Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, and Aviral Kumar. Rewarding progress: Scaling automated process verifiers for LLM reasoning. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id= A6Y7AqlzLW. 





Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 





Idan Shenfeld, Mehul Damani, Jonas Hübotter, and Pulkit Agrawal. Self-distillation enables continual learning. arXiv preprint arXiv:2601.19897, 2026. 





Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pp. 1279–1297, 2025. 





Charlie Snell, Dan Klein, and Ruiqi Zhong. Learning by distilling context. arXiv preprint arXiv:2209.15189, 2022. 





Yuda Song, Lili Chen, Fahim Tajwar, Remi Munos, Deepak Pathak, J Andrew Bagnell, Aarti Singh, and Andrea Zanette. Expanding the capabilities of reinforcement learning via text feedback. arXiv preprint arXiv:2602.02482, 2026. 





Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao, and Le Sun. Toolalpaca: Generalized tool learning for language models with 3000 simulated cases. arXiv preprint arXiv:2306.05301, 2023. 





Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025. 





Yinjie Wang, Xuyang Chen, Xiaolong Jin, Mengdi Wang, and Ling Yang. Openclaw-rl: Train any agent simply by talking. arXiv preprint arXiv:2603.10165, 2026. 





Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992. 





An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 





Feng Yao, Liyuan Liu, Dinghuai Zhang, Chengyu Dong, Jingbo Shang, and Jianfeng Gao. Your efficient rl framework secretly brings you off-policy rl training, august 2025. URL https://fengyao. notion. site/off-policy-rl, 2025. 





Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, and Furu Wei. On-policy context distillation for language models. arXiv preprint arXiv:2602.12275, 2026. 





Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. 





Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, and Jie Tang. Rest-mcts*: Llm self-training via process reward guided tree search. Advances in Neural Information Processing Systems, 37:64735–64772, 2024. 





Dan Zhang, Min Cai, Jonathan Light, Ziniu Hu, Yisong Yue, and Jie Tang. Tdrm: Smooth reward models with temporal difference for llm rl and inference. arXiv preprint arXiv:2509.15110, 2025. 





Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026. 





Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025. 





Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody H Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Sglang: Efficient execution of structured language model programs. Advances in neural information processing systems, 37:62557–62583, 2024. 





Sining Zhoubian, Dan Zhang, and Jie Tang. Rest-rl: Achieving accurate code reasoning of llms with optimized self-training and decoding. arXiv preprint arXiv:2508.19576, 2025. 



## A Related Work

## A.1 Reinforcement Learning with Verifiable Rewards

Post-training with verifiable rewards has become a central paradigm for LLM alignment and adaptation, building on policy-gradient foundations such as REINFORCE and PPO (Williams, 1992; Schulman et al., 2017). A growing body of work applies these ideas to LLM post-training, where sequence-level outcome rewards guide optimization on modelsampled trajectories (Guo et al., 2025; Shao et al., 2024; Yu et al., 2025; Liu et al., 2025; Zheng et al., 2025; Zhang et al., 2025; Zhoubian et al., 2025). Among them, GRPO estimates advantages from group-relative rewards without requiring a separate critic, making it a strong and scalable baseline (Shao et al., 2024). 

However, these methods typically assign a single scalar advantage uniformly to every token, so credit assignment remains coarse. Recent analyses have shown that this uniform assignment dilutes gradients across causally irrelevant tokens (Khandoga et al., 2026), hinders localization of semantic errors in near-correct programs (Kumar et al., 2026), and introduces bias that grows with sequence length (Parthasarathi et al., 2025). A complementary line of work seeks to improve credit assignment through process supervision or process reward models, which provide denser step-level signals derived from intermediate states or feedback (Lightman et al., 2023; Setlur et al., 2025; Cui et al., 2025). These approaches offer finer-grained guidance but usually require additional learned reward estimators. This tradeoff motivates methods that provide denser supervision without introducing an additional reward model. 

## A.2 On-Policy Distillation and Self-Distillation

Distillation transfers behavior from a teacher to a student by matching output distributions or intermediate representations (Hinton et al., 2015; Kim & Rush, 2016; Sanh et al., 2019). More recent on-policy distillation methods reduce train-test mismatch by training the student on its own trajectories while receiving teacher guidance on those same trajectories (Agarwal et al., 2024; Gu et al., 2023; Lu & Lab, 2025). Relative to reward-only RL, these methods provide denser supervision, but they typically rely on a separate and often stronger external teacher. 

Self-distillation removes the need for an external teacher by supervising the model with a conditioned version of itself. Context distillation first showed that a model can internalize behavior induced by privileged context into its parameters (Snell et al., 2022). More recent work extends this idea to self-improvement and on-policy self-distillation settings, including learning from self-generated trajectories or richer conditioning information (Mitra & Ulukus, 2025; Hübotter et al., 2025; 2026; Shenfeld et al., 2026; Zhao et al., 2026; Buening et al., 2026; Wang et al., 2026). A representative example is SDPO (Hübotter et al., 2026), which samples rollouts from the current policy and distills the logit-level distribution of a feedback conditioned self-teacher back into the same policy. However, feedback-conditioned onpolicy self-distillation can exhibit late-stage degradation: concurrent work by Kim et al. (2026) attributes this to the suppression of epistemic verbalization, while our analysis (Section 1) traces it to ambiguity on correct samples and progressive degradation of the self-teacher signal. 

Overall, prior RL-based post-training methods provide reward alignment but rely on coarse sequence-level supervision. Distillation-based methods provide denser logit-level guidance, and self-distillation removes the need for an external teacher, but feedback-conditioned on-policy self-distillation may suffer from sample-dependent ambiguity and degraded signal quality in later training. To address this gap, our work studies how reward-driven and self-distillation-based supervision can be combined within a unified framework based on sample routing, thereby leveraging the strengths of both post-training paradigms. 

## B Experimental Details

## B.1 Technical Setup

All experiments were conducted on a single node equipped with 8 NVIDIA H20 GPUs interconnected via NVLink, providing a total of 768 GB VRAM. Our software environment uses GPU driver version 550.144.03, CUDA 12.4, and PyTorch 2.8.0. 

Our implementation is based on the verl library (Sheng et al., 2025). We use PyTorch Fully Sharded Data Parallel (FSDP2) for distributed training across GPUs. For rollout generation, we employ SGLang (Zheng et al., 2024) instead of the vLLM backend (Kwon et al., 2023) used in the original SDPO implementation, as SGLang provides better compatibility with our environment. Since both engines implement the same sampling algorithms and support identical temperature, top-p, and other decoding parameters, the choice of inference backend affects only throughput and does not alter the sampling, preserving a fair comparison with SDPO. 

## B.2 Hyperparameters

Table 3 summarizes the hyperparameters for all three methods. For the two baselines (GRPO and SDPO), we directly adopt the configurations selected via grid search in the original SDPO work (Hübotter et al., 2026); see that paper for details on the search procedure. For SRPO, we inherit all non-learning-rate hyperparameters from SDPO and set the learning rate to $5 \times 1 0 ^ { - 6 } .$ , halfway between the GRPO and SDPO rates, to balance the reward-driven and self-distillation signals within a unified framework. The GRPO branch within SRPO uses the same loss-specific parameters as the standalone GRPO baseline, and the SDPO branch uses the same loss-specific parameters as the standalone SDPO baseline. The only additional hyperparameter introduced by SRPO is the dynamic-weighting temperature ${ \bar { \boldsymbol { \beta } } } ,$ which we set to 1 as default. 

## B.3 Prompt Templates

We use the same prompt templates as SDPO (Hübotter et al., 2026) without any modification, ensuring a fair comparison across all methods. The Science Q&A benchmarks (Chemistry, Physics, Biology, Materials) share a common multiple-choice format, while Tool Use follows a separate tool-calling format. We reproduce both templates below. 

<table><tr><td>Parameters</td><td>GRPO</td><td>SDPO</td><td>SRPO</td></tr><tr><td colspan="4">General</td></tr><tr><td>Model</td><td>Qwen3-{4B, 8B}</td><td>Qwen3-{4B, 8B}</td><td>Qwen3-{4B, 8B}</td></tr><tr><td>Thinking</td><td>False</td><td>False</td><td>False</td></tr><tr><td colspan="4">Data</td></tr><tr><td>Max. prompt length</td><td>2048</td><td>2048</td><td>2048</td></tr><tr><td>Max. response length</td><td>8192</td><td>8192</td><td>8192</td></tr><tr><td colspan="4">Batching</td></tr><tr><td>Question batch size</td><td>32</td><td>32</td><td>32</td></tr><tr><td>Mini batch size</td><td>8</td><td>32</td><td>32</td></tr><tr><td>Number of rollouts</td><td>8</td><td>8</td><td>8</td></tr><tr><td colspan="4">Rollout</td></tr><tr><td>Inference engine</td><td>SGLang</td><td>SGLang</td><td>SGLang</td></tr><tr><td>Temperature</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td colspan="4">Validation</td></tr><tr><td>Number of rollouts</td><td>16</td><td>16</td><td>16</td></tr><tr><td>Temperature</td><td>0.6</td><td>0.6</td><td>0.6</td></tr><tr><td>Top-p</td><td>0.95</td><td>0.95</td><td>0.95</td></tr><tr><td colspan="4">GRPO loss</td></tr><tr><td>ε-high (asymmetric clip)</td><td>0.28</td><td>-</td><td>0.28</td></tr><tr><td>Rollout IS clip (ρ)</td><td>2</td><td>-</td><td>2</td></tr><tr><td>KL coefficient</td><td>0.0</td><td>-</td><td>0.0</td></tr><tr><td colspan="4">SDPO loss</td></tr><tr><td>Top-K distillation</td><td>-</td><td>100</td><td>100</td></tr><tr><td>Distillation divergence</td><td>-</td><td>Jensen-Shannon</td><td>Jensen-Shannon</td></tr><tr><td>Teacher-EMA update rate</td><td>-</td><td>0.05</td><td>0.05</td></tr><tr><td>Rollout IS clip (ρ)</td><td>-</td><td>2</td><td>2</td></tr><tr><td colspan="4">Dynamic weighting</td></tr><tr><td>β</td><td>-</td><td>-</td><td>1</td></tr><tr><td colspan="4">Training</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning rate</td><td><eq>1 \times 10^{-6}</eq></td><td><eq>1 \times 10^{-5}</eq></td><td><eq>5 \times 10^{-6}</eq></td></tr><tr><td>Warmup steps</td><td>10</td><td>10</td><td>10</td></tr><tr><td>Weight decay</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td>Gradient clip norm</td><td>1.0</td><td>1.0</td><td>1.0</td></tr></table>


Table 3: Hyperparameters for GRPO, SDPO, and SRPO. For GRPO and SDPO, we use the configurations from Hübotter et al. (2026). For SRPO, the GRPO-branch and SDPO-branch loss parameters are inherited from the respective baselines; only the learning rate and the dynamic-weighting temperature $\beta$ differ. Entries marked $^ { \prime \prime } - { } ^ { \prime \prime }$ indicate parameters not applicable to that method. 


```txt
Listing 1: System prompt for Science Q&A.

Given a question and four options, please select the right answer. Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>

For the answer, only output the letter corresponding to the correct option (A, B, C, or D), and nothing else. Do not restate the answer text. For example, if the answer is "A", just output:
<answer>
A
</answer> 
```


Listing 2: User prompt for Science Q&A.


```txt
{question}
Please reason step by step. 
```


Listing 3: System prompt for Tool Use.


```txt
You are a tool-use assistant. Solve each request by reasoning about the task and calling the provided tools when needed.
Use only the tools provided in the user message.
Follow the required response format exactly. 
```

## Listing 4: User prompt for Tool Use.

```txt
Your task is to answer the user's question using available tools.
You have access to the following tools:
Name: Axolotl
Description: Collection of axolotl pictures and facts
Documentation:
getRandomAxolotlImage: Retrieve a random axolotl image with information on the image source.
Parameters: {}
Output: Successful response.
- Format: application/json
- Structure: Object{url, source, description}
searchAxolotlImages: Search for axolotl images based on specific criteria such as color, gender, and size.
Parameters: {"color": "string. One of: [wild, leucistic, albino]. The color of the axolotl (e.g., 'wild', 'leucistic', 'albino', etc.).", "gender": "string. One of: [male, female]. The gender of the axolotl ('male', 'female').", "size": "string. One of: [small, medium, large]. The size of the axolotl ('small', 'medium', 'large').", "page": "integer. The page number for pagination purposes."}
Output: Successful response.
- Format: application/json
- Structure: Object{results: Array[Object{url, source, description}], pagination: Object{current_page, total_pages, total_results}}
getAxolotlFacts: Retrieve interesting facts about axolotls such as their habits, habitats, and physical characteristics.
Parameters: {"category": "string. One of: [habits, habitat, physical characteristics]. The category of facts to retrieve (e.g., 'habits', 'habitat', 'physical characteristics').", "limit": "integer. The maximum number of facts to return."}
Output: Successful response.
- Format: application/json
- Structure: Array[Object{fact, source}]
Use the following format: 
```

```txt
Thought: you should always think about what to do  
Action: the action to take, should be one of the tool names.  
Action Input: the input to the action, must be in JSON format. All of the action input must be realistic and from the user.  
Begin!  
Question: Hey, can you show me a random picture of an axolotl? 
```

## B.4 Benchmark Details

We use the exact train/test splits provided in the official SDPO github repository to ensure full comparability. Table 4 summarizes the dataset statistics. 

<table><tr><td>Benchmark</td><td>Source</td><td>Train</td><td>Test</td><td>Total</td></tr><tr><td>Chemistry</td><td>SciKnowEval</td><td>1,890</td><td>210</td><td>2,100</td></tr><tr><td>Physics</td><td>SciKnowEval</td><td>720</td><td>80</td><td>800</td></tr><tr><td>Biology</td><td>SciKnowEval</td><td>450</td><td>50</td><td>500</td></tr><tr><td>Materials</td><td>SciKnowEval</td><td>841</td><td>94</td><td>935</td></tr><tr><td>Tool Use</td><td>ToolAlpaca</td><td>4,046</td><td>68</td><td>4,114</td></tr></table>


Table 4: Dataset statistics for all five benchmarks. The four Science Q&A benchmarks are drawn from the reasoning subset (Level 3) of SciKnowEval (Feng et al., 2024); Tool Use is drawn from ToolAlpaca (Tang et al., 2023). All splits are identical to those used by SDPO (Hübotter et al., 2026).


The four Science Q&A benchmarks are formatted as four-option single-choice questions targeting undergraduate-level scientific reasoning. Each question presents a problem statement (often involving domain-specific notation such as SMILES strings in Chemistry, physical equations in Physics, protein sequences in Biology, or crystal lattice parameters in Materials) followed by four candidate answers. The Tool Use benchmark pairs a natural-language user request with a tool-API specification (including function names, parameter schemas, and output types); the model must produce the correct tool call in a structured Thought / Action / Action Input format. 


Table 5 shows one representative example from each benchmark.


<table><tr><td>Benchmark</td><td>Question (excerpt)</td><td>Answer</td></tr><tr><td>Chemistry</td><td>What is the correct logarithmic solubility value of the molecule “Cc1cc(=O)[nH]c(=S)[nH]1” in aqueous solutions?A: -3.01 B: -2.436 C: -4.576 D: 1.1</td><td>B</td></tr><tr><td>Physics</td><td>A charged particle produces an electric field with a magnitude of <eq>{2.0}\mathrm{\;N}/\mathrm{C}</eq> at a point that is <eq>{50}\mathrm{\;{cm}}</eq> away from the particle. What is the magnitude of the particle&#x27;s charge?A: 50 pC B: 56 pC C: 60 pC D: 64 pC</td><td>B</td></tr><tr><td>Biology</td><td>What is the folding stability score of the protein sequence “GSSTTRYR-FLDEEEARRAAKEWARRGYQVHVTQNGTYWEVEVR”?A: -0.01 B: 1.69 C: 2.49 D: 0.45</td><td>B</td></tr><tr><td>Materials</td><td>Given the following crystal structure parameters for the material <eq>{\mathrm{{RbLa}}}_{9}{\left( {\mathrm{{IrO}}}_{6}\right) }_{4}</eq> (Material ID: mp-560657), calculate the volume of the unit cell (in <eq>{\mathring{\mathrm{A}}}^{3}</eq> ). Lattice: <eq>a = {7.82},b = {7.82},c = {17.88Å};\alpha = \beta = \gamma = {90}^{ \circ }</eq> . A: 1025.67 B: 1094.31 C: 1200.45 D: 1150.78</td><td>B</td></tr><tr><td>Tool Use</td><td>(Given the Axolotl API specification) Question: “I&#x27;m looking for an axolotl that is wild in color and medium in size. Can you help me find some pictures?”</td><td>searchAxolotlImages((&quot;color&quot;: &quot;wild&quot;, &quot;gender&quot;: &quot;&quot;, &quot;size&quot;: &quot;medium&quot;, &quot;page&quot;: 1))</td></tr></table>


Table 5: One representative example from each benchmark. Science Q&A examples show the question stem and four answer options; the Tool Use example shows the user query and the expected structured tool call (API specification and answer omitted for brevity; see Appendix B.3 for the full template).


## B.5 Teacher Information Construction

As described in Section 3, the SDPO branch requires teacher information $f _ { i }$ for each rollout y<sub>i</sub> to construct the feedback-conditioned self-teacher distribution $\pi _ { \theta } ( \cdot \mid x , \dot { f } _ { i } , y _ { i , < t } )$ . Following SDPO (Hübotter et al., 2026), we use successful sibling rollouts within the same group as teacher information. Since our experimental setting does not include rich environment feed back (e.g., runtime errors in coding tasks), the only available source of teacher information is a correct sibling rollout from the same prompt. 

Construction procedure. For each prompt $x ,$ the policy generates a group of $G = 8$ rollouts $\{ y _ { 1 } , \dotsc , y _ { G } \}$ . We identify all correct rollouts in the group (those with reward $r _ { i } \geq 0 . 5 )$ . For each rollout $y _ { i } ,$ the teacher information $f _ { i }$ is constructed as follows: 

1. Collect the indices of all correct rollouts for the same prompt, excluding rollout i itself (to prevent a sample from serving as its own teacher). 

2. If at least one correct sibling exists, select one and use its full response text as the teacher information. The teacher prompt is then formatted as: 


Listing 5: Teacher prompt template. {question} is the original prompt and {sibling_response} is the full text of a correct sibling rollout.


```txt
{question}
Correct solution:
{sibling_response}
Correctly solve the original question. 
```

The self-teacher processes this enriched prompt concatenated with the student’s own response tokens $y _ { i , < t } ,$ producing a logit-level distribution at each position that serves as the distillation target. Crucially, the self-teacher does not generate a new response; it re-scores the student’s existing trajectory under the enriched context. 


Illustrative example. Consider a prompt with $G = 8$ rollouts, of which rollouts $y _ { 2 }$ and y are correct (reward = 1.0) and the remaining six are incorrect (reward = 0.0). Table 6 shows the resulting routing decision for representative rollouts.


<table><tr><td>Rollout</td><td>Correct? (ci)</td><td>Teacher avail.? (mi)</td><td>Route</td><td>Explanation</td></tr><tr><td>y0 (incorrect)</td><td>0</td><td>1</td><td>SDPO</td><td>Uses y2&#x27;s response as teacher info</td></tr><tr><td>y2 (correct)</td><td>1</td><td>1</td><td>GRPO</td><td>Correct ⇒ GRPO; y5 available but unused</td></tr><tr><td>y5 (correct)</td><td>1</td><td>1</td><td>GRPO</td><td>Correct ⇒ GRPO; y2 available but unused</td></tr><tr><td>y7 (incorrect)</td><td>0</td><td>1</td><td>SDPO</td><td>Uses y2&#x27;s response as teacher info</td></tr></table>


Table 6: Routing decisions for a prompt with two correct rollouts $( y _ { 2 } , y _ { 5 } )$ and six incorrect ones. All incorrect rollouts have teacher information available $( m _ { i } = 1 )$ because at least one correct sibling exists.


Fallback to GRPO when no teacher information is available. When all G rollouts for a prompt are incorrect, no correct sibling exists, so $m _ { i } = 0$ for every rollout. By the routing rule $\begin{array} { r } { \dot { z } _ { i } ^ { \mathrm { S D P O } } = ( 1 - c _ { i } ) m _ { i } , } \end{array}$ all rollouts are assigned to the GRPO branch despite being incorrect. $\mathrm { \Delta N o t a b l y , }$ when a rollout is the only correct one in its group, it is excluded from being its own teacher, so $m _ { i } = 0$ for that rollout. Since it is correct $( c _ { i } = 1 )$ , it is routed to GRPO regardless. Table 7 summarizes the complete decision logic. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-28/ced84c0c-2995-40f4-9aa7-cd7c8c72d160/9d7281ab0e88225f6813b91ee7811a93045f9dbf349cce08553c318ce5407bd8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-28/ced84c0c-2995-40f4-9aa7-cd7c8c72d160/cd809dcabffa06b5c63c2a13fd4d639e8e4a7ccca401f4d60c83902f738bfc23.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-28/ced84c0c-2995-40f4-9aa7-cd7c8c72d160/129f854245ea2d8aa564be9590a2f0efeb0bda5ce9a5562812929ab59ebbb0e0.jpg)



Figure 5: Routing statistics during SRPO training of Qwen3-8B in Chemistry. (a) Fraction of samples routed to the GRPO branch. (b) Fraction of samples routed to the SDPO branch. (c) Fraction of samples in each batch for which teacher information can be constructed. As training progresses, the policy improves and generates more correct rollouts, causing the SDPO fraction to decrease steadily while the GRPO fraction increases correspondingly. All curves show a 5-step rolling mean and shaded bands denote ±1 std.


<table><tr><td>Correct? (ci)</td><td>Teacher avail.? (mi)</td><td>Teacher prompt content</td><td>Route</td></tr><tr><td>✓</td><td>✓</td><td>Question + sibling solution</td><td>GRPO</td></tr><tr><td>✓</td><td>✕</td><td>Question only (no sibling)</td><td>GRPO</td></tr><tr><td>✕</td><td>✓</td><td>Question + sibling solution</td><td>SDPO</td></tr><tr><td>✕</td><td>✕</td><td>Question only (no sibling)</td><td>GRPO (fallback)</td></tr></table>


Table 7: Complete routing decision matrix. Only incorrect rollouts with available teacher information are routed to the SDPO branch; all other cases default to GRPO.


This design ensures that the SDPO branch is activated only when dense logit-level correction is both needed (the rollout is incorrect) and feasible (a correct sibling provides informative teacher context). In all other cases, the update falls back to GRPO’s reward-aligned advantage signal. 

## C Routing Statistics Over Training

Figure 5 visualizes how the sample-routing composition of SRPO evolves over the course of training. At the beginning of training, approximately 40% of samples are routed to the SDPO branch and 60% to the GRPO branch, reflecting the substantial fraction of incorrect rollouts that benefit from dense logit-level correction. As training progresses and the policy improves, the fraction of correct rollouts increases, causing more samples to be routed to the GRPO branch. 

This dynamic shift has two important implications. First, it provides direct empirical support for the adaptive mixing behavior described in Section 3.3. SDPO branch contributes a substantial share in the early stage, providing meaningful dense logit-level correction when the policy is weaker and incorrect rollouts are frequent. As training proceeds and the policy improves, this contribution gradually diminishes while an increasing share of samples is handled by the GRPO branch, whose reward-aligned advantages provide a more stable and unbiased optimization signal for already-correct rollouts. The net effect is that SRPO automatically modulates the influence of self-distillation—leveraging it most when it is most beneficial and changing it to reward-aligned reinforcement for stability as the policy matures—without requiring any manual scheduling of the mixing ratio. 

Second, the decreasing SDPO fraction directly explains the compute-time trend observed in Section 4.4 (Figure 4(b)): since the self-teacher log-probability computation is only performed for samples on the SDPO branch, the per-step overhead of this additional forward pass diminishes as fewer samples require it. This accounts for why SRPO’s per-step compute time decreases steadily over training and eventually falls below that of both standalone GRPO and SDPO. 

Figure 5(c) further shows that the fraction of samples with constructable teacher information remains high throughout training. This indicates that the fallback to GRPO due to teacher unavailability $( m _ { i } = 0 )$ is relatively infrequent; the primary driver of the routing shift is the increasing correctness of rollouts $\dot { ( } c _ { i } = 1 \dot { ) }$ , not the absence of teacher information. 