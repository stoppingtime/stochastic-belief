<!-- MODEL-ID: SP-P0001-Q38-M1U-Q4-v1 -->
<!-- STATIC-FLOOR-BYTES: 15660093440 -->
<!-- KV-BYTES-PER-CONTEXT-TOKEN: 65536 -->
<!-- RAW-BUS-BPS: 819200000000 -->
<!-- APPLE-RATED-BPS: 800000000000 -->
<!-- RAW-CTX0-MILLI-TPS: 52311 -->
<!-- APPLE-CTX0-MILLI-TPS: 51085 -->
<!-- RAW-CTX262144-MILLI-TPS: 24945 -->
<!-- APPLE-CTX262144-MILLI-TPS: 24360 -->

# M1 Ultra 上 Qwen3.8-27B Q4_K_M 单 token 解码的带宽上界

**模型编号：SP-P0001-Q38-M1U-Q4-v1**

## 摘要

本文研究一个很具体的问题：在 M1 Ultra 上，以 batch=1、非 speculative、非 MTP 的方式执行 Qwen3.8-27B Q4_K_M 自回归解码时，单 token 解码速度在给定内存带宽和给定表示流量下最多能有多快。

我们不把 GGUF 文件总大小直接当成“每生成一个 token 必须从 DRAM 读取的字节数”，也不把某个实测速度反推成物理极限。证明先从模型结构和量化块格式构造一个保守的表示字节下界，再把“实际 DRAM 流量不小于这个下界”作为显式物理前提。随后只使用数据传输守恒：若每 token 至少需要跨 DRAM 传输 \(D\) 字节，而内存通路每秒至多传输 \(B\) 字节，则吞吐率 \(r\) 必须满足

\[
rD\le B.
\]

为避免浮点舍入，形式化证明使用 milli-token/s，即 \(1/1000\) token/s 为一个整数单位。对短上下文、`cacheCarry=0` 的主实例，得到：

- 使用 Apple 公布的 \(800\times10^9\) byte/s：模型内精确离散上界为 **51.085 token/s**；
- 使用更宽松的 \(819.2\times10^9\) byte/s 原始线速：模型内精确离散上界为 **52.311 token/s**。

在 \(262144\) token 上下文时，相应上界降为 **24.360 token/s** 和 **24.945 token/s**。

这里的“精确”只指本文抽象模型在 milli-token/s 粒度下的精确最大整数。它不表示真实 M1 Ultra 必然达到该速度，也不表示 Lean 已经证明 Apple 规格、GGUF tensor layout 或具体运行时的 DRAM 流量。

---

## 1. 问题、对象与单位

我们只讨论一次普通 target-model decode step 产生一个 target token 的情形。下列条件固定不变：

- batch size 为 1；
- 不使用 speculative decoding；
- 不使用 MTP 接受多个 token；
- full-attention KV cache 使用 FP16；
- 主定理取 `cacheCarry=0`，即不把任何跨 token 的片上常驻字节预先从 DRAM 下界中扣除。

设：

- \(B\in\mathbb N\)：内存通路带宽上限，单位 byte/s；
- \(r\in\mathbb N\)：吞吐率，单位 milli-token/s；
- \(A\in\mathbb N\)：真实执行每生成一个 token 实际跨 DRAM 的字节数，单位 byte/token；
- \(D(L,C)\in\mathbb N\)：本文根据模型表示构造出的 DRAM 流量下界，其中 \(L\) 是已有上下文长度，\(C\) 是允许跨 step 常驻片上的有效字节数。

由于 \(r\) 的单位是 milli-token/s，带宽守恒写成整数式：

\[
rA\le 1000B. \tag{1}
\]

式 (1) 没有浮点数，也没有“四舍五入以后差不多”的问题。

---

## 2. 外部事实与数学前提必须分开

形式证明只负责“从前提推出结论”。本文使用的外部事实包括：M1 Ultra 的公布带宽、Qwen3.8-27B 的结构参数、GGML 量化块的编码方式，以及选定 GGUF 制品采用哪些 tensor 编码。这些事实的来源和证据等级记录在 `evidence.md`。

Lean 并不证明这些网页或制品是真的。因此，物理世界到数学模型之间必须有一条明确的桥：

> **DRAM 流量前提。** 对给定 \(L,C\)，真实执行的每 token DRAM 流量 \(A\) 满足
>
> \[
> D(L,C)\le A. \tag{2}
> \]

后面的 theorem 证明的是：只要 (1) 与 (2) 成立，速度上界就必然成立。

这一区分很重要。例如，“GGUF 文件有 16.5 GiB”不能直接推出“每 token 一定从 DRAM 读取 16.5 GiB”。文件还可能包含 metadata、padding；更重要的是，片上缓存和实现细节会改变真正的 DRAM 流量。本文因此从参与计算的大 tensor 逐类构造一个更小、方向更安全的下界。

---

## 3. 从模型结构构造静态表示字节下界

Qwen3.8-27B 文本模型取以下结构参数：

\[
H=5120,\qquad I=17408,\qquad N_{\mathrm{layer}}=64.
\]

64 层由 48 个 Gated DeltaNet 层和 16 个 full-attention 层组成。词表大小为

\[
V=248320.
\]

### 3.1 FFN

每层 SwiGLU 有 `gate`、`up`、`down` 三个大矩阵，因此 64 层 FFN 权重数为

\[
N_{\mathrm{FFN}}
=64\times 3\times5120\times17408
=17\,112\,760\,320. \tag{3}
\]

### 3.2 Gated DeltaNet 大投影

线性注意力的 Q/K head 数为 16，V head 数为 48，head dimension 为 128，所以

\[
d_{QK}=16\times128=2048,
\qquad
d_V=48\times128=6144.
\]

每个 GDN 层只计入三个必须参与当前建模的大投影：

\[
W_{qkv}:5120\to(2d_{QK}+d_V),
\]

\[
W_z:5120\to d_V,
\qquad
W_o:d_V\to5120.
\]

48 层合计

\[
N_{\mathrm{GDN,Q4}}=5\,536\,481\,280. \tag{4}
\]

另外，`in_proj_a` 与 `in_proj_b` 在选定量化配置中按 F32 计，数量为

\[
N_{\mathrm{SSM,F32}}
=48\times2\times5120\times48
=23\,592\,960. \tag{5}
\]

### 3.3 Full attention

full attention 有 24 个 Q head、4 个 KV head，head dimension 为 256。由于 Q projection 同时包含 output gate，Q 输出宽度按 \(2\times24\times256\) 计算。16 层 Q/K/V/O 合计

\[
N_{\mathrm{FA,Q8}}=1\,677\,721\,600. \tag{6}
\]

### 3.4 输出头

模型不共享 input embedding 与 LM output head。一次解码不需要扫描整个 input embedding table，只需查当前 token 的一行；但要得到完整 logits，输出头必须参与矩阵乘。因此保留

\[
N_{\mathrm{out,Q6}}
=248320\times5120
=1\,271\,398\,400. \tag{7}
\]

### 3.5 从权重数换成编码字节数

本文使用的编码块大小是：

\[
\mathrm{Q4_K}:\;256\text{ weights}\to144\text{ bytes},
\]

\[
\mathrm{Q6_K}:\;256\text{ weights}\to210\text{ bytes},
\]

\[
\mathrm{Q8_0}:\;32\text{ weights}\to34\text{ bytes},
\qquad
\mathrm{F32}:\;1\to4\text{ bytes}.
\]

相应 tensor 数量都满足块对齐，因此不用处理尾块。静态表示下界定义为

\[
\begin{aligned}
D_{\mathrm{static}}
={}&144\frac{N_{\mathrm{FFN}}+N_{\mathrm{GDN,Q4}}}{256}\\
&+34\frac{N_{\mathrm{FA,Q8}}}{32}
+4N_{\mathrm{SSM,F32}}
+210\frac{N_{\mathrm{out,Q6}}}{256}.
\end{aligned} \tag{8}
\]

把 (3)--(7) 代入：

\[
\boxed{D_{\mathrm{static}}=15\,660\,093\,440\ \text{byte/token}}. \tag{9}
\]

这是**表示层的保守下界**，不是对真实 DRAM 流量的精确测量。我们故意没有加入 RMSNorm、卷积、小型 bias/state、kernel 中间读写等流量，因为漏掉真实流量会使 \(D\) 变小，使最后得到的吞吐上界变大。对于“证明速度不能超过某个数”，这是保守方向。

---

## 4. KV cache 随上下文增长的项

只有 16 个 full-attention 层的 KV 随上下文长度线性增长。每个历史 token、每个 full-attention 层需要

\[
4\text{ KV heads}\times256
\times2\text{ tensors}\times2\text{ byte}
\]

的 FP16 KV 数据。因此全部 16 层合计：

\[
\boxed{
D_{\mathrm{KV/token}}
=16\times4\times256\times2\times2
=65\,536\ \text{byte/context-token}
}. \tag{10}
\]

于是上下文长度为 \(L\) 时，表示层工作集下界为

\[
R(L)=15\,660\,093\,440+65\,536L. \tag{11}
\]

如果允许有 \(C\) 字节能够跨 decode step 留在片上且无需重新跨 DRAM，则定义

\[
D(L,C)=\max\{0,R(L)-C\}. \tag{12}
\]

Lean 中使用自然数减法表达同一截断。主结果取 \(C=0\)。这样做不是宣称缓存不存在，而是把“多少字节能够稳定跨 token 常驻”从隐藏假设变成可替换参数。

---

## 5. 一般带宽上界定理

### 定理 1（带宽守恒上界）

设 \(B,r,A,D,T\in\mathbb N\)。若

\[
D\le A, \tag{13}
\]

\[
rA\le1000B, \tag{14}
\]

且

\[
1000B<TD, \tag{15}
\]

则

\[
r<T. \tag{16}
\]

### 证明

反设 \(r\ge T\)。由 \(r\ge T\) 与 \(A\ge D\)，自然数乘法的单调性给出

\[
TD\le rA. \tag{17}
\]

由带宽可行性 (14)，

\[
rA\le1000B. \tag{18}
\]

合并 (17)、(18)：

\[
TD\le1000B. \tag{19}
\]

但假设 (15) 同时给出

\[
1000B<TD. \tag{20}
\]

(19) 与 (20) 矛盾。因此 \(r<T\)。证毕。

这个证明没有用到神经网络的特殊性质。神经网络结构只负责给出 \(D\)；一旦 \(D\) 建立，上界来自纯粹的数据传输守恒。

---

## 6. 为什么可以得到“精确离散上界”

只证明 \(r<T\) 还不能说明某个较小整数是模型内的精确最大值。为此再检查相邻两个整数。

若某个整数 \(q\) 满足

\[
qD\le1000B<(q+1)D, \tag{21}
\]

那么在“实际流量恰等于下界 \(D\)”的抽象 floor-traffic 模型中：

1. \(q\) 满足带宽可行性；
2. \(q+1\) 已违反带宽上限；
3. 因而 \(q\) 是 milli-token/s 粒度下的精确最大整数。

注意这里的“\(q\) 可行”只发生在抽象模型中，取的是最有利的 \(A=D\)。它并不证明真实机器达到 \(q\)。真实运行时通常还有额外流量和利用率损失。

---

## 7. 数值实例

### 7.1 Apple 公布的 800 GB/s

取

\[
B_{\mathrm{Apple}}=800\,000\,000\,000\ \text{byte/s},
\qquad L=0,\ C=0.
\]

此时 \(D=15\,660\,093\,440\)。检查：

\[
51\,085\,D\le1000B_{\mathrm{Apple}}<51\,086\,D. \tag{22}
\]

因此模型内精确离散上界为

\[
\boxed{51.085\ \text{token/s}}. \tag{23}
\]

### 7.2 更宽松的 819.2 GB/s 原始线速

取

\[
B_{\mathrm{raw}}=819\,200\,000\,000\ \text{byte/s}.
\]

短上下文时：

\[
52\,311\,D\le1000B_{\mathrm{raw}}<52\,312\,D. \tag{24}
\]

所以

\[
\boxed{52.311\ \text{token/s}}. \tag{25}
\]

这是比 800 GB/s 口径更宽松的上界：我们给机器更多带宽，所得最大速度也更高。若连这一上界都无法支持某个声称的普通单-token decode 速度，该声称就必须改变模型前提，例如使用 speculative/MTP、batching，或证明存在足够大的跨 token 片上驻留量。

### 7.3 原生最大上下文 \(L=262144\)

由 (11)：

\[
R(262144)
=15\,660\,093\,440+65\,536\times262\,144
=32\,839\,962\,624\ \text{byte/token}. \tag{26}
\]

相邻整数检查得到：

\[
\boxed{
\begin{aligned}
B=800\text{ GB/s}:&\quad 24.360\text{ token/s},\\
B=819.2\text{ GB/s}:&\quad 24.945\text{ token/s}.
\end{aligned}}
\tag{27}
\]

因此长上下文把每步需要访问的历史 KV 数据提高到与静态权重表示同一数量级，带宽上限随之显著下降。

---

## 8. 本文没有证明什么

本文的结论必须连同前提一起读。

**第一，800 GB/s 是 Apple 的产品规格。** 它不是本文从 DRAM 电气时序第一性原理推导出来的不可突破总线定理。

**第二，819.2 GB/s 是原始线速情景。** 它来自 LPDDR5-6400 与 1024-bit 聚合接口的乘法，是为了给上界更宽松的预算；它不表示应用可以持续获得 819.2 GB/s 有效载荷。

**第三，15,660,093,440 byte/token 是表示层下界。** Lean 证明该数字由本文冻结的 tensor 类别和编码公式得到，但“某个具体 GGUF 的全部相关 tensor 确实属于这些类别”仍需制品解析器独立核验。

**第四，式 (2) 是物理桥梁。** 如果未来证明 M1 Ultra 能让大量相关权重在连续 token 间稳定驻留片上，使实际 DRAM 流量低于本文的 `dramFloor`，那么应修改 \(C\)，重新计算，而不是声称原 theorem 被推翻。

**第五，MTP/speculative decoding 不在作用域内。** 它们允许一次 target verification 接受多个输出 token，改变了“每次完整 target step 对应一个输出 token”的计量关系，因此 output token/s 可以超过本文的单-step ceiling，而不构成矛盾。

---

## 9. 形式验证状态

对应机器证明位于 `Proof.lean`。证明把吞吐率离散为 milli-token/s，并在自然数上完成所有关键不等式；核心结论由 `audit_conclusion` 汇总。

验收流程要求：

```bash
lake build
lake env leanchecker --fresh Problems.Qwen38M1UltraQ4.Proof
lake env lean Problems/Qwen38M1UltraQ4/Proof.lean
```

最后一步用于检查 `#print axioms`。允许集合限制在 Lean 标准基础 `{propext, Classical.choice, Quot.sound}` 的子集，并拒绝 `sorryAx` 与 `Lean.trustCompiler`。

本文因此把结论分成两层：

\[
\text{外部证据与物理前提}
\quad+\quad
\text{Lean 已验证的数学推导}.
\]

只有前一层也经过独立制品与硬件审计后，才能把模型内上界提升为更强的真实硬件断言。
