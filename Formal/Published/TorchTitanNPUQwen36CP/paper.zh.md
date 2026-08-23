<!-- MODEL-ID: SP-P0002-TTNPU-Q36-CP-v1 -->
<!-- OVERRIDE-COMMIT: 7864d5dc17930667d663bbadd1ce2bc722de2753 -->
<!-- TORCHTITAN-COMMIT: c91448d20480c7b294314e68976823050002ebec -->
<!-- CP-DEGREE: 4 -->
<!-- GLOBAL-SEQUENCE-LENGTH: 32768 -->
<!-- LOCAL-SEQUENCE-LENGTH: 8192 -->
<!-- FULL-ATTN-Q-HEADS: 24 -->
<!-- FULL-ATTN-KV-HEADS: 4 -->
<!-- GDN-QK-HEADS: 16 -->
<!-- GDN-V-HEADS: 48 -->

# TorchTitan-NPU `override-refactor` 中 Qwen3.6 Context Parallel 的语义正确性

**模型编号：`SP-P0002-TTNPU-Q36-CP-v1`**

## 摘要

本文讨论一个边界明确的工程数学问题：在 TorchTitan-NPU `override-refactor` 的固定代码版本中，Qwen3.6 长文本训练把序列切到多个 context-parallel rank，随后又把 Q、K、V 及 Gated DeltaNet 的若干投影从“按序列分片”重排为“按 head 分片”，在每个 rank 上对完整序列的一部分 head 做计算，最后恢复为按序列分片。怎样证明这组重排在精确算术模型中没有丢失、重复或错配元素，并且不会改变按 head 独立的注意力或 GDN 运算？变长打包训练中，局部序列块跨越文档边界时，代码重建 reset 标记和因果 K/V 前缀的做法又为什么正确？

证明固定在以下实现证据上：TorchTitan-NPU 镜像分支 `override-refactor` 的提交 `7864d5dc17930667d663bbadd1ce2bc722de2753`，以及它在 `requirements.txt` 中固定的 TorchTitan 提交 `c91448d20480c7b294314e68976823050002ebec`。仓库使用 `qwen3_5` 目录承载 Qwen3.5/3.6 共用实现；其中 attention override 的说明直接写明适用于 “Qwen3.5/3.6 Context Parallel”。本文因此沿用工程侧的 Qwen3.6 名称，同时把文件路径和 commit 全部写清，避免把一个会移动的分支名当成证明对象。

形式化模型把 CP 交换写成四维坐标的转置：

\[
(s,\ell,h,r)\longmapsto(h,s,\ell,r),
\]

其中 \(s\) 是序列 rank，\(\ell\) 是该 rank 内的序列位置，\(h\) 是 head rank，\(r\) 是 head rank 内的局部 head。Lean 4 证明这个映射和逆映射互为反函数。由此进一步证明：对任何只读取本 head 完整序列、且不读取其他 head 的精确算子 \(F\)，都有

\[
\operatorname{HeadToSequence}
\bigl(F_{\mathrm{head}}(\operatorname{SequenceToHead}(x))\bigr)
=F_{\mathrm{reference}}(x).
\]

该定理同时覆盖 full attention 与 Gated DeltaNet 的布局层语义；算子的内部可以沿序列维非线性、递归或带状态，证明只使用“head 之间没有数据依赖”这一结构条件。

对于变长打包元数据，设一个本地连续片段的文档起点为 \(d\)，本地片段起点为 \(a\)，片段右端点为 \(b\)，且 \(d\le a<b\)。代码用“本地 Q 长度等于因果 K 前缀长度”识别真正的文档起点。其数学内容是

\[
b-a=b-d\iff a=d.
\]

Lean 还证明代码构造的 K 索引 \(d+j\;(0\le j<b-d)\) 恰好覆盖半开区间 \([d,b)\)，既不越过文档起点，也不遗漏当前查询可见的因果前缀。

结论是条件性的：**只要 DTensor/HCCL 实现上述坐标转置，NPU attention 与 GDN kernel 实现各自声明的 head-local 数学算子，CP 布局本身就与未切分参考计算语义等价。** 本文没有把这一结论扩大成对 CANN、HCCL、Triton、BF16 舍入、autograd、checkpoint、optimizer step 或训练收敛的证明。

---

## 1. 证明对象和不证明的对象

### 1.1 固定代码版本

本文只针对以下版本组合：

| 对象 | 固定值 |
|---|---|
| TorchTitan-NPU 镜像 | `botcanlearn/torchtitan-npu-upstream` |
| 目标分支 | `override-refactor` |
| 目标提交 | `7864d5dc17930667d663bbadd1ce2bc722de2753` |
| Git tree | `a2083d3d601007d3a47a7200022d55ee89f90608` |
| TorchTitan 依赖提交 | `c91448d20480c7b294314e68976823050002ebec` |
| Lean | `4.33.0`，只导入 `Init` |

代码证据和 blob SHA 记录在 [`evidence.md`](evidence.md)。证明不使用“最新分支”“目前主仓”一类无法复现的指代。

### 1.2 具体运行配置

`qwen35_27b_long_text_sft` 配置给出：

\[
P=4,\qquad L=32768,\qquad L_{\mathrm{local}}=8192,
\]

并设置：

- `context_parallel_degree = 4`；
- `context_parallel_load_balancer = None`；
- `attn_backend = "varlen"`；
- 导入 Qwen CP full-attention 与 GDN override。

这里先证明无 load balancer 的连续序列分片。`CPVarlenMetadata` 本身还支持 head-tail 重排，但具体 Qwen3.6 长文本配置把它关闭；把未启用的路径强行塞进主定理，只会扩大论证范围而不提高当前结论的可信度。

### 1.3 证明分三层

为了避免把不同性质混在一起，本文把结论拆成三层。

**布局层。** 序列分片与 head 分片之间的交换是双射，往返后每个逻辑元素回到原坐标。

**算子层。** 若算子按 head 独立工作，则在 head layout 上执行再恢复，与在全局逻辑 tensor 上直接执行相同。

**元数据层。** 对变长打包中的每个本地连续片段，reset 判定和 K/V 因果前缀的整数区间构造正确。

这三层都能在不依赖浮点数的情况下写成离散数学命题。

### 1.4 本文没有验证 Python 和 NPU kernel

Lean 文件不会读取 Python AST，也不会调用 NPU。所以下列命题仍是工程义务：

1. DTensor 的 `redistribute` 和底层 HCCL 确实实现模型里的坐标转置；
2. `npu_fusion_attention_v3` 的 TND、因果 mask 和变长边界实现符合数学 attention；
3. Triton-Ascend GDN kernel、因果卷积及其 backward 正确；
4. BF16 类型转换不会造成不可接受的数值偏差；
5. 参数、梯度、optimizer state 和 checkpoint 在所有并行维度上正确更新与保存。

形式证明解决的是其中最基础、也最容易因索引错误而出问题的一层：**通信前后的逻辑张量和 head-local 运算是否仍是同一个函数。**

---

## 2. 实现路径与数学抽象

### 2.1 `exchange_sequence_heads` 在做什么

`parallelize.py` 中的 `exchange_sequence_heads` 接收若干形状相同、只在 head 数上可能不同的投影 tensor。对每个 tensor，它先沿 head 维分成 \(P\) 份；随后把各 tensor 的第 0 份、第 1 份……按 rank 顺序拼起来，再把本地 tensor 从 `Shard(1)` 重新分布成 `Shard(head_dim)`。最后按照各 tensor 的局部 head 宽度重新拆分。

抽掉打包细节后，其逻辑效果是：

- 交换前，每个 CP rank 持有全体 head 的一段序列；
- 交换后，每个 CP rank 持有部分 head 的完整序列；
- tensor 种类仍然保持分开，Q 不会与 K 混合，K 不会与 V 混合；
- 还原交换执行相反方向。

“先拼后拆”是通信实现，不是数学本体。数学本体只需要记录每个元素的四个离散坐标。

### 2.2 Qwen3.6 的两类序列算子

Qwen3.6 混合了 full attention 与 Gated DeltaNet。两者内部算法不同，但在 CP 布局上有同一个结构：

- 每个 head 读取该 head 的完整逻辑序列；
- 计算可以依赖同一 head 的过去位置；
- 不需要读取别的 head 的值。

full attention 对每个查询 head 计算因果注意力。GQA 会先让一个 KV head 服务多个 Q head；一旦这种对应关系已经建立，单个 Q head 的 attention 仍然是 head-local 算子。

GDN 沿序列维执行因果卷积和 gated-delta recurrence。它比普通 attention 更容易让人误以为“序列切开后各 rank 独立算即可”。实际代码并没有这样做，而是先把完整序列聚到 head shard，再运行本地 head 的 recurrence。因此本文的完整序列 head-fiber 模型与实现路径一致。

### 2.3 token-local 投影为什么可以放在交换两侧

Q、K、V、decay、beta 和 output gate 的线性投影对每个 token 独立执行。设 \(f\) 是任意逐元素映射，则坐标转置满足

\[
T(f(x))=f(T(x)).
\]

Lean 文件把这一点单独证明为 `sequenceToHead_map_commutes` 和 `headToSequence_map_commutes`。这样，完整 pipeline 的论证不会把“投影是局部运算”藏在一句口头说明里。

---

## 3. 两种布局的严格定义

设：

- \(P\in\mathbb N\) 为 CP degree；
- \(S\in\mathbb N\) 为每个 sequence rank 的本地序列长度；
- \(H\in\mathbb N\) 为每个 head rank 的本地 head 数；
- \(\alpha\) 为元素类型。

按序列分片的逻辑布局定义为函数

\[
X:\operatorname{Fin}(P)\times\operatorname{Fin}(S)
\times\operatorname{Fin}(P)\times\operatorname{Fin}(H)\to\alpha.
\]

坐标依次记为

\[
(s,\ell,h,r),
\]

其中 \(s\) 指 sequence rank，\(h\) 指 head rank。

按 head 分片的布局保存同一批元素，但把 rank 坐标顺序换成

\[
Y:\operatorname{Fin}(P)\times\operatorname{Fin}(P)
\times\operatorname{Fin}(S)\times\operatorname{Fin}(H)\to\alpha,
\]

坐标记为

\[
(h,s,\ell,r).
\]

定义交换 \(T\) 与恢复 \(T^{-1}\)：

\[
(TX)(h,s,\ell,r)=X(s,\ell,h,r), \tag{1}
\]

\[
(T^{-1}Y)(s,\ell,h,r)=Y(h,s,\ell,r). \tag{2}
\]

这里没有用 tensor reshape 的模糊直觉。每一个索引域都由 `Fin` 表示，越界坐标根本无法构造。

---

## 4. 布局双射定理

### 定理 1（sequence-to-head 往返恒等）

对任意 \(X\)，

\[
T^{-1}(TX)=X. \tag{3}
\]

### 证明

任取合法坐标 \((s,\ell,h,r)\)。由定义 (2)：

\[
(T^{-1}(TX))(s,\ell,h,r)
=(TX)(h,s,\ell,r).
\]

再用定义 (1)：

\[
(TX)(h,s,\ell,r)=X(s,\ell,h,r).
\]

所以两个函数在每个输入坐标上相等，由函数外延性得到 (3)。证毕。

### 定理 2（head-to-sequence 往返恒等）

对任意 \(Y\)，

\[
T(T^{-1}Y)=Y. \tag{4}
\]

证明完全对称。

定理 1 与定理 2 共同说明 \(T\) 是双射。这比只证明“元素总数相同”强得多：元素数相同仍可能发生重复一个元素、漏掉另一个元素；函数级逆关系排除了这种情况。

Lean 对应声明是：

```lean
headToSequence_sequenceToHead
sequenceToHead_headToSequence
roundTrip_preserves_entry
```

前两个证明完整函数相等，第三个把结论展开到任意单个元素坐标。

---

## 5. Head-local 算子交换定理

### 5.1 一个 head 的完整序列

固定 \((h,r)\) 后，该 head 的完整逻辑序列是

\[
x_{h,r}:\operatorname{Fin}(P)\times\operatorname{Fin}(S)\to\alpha,
\qquad
x_{h,r}(s,\ell)=X(s,\ell,h,r).
\]

设

\[
F:(\operatorname{Fin}(P)\times\operatorname{Fin}(S)\to\alpha)
\to
(\operatorname{Fin}(P)\times\operatorname{Fin}(S)\to\beta)
\]

是任意 head-local 算子。这里不要求 \(F\) 线性、无状态或只看固定窗口。它可以是：

- 因果 softmax attention；
- Gated DeltaNet recurrence；
- 带文档 reset 的因果卷积；
- 任何只依赖该 head 全序列的精确函数。

### 5.2 两种执行方式

参考执行直接对每个 \((h,r)\) 的序列应用 \(F\)：

\[
(F_{\mathrm{ref}}X)(s,\ell,h,r)
=F(x_{h,r})(s,\ell). \tag{5}
\]

CP 执行先转成 head layout。在 head rank \(h\) 上，同一局部 head \(r\) 的序列是

\[
(TX)(h,s,\ell,r)=x_{h,r}(s,\ell).
\]

对它应用 \(F\)，再用 \(T^{-1}\) 恢复。

### 定理 3（head-local CP 语义等价）

\[
T^{-1}\bigl(F_{\mathrm{head}}(TX)\bigr)=F_{\mathrm{ref}}X. \tag{6}
\]

### 证明

任取 \((s,\ell,h,r)\)。左侧为

\[
F\bigl((s',\ell')\mapsto(TX)(h,s',\ell',r)\bigr)(s,\ell).
\]

由 (1)，括号里的函数逐点等于

\[
(s',\ell')\mapsto X(s',\ell',h,r)=x_{h,r}.
\]

因此左侧等于 \(F(x_{h,r})(s,\ell)\)，即右侧定义 (5)。证毕。

Lean 中这不是针对某个玩具矩阵的有限枚举，而是对任意类型、任意 \(P,S,H\) 和任意 head-local 函数的全称定理：

```lean
headwise_context_parallel_correct
contextParallelFunction_eq_reference
```

所以 full attention 与 GDN 不需要各写一份同构证明。二者真正需要分别验证的是“kernel 是否实现声明的 head-local 算子”，那属于实现层证据。

---

## 6. 对 backward 的有限推论

定理 3 证明的是两个 forward 函数严格相等，而不只是某组输入数值接近。若 \(D\) 是一个以函数语义为输入的、外延一致的导数或梯度构造，那么函数相等蕴含

\[
D\!\left(T^{-1}\circ F_{\mathrm{head}}\circ T\right)
=D(F_{\mathrm{ref}}). \tag{7}
\]

Lean 用更一般的 `every_extensional_observer_agrees` 表示：任何对相等函数给出相等结果的观察者，都无法区分 CP 函数和参考函数。数学导数是这类观察者之一。

式 (7) 不能被误读成“PyTorch autograd 和 NPU backward 已被 Lean 验证”。从数学函数相等到实际梯度 tensor 相等，还需要：

- 每个 kernel 的 backward 与其 forward 导数一致；
- collective backward 实现的是逆向线性映射；
- 浮点舍入、规约顺序和混合精度策略在允许误差范围内；
- 参数更新没有被 `no_grad`、hook、checkpoint 或 optimizer wiring 截断。

这也解释了为什么“权重保存后与初始权重相同”不能由本证明直接排除。该现象可能发生在 optimizer step、梯度连接、checkpoint 保存或训练循环层，而这些都不在布局双射定理的结论里。

---

## 7. 变长文档元数据

### 7.1 为什么局部片段需要自己的 K 前缀

打包序列把多个文档首尾相接。CP 按固定长度切序列时，一个 rank 的本地 shard 可能从某个文档中间开始。例如某文档占全局位置 \([d,e)\)，某个本地连续片段只持有 \([a,b)\)，其中

\[
d\le a<b\le e.
\]

本地 Q 只有 \([a,b)\)，长度为

\[
q=b-a. \tag{8}
\]

但因果 attention 或递归状态所需的可见 K/V 前缀从文档起点开始，是 \([d,b)\)，长度为

\[
k=b-d. \tag{9}
\]

若 \(a>d\)，本地片段不是新文档开头，不能 reset；它必须继承该文档之前的 K/V 或递归状态。若 \(a=d\)，本地片段恰好从文档起点开始，应当 reset。

### 定理 4（reset 判定充要）

在 \(d\le a<b\) 下，

\[
q=k\iff a=d. \tag{10}
\]

### 证明

由 (8)、(9)，命题左侧为

\[
b-a=b-d.
\]

因为 \(a,d\le b\)，自然数截断减法在这里就是普通差。两边从相同的 \(b\) 减去的数相等，当且仅当 \(a=d\)。反向代入立即成立。Lean 使用 Presburger 算术求解这一离散命题，并显式携带 \(d\le a<b\) 的前提。

这正对应 `build_sequence_metadata` 中比较 `cu_seq_q.diff()` 与 `cu_seq_k.diff()` 的条件。等长片段被汇总为真正的 document-start reset。

### 7.2 因果 K 前缀是否完整

定义

\[
K(j)=d+j,\qquad 0\le j<b-d. \tag{11}
\]

需要同时证明两个方向。

**可靠性。** 任意合法 \(j\) 都满足

\[
d\le K(j)<b. \tag{12}
\]

**完备性。** 任意 \(x\) 若满足

\[
d\le x<b,
\]

都可以取 \(j=x-d\)，使得 \(0\le j<b-d\) 且 \(K(j)=x\)。

于是得到集合相等：

\[
\{K(j)\mid 0\le j<b-d\}
=\{x\in\mathbb N\mid d\le x<b\}. \tag{13}
\]

Lean 声明 `prefixKey_exact` 同时包含可靠性和完备性。它排除了两个常见 off-by-one 错误：把 \(b\) 错误地纳入 K 前缀，或者漏掉 \(b-1\)。

---

## 8. 具体 CP=4 实例

固定配置满足：

\[
32768=4\times8192.
\]

相关 head 数也全部能被 4 整除：

| head 类型 | 全局 head 数 | 每个 CP rank 的 head 数 |
|---|---:|---:|
| full-attention Q | 24 | 6 |
| full-attention KV | 4 | 1 |
| GDN Q/K | 16 | 4 |
| GDN V | 48 | 12 |

Lean 以闭合整数命题检查商和余数：

```lean
concrete_sequence_partition
concrete_head_partitions
```

这些算术事实很简单，却不能省略。若某个 head 数不能整除 CP degree，实际代码必须复制、补齐或采用不等长 shard；那时本文的等长 `Fin(P) × Fin(H)` 模型就不再直接适用。

---

## 9. 与代码的逐段对应

| 数学对象 | 固定实现位置 | 需要的桥接前提 |
|---|---|---|
| \(T\)：sequence → head | `sequence_to_head_shard`、`exchange_sequence_heads` | DTensor/HCCL 实现坐标转置 |
| \(T^{-1}\)：head → sequence | `head_to_sequence_shard` | 实现与 \(T\) 互逆 |
| full-attention head 算子 | `AscVarlenAttention.forward` | fused attention 符合声明语义 |
| GDN head 算子 | `ContextParallelGatedDeltaNet.forward` | 卷积、GDN kernel 和 reset 正确 |
| 本地片段 \((d,a,b)\) | `CPVarlenMetadata.from_global` | global-to-local 索引映射正确 |
| reset 判定 \(b-a=b-d\) | `build_sequence_metadata` | diff 数组对应同一片段 |
| K 前缀 \([d,b)\) | `k_global_gather_indices` | gather 按索引取值且不重排错位 |

这张表说明形式证明的作用方式：它没有替代码“背书”，而是给每个工程模块规定一个可审查的数学契约。实现测试应围绕这些契约设计，而不是只看 loss 是否下降。

---

## 10. 应当怎样验证剩余工程义务

要把本文的条件定理推进到具体训练正确性，至少还需要以下测试层次。

### 10.1 纯索引测试

构造带唯一 token/head 标识的整数 tensor，分别执行 sequence→head→sequence，逐元素检查身份映射；覆盖 CP=2、4、8，Q/K/V/GDN 不同 head 数，以及多个 tensor 一起 pack/split 的情形。

### 10.2 Kernel 参考对齐

在 FP64 或 FP32 小尺寸输入上，把 CP full attention、GDN 和 causal convolution 与无 CP reference 对齐，分别检查 forward、输入梯度和参数梯度。BF16 测试应单独报告绝对误差、相对误差和 ULP/末位漂移，不能把它与精确语义测试混为一组。

### 10.3 变长边界穷举

对较短序列枚举文档划分和 CP 切点，验证：

- reset 只出现在 document start；
- 每个 Q 看到的 K 索引正好是同文档的因果前缀；
- 空文档、长度 1 文档、切点恰落在文档边界、切点落在文档内部都被覆盖。

### 10.4 训练循环测试

需要直接断言：

- 至少一个预期参数有非零梯度；
- optimizer step 前后参数发生变化；
- 保存的 checkpoint 与内存中 step 后参数一致；
- 恢复训练后的下一步与不中断参考轨迹在容差内一致。

这些测试对排查“保存权重与初始权重相同”比单看 loss 更直接。

---

## 11. 结论边界

本文已经机器验证的结论是：在给定离散坐标模型中，Qwen3.6 CP 使用的 sequence/head 交换是双射；任意 head-local 精确算子在交换前后语义不变；变长片段的 reset 判定与因果 K 前缀构造满足相应的充要条件和集合等价。

因此，若具体实现满足下列桥接前提：

1. collective 精确实现坐标转置；
2. full attention 和 GDN 各自实现声明的 head-local 函数；
3. 变长元数据中的 \(d,a,b\) 与真实文档和 shard 边界一致；

那么 CP forward 与未切分 reference 是同一个数学函数。

这个结论足以把一大类“CP 因索引错位而改变模型语义”的错误排除在模型之外，也给出了针对实现的精确测试契约。它不排除 kernel 数值错误、autograd 断链、optimizer 未执行、checkpoint 保存错误或分布式运行时故障。对这些问题，应继续使用对应层级的可执行证据，而不是扩大形式定理的表述范围。
