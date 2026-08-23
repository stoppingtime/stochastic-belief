<!-- MODEL-ID: SP-P0002-TTNPU-Q36-CP-v1 -->
<!-- UPSTREAM-COMMIT: 7864d5dc17930667d663bbadd1ce2bc722de2753 -->
<!-- CP-DEGREE: 4 -->
<!-- GLOBAL-SEQ-LEN: 32768 -->
<!-- LOCAL-SEQ-LEN: 8192 -->
<!-- FULL-Q-HEADS: 24 -->
<!-- FULL-KV-HEADS: 4 -->
<!-- GDN-QK-HEADS: 16 -->
<!-- GDN-V-HEADS: 48 -->

# TorchTitan-NPU `override-refactor` 中 Qwen3.6 Context Parallel 的条件正确性证明

**模型编号：SP-P0002-TTNPU-Q36-CP-v1**

## 摘要

本文讨论 `cann/torchtitan-npu` 的 `override-refactor` 分支中，Qwen3.5/3.6 长文本训练所采用的 Context Parallel（CP）实现。审阅对象固定在提交 `7864d5dc17930667d663bbadd1ce2bc722de2753`。该配置把长度为 32768 的序列按 CP=4 切成每 rank 8192 token，并同时覆盖两类完全不同的序列算子：16 个 full-attention 层使用变长 TND 融合注意力；48 个 Gated DeltaNet（GDN）层使用因果卷积与递归状态更新。

证明不把“训练能运行”当成算法正确，也不把一组 loss 接近当成数学等价。我们把实现拆成三层：

1. **通信层**：把张量从“序列维分片”变成“头块维分片”，计算后再逆变换；
2. **局部算子层**：每个头块在拿到完整序列后执行 full attention 或 GDN；
3. **实现映射层**：DTensor、HCCL、CANN 融合算子和 metadata 构造是否真正满足前两层的数学语义。

Lean 4 形式化验证了第一层，并证明：只要通信确实是无损的 rank 轴交换，局部 kernel 与非 CP 基线实现同一个头块可分算子，而且两边使用相同的全局序列边界，那么 CP 输出与非 CP 输出逐元素相等。这个定理对局部 kernel 的内部形式没有限制，所以同时适用于因果 softmax attention、变长 attention，以及把因果卷积、reset 和递归更新整体看作一个头块算子的 GDN。

这是一条**条件正确性定理**。它比“代码看起来像 All-to-All”更强，因为通信往返与整体函数等价已经由 Lean 检查；又比“证明了整套 NPU 训练绝对正确”更克制，因为 DTensor/HCCL 的具体语义、融合算子的数值实现和 metadata 的代码到模型映射仍是外部证明义务。

---

## 1. 外部事实：审阅的代码到底做了什么

### 1.1 固定分支与文件

审阅基线是 `override-refactor` 提交：

```text
7864d5dc17930667d663bbadd1ce2bc722de2753
```

与本证明直接相关的文件为：

```text
torchtitan_npu/models/qwen3_5/config_registry.py
torchtitan_npu/override/qwen3_5/parallelize.py
torchtitan_npu/override/qwen3_5/varlen_attention.py
torchtitan_npu/override/qwen3_5/gated_delta.py
```

目录仍叫 `qwen3_5`，但 override 的说明明确写的是 “Qwen3.5/3.6 Context Parallel”。本文沿用用户侧名称“Qwen3.6”，同时保留源码路径，避免把产品名和 Python package 名混为一谈。

### 1.2 长文本配置

`qwen35_27b_long_text_sft()` 固定：

\[
L=32768,\qquad P=4,
\]

其中 \(L\) 是全局序列长度，\(P\) 是 CP degree。每个序列 rank 初始持有

\[
L_{\mathrm{local}}=\frac{32768}{4}=8192
\]

个 token。配置还加载两个 override：

- `varlen_attention.asc_cp`：full attention 的 CP 路径；
- `gated_delta.context_parallel`：GDN 的 CP 路径。

### 1.3 通信骨架

`sequence_to_head_shard` 先把本地张量声明为 `Shard(1)`，即沿序列维分片，再通过 DTensor `redistribute` 变成沿 head 维分片。`head_to_sequence_shard` 做反向变换。

`exchange_sequence_heads` 还会先把多个 projection 的 head chunk 打包后一次通信。这个打包只应改变通信次数，不能改变任何元素的归属。形式化证明专门给出了“先打包再交换”等价于“分别交换再打包”的逐点定理。

### 1.4 full attention 路径

`AscVarlenAttention.forward` 的顺序是：

1. 必要时按 GQA 语义扩展 K/V；
2. 对 Q、K、V 执行序列分片到头分片的交换；
3. 每个 rank 在自己的头块上拿到完整序列；
4. 调用 `npu_fusion_attention_v3`，输入布局为 TND；
5. 把输出从头分片交换回序列分片。

Qwen3.6 27B 的 full-attention 头数为：

\[
H_Q=24,\qquad H_{KV}=4.
\]

在 \(P=4\) 下，每个 rank 对应：

\[
H_Q^{\mathrm{local}}=6,\qquad H_{KV}^{\mathrm{local}}=1.
\]

因此每个 rank 恰好接收一组完整的 GQA 头块：一个 KV head 对应六个 Q heads。这个整除关系由 Lean 中的封闭整数定理检查。

### 1.5 GDN 路径

`ContextParallelGatedDeltaNet.forward` 对 Q、K、V、decay 和 beta 五类 projection 做同样的序列到头块交换。之后：

- 因果卷积只使用本 rank 对应的卷积头权重；
- `cu_seqlens` 生成 reset 边界，卷积不会跨样本段传播；
- `A_log` 和 `dt_bias` 按头块切分；
- GDN kernel 在完整序列上执行递归；
- 输出再交换回序列分片，与仍按序列分片保存的 output gate 做 norm，最后进入 `out_proj`。

头数为：

\[
H_{QK}=16,\qquad H_V=48.
\]

在 \(P=4\) 下，本地头数分别为 4 和 12。它们虽然不相等，但都能按同一个 CP rank 划分。形式模型因此不把“一个 head”写成标量，而把每个 rank 对应的整个 head chunk 当作抽象 payload。这样可以在不伪造 Q/K/V 形状相同的前提下证明通信等价。

---

## 2. 建模假设

形式化定理不直接读取 Python、HCCL trace 或 NPU 内存。它使用以下三个明确前提。

### 假设 A：通信是 rank 轴的无损交换

把全局序列位置写成二元索引

\[
(r_s,i),\qquad r_s\in\{0,\ldots,P-1\},\quad
i\in\{0,\ldots,L_{\mathrm{local}}-1\},
\]

把全局头块写成

\[
r_h\in\{0,\ldots,P-1\}.
\]

序列分片布局中的元素记为

\[
X[r_s,i,r_h].
\]

序列到头分片的数学语义是

\[
(\mathcal T X)[r_h,r_s,i]=X[r_s,i,r_h]. \tag{1}
\]

逆变换为

\[
(\mathcal T^{-1}Y)[r_s,i,r_h]=Y[r_h,r_s,i]. \tag{2}
\]

这里的 payload 可以是一个 Q 头块、一个 KV 头块，也可以是 GDN 的五元 projection 组合。假设 A 要求真实 collective 与式 (1) 的元素映射一致，没有丢失、复制、错序或错误 chunk 边界。

### 假设 B：局部 kernel 与基线头块算子相同

对任意头块 \(r_h\)，记完整序列上的局部算子为

\[
\mathcal K_m(Q_{r_h},K_{r_h},V_{r_h}),
\]

其中 \(m\) 是全局 metadata，包括因果边界、累计序列长度、reset 信息和 scale。这个 \(\mathcal K_m\) 可以是：

- 因果 full attention；
- varlen TND attention；
- 包含因果卷积、gate、beta 和递归状态更新的完整 GDN 头块；
- 任何只依赖当前头块完整序列、不跨头块混合的确定性算子。

假设 B 要求 CP 路径调用的融合 kernel，与非 CP 基线在相同输入和相同 metadata 下实现同一个 \(\mathcal K_m\)。

### 假设 C：metadata 是同一个全局序列分段

CP 不能把不同样本拼成一条连续因果序列，也不能漏掉 reset。形式证明把 metadata 作为同一个参数同时传给 dense 与 CP 算子。真实代码要满足这个模型，就必须证明 `build_sequence_metadata` 所重建的 `cu_seqlens` 与基线全局 batch 的段边界一致。

---

## 3. 一般等价定理

定义非 CP 的头块可分计算：

\[
Y_{\mathrm{dense}}[r_s,i,r_h]
=
\mathcal K_m
\left(
Q[\cdot,\cdot,r_h],
K[\cdot,\cdot,r_h],
V[\cdot,\cdot,r_h]
\right)[r_s,i].
\tag{3}
\]

CP 算法先应用 \(\mathcal T\)，在每个 \(r_h\) 上计算完整序列，再应用 \(\mathcal T^{-1}\)：

\[
Y_{\mathrm{cp}}
=
\mathcal T^{-1}
\left(
r_h\mapsto
\mathcal K_m
\left(
(\mathcal TQ)[r_h,\cdot,\cdot],
(\mathcal TK)[r_h,\cdot,\cdot],
(\mathcal TV)[r_h,\cdot,\cdot]
\right)
\right).
\tag{4}
\]

### 定理 1（通信往返）

对任意张量 \(X\)，

\[
\mathcal T^{-1}(\mathcal TX)=X,
\qquad
\mathcal T(\mathcal T^{-1}X)=X.
\tag{5}
\]

**证明。** 对任意 \((r_s,i,r_h)\)，由定义：

\[
(\mathcal T^{-1}\mathcal TX)[r_s,i,r_h]
=(\mathcal TX)[r_h,r_s,i]
=X[r_s,i,r_h].
\]

另一方向同理。Lean 用函数外延逐个展开三个索引，证明项归约为 `rfl`。证毕。

### 定理 2（CP 与非 CP 前向等价）

在假设 A、B、C 对应的数学模型中，对任意 Q、K、V 和任意头块可分 kernel \(\mathcal K_m\)，

\[
\boxed{Y_{\mathrm{cp}}=Y_{\mathrm{dense}}}. \tag{6}
\]

**证明。** 固定任意输出索引 \((r_s,i,r_h)\)。由式 (4) 和逆交换定义：

\[
Y_{\mathrm{cp}}[r_s,i,r_h]
=
\mathcal K_m
\left(
(\mathcal TQ)[r_h,\cdot,\cdot],
(\mathcal TK)[r_h,\cdot,\cdot],
(\mathcal TV)[r_h,\cdot,\cdot]
\right)[r_s,i].
\]

再由式 (1)：

\[
(\mathcal TQ)[r_h,r'_s,i']=Q[r'_s,i',r_h],
\]

K、V 同理，所以右侧恰好等于式 (3)。因为索引任意，函数外延给出整体相等。Lean 中这条定理同样在展开定义后归约到逐点 `rfl`。证毕。

### 定理 3（打包通信不改变语义）

令

\[
P(Q,K,V)[r_s,i,r_h]
=
(Q[r_s,i,r_h],K[r_s,i,r_h],V[r_s,i,r_h]).
\]

则

\[
\mathcal T(P(Q,K,V))
=
P(\mathcal TQ,\mathcal TK,\mathcal TV).
\tag{7}
\]

因此 `exchange_sequence_heads` 把多个 projection 先拼接再 collective，只要 chunk 和 split 边界符合 payload 结构，就不会改变算法语义。

### 推论（下游目标一致）

对任意纯函数观测量 \(\Phi\)，例如把模型输出继续送入后续层并最终形成标量 loss，

\[
\Phi(Y_{\mathrm{cp}})=\Phi(Y_{\mathrm{dense}}). \tag{8}
\]

Lean 通过对式 (6) 使用函数同余得到该推论。若把两边视为同一个可微实函数，那么在导数存在处数学梯度相同；但具体 autograd kernel、浮点舍入和分布式反向通信不在这份 Lean 文件中建模。

---

## 4. 对 full attention 的实例化

full attention 的 payload 不要求 Q、K、V 头数相同。对 CP rank \(r_h\)，可以令：

- Q payload 是 6 个连续 Q heads；
- K payload 是 1 个 KV head；
- V payload 是 1 个 KV head。

局部 \(\mathcal K_m\) 内部负责 GQA 映射，即让该 KV head 服务对应的 6 个 Q heads。因为

\[
24=4\times6,\qquad
4=4\times1,\qquad
6=1\times6,
\]

Q 与 KV 的分组边界和 CP rank 完全对齐。形式证明不需要把 GQA 重复写成虚假的逐头复制，只需把一组头当作 payload，并要求局部 kernel 的 GQA 语义与基线一致。

因果 mask、TND layout 和 varlen 边界都包含在 metadata 与 \(\mathcal K_m\) 中。通信定理说明：只要每个本地头块看到的全局 token 顺序与基线相同，CP 不会改变 attention 的数学输入。

---

## 5. 对 GDN 的实例化

GDN 比 full attention 更容易被误判，因为它不是一次普通 attention，而是包含时间递归。证明的关键不是把递归公式删掉，而是确认递归只在头块内部沿完整序列发生。

把一个 GDN 头块 payload 定义为：

\[
Z=(Q,K,V,a,\beta,\theta_{\mathrm{conv}},A_{\log},b_{\Delta t}),
\]

并令 \(\mathcal G_m(Z)\) 表示：

1. 按段边界执行因果卷积；
2. 计算 decay gate；
3. 按 reset 边界执行 Gated Delta recurrence；
4. 返回该头块的完整序列输出。

那么 GDN 的 CP 路径正是定理 2 的一元版本：

\[
\mathcal T^{-1}\bigl(\mathcal G_m(\mathcal TZ)\bigr)
=
\mathcal G_m(Z).
\tag{9}
\]

这里有两个不能省略的实现条件：

- `shard_local_heads` 必须让卷积权重、`A_log`、`dt_bias` 与交换后激活属于同一头块；
- `cu_seqlens` 必须在全局序列坐标上给出与基线相同的 reset。

Lean 证明了“若两条条件成立，交换本身不会改变 GDN”；它没有替代对这两条代码映射的审查。

---

## 6. 形式验证

`Proof.lean` 使用 Lean 4.33.0，仅导入 `Init`。形式化对象包括：

- CP=4、32768=4×8192 的整数分解；
- full attention 24/4 heads 在 CP=4 下的 6/1 GQA 分组；
- GDN 16/48 heads 在 CP=4 下的 4/12 分组；
- `sequenceToHead` 与 `headToSequence` 的互逆；
- 打包交换与逐 tensor 交换的等价；
- 任意 metadata、任意三输入头块 kernel 的 CP/dense 等价；
- 任意一输入复合 kernel 的 CP/dense 等价，用于整体封装 GDN；
- 任意下游纯函数观测量的保持性。

证明不使用 `sorry`、`admit`、自定义公理或 `native_decide`。CI 需要执行：

```bash
lake build
lake env leanchecker --fresh Problems.TorchTitanNPUQwen36CP.Proof
lake env lean Problems/TorchTitanNPUQwen36CP/Proof.lean
```

并审查 `#print axioms` 输出。

---

## 7. 解释边界：本文没有证明什么

这份证明不能单独推出“override-refactor 的 Qwen3.6 CP 已经无条件正确”。它没有证明：

1. DTensor `redistribute` 在当前 torch/torch_npu/HCCL 组合上绝不会错序；
2. `torch.chunk`、拼接和 `split` 的实际边界与模型 payload 分组永远一致；
3. `build_sequence_metadata` 对 packing、padding、跨 rank 样本边界的所有情况都与基线相同；
4. `npu_fusion_attention_v3` 与参考 attention 在 BF16 下逐 bit 相等；
5. Triton-Ascend GDN kernel 与参考递归、反向传播和 checkpoint 重算逐 bit 相等；
6. FSDP、梯度缩放、optimizer step 和权重保存路径没有其他独立 bug。

这些是代码到数学模型的桥梁。完整工程闭环应继续加入：

- CP1/CP4 同 checkpoint、同 global batch、确定性前向与反向对照；
- Q/K/V、GDN 中间状态和最终梯度的分层误差；
- packed sequence 中跨 rank 段边界的构造性测试；
- 对 `exchange_sequence_heads` 的可逆随机测试；
- 32K 与 64K 的 loss/grad/weight-update 对照；
- checkpoint 保存、恢复后继续训练的权重差异检查。

形式证明的作用是把这些测试应当验证的合同写清楚：测试失败时，可以定位是通信映射、metadata、局部 kernel，还是模型外围训练路径，而不是只看到“loss 不一样”。

---

## 8. 结论必须连同前提一起读

在审阅提交和固定配置下，Qwen3.6 CP 的核心算法可以抽象为：

\[
\text{sequence-shard}
\xrightarrow{\mathcal T}
\text{head-shard with full sequence}
\xrightarrow{\mathcal K_m}
\text{head-shard output}
\xrightarrow{\mathcal T^{-1}}
\text{sequence-shard}.
\]

Lean 已验证：

\[
\boxed{
\mathcal T^{-1}\circ \mathcal K_m\circ\mathcal T
=
\mathcal K_m
}
\]

在这里，等号按全局索引逐元素成立，前提是 \(\mathcal K_m\) 对头块可分，并且通信与 metadata 符合本文模型。full attention 与 GDN 都能作为该一般定理的实例；不同头数被保留为不同 payload，而没有被强行写成相同形状。

因此可以给出清楚的工程判断：

> `override-refactor` 的 Qwen3.6 CP **通信算法骨架在所列前提下是数学正确的**；当前剩余风险集中在代码到模型的映射，尤其是 DTensor chunk 边界、varlen metadata、融合 kernel 语义和反向路径。任何“已完全证明训练正确”的表述都超出了本证明的范围。
