import Init

/-!
# Conditional correctness of Qwen3.6 context parallel transport

This file formalizes the rank-axis permutation used by the Qwen3.5/3.6
`override-refactor` context-parallel path in torchtitan-npu.

The theorem is deliberately conditional at the code-to-model boundary:
* the collective `Shard(sequence) -> Shard(head)` must implement the modeled
  rank-axis transpose without loss, duplication, or reordering inside a shard;
* the local fused attention or Gated DeltaNet kernel must implement the same
  head-shard-local mathematical operator as the non-CP baseline;
* the same global sequence metadata must be supplied to both executions.

Under those premises, the CP forward result is extensionally equal to the
non-CP head-separable result. The theorem is independent of floating-point
rounding and does not certify CANN, HCCL, DTensor, or a concrete NPU kernel.
-/

namespace TorchTitanNPUQwen36CP

def cpDegree : Nat := 4
def globalSequenceLength : Nat := 32_768
def localSequenceLength : Nat := 8_192

def fullAttentionQHeads : Nat := 24
def fullAttentionKVHeads : Nat := 4
def fullQHeadsPerRank : Nat := 6
def fullKVHeadsPerRank : Nat := 1
def gqaGroupSize : Nat := 6

def gdnQKHeads : Nat := 16
def gdnVHeads : Nat := 48
def gdnQKHeadsPerRank : Nat := 4
def gdnVHeadsPerRank : Nat := 12

theorem sequence_partition_exact :
    globalSequenceLength = cpDegree * localSequenceLength := by
  decide +kernel

theorem full_attention_partition_exact :
    fullAttentionQHeads = cpDegree * fullQHeadsPerRank ∧
    fullAttentionKVHeads = cpDegree * fullKVHeadsPerRank ∧
    fullQHeadsPerRank = fullKVHeadsPerRank * gqaGroupSize := by
  decide +kernel

theorem gated_delta_partition_exact :
    gdnQKHeads = cpDegree * gdnQKHeadsPerRank ∧
    gdnVHeads = cpDegree * gdnVHeadsPerRank := by
  decide +kernel

universe u v w x m z

abbrev GlobalSequence (P L : Nat) (α : Type u) :=
  Fin P → Fin L → α

abbrev SequenceSharded (P L : Nat) (α : Type u) :=
  Fin P → Fin L → Fin P → α

abbrev HeadSharded (P L : Nat) (α : Type u) :=
  Fin P → Fin P → Fin L → α

def sequenceToHead
    (tensor : SequenceSharded P L α) : HeadSharded P L α :=
  fun headRank sequenceRank localToken =>
    tensor sequenceRank localToken headRank

def headToSequence
    (tensor : HeadSharded P L α) : SequenceSharded P L α :=
  fun sequenceRank localToken headRank =>
    tensor headRank sequenceRank localToken

theorem head_to_sequence_after_sequence_to_head
    (tensor : SequenceSharded P L α) :
    headToSequence (sequenceToHead tensor) = tensor := by
  funext sequenceRank
  funext localToken
  funext headRank
  rfl

theorem sequence_to_head_after_head_to_sequence
    (tensor : HeadSharded P L α) :
    sequenceToHead (headToSequence tensor) = tensor := by
  funext headRank
  funext sequenceRank
  funext localToken
  rfl

def packThree
    (q : SequenceSharded P L α)
    (k : SequenceSharded P L β)
    (v : SequenceSharded P L γ) :
    SequenceSharded P L (α × β × γ) :=
  fun sequenceRank localToken headRank =>
    (q sequenceRank localToken headRank,
     k sequenceRank localToken headRank,
     v sequenceRank localToken headRank)

theorem packed_exchange_is_componentwise
    (q : SequenceSharded P L α)
    (k : SequenceSharded P L β)
    (v : SequenceSharded P L γ) :
    sequenceToHead (packThree q k v) =
      fun headRank sequenceRank localToken =>
        (sequenceToHead q headRank sequenceRank localToken,
         sequenceToHead k headRank sequenceRank localToken,
         sequenceToHead v headRank sequenceRank localToken) := by
  funext headRank
  funext sequenceRank
  funext localToken
  rfl

def denseThree
    (kernel :
      M →
      GlobalSequence P L α →
      GlobalSequence P L β →
      GlobalSequence P L γ →
      GlobalSequence P L δ)
    (metadata : M)
    (q : SequenceSharded P L α)
    (k : SequenceSharded P L β)
    (v : SequenceSharded P L γ) :
    SequenceSharded P L δ :=
  fun sequenceRank localToken headRank =>
    kernel metadata
      (fun sourceRank sourceToken => q sourceRank sourceToken headRank)
      (fun sourceRank sourceToken => k sourceRank sourceToken headRank)
      (fun sourceRank sourceToken => v sourceRank sourceToken headRank)
      sequenceRank localToken

def contextParallelThree
    (kernel :
      M →
      GlobalSequence P L α →
      GlobalSequence P L β →
      GlobalSequence P L γ →
      GlobalSequence P L δ)
    (metadata : M)
    (q : SequenceSharded P L α)
    (k : SequenceSharded P L β)
    (v : SequenceSharded P L γ) :
    SequenceSharded P L δ :=
  let qHead := sequenceToHead q
  let kHead := sequenceToHead k
  let vHead := sequenceToHead v
  headToSequence
    (fun headRank sequenceRank localToken =>
      kernel metadata
        (fun sourceRank sourceToken =>
          qHead headRank sourceRank sourceToken)
        (fun sourceRank sourceToken =>
          kHead headRank sourceRank sourceToken)
        (fun sourceRank sourceToken =>
          vHead headRank sourceRank sourceToken)
        sequenceRank localToken)

theorem context_parallel_three_eq_dense
    (kernel :
      M →
      GlobalSequence P L α →
      GlobalSequence P L β →
      GlobalSequence P L γ →
      GlobalSequence P L δ)
    (metadata : M)
    (q : SequenceSharded P L α)
    (k : SequenceSharded P L β)
    (v : SequenceSharded P L γ) :
    contextParallelThree kernel metadata q k v =
      denseThree kernel metadata q k v := by
  funext sequenceRank
  funext localToken
  funext headRank
  rfl

def denseUnary
    (kernel : M → GlobalSequence P L α → GlobalSequence P L β)
    (metadata : M)
    (input : SequenceSharded P L α) :
    SequenceSharded P L β :=
  fun sequenceRank localToken headRank =>
    kernel metadata
      (fun sourceRank sourceToken =>
        input sourceRank sourceToken headRank)
      sequenceRank localToken

def contextParallelUnary
    (kernel : M → GlobalSequence P L α → GlobalSequence P L β)
    (metadata : M)
    (input : SequenceSharded P L α) :
    SequenceSharded P L β :=
  let headInput := sequenceToHead input
  headToSequence
    (fun headRank sequenceRank localToken =>
      kernel metadata
        (fun sourceRank sourceToken =>
          headInput headRank sourceRank sourceToken)
        sequenceRank localToken)

theorem context_parallel_unary_eq_dense
    (kernel : M → GlobalSequence P L α → GlobalSequence P L β)
    (metadata : M)
    (input : SequenceSharded P L α) :
    contextParallelUnary kernel metadata input =
      denseUnary kernel metadata input := by
  funext sequenceRank
  funext localToken
  funext headRank
  rfl

theorem downstream_observation_preserved
    (observe : SequenceSharded P L δ → ζ)
    (kernel :
      M →
      GlobalSequence P L α →
      GlobalSequence P L β →
      GlobalSequence P L γ →
      GlobalSequence P L δ)
    (metadata : M)
    (q : SequenceSharded P L α)
    (k : SequenceSharded P L β)
    (v : SequenceSharded P L γ) :
    observe (contextParallelThree kernel metadata q k v) =
      observe (denseThree kernel metadata q k v) := by
  rw [context_parallel_three_eq_dense]

theorem full_attention_transport_correct
    (kernel :
      M →
      GlobalSequence P L α →
      GlobalSequence P L β →
      GlobalSequence P L γ →
      GlobalSequence P L δ)
    (metadata : M)
    (q : SequenceSharded P L α)
    (k : SequenceSharded P L β)
    (v : SequenceSharded P L γ) :
    contextParallelThree kernel metadata q k v =
      denseThree kernel metadata q k v :=
  context_parallel_three_eq_dense kernel metadata q k v

theorem gated_delta_transport_correct
    (kernel : M → GlobalSequence P L α → GlobalSequence P L β)
    (metadata : M)
    (input : SequenceSharded P L α) :
    contextParallelUnary kernel metadata input =
      denseUnary kernel metadata input :=
  context_parallel_unary_eq_dense kernel metadata input

#print axioms head_to_sequence_after_sequence_to_head
#print axioms sequence_to_head_after_head_to_sequence
#print axioms packed_exchange_is_componentwise
#print axioms context_parallel_three_eq_dense
#print axioms context_parallel_unary_eq_dense
#print axioms downstream_observation_preserved
#print axioms full_attention_transport_correct
#print axioms gated_delta_transport_correct

end TorchTitanNPUQwen36CP
