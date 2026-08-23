import Init

/-!
# TorchTitan-NPU Qwen3.6 context-parallel semantic correctness

Model ID: SP-P0002-TTNPU-Q36-CP-v1

Frozen implementation evidence:
* torchtitan-npu repository mirror: botcanlearn/torchtitan-npu-upstream
* branch: override-refactor
* commit: 7864d5dc17930667d663bbadd1ce2bc722de2753
* pinned TorchTitan commit: c91448d20480c7b294314e68976823050002ebec

This file proves the mathematical core of the Qwen3.6 context-parallel
construction used by the frozen implementation:

1. sequence-sharded/head-replicated and head-sharded/sequence-replicated
   layouts are related by an exact coordinate transposition;
2. the two transpositions are mutual inverses, so the exchange cannot lose,
   duplicate, or misaddress an entry in the abstract model;
3. every operator that is independent across attention/GDN heads commutes with
   this transposition, hence the CP forward map equals the unsharded reference;
4. the local variable-length metadata criterion identifies a document reset
   exactly when a local segment starts at the document boundary;
5. the generated key-prefix range is sound and complete for the causal prefix
   [document start, segment stop).

The theorem is conditional on an implementation bridge: DTensor/HCCL must
realize the modeled transposition, and the NPU attention/GDN kernels must realize
the stated head-local mathematical operators.  Lean checks the deduction from
those premises; it does not verify Python, CANN, HCCL, Triton, BF16 rounding, or
a training run.
-/

namespace TorchTitanNPUQwen36CP

universe u v w

/-! ## 1. Two coordinate views of the same tensor

The concrete implementation uses CP degree `P`.  In the sequence view, the
first rank coordinate chooses a contiguous sequence shard and the second rank
coordinate chooses a head shard.  In the head view those two rank coordinates
are transposed.  `S` is the local sequence length and `H` the number of local
heads inside one head shard.
-/

abbrev SequenceLayout (α : Type u) (P S H : Nat) :=
  Fin P → Fin S → Fin P → Fin H → α

abbrev HeadLayout (α : Type u) (P S H : Nat) :=
  Fin P → Fin P → Fin S → Fin H → α

/-- Abstract semantics of sequence-shard to head-shard redistribution. -/
def sequenceToHead
    {α : Type u} {P S H : Nat}
    (x : SequenceLayout α P S H) : HeadLayout α P S H :=
  fun headRank seqRank localSeq localHead =>
    x seqRank localSeq headRank localHead

/-- Abstract semantics of the inverse head-shard to sequence-shard redistribution. -/
def headToSequence
    {α : Type u} {P S H : Nat}
    (x : HeadLayout α P S H) : SequenceLayout α P S H :=
  fun seqRank localSeq headRank localHead =>
    x headRank seqRank localSeq localHead

/-- A round trip starting in sequence layout returns every entry unchanged. -/
theorem headToSequence_sequenceToHead
    {α : Type u} {P S H : Nat}
    (x : SequenceLayout α P S H) :
    headToSequence (sequenceToHead x) = x := by
  funext seqRank localSeq headRank localHead
  rfl

/-- A round trip starting in head layout returns every entry unchanged. -/
theorem sequenceToHead_headToSequence
    {α : Type u} {P S H : Nat}
    (x : HeadLayout α P S H) :
    sequenceToHead (headToSequence x) = x := by
  funext headRank seqRank localSeq localHead
  rfl

/-- Pointwise form of the no-loss/no-duplication round-trip theorem. -/
theorem roundTrip_preserves_entry
    {α : Type u} {P S H : Nat}
    (x : SequenceLayout α P S H)
    (seqRank : Fin P) (localSeq : Fin S)
    (headRank : Fin P) (localHead : Fin H) :
    headToSequence (sequenceToHead x) seqRank localSeq headRank localHead =
      x seqRank localSeq headRank localHead := by
  rfl

/-! ## 2. Head-separable operators commute with the redistribution

A full attention head and a Gated DeltaNet value head both consume the complete
sequence for that head and produce another complete sequence for the same head.
They may be nonlinear and stateful along the sequence axis; the only structural
condition used here is that one head does not read another head's coordinates.
-/

abbrev SequenceFiber (α : Type u) (P S : Nat) :=
  Fin P → Fin S → α

abbrev HeadOperator (α : Type u) (β : Type v) (P S : Nat) :=
  SequenceFiber α P S → SequenceFiber β P S

/-- Apply a head-local operator directly to the global logical tensor. -/
def applyGlobal
    {α : Type u} {β : Type v} {P S H : Nat}
    (op : HeadOperator α β P S)
    (x : SequenceLayout α P S H) : SequenceLayout β P S H :=
  fun seqRank localSeq headRank localHead =>
    op (fun sourceRank sourceSeq =>
      x sourceRank sourceSeq headRank localHead) seqRank localSeq

/-- Apply the same operator after sequence-to-head redistribution. -/
def applyOnHeadShards
    {α : Type u} {β : Type v} {P S H : Nat}
    (op : HeadOperator α β P S)
    (x : HeadLayout α P S H) : HeadLayout β P S H :=
  fun headRank seqRank localSeq localHead =>
    op (fun sourceRank sourceSeq =>
      x headRank sourceRank sourceSeq localHead) seqRank localSeq

/--
Main forward-equivalence theorem.  Exchange to head layout, evaluate the
head-local kernel, and exchange back is extensionally equal to evaluating the
same kernel on the unsharded logical tensor.
-/
theorem headwise_context_parallel_correct
    {α : Type u} {β : Type v} {P S H : Nat}
    (op : HeadOperator α β P S)
    (x : SequenceLayout α P S H) :
    headToSequence (applyOnHeadShards op (sequenceToHead x)) =
      applyGlobal op x := by
  funext seqRank localSeq headRank localHead
  rfl

/-- Function-level equality, useful for any downstream semantic observer. -/
theorem contextParallelFunction_eq_reference
    {α : Type u} {β : Type v} {P S H : Nat}
    (op : HeadOperator α β P S) :
    (fun x : SequenceLayout α P S H =>
      headToSequence (applyOnHeadShards op (sequenceToHead x))) =
    applyGlobal op := by
  funext x
  exact headwise_context_parallel_correct op x

/--
Any extensional observer sees the same result on CP and reference functions.
An independently defined derivative/gradient operator is one possible observer;
thus exact forward function equality is the mathematical premise needed for an
exact backward corollary, apart from implementation-level floating-point and
autograd correctness.
-/
theorem every_extensional_observer_agrees
    {α : Type u} {β : Type v} {γ : Type w} {P S H : Nat}
    (op : HeadOperator α β P S)
    (observer :
      (SequenceLayout α P S H → SequenceLayout β P S H) → γ) :
    observer
        (fun x => headToSequence
          (applyOnHeadShards op (sequenceToHead x))) =
      observer (applyGlobal op) := by
  exact congrArg observer (contextParallelFunction_eq_reference op)

/-! ## 3. Token-local maps also commute with the exchange

The Q/K/V, decay, beta, gate and output projections are token-local maps before
or after the head-local kernel.  Their commutation is recorded separately so
that the full implementation correspondence does not hide this step.
-/

def mapSequence
    {α : Type u} {β : Type v} {P S H : Nat}
    (f : α → β) (x : SequenceLayout α P S H) :
    SequenceLayout β P S H :=
  fun seqRank localSeq headRank localHead =>
    f (x seqRank localSeq headRank localHead)


def mapHead
    {α : Type u} {β : Type v} {P S H : Nat}
    (f : α → β) (x : HeadLayout α P S H) :
    HeadLayout β P S H :=
  fun headRank seqRank localSeq localHead =>
    f (x headRank seqRank localSeq localHead)


theorem sequenceToHead_map_commutes
    {α : Type u} {β : Type v} {P S H : Nat}
    (f : α → β) (x : SequenceLayout α P S H) :
    sequenceToHead (mapSequence f x) = mapHead f (sequenceToHead x) := by
  funext headRank seqRank localSeq localHead
  rfl


theorem headToSequence_map_commutes
    {α : Type u} {β : Type v} {P S H : Nat}
    (f : α → β) (x : HeadLayout α P S H) :
    headToSequence (mapHead f x) = mapSequence f (headToSequence x) := by
  funext seqRank localSeq headRank localHead
  rfl

/-! ## 4. Variable-length segment metadata

A local CP shard can cut through a packed document.  For one contiguous local
run, `docStart` is the global document start, `localStart` the first global token
owned by the run, and `stop` the exclusive global end of the run.
-/

structure LocalSegment where
  docStart : Nat
  localStart : Nat
  stop : Nat
  doc_le_local : docStart ≤ localStart
  local_lt_stop : localStart < stop

/-- Number of Q tokens physically present in the local run. -/
def qLength (s : LocalSegment) : Nat :=
  s.stop - s.localStart

/-- Number of K/V tokens in the exact causal prefix ending at this run. -/
def kPrefixLength (s : LocalSegment) : Nat :=
  s.stop - s.docStart

/-- Criterion used by the Qwen metadata rebuild: local Q and K lengths agree. -/
def resetMarked (s : LocalSegment) : Prop :=
  qLength s = kPrefixLength s

/--
The equality-of-lengths criterion is exact: it marks precisely those local runs
whose first token is also the first token of the packed document.
-/
theorem resetMarked_iff_documentStart
    (s : LocalSegment) :
    resetMarked s ↔ s.localStart = s.docStart := by
  constructor
  · intro h
    have hDoc := s.doc_le_local
    have hStop := s.local_lt_stop
    unfold resetMarked qLength kPrefixLength at h
    omega
  · intro h
    unfold resetMarked qLength kPrefixLength
    omega

/-- The j-th key in the generated causal prefix. -/
def prefixKey (s : LocalSegment) (j : Fin (kPrefixLength s)) : Nat :=
  s.docStart + j.val

/-- Every generated key lies at or after the document start. -/
theorem prefixKey_lowerBound
    (s : LocalSegment) (j : Fin (kPrefixLength s)) :
    s.docStart ≤ prefixKey s j := by
  unfold prefixKey
  omega

/-- Every generated key lies strictly before the segment stop. -/
theorem prefixKey_upperBound
    (s : LocalSegment) (j : Fin (kPrefixLength s)) :
    prefixKey s j < s.stop := by
  have hj := j.isLt
  unfold prefixKey
  unfold kPrefixLength at hj
  omega

/-- Every key in [docStart, stop) appears in the generated prefix. -/
theorem prefixKey_complete
    (s : LocalSegment) (k : Nat)
    (hLower : s.docStart ≤ k)
    (hUpper : k < s.stop) :
    ∃ j : Fin (kPrefixLength s), prefixKey s j = k := by
  have hlt : k - s.docStart < kPrefixLength s := by
    unfold kPrefixLength
    omega
  let j : Fin (kPrefixLength s) := ⟨k - s.docStart, hlt⟩
  refine ⟨j, ?_⟩
  unfold prefixKey
  change s.docStart + (k - s.docStart) = k
  omega

/-- Soundness and completeness of the causal-prefix gather range. -/
theorem prefixKey_exact
    (s : LocalSegment) (k : Nat) :
    (∃ j : Fin (kPrefixLength s), prefixKey s j = k) ↔
      s.docStart ≤ k ∧ k < s.stop := by
  constructor
  · intro h
    cases h with
    | intro j hj =>
      rw [← hj]
      exact ⟨prefixKey_lowerBound s j, prefixKey_upperBound s j⟩
  · intro h
    exact prefixKey_complete s k h.1 h.2

/-! ## 5. Frozen concrete Qwen3.6 long-text instance

The `qwen35_27b_long_text_sft` configuration in the frozen override branch uses
sequence length 32768, CP degree 4 and no head-tail load balancer.  The same
implementation names cover Qwen3.5/3.6 in the repository.  The relevant head
counts are divisible by four, so no partial head shard exists in this instance.
-/

def cpDegree : Nat := 4
def globalSequenceLength : Nat := 32_768
def localSequenceLength : Nat := 8_192

def fullAttentionQHeads : Nat := 24
def fullAttentionKVHeads : Nat := 4
def gatedDeltaQKHeads : Nat := 16
def gatedDeltaVHeads : Nat := 48


def overrideCommit : String :=
  "7864d5dc17930667d663bbadd1ce2bc722de2753"


def torchTitanCommit : String :=
  "c91448d20480c7b294314e68976823050002ebec"


theorem concrete_sequence_partition :
    globalSequenceLength / cpDegree = localSequenceLength ∧
    globalSequenceLength % cpDegree = 0 := by
  decide +kernel


theorem concrete_head_partitions :
    fullAttentionQHeads / cpDegree = 6 ∧
    fullAttentionQHeads % cpDegree = 0 ∧
    fullAttentionKVHeads / cpDegree = 1 ∧
    fullAttentionKVHeads % cpDegree = 0 ∧
    gatedDeltaQKHeads / cpDegree = 4 ∧
    gatedDeltaQKHeads % cpDegree = 0 ∧
    gatedDeltaVHeads / cpDegree = 12 ∧
    gatedDeltaVHeads % cpDegree = 0 := by
  decide +kernel

/-! ## 6. Bundled audit statement -/

theorem audit_conclusion :
    (∀ {α : Type u} (x : SequenceLayout α 4 8_192 6),
      headToSequence (sequenceToHead x) = x) ∧
    (∀ {α : Type u} {β : Type v}
      (op : HeadOperator α β 4 8_192)
      (x : SequenceLayout α 4 8_192 6),
      headToSequence (applyOnHeadShards op (sequenceToHead x)) =
        applyGlobal op x) ∧
    (∀ s : LocalSegment,
      resetMarked s ↔ s.localStart = s.docStart) ∧
    globalSequenceLength / cpDegree = localSequenceLength := by
  exact ⟨
    fun x => headToSequence_sequenceToHead x,
    fun op x => headwise_context_parallel_correct op x,
    resetMarked_iff_documentStart,
    by decide +kernel
  ⟩

/-! ## 7. Trust-base audit hooks -/

#print axioms headToSequence_sequenceToHead
#print axioms sequenceToHead_headToSequence
#print axioms headwise_context_parallel_correct
#print axioms contextParallelFunction_eq_reference
#print axioms every_extensional_observer_agrees
#print axioms resetMarked_iff_documentStart
#print axioms prefixKey_exact
#print axioms concrete_sequence_partition
#print axioms concrete_head_partitions
#print axioms audit_conclusion

end TorchTitanNPUQwen36CP
