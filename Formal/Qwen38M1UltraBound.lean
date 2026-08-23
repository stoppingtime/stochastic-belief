import Init

/-!
# Qwen3.8-27B Q4_K_M decode roofline on M1 Ultra

Target: Lean 4.33.0 (core library only; no Mathlib).

Pinned external artifact for the instantiated Q4_K_M scenario:
  repo: 6block/Qwen3.8-27B-GGUF
  file: Qwen3.8-27B-Q4_K_M.gguf
  SHA-256: 038b8d86da2e388e4c3f5bafdf5a8aa4dcb630861a18d430b3a79c538c1a3beb

Lean does NOT certify that hash-to-tensor-layout relationship; the hash is a
provenance anchor for the external evidence/review procedure.

Scope: batch=1, one target-model token per decode step, no speculative/MTP,
       exact standard attention over a growing FP16 KV cache.

What this file proves:
  * exact integer parameter/encoding counts for the stated model/quantization model;
  * an abstract bandwidth conservation theorem;
  * certified milli-token/s ceilings for selected context lengths;
  * cache-carry is explicit instead of silently assumed away.

What this file does NOT prove:
  * that Apple's published/derived hardware numbers are true;
  * that a particular GGUF file actually has the declared tensor types;
  * that a runtime must satisfy the stated DRAM-byte lower-bound assumption.
Those are external premises.  Lean checks the deduction from those premises.
-/

namespace Qwen38M1Ultra

/-! ## 1. Frozen architecture facts -/

def hidden : Nat := 5_120
def intermediate : Nat := 17_408
def layers : Nat := 64

def gdnLayers : Nat := 48
def fullAttnLayers : Nat := 16

def linearQKHeads : Nat := 16
def linearVHeads : Nat := 48
def linearHeadDim : Nat := 128

def attnQHeads : Nat := 24
def attnKVHeads : Nat := 4
def attnHeadDim : Nat := 256

def vocab : Nat := 248_320

/-! ## 2. Parameter classes used in the conservative byte floor

The floor deliberately omits tiny tensors such as RMSNorm vectors, convolution
weights, A_log/dt_bias, etc.  Omitting required bytes makes the lower bound on
traffic smaller and therefore makes the throughput ceiling more permissive.
This is the safe direction for an upper-bound proof.
-/

/-- gate_proj + up_proj + down_proj in all 64 FFNs. -/
def ffnQ4Weights : Nat :=
  layers * (3 * hidden * intermediate)

/--
For each Gated DeltaNet layer, only the three large tier-quantized projections:
  in_proj_qkv : hidden -> 2*keyDim + valueDim
  in_proj_z   : hidden -> valueDim
  out_proj    : valueDim -> hidden
-/
def gdnQ4Weights : Nat :=
  let keyDim := linearQKHeads * linearHeadDim
  let valueDim := linearVHeads * linearHeadDim
  gdnLayers *
    (hidden * (2 * keyDim + valueDim) +
     hidden * valueDim +
     valueDim * hidden)

/-- in_proj_a and in_proj_b; protected as F32 in the selected GGUF recipe. -/
def ssmF32Weights : Nat :=
  gdnLayers * (2 * hidden * linearVHeads)

/--
Full-attention q/k/v/o weights. q has a second head-sized half for output gating.
These are protected as Q8_0 in the selected GGUF recipe.
-/
def fullAttnQ8Weights : Nat :=
  fullAttnLayers *
    (hidden * (2 * attnQHeads * attnHeadDim) +
     hidden * (attnKVHeads * attnHeadDim) +
     hidden * (attnKVHeads * attnHeadDim) +
     (attnQHeads * attnHeadDim) * hidden)

/-- Untied LM output projection only.  The input embedding table is not streamed whole per token. -/
def outputQ6Weights : Nat :=
  vocab * hidden

/-! ## 3. GGML/GGUF block encodings

Q4_K : 256 weights -> 144 bytes  (4.5 bpw)
Q6_K : 256 weights -> 210 bytes  (6.5625 bpw)
Q8_0 :  32 weights ->  34 bytes  (32 int8 + one fp16 scale)
F32   : one weight   ->   4 bytes
-/

def q4KBytes (n : Nat) : Nat := (n / 256) * 144
def q6KBytes (n : Nat) : Nat := (n / 256) * 210
def q8_0Bytes (n : Nat) : Nat := (n / 32) * 34
def f32Bytes (n : Nat) : Nat := n * 4

/-- Conservative static representation bytes that must participate in one decode step. -/
def staticRepresentationFloor : Nat :=
  q4KBytes (ffnQ4Weights + gdnQ4Weights) +
  q8_0Bytes fullAttnQ8Weights +
  f32Bytes ssmF32Weights +
  q6KBytes outputQ6Weights

/-! ## 4. KV growth

There are 16 growing full-attention layers.
For each context token and each such layer:
  4 KV heads * 256 dims * 2 tensors (K,V) * 2 bytes (FP16)
-/

def kvBytesPerContextToken : Nat :=
  fullAttnLayers * attnKVHeads * attnHeadDim * 2 * 2

/-- Representation-level working set touched by a standard exact decode step. -/
def representationBytes (ctx : Nat) : Nat :=
  staticRepresentationFloor + kvBytesPerContextToken * ctx

/--
`cacheCarry` is an explicit allowance for bytes that need not cross DRAM because
an execution model lets them persist on chip from the previous token.

This is deliberately a parameter: public M1 Ultra documentation is not strong
enough to turn a particular SLC number into an unconditional theorem about GPU
residency.  The physical premise supplied to the theorem is that actual DRAM
traffic is at least this number.
-/
def dramFloor (ctx cacheCarry : Nat) : Nat :=
  representationBytes ctx - cacheCarry

/-! ## 5. Hardware caps

`appleRatedBps` is Apple's published 800 GB/s product figure.
`rawBusBps` is the more permissive LPDDR5-6400 x 1024-bit line-rate figure.
For a "cannot exceed" argument, using the larger number is safer.
-/

def appleRatedBps : Nat := 800_000_000_000

/-- External bus facts: LPDDR5-6400 and aggregate 1024-bit width. -/
def lpddrTransfersPerSec : Nat := 6_400_000_000
def memoryBusWidthBits : Nat := 1_024

/-- 6.4e9 transfers/s * 1024 bits / 8 = 819.2e9 bytes/s. -/
def rawBusBps : Nat :=
  lpddrTransfersPerSec * memoryBusWidthBits / 8

theorem raw_bus_bytes_per_second :
    rawBusBps = 819_200_000_000 := by
  decide +kernel

/-! ## 6. Integer definition of feasible throughput

`rateMilli` means 1/1000 token/s.  Thus

    rateMilli * bytesPerToken <= bandwidthBytesPerSecond * 1000

contains no floating point and no rounding assumptions.
-/

def MemoryFeasible
    (bandwidthBps rateMilli actualDramBytesPerToken : Nat) : Prop :=
  rateMilli * actualDramBytesPerToken ≤ bandwidthBps * 1000

/--
Generic conservation theorem.
If every token needs at least `floor` DRAM bytes and a proposed threshold would
already demand more than the bandwidth cap, then any feasible rate is below
that threshold.
-/
theorem threshold_impossible
    {bandwidth rate actual floor threshold : Nat}
    (hFloor : floor ≤ actual)
    (hFeasible : MemoryFeasible bandwidth rate actual)
    (hTooFast : bandwidth * 1000 < threshold * floor) :
    rate < threshold := by
  have hNot : ¬ threshold ≤ rate := by
    intro hRate
    have hProd : threshold * floor ≤ rate * actual :=
      Nat.mul_le_mul hRate hFloor
    have hCap : threshold * floor ≤ bandwidth * 1000 :=
      Nat.le_trans hProd hFeasible
    exact (Nat.not_le_of_lt hTooFast) hCap
  omega

/-- Meaning of a certified integer milli-token/s ceiling. -/
def CeilingAt
    (bandwidth ctx cacheCarry ceilingMilli : Nat) : Prop :=
  ∀ rate actual,
    dramFloor ctx cacheCarry ≤ actual →
    MemoryFeasible bandwidth rate actual →
    rate ≤ ceilingMilli

/-- Reduce a concrete ceiling certificate to one closed integer inequality. -/
theorem certifyCeiling
    (bandwidth ctx cacheCarry ceilingMilli : Nat)
    (hNumeric :
      bandwidth * 1000 <
        (ceilingMilli + 1) * dramFloor ctx cacheCarry) :
    CeilingAt bandwidth ctx cacheCarry ceilingMilli := by
  intro rate actual hFloor hFeasible
  have hlt : rate < ceilingMilli + 1 :=
    threshold_impossible
      (bandwidth := bandwidth)
      (rate := rate)
      (actual := actual)
      (floor := dramFloor ctx cacheCarry)
      (threshold := ceilingMilli + 1)
      hFloor hFeasible hNumeric
  omega

/--
An exact ceiling certificate at milli-token/s granularity.  This strengthens an
upper bound by also exhibiting that the stated ceiling itself is feasible in
the abstract model when actual DRAM traffic is exactly the declared floor.

This is *not* a claim that real hardware reaches the ceiling; it proves that,
inside the conservation model, the integer ceiling cannot be lowered without
adding another premise.
-/
def ExactCeilingAt
    (bandwidth ctx cacheCarry ceilingMilli : Nat) : Prop :=
  CeilingAt bandwidth ctx cacheCarry ceilingMilli ∧
  MemoryFeasible bandwidth ceilingMilli (dramFloor ctx cacheCarry)

/--
A pair of adjacent integer inequalities certifies the exact maximum milli-token
rate in the abstract floor-traffic model:

  ceiling * floor ≤ cap < (ceiling + 1) * floor.
-/
theorem certifyExactCeiling
    (bandwidth ctx cacheCarry ceilingMilli : Nat)
    (hAtCeiling :
      ceilingMilli * dramFloor ctx cacheCarry ≤ bandwidth * 1000)
    (hNextImpossible :
      bandwidth * 1000 <
        (ceilingMilli + 1) * dramFloor ctx cacheCarry) :
    ExactCeilingAt bandwidth ctx cacheCarry ceilingMilli := by
  constructor
  · exact certifyCeiling bandwidth ctx cacheCarry ceilingMilli hNextImpossible
  · exact hAtCeiling

/-! ## 7. Kernel-checked closed arithmetic facts -/

theorem ffn_count : ffnQ4Weights = 17_112_760_320 := by
  decide +kernel

theorem gdn_q4_count : gdnQ4Weights = 5_536_481_280 := by
  decide +kernel

theorem ssm_f32_count : ssmF32Weights = 23_592_960 := by
  decide +kernel

theorem full_attn_q8_count : fullAttnQ8Weights = 1_677_721_600 := by
  decide +kernel

theorem output_q6_count : outputQ6Weights = 1_271_398_400 := by
  decide +kernel

/-- Block alignment checks, so the block byte formulas have no partial block. -/
theorem q4_alignment :
    (ffnQ4Weights + gdnQ4Weights) % 256 = 0 := by
  decide +kernel

theorem q8_alignment : fullAttnQ8Weights % 32 = 0 := by
  decide +kernel

theorem q6_alignment : outputQ6Weights % 256 = 0 := by
  decide +kernel

/-- 15.660093440 GB decimal = about 14.5846 GiB. -/
theorem static_floor_bytes :
    staticRepresentationFloor = 15_660_093_440 := by
  decide +kernel

/-- Exactly 64 KiB of FP16 KV per context token. -/
theorem kv_per_token_bytes :
    kvBytesPerContextToken = 65_536 := by
  decide +kernel

/-! ## 8. Certified streaming ceilings, cacheCarry = 0

The numbers are milli-token/s.  Example: 52_311 means 52.311 token/s.
The raw-bus family is the more permissive physical line-rate roof.
-/

theorem raw_ctx_0 : CeilingAt rawBusBps 0 0 52_311 := by
  apply certifyCeiling
  decide +kernel

/-- Exact discrete maximum of the abstract raw-bus/floor model at ctx=0. -/
theorem raw_ctx_0_exact : ExactCeilingAt rawBusBps 0 0 52_311 := by
  apply certifyExactCeiling
  · decide +kernel
  · decide +kernel

theorem raw_ctx_1k : CeilingAt rawBusBps 1_024 0 52_088 := by
  apply certifyCeiling
  decide +kernel

theorem raw_ctx_4k : CeilingAt rawBusBps 4_096 0 51_429 := by
  apply certifyCeiling
  decide +kernel

theorem raw_ctx_8k : CeilingAt rawBusBps 8_192 0 50_577 := by
  apply certifyCeiling
  decide +kernel

theorem raw_ctx_16k : CeilingAt rawBusBps 16_384 0 48_954 := by
  apply certifyCeiling
  decide +kernel

theorem raw_ctx_32k : CeilingAt rawBusBps 32_768 0 46_002 := by
  apply certifyCeiling
  decide +kernel

theorem raw_ctx_64k : CeilingAt rawBusBps 65_536 0 41_052 := by
  apply certifyCeiling
  decide +kernel

theorem raw_ctx_128k : CeilingAt rawBusBps 131_072 0 33_781 := by
  apply certifyCeiling
  decide +kernel

theorem raw_ctx_262k : CeilingAt rawBusBps 262_144 0 24_945 := by
  apply certifyCeiling
  decide +kernel

/-- Exact discrete maximum of the abstract raw-bus/floor model at native max context. -/
theorem raw_ctx_262k_exact :
    ExactCeilingAt rawBusBps 262_144 0 24_945 := by
  apply certifyExactCeiling
  · decide +kernel
  · decide +kernel

/-! Apple's published 800 GB/s figure, same conservative representation floor. -/
theorem apple_rated_ctx_0 : CeilingAt appleRatedBps 0 0 51_085 := by
  apply certifyCeiling
  decide +kernel

theorem apple_rated_ctx_0_exact :
    ExactCeilingAt appleRatedBps 0 0 51_085 := by
  apply certifyExactCeiling
  · decide +kernel
  · decide +kernel

theorem apple_rated_ctx_262k : CeilingAt appleRatedBps 262_144 0 24_360 := by
  apply certifyCeiling
  decide +kernel

theorem apple_rated_ctx_262k_exact :
    ExactCeilingAt appleRatedBps 262_144 0 24_360 := by
  apply certifyExactCeiling
  · decide +kernel
  · decide +kernel

/-! ## 9. Cache-carry sensitivity, explicitly hypothetical

96 MiB is provided only as a scenario.  The theorem itself does not assert that
M1 Ultra's GPU can actually retain exactly this many useful bytes between steps.
-/

def cache96MiB : Nat := 96 * 1024 * 1024

theorem raw_ctx_0_cache96_scenario :
    CeilingAt rawBusBps 0 cache96MiB 52_649 := by
  apply certifyCeiling
  decide +kernel

/-! ## 10. Compute-side model, kept parametric

For quantized GEMV, an FP32 TFLOP headline is not a rigorous cap on the mixed
integer/dequantize/FP execution path.  We therefore formalize compute
conservation without pretending that a public FP32 number is the exact Q4 cap.
-/

def ComputeFeasible
    (peakOpsPerSec rateMilli actualOpsPerToken : Nat) : Prop :=
  rateMilli * actualOpsPerToken ≤ peakOpsPerSec * 1000

theorem compute_threshold_impossible
    {peak rate actualOps floorOps threshold : Nat}
    (hFloor : floorOps ≤ actualOps)
    (hFeasible : ComputeFeasible peak rate actualOps)
    (hTooFast : peak * 1000 < threshold * floorOps) :
    rate < threshold := by
  have hNot : ¬ threshold ≤ rate := by
    intro hRate
    have hProd : threshold * floorOps ≤ rate * actualOps :=
      Nat.mul_le_mul hRate hFloor
    have hCap : threshold * floorOps ≤ peak * 1000 :=
      Nat.le_trans hProd hFeasible
    exact (Nat.not_le_of_lt hTooFast) hCap
  omega

/-- Large linear projections alone: one multiply + one add per matrix weight. -/
def largeProjectionWeights : Nat :=
  ffnQ4Weights + gdnQ4Weights + ssmF32Weights +
  fullAttnQ8Weights + outputQ6Weights

def classicalMacOpsFloor : Nat := 2 * largeProjectionWeights

theorem large_projection_weight_count :
    largeProjectionWeights = 25_621_954_560 := by
  decide +kernel

theorem classical_mac_ops_floor :
    classicalMacOpsFloor = 51_243_909_120 := by
  decide +kernel

/-! ## 11. One top-level audit conclusion

This bundles the principal certified numerical results into one declaration.
It is intentionally a statement *inside the frozen model*.  The bridge from
real hardware/file/runtime to the model remains the external-evidence layer.
-/

theorem audit_conclusion :
    ExactCeilingAt rawBusBps 0 0 52_311 ∧
    ExactCeilingAt rawBusBps 262_144 0 24_945 ∧
    ExactCeilingAt appleRatedBps 0 0 51_085 ∧
    ExactCeilingAt appleRatedBps 262_144 0 24_360 := by
  exact ⟨raw_ctx_0_exact, raw_ctx_262k_exact,
    apple_rated_ctx_0_exact, apple_rated_ctx_262k_exact⟩

/-! ## 12. Audit hooks

Run `#print axioms` after build.  No `sorry`, `admit`, `native_decide`, custom
axiom, or unsafe theorem is used in this file.
-/

#print axioms threshold_impossible
#print axioms certifyCeiling
#print axioms certifyExactCeiling
#print axioms raw_ctx_0
#print axioms raw_ctx_0_exact
#print axioms raw_ctx_262k
#print axioms apple_rated_ctx_0
#print axioms compute_threshold_impossible
#print axioms audit_conclusion

end Qwen38M1Ultra
