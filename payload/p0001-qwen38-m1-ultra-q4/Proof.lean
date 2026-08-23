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
Those are external premises. Lean checks the deduction from those premises.
-/

namespace Qwen38M1Ultra

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

/-- gate_proj + up_proj + down_proj in all 64 FFNs. -/
def ffnQ4Weights : Nat :=
  layers * (3 * hidden * intermediate)

/-- Large Gated DeltaNet projections that remain in the tier quantization. -/
def gdnQ4Weights : Nat :=
  let keyDim := linearQKHeads * linearHeadDim
  let valueDim := linearVHeads * linearHeadDim
  gdnLayers *
    (hidden * (2 * keyDim + valueDim) +
     hidden * valueDim +
     valueDim * hidden)

/-- in_proj_a and in_proj_b, represented as F32 in the selected recipe. -/
def ssmF32Weights : Nat :=
  gdnLayers * (2 * hidden * linearVHeads)

/-- Full-attention q/k/v/o weights, represented as Q8_0 in the selected recipe. -/
def fullAttnQ8Weights : Nat :=
  fullAttnLayers *
    (hidden * (2 * attnQHeads * attnHeadDim) +
     hidden * (attnKVHeads * attnHeadDim) +
     hidden * (attnKVHeads * attnHeadDim) +
     (attnQHeads * attnHeadDim) * hidden)

/-- Untied LM output projection only. -/
def outputQ6Weights : Nat :=
  vocab * hidden

/-- Q4_K: 256 weights -> 144 bytes. -/
def q4KBytes (n : Nat) : Nat := (n / 256) * 144
/-- Q6_K: 256 weights -> 210 bytes. -/
def q6KBytes (n : Nat) : Nat := (n / 256) * 210
/-- Q8_0: 32 weights -> 34 bytes. -/
def q8_0Bytes (n : Nat) : Nat := (n / 32) * 34
/-- F32: one weight -> four bytes. -/
def f32Bytes (n : Nat) : Nat := n * 4

/-- Conservative static representation floor used by the bandwidth theorem. -/
def staticRepresentationFloor : Nat :=
  q4KBytes (ffnQ4Weights + gdnQ4Weights) +
  q8_0Bytes fullAttnQ8Weights +
  f32Bytes ssmF32Weights +
  q6KBytes outputQ6Weights

/-- FP16 KV bytes added by one context token across the 16 full-attention layers. -/
def kvBytesPerContextToken : Nat :=
  fullAttnLayers * attnKVHeads * attnHeadDim * 2 * 2

def representationBytes (ctx : Nat) : Nat :=
  staticRepresentationFloor + kvBytesPerContextToken * ctx

/--
`cacheCarry` is an explicit allowance for representation bytes that need not
cross DRAM because they persist on chip from the previous token.
-/
def dramFloor (ctx cacheCarry : Nat) : Nat :=
  representationBytes ctx - cacheCarry

/-- Apple's published M1 Ultra memory-bandwidth figure. -/
def appleRatedBps : Nat := 800_000_000_000

/-- External line-rate premises used only for the more permissive bus scenario. -/
def lpddrTransfersPerSec : Nat := 6_400_000_000
def memoryBusWidthBits : Nat := 1_024

def rawBusBps : Nat :=
  lpddrTransfersPerSec * memoryBusWidthBits / 8

theorem raw_bus_bytes_per_second :
    rawBusBps = 819_200_000_000 := by
  decide +kernel

/-- `rateMilli` is measured in 1/1000 token/s. -/
def MemoryFeasible
    (bandwidthBps rateMilli actualDramBytesPerToken : Nat) : Prop :=
  rateMilli * actualDramBytesPerToken ≤ bandwidthBps * 1000

/-- Generic conservation theorem: a threshold whose required byte rate exceeds
    the bandwidth cap cannot be reached by any feasible execution. -/
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
Exact discrete ceiling in the abstract floor-traffic model. The second conjunct
only witnesses feasibility when actual traffic equals the declared floor; it is
not an empirical attainability claim for real hardware.
-/
def ExactCeilingAt
    (bandwidth ctx cacheCarry ceilingMilli : Nat) : Prop :=
  CeilingAt bandwidth ctx cacheCarry ceilingMilli ∧
  MemoryFeasible bandwidth ceilingMilli (dramFloor ctx cacheCarry)

/-- Adjacent integer inequalities certify the exact maximum milli-token rate. -/
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

/-! Closed arithmetic facts. -/
theorem ffn_count : ffnQ4Weights = 17_112_760_320 := by decide +kernel
theorem gdn_q4_count : gdnQ4Weights = 5_536_481_280 := by decide +kernel
theorem ssm_f32_count : ssmF32Weights = 23_592_960 := by decide +kernel
theorem full_attn_q8_count : fullAttnQ8Weights = 1_677_721_600 := by decide +kernel
theorem output_q6_count : outputQ6Weights = 1_271_398_400 := by decide +kernel

theorem q4_alignment :
    (ffnQ4Weights + gdnQ4Weights) % 256 = 0 := by decide +kernel
theorem q8_alignment : fullAttnQ8Weights % 32 = 0 := by decide +kernel
theorem q6_alignment : outputQ6Weights % 256 = 0 := by decide +kernel

theorem static_floor_bytes :
    staticRepresentationFloor = 15_660_093_440 := by decide +kernel

theorem kv_per_token_bytes :
    kvBytesPerContextToken = 65_536 := by decide +kernel

/-! Raw-line-rate ceilings, cacheCarry = 0. -/
theorem raw_ctx_0 : CeilingAt rawBusBps 0 0 52_311 := by
  apply certifyCeiling
  decide +kernel

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

theorem raw_ctx_262k_exact :
    ExactCeilingAt rawBusBps 262_144 0 24_945 := by
  apply certifyExactCeiling
  · decide +kernel
  · decide +kernel

/-! Apple-rated 800 GB/s ceilings. -/
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

/-! Explicitly hypothetical cache-residency sensitivity. -/
def cache96MiB : Nat := 96 * 1024 * 1024

theorem raw_ctx_0_cache96_scenario :
    CeilingAt rawBusBps 0 cache96MiB 52_649 := by
  apply certifyCeiling
  decide +kernel

/-! Compute-side conservation is kept parametric. -/
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
    largeProjectionWeights = 25_621_954_560 := by decide +kernel

theorem classical_mac_ops_floor :
    classicalMacOpsFloor = 51_243_909_120 := by decide +kernel

/-- Principal model-internal exact ceilings. -/
theorem audit_conclusion :
    ExactCeilingAt rawBusBps 0 0 52_311 ∧
    ExactCeilingAt rawBusBps 262_144 0 24_945 ∧
    ExactCeilingAt appleRatedBps 0 0 51_085 ∧
    ExactCeilingAt appleRatedBps 262_144 0 24_360 := by
  exact ⟨raw_ctx_0_exact, raw_ctx_262k_exact,
    apple_rated_ctx_0_exact, apple_rated_ctx_262k_exact⟩

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
