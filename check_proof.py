from nemo_skills.code_execution.sandbox import LocalSandbox
sandbox = LocalSandbox()

proof = """
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Tactic

set_option maxHeartbeats 0

open BigOperators Complex

theorem amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + I) / Real.sqrt 2) :
    ((∑ k : ℤ in Finset.Icc 1 12, z ^ k ^ 2) * (∑ k : ℤ in Finset.Icc 1 12, 1 / z ^ k ^ 2)) = 36 := by
  -- First show z is an 8th root of unity
  have hz : z ^ 8 = 1 := by
    rw [h₀]
    have : (1 + I) ^ 8 = 16 := by
      rw [← Complex.norm_eq_abs, ← norm_pow, norm_sq_add_mul_I, pow_mul']
      norm_num
    rw [← this, div_pow, ← pow_mul, mul_comm]
    simp only [Real.sqrt_mul_self zero_le_two, Real.sqrt_sq zero_le_two]
    norm_num
  
  -- The exponents are periodic mod 8
  have h_periodic : ∀ n : ℤ, z ^ (n + 8) = z ^ n := by
    intro n
    rw [← zpow_mul, ← zpow_add, add_comm, zpow_add, hz, one_zpow, mul_comm, zpow_mul, hz, one_zpow]
  
  -- Reduce the sums to just 1..8 using periodicity
  let s1 := ∑ k : ℤ in Finset.Icc 1 12, z ^ k ^ 2
  let s2 := ∑ k : ℤ in Finset.Icc 1 12, 1 / z ^ k ^ 2
  have h_s2 : s2 = ∑ k : ℤ in Finset.Icc 1 12, z ^ (-k ^ 2) := by
    simp_rw [one_div, zpow_neg]
  
  have h_sum1 : s1 = ∑ k : ℤ in Finset.Icc 1 8, z ^ k ^ 2 := by
    rw [← Finset.sum_union (Finset.disjoint_Icc_Icc.mpr (by norm_num))]
    congr
    · apply Finset.sum_congr
      · apply Finset.ext; intro x; simp; omega
      · simp
    · apply Finset.sum_bij' (fun k _ => k - 8) (by simp; omega) (by simp [h_periodic]) 
        (fun k _ => k + 8) (by simp; omega) (by simp) (by simp; omega)
  
  have h_sum2 : s2 = ∑ k : ℤ in Finset.Icc 1 8, z ^ (-k ^ 2) := by
    rw [h_s2]
    apply h_sum1.trans
    congr
    ext k
    rw [← Int.pow_mod, ← Int.pow_mod, h_periodic]
  
  rw [h_sum1, h_sum2]
  
  -- Now compute the product of sums
  have h_sums : (∑ k in Finset.Icc 1 8, z ^ k ^ 2) * (∑ k in Finset.Icc 1 8, z ^ (-k ^ 2)) =
      ∑ k in Finset.Icc 1 8, ∑ l in Finset.Icc 1 8, z ^ (k ^ 2 - l ^ 2) := by
    simp_rw [Finset.mul_sum, Finset.sum_mul, ← zpow_sub, div_eq_mul_inv, inv_zpow]
  
  rw [h_sums]
  
  -- The key observation: z^(k²-l²) = 1 when k=l or k+l=8
  have h_pairwise : ∀ k l ∈ Finset.Icc 1 8, (z ^ (k ^ 2 - l ^ 2) = 1 ↔ k = l ∨ k + l = 8) := by
    intro k l hk hl
    rw [← sub_eq_zero, ← mul_eq_zero, sub_mul]
    have h8 : z ^ 8 = 1 := by rw [hz]
    have h4 : z ^ 4 = -1 := by
      rw [h₀, div_pow, ← pow_mul, mul_comm]
      norm_num
      rw [← pow_mul, (by norm_num : 4 * 2 = 8)]
      exact hz
    cases' Int.modEq_or_modEq_of_modEq_mul (k ^ 2 - l ^ 2) 0 8 with h h
    · left; omega
    · right; omega
  
  -- Count the diagonal terms where z^(k²-l²) = 1
  have h_diag : ∑ k in Finset.Icc 1 8, ∑ l in Finset.Icc 1 8, if k = l ∨ k + l = 8 then 1 else 0 = 16 := by
    simp_rw [Finset.sum_comm, ← Finset.sum_add_distrib]
    let diag1 := ∑ k in Finset.Icc 1 8, if k = k then 1 else 0
    let diag2 := ∑ k in Finset.Icc 1 8, if k + k = 8 then 1 else 0
    have : diag1 = 8 := by simp [diag1]
    have : diag2 = 8 := by
      simp [diag2]
      have : (Finset.Icc 1 8).filter (fun k => k + k = 8) = {4} := by
        apply Finset.ext; intro x; simp; omega
      rw [this]; simp
    simp [*]
  
  -- The off-diagonal terms cancel out in pairs
  have h_rest : ∑ k in Finset.Icc 1 8, ∑ l in Finset.Icc 1 8, if ¬(k = l ∨ k + l = 8) then z ^ (k ^ 2 - l ^ 2) else 0 = 0 := by
    apply Finset.sum_eq_zero; intro k hk
    apply Finset.sum_eq_zero; intro l hl
    by_cases h : k = l ∨ k + l = 8
    · simp [h]
    · simp [h]
      have : ∃ m n, m ∈ Finset.Icc 1 8 ∧ n ∈ Finset.Icc 1 8 ∧ ¬(m = n ∨ m + n = 8) ∧ 
          z ^ (k ^ 2 - l ^ 2) = -z ^ (m ^ 2 - n ^ 2) := by
        refine ⟨8 - l, 8 - k, ?_, ?_, ?_⟩
        · simp; omega
        · simp; omega
        · constructor
          · intro h'; cases h' with | inl h' => omega | inr h' => omega
          · have : (k ^ 2 - l ^ 2) + ((8 - l) ^ 2 - (8 - k) ^ 2) = 0 := by ring
            rw [← this, zpow_add, eq_neg_iff_add_eq_zero]
            rw [h₀, ← div_pow, ← div_pow, div_eq_mul_inv, div_eq_mul_inv]
            ring
      rcases this with ⟨m, n, hm, hn, hmn, heq⟩
      rw [heq]
  
  -- Combine the results
  rw [← Finset.sum_add_distrib, h_diag, h_rest]
  norm_num
"""

sandbox_output = sandbox.is_proof_correct(proof)

print(f'Status: {sandbox_output["process_status"]}')
print(sandbox_output['stdout'])
