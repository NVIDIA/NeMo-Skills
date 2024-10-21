minif2f_deepseek_fewshot = [
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "informal_prefix": "/-- Expand the following expression: $7(3y+2)$ Show that it is 21y+14.-/\n",
        "formal_statement": "theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n",
        "formal_proof": "  /- We apply the distributive property to get\\begin{align*}\n  7(3y+2) &= 7\\cdot 3y+7\\cdot 2\\\\\n  &= 21y+14.\n  \\end{align*}\n  -/\n  ring",
    },
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "informal_prefix": "/-- For what real value of $k$ is $\\frac{13-\\sqrt{131}}{4}$ a root of $2x^2-13x+k$? Show that it is $\\frac{19}{4}$.-/\n",
        "formal_statement": "theorem mathd_algebra_116 (k x : ℝ) (h₀ : x = (13 - Real.sqrt 131) / 4)\n    (h₁ : 2 * x ^ 2 - 13 * x + k = 0) : k = 19 / 4 := by\n",
        "formal_proof": "  /- We could substitute $(13-\\sqrt{131})/4$ for $x$ in the equation, but the quadratic formula suggests a quicker approach. Substituting $2$, $-13$, and $k$ into the quadratic formula gives  \\[\n  \\frac{-(-13)\\pm\\sqrt{(-13)^2-4(2)(k)}}{2(2)}= \\frac{13\\pm\\sqrt{169-8k}}{4}.\n  \\]Setting $(13+\\sqrt{169-8k})/4$ and $(13-\\sqrt{169-8k})/4$ equal to $(13-\\sqrt{131})/4$, we find no solution in the first case and $169-8k=131$ in the second case.  Solving yields $k=(169-131)/8=38/8=\\frac{19}{4}$.\n  -/\n  rw [h₀] at h₁\n  rw [eq_comm.mp (add_eq_zero_iff_neg_eq.mp h₁)]\n  norm_num\n  rw [pow_two]\n  rw [mul_sub]\n  rw [sub_mul, sub_mul]\n  rw [Real.mul_self_sqrt _]\n  ring\n  linarith",
    },
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "informal_prefix": "/-- What is the greatest common factor of $20 !$ and $200,\\!000$?  (Reminder: If $n$ is a positive integer, then $n!$ stands for the product $1\\cdot 2\\cdot 3\\cdot \\cdots \\cdot (n-1)\\cdot n$.) Show that it is 40,\\!000.-/\n",
        "formal_statement": "theorem mathd_numbertheory_169 : Nat.gcd 20! 200000 = 40000 := by\n",
        "formal_proof": "  /- The prime factorization of $200,000$ is $2^6 \\cdot 5^5$. Then count the number of factors of $2$ and $5$ in $20!$. Since there are $10$ even numbers, there are more than $6$ factors of $2$. There are $4$ factors of $5$. So the greatest common factor is $2^6 \\cdot 5^4=40,\\!000$.\n  -/\n  apply Eq.refl",
    },
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "informal_prefix": "/-- A group of $N$ students, where $N < 50$, is on a field trip. If their teacher puts them in groups of 8, the last group has 5 students. If their teacher instead puts them in groups of 6, the last group has 3 students. What is the sum of all possible values of $N$? Show that it is 66.-/\n",
        "formal_statement": "theorem mathd_numbertheory_149 :\n  (∑ k in Finset.filter (fun x => x % 8 = 5 ∧ x % 6 = 3) (Finset.range 50), k) = 66 := by\n",
        "formal_proof": "  /- We are given that $N\\equiv 5\\pmod{8}$ and $N\\equiv 3\\pmod{6}$.  We begin checking numbers which are 5 more than a multiple of 8, and we find that 5 and 13 are not 3 more than a multiple of 6, but 21 is 3 more than a multiple of 6. Thus 21 is one possible value of $N$. By the Chinese Remainder Theorem, the integers $x$ satisfying $x\\equiv 5\\pmod{8}$ and $x\\equiv 3\\pmod{6}$ are those of the form $x=21+\\text{lcm}(6,8)k = 21 + 24 k$, where $k$ is an integer. Thus the 2 solutions less than $50$ are 21 and $21+24(1) = 45$, and their sum is $21+45=66$.\n  -/\n  apply Eq.refl",
    },
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "informal_prefix": "/-- Evaluate: $\\left( \\frac{1}{2} + \\frac{1}{3} \\right) \\left( \\frac{1}{2} - \\frac{1}{3} \\right)$ Show that it is \\frac{5}{36}.-/\n",
        "formal_statement": "theorem mathd_algebra_462 : ((1 : ℚ) / 2 + 1 / 3) * (1 / 2 - 1 / 3) = 5 / 36 := by\n",
        "formal_proof": "  /- For any $x$ and $y$, $(x+y)(x-y)=x^2-y^2+xy-xy=x^2-y^2$, so \\begin{align*}\n  \\left( \\frac{1}{2} + \\frac{1}{3} \\right) \\left( \\frac{1}{2} - \\frac{1}{3} \\right)&=\\left(\\frac12\\right)^2-\\left(\\frac13\\right)^2\\\\\n  &=\\frac14-\\frac19\\\\\n  &=\\frac{9}{36}-\\frac{4}{36}\\\\\n  &=\\frac{5}{36}\n  \\end{align*}\n  -/\n  simp_all only [one_div]\n  norm_num",
    },
]

examples_map = {
    "minif2f_deepseek_fewshot": minif2f_deepseek_fewshot,
}
