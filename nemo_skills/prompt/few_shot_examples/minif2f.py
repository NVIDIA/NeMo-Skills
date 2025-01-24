minif2f_deepseek_fewshot = [
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "informal_prefix": "/-- Expand the following expression: $7(3y+2)$ Show that it is 21y+14.-/\n",
        "formal_statement": "theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n",
        "formal_proof": "/- We apply the distributive property to get\\begin{align*}\n  7(3y+2) &= 7\\cdot 3y+7\\cdot 2\\\\\n  &= 21y+14.\n  \\end{align*}\n  -/\nring",
    },
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "informal_prefix": "/-- What is the units digit of $19^{19}+99^{99}$? Show that it is 8.-/\n",
        "formal_statement": "theorem mathd_numbertheory_202 : (19 ^ 19 + 99 ^ 99) % 10 = 8 := by\n",
        "formal_proof": "/- The units digit of a power of an integer is determined by the units digit of the integer; that is, the tens digit, hundreds digit, etc... of the integer have no effect on the units digit of the result. In this problem, the units digit of $19^{19}$ is the units digit of $9^{19}$. Note that $9^1=9$ ends in 9, $9^2=81$ ends in 1, $9^3=729$ ends in 9, and, in general, the units digit of odd powers of 9 is 9, whereas the units digit of even powers of 9 is 1. Since both exponents are odd, the sum of their units digits is $9+9=18$, the units digit of which is $8.$\n  -/\napply Eq.refl",
    },
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "informal_prefix": "/-- At each basketball practice last week, Jenny made twice as many free throws as she made at the previous practice.  At her fifth practice she made 48 free throws.  How many free throws did she make at the first practice? Show that it is 3.-/\n",
        "formal_statement": "theorem mathd_algebra_455 (x : ℝ) (h₀ : 2 * (2 * (2 * (2 * x))) = 48) : x = 3 := by\n",
        "formal_proof": "/- At Jenny's fourth practice she made $\\frac{1}{2}(48)=24$ free throws. At her third practice she made 12, at her second practice she made 6, and at her first practice she made $3$.\n  -/\nlinarith",
    },
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "informal_prefix": "/-- A group of $N$ students, where $N < 50$, is on a field trip. If their teacher puts them in groups of 8, the last group has 5 students. If their teacher instead puts them in groups of 6, the last group has 3 students. What is the sum of all possible values of $N$? Show that it is 66.-/\n",
        "formal_statement": "theorem mathd_numbertheory_149 :\n  (∑ k in Finset.filter (fun x => x % 8 = 5 ∧ x % 6 = 3) (Finset.range 50), k) = 66 := by\n",
        "formal_proof": "/- We are given that $N\\equiv 5\\pmod{8}$ and $N\\equiv 3\\pmod{6}$.  We begin checking numbers which are 5 more than a multiple of 8, and we find that 5 and 13 are not 3 more than a multiple of 6, but 21 is 3 more than a multiple of 6. Thus 21 is one possible value of $N$. By the Chinese Remainder Theorem, the integers $x$ satisfying $x\\equiv 5\\pmod{8}$ and $x\\equiv 3\\pmod{6}$ are those of the form $x=21+\\text{lcm}(6,8)k = 21 + 24 k$, where $k$ is an integer. Thus the 2 solutions less than $50$ are 21 and $21+24(1) = 45$, and their sum is $21+45=66$.\n  -/\napply Eq.refl",
    },
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "informal_prefix": "/-- Evaluate: $\\left( \\frac{1}{2} + \\frac{1}{3} \\right) \\left( \\frac{1}{2} - \\frac{1}{3} \\right)$ Show that it is \\frac{5}{36}.-/\n",
        "formal_statement": "theorem mathd_algebra_462 : ((1 : ℚ) / 2 + 1 / 3) * (1 / 2 - 1 / 3) = 5 / 36 := by\n",
        "formal_proof": "/- For any $x$ and $y$, $(x+y)(x-y)=x^2-y^2+xy-xy=x^2-y^2$, so \\begin{align*}\n  \\left( \\frac{1}{2} + \\frac{1}{3} \\right) \\left( \\frac{1}{2} - \\frac{1}{3} \\right)&=\\left(\\frac12\\right)^2-\\left(\\frac13\\right)^2\\\\\n  &=\\frac14-\\frac19\\\\\n  &=\\frac{9}{36}-\\frac{4}{36}\\\\\n  &=\\frac{5}{36}\n  \\end{align*}\n  -/\nsimp_all only [one_div]\nnorm_num",
    },
]

examples_map = {
    "minif2f_deepseek_fewshot": minif2f_deepseek_fewshot,
}
