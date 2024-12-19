# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


mmlu_pro_few_shot_llama_math = [
    {
        "question": "The symmetric group $S_n$ has $factorial{n}$ elements, hence it is not true that $S_{10}$ has 10 elements.\nFind the characteristic of the ring 2Z.",
        "options": "A. 0\nB. 30\nC. 3\nD. 10\nE. 12\nF. 50\nG. 2\nH. 100\nI. 20\nJ. 5",
        "solution": "A characteristic of a ring is R is $n$ if the statement $ka = 0$ for all $a\\in 2Z$ implies that $k$ is a multiple of $n$. Assume that $ka = 0$ for all $a\\in 2Z$ for some $k$. In particular $2k = 0$. Hence $k=0$ and $n=0$. The best answer is A.",
        "topic": "math",
    },
    {
        "question": "Let V be the set of all real polynomials p(x). Let transformations T, S be defined on V by T:p(x) -> xp(x) and S:p(x) -> p'(x) = d/dx p(x), and interpret (ST)(p(x)) as S(T(p(x))). Which of the following is true?",
        "options": "A. ST + TS is the identity map of V onto itself.\nB. TS = 0\nC. ST = 1\nD. ST - TS = 0\nE. ST = T\nF. ST = 0\nG. ST = TS\nH. ST - TS is the identity map of V onto itself.\nI. TS = T\nJ. ST = S",
        "solution": "For a given polynomial $p$ we have\n\\[ST(p) = (xp(x))’ = p(x) + xp’(x)\\]\nand\n\\[TS(p) = xp’(x).\\]\nHence \\[ST(p) - TS(p) = p(x) + xp’(x) - xp’(x).\\] The best answer is H.",
        "topic": "math",
    },
    {
        "question": "Let A be the set of all ordered pairs of integers (m, n) such that 7m + 12n = 22. What is the greatest negative number in the set B = {m + n : (m, n) \\in A}?",
        "options": "A. -5\nB. 0\nC. -3\nD. -7\nE. -4\nF. -6\nG. -1\nH. -2\nI. -9\nJ. N/A",
        "solution": "We have 12n = 22 - 7m and one of the solutions is $m = -2$, $n = 3$. Then $m + n = 1$, hence we need to look for smaller $m$ in order to make $m + n$ negative. The next solution is $m = -14$ and $n = 10$. For smaller $m$ we have $m + n$ smaller than $-4$. The best answer is E.",
        "topic": "math",
    },
    {
        "question": "A tank initially contains a salt solution of 3 grams of salt dissolved in 100 liters of water. A salt solution containing 0.02 grams of salt per liter of water is sprayed into the tank at a rate of 4 liters per minute. The sprayed solution is continually mixed with the salt solution in the tank, and the mixture flows out of the tank at a rate of 4 liters per minute. If the mixing is instantaneous, how many grams of salt are in the tank after 100 minutes have elapsed?",
        "options": "A. 3 + e^-2\nB. 2 - e^-4\nC. 2 - e^-2\nD. 3 + e^-4\nE. 2 + e^-3\nF. 2 - e^-3\nG. 3 - e^-2\nH. 2 + e^-2\nI. 2 + e^-4\nJ. 2",
        "solution": "For all $t \\in \\mathbb{R}$, let $s(t)$ denote the number grams of salt in the tank at the $t$ minute mark. Then $s(0) = 3$.\nWe use $s$ and $s(t)$ interchangeably. We also use $s^{\\prime}$ and $s^{\\prime}(t)$ interchangeably. The solution sprayed into the tank adds $(0.02) 4=2 / 25$ grams of salt per minute. There are always 100 liters of liquid in the tank, containing $s$ grams of salt. So the density of salt in the tank is $s / 100$ grams per liter. The flow of water out of the tank therefore subtracts $4(s / 100)=s / 25$ grams of salt per minute. Then, for all $t \\in \\mathbb{R}$, we have $s^{\\prime}(t)=(2 / 25)-(s / 25)=(2-s) / 25$, and so $[s(t)=2] \\Rightarrow\\left[s^{\\prime}(t)=0ight]$. For all $t \\in \\mathbb{R}$,\n$$\n\x0crac{d}{d t}[\\ln (s-2)]=\x0crac{s^{\\prime}}{s-2}=\x0crac{-1}{25}=\x0crac{d}{d t}\\left[-\x0crac{t}{25}ight] .\n$$\nChoose $C \\in \\mathbb{R}$ such that, for all $t \\in \\mathbb{R}, \\ln ((s(t)-2))=-[t / 25]+C$. Let $K:=e^{C}$. Then, for all $t \\in \\mathbb{R}$, we have $(s(t))-2=K e^{-t / 25}$, and so $s(t)=2+K e^{-t / 25}$. Then $3=s(0)=2+K e^{0}=2+K$, so $K=1$. Then $s(100)=2+K e^{-100 / 25}=2+1 \\cdot e^{-4}=2+e^{-4}$. The best answer is I.",
        "topic": "math",
    },
    {
        "question": "A total of 30 players will play basketball at a park. There will be exactly 5 players on each team. Which statement correctly explains how to find the number of teams needed?",
        "options": "A. Multiply 5 by 5 to find 25 teams.\nB. Divide 30 by 5 to find 6 teams.\nC. Add 5 to 30 to find 35 teams.\nD. Subtract 30 from 5 to find -25 teams.\nE. Divide 5 by 30 to find 0.1667 teams.\nF. Add 5 to 30 then divide by 2 to find 17.5 teams.\nG. N/A\nH. N/A\nI. N/A\nJ. N/A",
        "solution": "We want to find the number of teams. We know that there are 5 players/team, and 30 players. Thus to get the number of teams we divide players by players/team, so 30 players / 5 players/team = 6 teams. The best answer is B.",
        "topic": "math",
    },
]

mmlu_pro_few_shot_llama_health = [
    {
        "question": "Which of the following is the body cavity that contains the pituitary gland?",
        "options": "A. Ventral\nB. Dorsal\nC. Buccal\nD. Thoracic\nE. Pericardial\nF. Abdominal\nG. Spinal\nH. Pelvic\nI. Pleural\nJ. Cranial",
        "solution": "We refer to Wikipedia articles on anatomy for help. Let’s solve this problem step by step. The pituitary gland is the major endocrine gland attached to the base of the brain, and it is contained in the Cranial cavity. The best answer is J.",
        "topic": "health",
    },
    {
        "question": "What is the embryological origin of the hyoid bone?",
        "options": "A. The third and fourth pharyngeal arches\nB. The fourth pharyngeal arch\nC. The third pharyngeal arch\nD. The second pharyngeal arch\nE. The second, third and fourth pharyngeal arches\nF. The first pharyngeal arch\nG. The second and third pharyngeal arches\nH. The first and third pharyngeal arches\nI. The first, second and third pharyngeal arches\nJ. The first and second pharyngeal arches",
        "solution": "We refer to Wikipedia articles on anatomy for help. Let’s solve this problem step by step. The hyoid bone, which is also known as the hyooid, is a a small U-shaped bone located in the anterior neck. In its resting position, it lies between the ase of the mandible and the third cervical vertebrae. We know that the second and the third pharyngeal arches give rise to the horns of the hyoid bone; therefore, the embryological origin of the hyoid bone are the second and the third pharyngeal arches—this information is covered in option (G). Therefore, we conclude that (G) must be the correct answer. The best answer is G.",
        "topic": "health",
    },
    {
        "question": "What is the difference between a male and a female catheter?",
        "options": "A. Female catheters are used more frequently than male catheters.\nB. Male catheters are bigger than female catheters.\nC. Male catheters are more flexible than female catheters.\nD. Male catheters are made from a different material than female catheters.\nE. Female catheters are longer than male catheters.\nF. Male catheters are longer than female catheters.\nG. Female catheters are bigger than male catheters.\nH. Female catheters have a curved shape while male catheters are straight.\nI. Male and female catheters are different colours.\nJ. Male catheters have a smaller diameter than female catheters.",
        "solution": "We refer to Wikipedia articles on clinical knowledge for help. The difference between a male and female catheter is that male catheters tend to be longer than female catheters. The best answer is F.",
        "topic": "health",
    },
    {
        "question": "How many attempts should you make to cannulate a patient before passing the job on to a senior colleague, according to the medical knowledge of 2020?",
        "options": "A. 1\nB. Unlimited attempts\nC. 5\nD. 0, always pass on to a senior colleague\nE. 7\nF. 2\nG. 4\nH. 6\nI. 3\nJ. 8",
        "solution": "We refer to Wikipedia articles on clinical knowledge for help. According to the medical protocol as of 2020, you should make two attempts to cannulate a patient before passing the job on to a more-senior practitioner. The best answer is F.",
        "topic": "health",
    },
    {
        "question": "Why are parvoviruses a highly impactful parasite?",
        "options": "A. They are able to alter the host's DNA\nB. Because they have no nucleic acid\nC. They can survive in extreme temperatures\nD. Only replicate in dividing cells\nE. They can infect multiple species\nF. They don't require a host to survive\nG. Can integrate into host chromosomes\nH. N/A\nI. N/A\nJ. N/A",
        "solution": "We refer to Wikipedia articles on virology for help. Paroviruses are highly impactful because they do not have nucleic acid. The best answer is B.",
        "topic": "health",
    },
]

mmlu_pro_few_shot_llama_physics = [
    {
        "question": "Say the pupil of your eye has a diameter of 5 mm and you have a telescope with an aperture of 50 cm. How much more light can the telescope gather than your eye?",
        "options": "A. 1000 times more\nB. 50 times more\nC. 5000 times more\nD. 500 times more\nE. 10000 times more\nF. 20000 times more\nG. 2000 times more\nH. 100 times more\nI. 10 times more\nJ. N/A",
        "solution": "The amount of light is proportional to the aperture area $A = \\pi D^2/4$ for a lens with diameter $D$, so the relative amounts of light between the eye with diameter 5mm and the telescope with diameter 50mm is $(50 cm)^2/(5mm)^2 = 10000$. The best answer is E.",
        "topic": "physics",
    },
    {
        "question": "Where do most short-period comets come from and how do we know?",
        "options": "A. The Kuiper belt; short period comets tend to be in the plane of the solar system just like the Kuiper belt.\nB. The asteroid belt; short period comets tend to come from random directions indicating a spherical distribution of comets called the asteroid belt.\nC. The asteroid belt; short period comets tend to be in the plane of the solar system just like the asteroid belt.\nD. The Oort cloud; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the Oort cloud.\nE. The Oort Cloud; short period comets tend to come from random directions indicating a spherical distribution of comets called the Oort Cloud.\nF. The Oort cloud; short period comets tend to be in the plane of the solar system just like the Oort cloud.\nG. The asteroid belt; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the asteroid belt.\nH. N/A\nI. N/A\nJ. N/A",
        "solution": "Most short-period comets come from the Kuiper belt, and we know because short period coments tend to be in the plane of the solar system, just like the Kuiper belt is. The best answer is A.",
        "topic": "physics",
    },
    {
        "question": "A refracting telescope consists of two converging lenses separated by 100 cm. The eye-piece lens has a focal length of 20 cm. The angular magnification of the telescope is",
        "options": "A. 10\nB. 40\nC. 6\nD. 25\nE. 15\nF. 50\nG. 30\nH. 4\nI. 5\nJ. 20",
        "solution": "In a refracting telescope, if both lenses are converging, the focus of both lenses must be between the two lenses, and thus the focal lengths of the two lenses must add up to their separation. Since the focal length of one lens is 20 cm, the focal length of the other must be 80 cm. The magnification is the ratio of these two focal lengths, or 4. The best answer is H.",
        "topic": "physics",
    },
    {
        "question": "Colors in a soap bubble result from light",
        "options": "A. dispersion\nB. deflection\nC. refraction\nD. reflection\nE. interference\nF. converted to a different frequency\nG. polarization\nH. absorption\nI. diffraction\nJ. transmission",
        "solution": "In a soap bubble film, the light bounces between the two soap-air interfaces many times, interfering with itself constructively or destructively depending on the width of the film. This results in different colors being visible. The best answer is E.",
        "topic": "physics",
    },
    {
        "question": "A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?",
        "options": "A. 240 W\nB. 120 W\nC. 10 W\nD. 480 W\nE. 360 W\nF. 200 W\nG. 30 W\nH. 150 W\nI. 60 W\nJ. 300 W",
        "solution": "Rate of energy usage is known as power; in an dissipative electrical circuit, power is given by voltage times current. So in our case, the power is 120 V times 2 amps, or 240 W. The best answer is A.",
        "topic": "physics",
    },
]

mmlu_pro_few_shot_llama_business = [
    {
        "question": "In contrast to _______, _______ aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of ___________, which allow campaigns to facilitate the company in achieving _________ .",
        "options": "A. Boycotts, Buyalls, Blockchain technology, Increased Sales\nB. Buycotts, Boycotts, Digital technology, Decreased Sales\nC. Boycotts, Buycotts, Digital technology, Decreased Sales\nD. Buycotts, Boycotts, Blockchain technology, Charitable donations\nE. Boycotts, Buyalls, Blockchain technology, Charitable donations\nF. Boycotts, Buycotts, Digital technology, Increased Sales\nG. Buycotts, Boycotts, Digital technology, Increased Sales\nH. Boycotts, Buycotts, Physical technology, Increased Sales\nI. Buycotts, Buyalls, Blockchain technology, Charitable donations\nJ. Boycotts, Buycotts, Blockchain technology, Decreased Sales",
        "solution": "We refer to Wikipedia articles on business ethics for help. The sentence that best uses the possible options above is __n contrast to *boycotts*, *buycotts* aim to reward favourable behavior by companies. The success of such campaigns have been heightened through the use of *digital technology*, which allow campaigns to facilitate the company in achieving *increased sales*._ The best answer is F.",
        "topic": "business",
    },
    {
        "question": "_______ is the direct attempt to formally or informally manage ethical issues or problems, through specific policies, practices and programmes.",
        "options": "A. Operational management\nB. Corporate governance\nC. Environmental management\nD. Business ethics management\nE. Sustainability\nF. Stakeholder management\nG. Social marketing\nH. Human resource management\nI. N/A\nJ. N/A",
        "solution": "We refer to Wikipedia articles on business ethics for help. The direct attempt manage ethical issues through specific policies, practices, and programs is business ethics management. The best answer is D.",
        "topic": "business",
    },
    {
        "question": "How can organisational structures that are characterised by democratic and inclusive styles of management be described?",
        "options": "A. Flat\nB. Bureaucratic\nC. Autocratic\nD. Hierarchical\nE. Functional\nF. Decentralized\nG. Matrix\nH. Network\nI. Divisional\nJ. Centralized",
        "solution": "We refer to Wikipedia articles on management for help. Flat organizational structures are characterized by democratic and inclusive styles of management, and have few (if any) levels of management between the workers and managers.  The best answer is A.",
        "topic": "business",
    },
    {
        "question": "Although the content and quality can be as controlled as direct mail, response rates of this medium are lower because of the lack of a personal address mechanism. This media format is known as:",
        "options": "A. Online banners.\nB. Television advertising.\nC. Email marketing.\nD. Care lines.\nE. Direct mail.\nF. Inserts.\nG. Door to door.\nH. Radio advertising.\nI. Billboards.\nJ. Social media advertising.",
        "solution": "We refer to Wikipedia articles on marketing for help. Door to door marketing delivers non-addressed items within all buildings within a geographic area. While it can control the content and quality as well as direct mail marketing, its response rate is lower because of the lack of a personal address mechanism. The best answer is G.",
        "topic": "business",
    },
    {
        "question": "In an organization, the group of people tasked with buying decisions is referred to as the _______________.",
        "options": "A. Procurement centre.\nB. Chief executive unit.\nC. Resources allocation group.\nD. Marketing department.\nE. Purchasing department.\nF. Supply chain management team.\nG. Outsourcing unit.\nH. Decision-making unit.\nI. Operations unit.\nJ. Financial management team.",
        "solution": "We refer to Wikipedia articles on marketing for help. In an organization, the group of the people tasked with buying decision is referred to as the decision-making unit. The best answer is H.",
        "topic": "business",
    },
]

mmlu_pro_few_shot_llama_biology = [
    {
        "question": "Which of the following represents an accurate statement concerning arthropods?",
        "options": "A. They possess an exoskeleton composed primarily of peptidoglycan.\nB. They possess an open circulatory system with a dorsal heart.\nC. They are members of a biologically unsuccessful phylum incapable of exploiting diverse habitats and nutrition sources.\nD. They lack paired, jointed appendages.\nE. N/A\nF. N/A\nG. N/A\nH. N/A\nI. N/A\nJ. N/A",
        "solution": "Peptidoglycan is known to comprise the plasma membrane of most bacteria, rather than the exoskeleton of arthropods, which is made of chitin, which rules out (A). The answer (C) is false because arthropods are a highly successful phylum. Likewise, arthropods have paired, jointed appendages, which rules out (D). The only remaining option is (B), as arthropods have an open circulatory system with a dorsal tubular heart. The best answer is B.",
        "topic": "biology",
    },
    {
        "question": "In a given population, 1 out of every 400 people has a cancer caused by a completely recessive allele, b. Assuming the population is in Hardy-Weinberg equilibrium, which of the following is the expected proportion of individuals who carry the b allele but are not expected to develop the cancer?",
        "options": "A. 19/400\nB. 1/400\nC. 40/400\nD. 38/400\nE. 2/400\nF. 1/200\nG. 20/400\nH. 50/400\nI. N/A\nJ. N/A",
        "solution": "According to the Hardy Weinberg Law, $p^2 + 2 p q + q^2 = 1$, and $p + q = 1$ where $p$ is the frequency of the dominant allele, $q$ is the frequency of the recessive allele, and $p^2$, $q^2$, and $2pq$ are the frequencies of dominant homozygous, recessive homozygous, and heterozygous individuals, respectively. \u200bThe frequency of the recessive allele (q) is $\\sqrt{\x0crac{1}{400}} = 0.05$. We have $p = 1 - q = 0.95$. The frequency of heterozygous individuals is $2pq = 2 \\cdot 0.05 \\cdot 0.95 = 0.095$. The number of heterozygous individuals is equal to the frequency of heterozygous individuals times the size of the population, or $0.095 * 400 = 38$. So we end up with 38/400. The best answer is D.",
        "topic": "biology",
    },
    {
        "question": "A mutation in a bacterial enzyme changed a previously polar amino acid into a nonpolar amino acid. This amino acid was located at a site distant from the enzyme’s active site. How might this mutation alter the enzyme’s substrate specificity?",
        "options": "A. By changing the enzyme’s pH optimum\nB. By changing the enzyme's molecular weight\nC. An amino acid change away from the active site increases the enzyme's substrate specificity.\nD. By changing the shape of the protein\nE. By changing the enzyme's temperature optimum\nF. By altering the enzyme's ability to be denatured\nG. By changing the enzyme’s location in the cell\nH. By changing the enzyme's color\nI. An amino acid change away from the active site cannot alter the enzyme’s substrate specificity.\nJ. By altering the enzyme's rate of reaction",
        "solution": "A change in an amino acid leads to a change in the primary structure of the protein. A change in the primary structure may lead to a change in the secondary and the tertiary structure of the protein. A change in the tertiary structure means a change in the shape of the protein, so (C) has to be correct. Since the change does not affect the active site of the enzyme, we do not expect the activity of the enzyme to be affected. The best answer is D.",
        "topic": "biology",
    },
    {
        "question": "Which of the following is not a way to form recombinant DNA?",
        "options": "A. Translation\nB. Conjugation\nC. Specialized transduction\nD. Transformation\nE. N/A\nF. N/A\nG. N/A\nH. N/A\nI. N/A\nJ. N/A",
        "solution": "The introduction of foreign DNA or RNA into bacteria or eukaryotic cells is a common technique in molecular biology and scientific research. There are multiple ways foreign DNA can be introduced into cells including transformation, transduction, conjugation, and transfection. In contrast, (A) is not a way to form DNA: during translation the ribosomes synthesize proteins from RNA. The best answer is A.",
        "topic": "biology",
    },
    {
        "question": "Which of the following is not known to be involved in the control of cell division?",
        "options": "A. Microtubules\nB. Checkpoints\nC. DNA polymerase\nD. Centrosomes\nE. Cyclins\nF. Mitochondria\nG. Protein kinases\nH. Fibroblast cells\nI. N/A\nJ. N/A",
        "solution": "Normal cells move through the cell cycle in a regulated way. At the checkpoint stage, they use information about their own internal state and cues from the environment around them to decide whether to proceed with cell division. Cues like these act by changing the activity of core cell cycle regulators inside the cell. The most common regulators are cyclins and cyclin-dependent kinases. Fibroblast cells do not play any role in cell division. The best answer is H.",
        "topic": "biology",
    },
]

mmlu_pro_few_shot_llama_chemistry = [
    {
        "question": "Predict the number of lines in the EPR spectrum of a solution of 13C-labelled methyl radical (13CH3•), assuming the lines do not overlap.",
        "options": "A. 10\nB. 8\nC. 4\nD. 20\nE. 12\nF. 3\nG. 16\nH. 5\nI. 24\nJ. 6",
        "solution": "The electron paramagnetic resonance spectrum will be split by two forms of interactions. The first is the hyperfine interaction with the 13C (nuclear spin $I = \n\x0crac{1}{2}$) which will split the spectrum into 2 lines. This will be further split into 4 lines by the interaction with three equivalent 1H nuclei. The total number of lines is therefore $2 \\cdot 4 = 8$. The best answer is B.",
        "topic": "chemistry",
    },
    {
        "question": "Which of the following lists the hydrides of group-14 elements in order of thermal stability, from lowest to highest?",
        "options": "A. GeH4 < SnH4 < PbH4 < SiH4 < CH4\nB. SiH4 < GeH4 < SnH4 < PbH4 < CH4\nC. PbH4 < CH4 < SnH4 < GeH4 < SiH4\nD. PbH4 < SnH4 < CH4 < GeH4 < SiH4\nE. SnH4 < GeH4 < SiH4 < PbH4 < CH4\nF. CH4 < GeH4 < SnH4 < PbH4 < SiH4\nG. SiH4 < SnH4 < PbH4 < GeH4 < CH4\nH. CH4 < SiH4 < GeH4 < SnH4 < PbH4\nI. CH4 < PbH4 < GeH4 < SnH4 < SiH4\nJ. PbH4 < SnH4 < GeH4 < SiH4 < CH4",
        "solution": "The thermal stability of group-14 hydrides decreases as we move from the top of group 14 to the bottom. The order of elements in the group from top to bottom is C, Si, Ge, Sn, Pb. Therefore in order of increasing thermal stability we have PbH4, SnH4, GeH4, SiH4, and CH4, or answer (J). The best answer is J.",
        "topic": "chemistry",
    },
    {
        "question": "Which of the following is considered an acid anhydride?",
        "options": "A. H2SO3\nB. NaCl\nC. CH4\nD. HNO3\nE. SO2\nF. Al(NO3)3\nG. CaCO3\nH. H2O\nI. HCl\nJ. N/A",
        "solution": "An acid anhydride is a compound that is derived by removing water from an acid. The chemical formula for water is H2O, which means that we need to determine which of these options, when combined with H2O, forms an acid. SO2, or Sulfur dioxide, when combined with H2O, makes H2SO4, or sulfuric acid. The best answer is E.",
        "topic": "chemistry",
    },
    {
        "question": "A new compound is synthesized and found to be a monoprotic acid with a molar mass of 248 g/mol. When 0.0050 mol of this acid are dissolved in 0.500 L of water, the pH is measured as 3.89. What is the pKa of this acid?",
        "options": "A. 5.78\nB. 4.78\nC. 4.56\nD. 6.89\nE. 7.78\nF. 3.89\nG. 1.23\nH. 2.89\nI. 2.33\nJ. 5.33",
        "solution": "Recall that $[A] = [H^{+}]$. Here, this is equal to $$10^{-3.89}$. Then we have $K_{a} = $\n\x0crac{[H^{+}][A^{-}]}{[HA]} = \n\x0crac{10^{-3.89} \\cdot 10^{-3.89}}{10^{-2}}. The resulting exponent is $-3.89 + (-3.89) - (-2) = 5.78$, therefore $K_a = 10^{-5.78}$. The $pK_a$ is the negative log of $K_a$, which is equal to $5.78$. The best answer is A.",
        "topic": "chemistry",
    },
    {
        "question": "A solution contains 2.00 mole of acetic acid, CH3COOH, and 1.00 mole of calcium acetate, Ca(CH3COO)2. The solution is able to resist the addition of a small amount of strong acid or strong base with only minor changes in the pH of the solution. Larger quantities of strong acid or strong base can cause a significant change in pH. How many moles of nitric acid, HNO3, may be added before the pH begins to change significantly?",
        "options": "A. 0.250 mole\nB. 0.500 mole\nC. 3.00 mole\nD. 1.00 mole\nE. 3.50 mole\nF. 1.50 mole\nG. 2.50 mole\nH. 4.00 mole\nI. 0.750 mole\nJ. 2.00 mole",
        "solution": "We would like to compute the buffer capacity of this solution. First we write the equation for the ionization of the weak acid, in this case of acetic acid. $CH_{3}COOH (aq) + H_{2}O \nightarrow H_{3}O^{+} + CH3COO^{-}$. The conjugate base is therefore the acetate ion. The added strong acid, Nitric acid, will react with the conjugate base. Therefore the maximum amount of acid that can be added will be equal to the amount of acetate ion, or 2 moles. The best answer is J.",
        "topic": "chemistry",
    },
]

mmlu_pro_few_shot_llama_computer_science = [
    {
        "question": "A certain pipelined RISC machine has 8 general-purpose registers R0, R1, . . . , R7 and supports the following operations:\nADD Rs1, Rs2, Rd (Add Rs1 to Rs2 and put the sum in Rd)\nMUL Rs1, Rs2, Rd (Multiply Rs1 by Rs2 and put the product in Rd)\nAn operation normally takes one cycle; however, an operation takes two cycles if it produces a result required by the immediately following operation in an operation sequence.\nConsider the expression AB + ABC + BC, where variables A, B, C are located in registers R0, R1, R2. If the contents of these three registers must not be modified, what is the minimum number of clock cycles required for an operation sequence that computes the value of AB + ABC + BC?",
        "options": "A. 10\nB. 6\nC. 9\nD. 5\nE. 12\nF. 4\nG. 7\nH. 8\nI. N/A\nJ. N/A",
        "solution": "First, we are given that A is in R0, B is in R1, and C is in R2.\nNext, we can see that we must compute three multiplies (AB, BC, and ABC) and two adds (AB + ABC, (AB + ABC) + BC) to compute our final answer, resulting in a minimum of five clock cycles.\nNext, we can see that there is no way to avoid at least one pipeline stall when computing our final answer, because to compute our final sum we must wait at least one cycle for the results from the previous stage to be ready. Thus, our minimum number of cycles must be 6.\nWe can verify that we can create a solution that requires only six cycles as follows:\ncompute AB: MUL R0, R1, R3\ncompute BC: MUL R1, R2, R4\ncompute ABC: MUL R3, R4, R5\ncompute AB + BC: ADD R3, R4, R6\nSTALL\ncompute AB + ABC + BC: ADD R5, R6, R7\nSo there are 6 cycles. The best answer is B.",
        "topic": "computer_science",
    },
    {
        "question": "A compiler generates code for the following assignment statement.\nG := (A + B) * C - (D + E) * F\nThe target machine has a single accumulator and a single-address instruction set consisting of instructions load, store, add, subtract, and multiply. For the arithmetic operations, the left operand is taken from the accumulator and the result appears in the accumulator. The smallest possible number of instructions in the resulting code is",
        "options": "A. 7\nB. 5\nC. 12\nD. 4\nE. 11\nF. 6\nG. 8\nH. 10\nI. 9\nJ. 3",
        "solution": "We can compute the final answer with the following sequence of operations:\n1. LOAD D  (accumulator = D)\n2. ADD E  (accumulator = D+E)\n3. MUL F  (accumulator = (D+E)*F)\n4. STORE X (X = (D+E)*F)\n5. LOAD A  (accumulator = A)\n6. ADD B  (accumulator = A+B)\n7. MUL C  (accumulator = (A+B)*C)\n8. SUB X  (accumulator = (A+B)*C - (D+E)*F)\n9. STORE G (G = (A+B)*C - (D+E)*F)\nThis sequence takes 9 instructions. The best answer is I.",
        "topic": "computer_science",
    },
    {
        "question": "SHA-1 has a message digest of",
        "options": "A. 628 bits\nB. 780 bits\nC. 160 bits\nD. 256 bits\nE. 2048 bits\nF. 820 bits\nG. 128 bits\nH. 512 bits\nI. 1024 bits\nJ. 64 bits",
        "solution": "Since SHA-1 is a hash function which takes an input and produces a 160-bit (20-byte) hash value, its message digest is 160 bits. The best answer is C.",
        "topic": "computer_science",
    },
    {
        "question": "_____________ can modify data on your system – so that your system doesn’t run correctly or you can no longer access specific data, or it may even ask for ransom in order to give your access.",
        "options": "A. Email Trojans\nB. Backdoor Trojans\nC. Rootkit Trojans\nD. Banking Trojans\nE. Trojan-Downloader\nF. Spyware Trojans\nG. Network Trojans\nH. Worm Trojans\nI. Ransom Trojan\nJ. IM – Trojans",
        "solution": "The system is asking for trojans, which are for ransom, which means ransom trojan. The best answer is I.",
        "topic": "computer_science",
    },
    {
        "question": 'In the program below, the initial value of X is 5 and the initial value of Y is 10.\nIF (X < 0){\n DISPLAY ("Foxtrot")\n} ELSE {\n IF (X > Y){\n  DISPLAY ("Hotel")\n } ELSE {\n  IF (Y > 0){\n   DISPLAY ("November")\n  } ELSE {\n   DISPLAY ("Yankee")\n  }\n }\n}\nWhat is displayed as a result of running the program?',
        "options": "A. November\nB. Foxtrot\nC. Zulu\nD. Alpha\nE. Charlie\nF. Bravo\nG. Yankee\nH. Echo\nI. Hotel\nJ. Delta",
        "solution": 'Because X has the value 5, the first conditional IF (X < 0) is false, so we move to the first ELSE clause. Because X is 5 and Y is 10, the second conditional IF (X > Y) is false, so we move to the following ELSE clause. Since Y is 10, the conditional IF (Y > 0) is true, so the command DISPLAY ("November") is executed. The best answer is A.',
        "topic": "computer_science",
    },
]

mmlu_pro_few_shot_llama_economics = [
    {
        "question": "For a stationary autoregressive process, shocks will",
        "options": "A. Never occur\nB. Persist indefinitely\nC. Cause a permanent change in the trend\nD. Grow exponentially\nE. Only affect the next period\nF. Eventually die away\nG. Cause the process to become non-stationary\nH. Immediately disappear\nI. Lead to a cyclical pattern\nJ. Be balanced out by subsequent shocks",
        "solution": "We refer to Wikipedia articles on econometrics for help. This is a formal logic problem about stationally process. For a stationary autoregressive process, shocks will eventually die away. The best answer is F.",
        "topic": "economics",
    },
    {
        "question": "Consider the following AR(1) model with the disturbances having zero mean and unit variance\nyt = 0.2 + 0.4 yt-1 + ut\nThe (unconditional) mean of y will be given by",
        "options": "A. 0.45\nB. 0.2\nC. 0.4\nD. 0.1\nE. 0.3\nF. 0.55\nG. 0.25\nH. 0.33\nI. 0.6\nJ. 0.5",
        "solution": "We refer to Wikipedia articles on econometrics for help. Let’s solve this problem step by step. If we have a an AR(1) model with the disturbances having zero mean and unit variance, then the unconditional mean of y is equal to the following:\nunconditional mean of y = (the intercept term) / (1 - autoregressive coefficient)\nWe know that the intercept term is 0.2 and the autoregressive coefficient is 0.4; thus, we have:\nunconditional mean of y = (0.2) / (1 - 0.4) = (0.2) / (0.6) = 2 / 6 = 1 / 3, which is approximately 0.33. That means that the answer should be (H) 0.33. The best answer is H.",
        "topic": "economics",
    },
    {
        "question": "Suppose that a test statistic has associated with it a p-value of 0.08. Which one of the following statements is true?\n(i) If the size of the test were exactly 8%, we would be indifferent between rejecting and not rejecting the null hypothesis\n(ii) The null would be rejected if a 10% size of test were used\n(iii) The null would not be rejected if a 1% size of test were used\n(iv) The null would be rejected if a 5% size of test were used.",
        "options": "A. (iii) and (iv) only\nB. (i) and (iv) only\nC. (i), (ii), and (iii) only\nD. (i) and (ii) only\nE. (i) and (iii) only\nF. (i), (ii), (iii), and (iv).\nG. (ii), (iii), and (iv) only\nH. (ii) and (iii) only\nI. (ii) and (iv) only\nJ. (i), (iii), and (iv) only",
        "solution": "We refer to Wikipedia articles on econometrics for help. Let’s reason about each of the options.\n(i) is a true statement.\n(ii) is a true statement.\n(iii) is a true statement.\n(iv) is not a true statement. Thus, (i), (ii), and (iii) are true. The best answer is C.",
        "topic": "economics",
    },
    {
        "question": "Which of the following policies best describes supply-side fiscal policy?",
        "options": "A. Higher taxes on household income\nB. Increased government spending\nC. Increased taxes on corporate profits\nD. Increased import tariffs\nE. Decreased interest rates\nF. Lower taxes on consumer goods\nG. Lower taxes on research and development of new technology\nH. Reduced government spending\nI. Higher taxes on capital gains\nJ. An increase in the money supply",
        "solution": "We refer to Wikipedia articles on macroeconomics for help. Supply-side fiscal policy stimulates the economy by encouraging more production of goods and services through reduction in taxes and deregulation. The best answer is G.",
        "topic": "economics",
    },
    {
        "question": "The concentration ratio for a monopoly is",
        "options": "A. 50\nB. 5\nC. 10\nD. 90\nE. 15\nF. 100\nG. 0\nH. 25\nI. 75\nJ. N/A",
        "solution": "We refer to Wikipedia articles on microeconomics for help. The concentration ratio is calculated as the sum of market share of a specific number of largest companies. Monopoly means one company or entity controls the entire market, therefore, the concentration ratio is 100 percent. The best answer is F.",
        "topic": "economics",
    },
]

mmlu_pro_few_shot_llama_engineering = [
    {
        "question": "A point pole has a strength of 4π * 10^-4 weber. The force in newtons on a point pole of 4π * 1.5 * 10^-4 weber placed at a distance of 10 cm from it will be",
        "options": "A. 25 N.\nB. 12.5 N.\nC. 15 N.\nD. 10 N.\nE. 7.5 N.\nF. 20 N.\nG. 5 N.\nH. 3.75 N.\nI. N/A\nJ. N/A",
        "solution": "The force between two point poles is given by m_1m_2/(mu_0 4 \\pi r^2), in analogy to Coulomb’s law. Plugging in the values given in the question, we calculate that the force is approximately 15 N. The best answer is C.",
        "topic": "engineering",
    },
    {
        "question": "The coil of a moving coil meter has 100 turns, is 40 mm long and 30 mm wide. The control torque is 240*10-6 N-m on full scale. If magnetic flux density is 1Wb/m2 range of meter is",
        "options": "A. 2 mA.\nB. 5 mA.\nC. 1.5 mA.\nD. 0.5 mA.\nE. 6 mA.\nF. 4 mA.\nG. 3 mA.\nH. 1 mA.\nI. 2.5 mA.\nJ. 3.5 mA.",
        "solution": "The torque on a coil in a uniform magnetic field is given by BANI, where B is the magnetic flux density, A is the area of the coil, N is the number of turns, and I is the current. So we have that I = (Torque)/(BAN), or 240e-6/(1200e-6 * 100 * 1) = 2e-3. The best answer is A.",
        "topic": "engineering",
    },
    {
        "question": "In an SR latch built from NOR gates, which condition is not allowed",
        "options": "A. S=0, R=2\nB. S=2, R=2\nC. S=1, R=1\nD. S=1, R=-1\nE. S=1, R=2\nF. S=0, R=0\nG. S=2, R=0\nH. S=1, R=0\nI. S=2, R=1\nJ. S=0, R=1",
        "solution": "An SR latch is a set-reset latch; in the case where S=1 and R=1, the circuit has no stable state; instead a race condition will be produced within the circuit, so the device will be in an undefined state. So S=1, R=1 is an illegal input. The best answer is C.",
        "topic": "engineering",
    },
    {
        "question": "Two long parallel conductors carry 100 A. If the conductors are separated by 20 mm, the force per meter of length of each conductor will be",
        "options": "A. 100 N.\nB. 0.01 N.\nC. 10 N.\nD. 0.5 N.\nE. 0.001 N.\nF. 5 N.\nG. 1 N.\nH. 50 N.\nI. 0.1 N.\nJ. 0.0001 N.",
        "solution": "The magnetic force-per-length between two current-carrying conductors is given by \\mu_0 I_1 I_2 / (2 \\pi r), where $r$ is the separation distance and I_1 and I_2 are the currents. Plugging in 100 A for I_1 and I_2, and 20 mm for r, gives 0.1 N. The best answer is I.",
        "topic": "engineering",
    },
    {
        "question": "In a 2 pole lap winding dc machine , the resistance of one conductor is 2Ω and total number of conductors is 100. Find the total resistance",
        "options": "A. 50Ω\nB. 1Ω\nC. 25Ω\nD. 200Ω\nE. 10Ω\nF. 100Ω\nG. 500Ω\nH. 150Ω\nI. 75Ω\nJ. 20Ω",
        "solution": "In lap winding, effectively two resistors are connected in parallel, so the actual resistance of each pair is 1 Ohm. Since we have 50 pairs, we get a total resistance of 50 Ohms. The best answer is A.",
        "topic": "engineering",
    },
]

mmlu_pro_few_shot_llama_philosophy = [
    {
        "question": "Which of the given formulas of PL is the best symbolization of the following sentence?\nTurtles live long lives and are happy creatures, unless they are injured.",
        "options": "A. (L • H) ≡ I\nB. (L • H) ∨ I\nC. L • (H ∨ I)\nD. L • (H ⊃ R).\nE. N/A\nF. N/A\nG. N/A\nH. N/A\nI. N/A\nJ. N/A",
        "solution": "We refer to Wikipedia articles on formal logic for help. Let’s solve this step by step. Let “L” denote “living long”, H “being happy”, and “I” “being injured”. Now, consider each choice:\n(A) means (living long AND being happy) is equivalent to (being injured). \n(B) means (living long AND being happy) OR (being injured). \n(C) means (living long) AND (being happy OR being injured). \n(D) means (living long) AND (being happy implies being R), but what R denotes is not clear.\nObviously, (B) is the best symbolization of the original sentence. The best answer is B.",
        "topic": "philosophy",
    },
    {
        "question": "Select the best translation into predicate logic.George borrows Hector's lawnmower. (g: George; h: Hector; l: Hector's lawnmower; Bxyx: x borrows y from z).",
        "options": "A. Bhgh\nB. Bggh\nC. Bhlh\nD. Bghl\nE. Bhlg\nF. Blhg\nG. Bllg\nH. Blgh\nI. Bhhg\nJ. Bglh",
        "solution": "We refer to Wikipedia articles on formal logic for help. Let’s solve this step by step. We are told that “Bxyx” means “x borrows y from z”. We can rewrite “George borrows Hector's lawnmower” as “George borrows a lawnmower from Hector”, which can then be translated into predicate logic as “Bglh”. The answer “Bglh” appears in (J); therefore, (J) must be the correct answer. The best answer is J.",
        "topic": "philosophy",
    },
    {
        "question": "The study of reality in the broadest sense, an inquiry into the elemental nature of the universe and the things in it, is known as _____.",
        "options": "A. quantum physics\nB. metaphysics\nC. ontology\nD. phenomenology\nE. cosmology\nF. teleology\nG. epistemology\nH. axiology\nI. N/A\nJ. N/A",
        "solution": "We refer to Wikipedia articles on philosophy for help. Among the options, only metaphysics studies the nature of reality and existence. The best answer is B.",
        "topic": "philosophy",
    },
    {
        "question": "According to Moore’s “ideal utilitarianism,” the right action is the one that brings about the greatest amount of:",
        "options": "A. wealth.\nB. virtue.\nC. fairness.\nD. pleasure.\nE. peace.\nF. justice.\nG. happiness.\nH. power.\nI. good.\nJ. knowledge.",
        "solution": "We refer to Wikipedia articles on philosophy for help. Moore's \"ideal utilitarianism\" states that one's actions should maximize intrinsic goods. The best answer is I.",
        "topic": "philosophy",
    },
    {
        "question": "What is the sign of the covenant for Jewish males?",
        "options": "A. Fasting on Yom Kippur\nB. Lighting Shabbat candles\nC. The rainbow\nD. Circumcision\nE. The Torah\nF. Bar mitzvah\nG. Keeping kosher\nH. Wearing a kippah\nI. A son\nJ. The Star of David",
        "solution": "We refer to Wikipedia articles on world religions for help. In Judaism, the most distinctive sign of the covenant is circumcision (brit milah). The best answer is D.",
        "topic": "philosophy",
    },
]

mmlu_pro_few_shot_llama_other = [
    {
        "question": "As of 2017, how many of the world’s 1-year-old children today have been vaccinated against some disease? *",
        "options": "A. 30%\nB. 60%\nC. 10%\nD. 90%\nE. 80%\nF. 40%\nG. 100%\nH. 50%\nI. N/A\nJ. N/A",
        "solution": "We refer to Wikipedia articles on global facts for help. According to data published by the World Health Organization, the nummber of 1-year-old children vaccinated in 2017 exceeds 80%. The best answer is E.",
        "topic": "other",
    },
    {
        "question": "Which one of the following items is an example of nonmaterial culture?",
        "options": "A. A dove feather\nB. Dove symbol\nC. Dove body lotion\nD. Dove deodorant\nE. Dove soap\nF. Dove candy bar\nG. Dove conditioner\nH. A dove (bird).\nI. Dove chocolate\nJ. Dove shampoo",
        "solution": "We refer to Wikipedia articles on geography for help. Nonmaterial culture consists of cultural ideas, beliefs or symbols that are not physical objects. The best answer is B.",
        "topic": "other",
    },
    {
        "question": "Which of the following cases established the precedent that a defendant must be informed of the right to remain silent, the right to a lawyer, and protection from self-incrimination?",
        "options": "A. Brown v. Board of Education\nB. Miranda v. Arizona\nC. Roe v. Wade\nD. Betts v. Brady\nE. Plessy v. Ferguson\nF. Dred Scott v. Sandford\nG. Weeks v. United States\nH. Gideon v. Wainwright\nI. Marbury v. Madison\nJ. Mapp v. Ohio",
        "solution": "We refer to Wikipedia articles on government and politics for help. In the landmark Miranda v. Arizona in 1966, the US Supreme Court, based on the Fifth and Sixth Amendment of the US Constitution, guaranteed a defendant's right to an attorney and protection from self-incrimination. The best answer is B.",
        "topic": "other",
    },
    {
        "question": "A woman who knows she has active herpes and untreated syphilis but continues to have sex without informing her partners of her condition has, in psychoanalytic terms:",
        "options": "A. a weak conscious mind\nB. a strong conscious mind\nC. a weak id\nD. a weak ego\nE. a weak unconscious mind\nF. a strong id\nG. a strong ego\nH. a strong superego\nI. a strong preconscious mind\nJ. a weak superego",
        "solution": "We refer to Wikipedia articles on human sexuality for help. A person with weak superego tends to be delinquent, criminal or have antisocial personality. The action of the woman who knows she has active venereal disease but still have sex with her partners indicate she may has antisocial personality. The best answer is J.",
        "topic": "other",
    },
    {
        "question": "What place is named in the title of the 1979 live album by rock legends Cheap Trick?",
        "options": "A. Brooklyn\nB. Beijing\nC. Budapest\nD. Boston\nE. Bhutan\nF. Barcelona\nG. Britain\nH. Brisbane\nI. Bruges\nJ. Budokan",
        "solution": "We refer to Wikipedia for help. Nippon Budokan is an indoor arena in Tokyo, Japan renowned for hosting rock music concerts including Cheap Trick in 1978. 'Cheap Trick at Budokan' became the name of their album. The best answer is J.",
        "topic": "other",
    },
]

mmlu_pro_few_shot_llama_history = [
    {
        "question": 'This question refers to the following information.\nIn Russia there was nothing going on well, and [Souvarine] was in despair over the news he had received. His old companions were all turning to the politicians; the famous Nihilists who made Europe tremble-sons of village priests, of the lower middle class, of tradesmen-could not rise above the idea of national liberation, and seemed to believe that the world would be delivered-when they had killed their despot&…\n"Foolery! They\'ll never get out of it with their foolery."\nThen, lowering his voice still more, in a few bitter words he described his old dream of fraternity. He had renounced his rank and his fortune; he had gone among workmen, only in the hope of seeing at last the foundation of a new society of labour in common. All the sous in his pockets had long gone to the urchins of the settlement; he had been as tender as a brother with the colliers, smiling at their suspicion, winning them over by his quiet workmanlike ways and his dislike of chattering. But decidedly the fusion had not taken place.\nHis voice changed, his eyes grew bright, he fixed them on étienne, directly addressing him:\n"Now, do you understand that? These hatworkers at Marseilles who have won the great lottery prize of a hundred thousand francs have gone off at once and invested it, declaring that they are going to live without doing anything! Yes, that is your idea, all of you French workmen; you want to unearth a treasure in order to devour it alone afterwards in some lazy, selfish corner. You may cry out as much as you like against the rich, you haven\'t got courage enough to give back to the poor the money that luck brings you. You will never be worthy of happiness as long as you own anything, and your hatred of the bourgeois proceeds solely from an angry desire to be bourgeois yourselves in their place."\némile Zola, French writer, Germinal, 1885\nThe passage displays the direct concern for the welfare of the working classes that was typically a part of which movement?',
        "options": "A. Communist\nB. Anarchist\nC. Feminist\nD. Fascist\nE. Imperialist\nF. Nationalist\nG. Enlightenment\nH. Existentialist\nI. N/A\nJ. N/A",
        "solution": "We refer to Wikipedia articles on european history for help. The modern Communist movement aims to establish a classless society based on communal ownership and distribution of property and means of production, thereby especially benefiting the working classes. The best answer is A.",
        "topic": "history",
    },
    {
        "question": "This question refers to the following information.\nThe following excerpt is from a pamphlet.\nYou will do me the justice to remember, that I have always strenuously supported the Right of every man to his own opinion, however different that opinion might be to mine. He who denies to another this right, makes a slave of himself to his present opinion, because he precludes himself the right of changing it.\nThe most formidable weapon against errors of every kind is Reason. I have never used any other, and I trust I never shall.\nThe circumstance that has now taken place in France of the total abolition of the whole national order of priesthood, and of everything appertaining to compulsive systems of religion, and compulsive articles of faith, has not only precipitated my intention, but rendered a work of this kind exceedingly necessary, lest in the general wreck of superstition, of false systems of government, and false theology, we lose sight of morality, of humanity, and of the theology that is true.\nI believe in one God, and no more; and I hope for happiness beyond this life.\nI believe in the equality of man; and I believe that religious duties consist in doing justice, loving mercy, and endeavoring to make our fellow-creatures happy.\nI do not believe in the creed professed by the Jewish church, by the Roman church, by the Greek church, by the Turkish church, by the Protestant church, nor by any church that I know of. My own mind is my own church.\nAll national institutions of churches, whether Jewish, Christian or Turkish, appear to me no other than human inventions, set up to terrify and enslave mankind, and monopolize power and profit.\nI do not mean by this declaration to condemn those who believe otherwise; they have the same right to their belief as I have to mine.\n—Thomas Paine, The Age of Reason, 1794–1795\nWhich of the following Enlightenment philosophes designed a system of checks and balances for government to avoid abuses of power?",
        "options": "A. Thomas Hobbes\nB. Jean Jacques Rousseau\nC. Baron Montesquieu\nD. Voltaire\nE. Denis Diderot\nF. Immanuel Kant\nG. David Hume\nH. Mary Wollstonecraft\nI. Adam Smith\nJ. John Locke",
        "solution": "We refer to Wikipedia articles on european history for help. Baron Montesquieu was a 18th centrury French philsopher who wrote extensively against the monoplization of power and advocated for a system of checks and balances in government to prevent the rise of despotism. The best answer is C.",
        "topic": "history",
    },
    {
        "question": 'This question refers to the following information.\n"The struggle against neo-colonialism is not aimed at excluding the capital of the developed world from operating in less developed countries. It is aimed at preventing the financial power of the developed countries being used in such a way as to impoverish the less developed.\nNon-alignment, as practiced by Ghana and many other countries, is based on co-operation with all States whether they be capitalist, socialist or have a mixed economy. Such a policy, therefore, involves foreign investment from capitalist countries, but it must be invested in accordance with a national plan drawn up by the government of the non-aligned State with its own interests in mind. The issue is not what return the foreign investor receives on his investments…The question is one of power. A State in the grip of neo-colonialism is not master of its own destiny."\nKwame Nkrumah, Neo-Colonialism, 1965\nWhich of the following provides the best context for Nkrumah\'s writings?',
        "options": "A. The spread of communism\nB. The Industrial Revolution\nC. The French Revolution\nD. The Cold War\nE. Decolonization\nF. World War II\nG. Autarky\nH. The rise of multinational corporations\nI. The American Civil Rights Movement\nJ. Regional Free Trade Associations",
        "solution": "We refer to Wikipedia articles on world history for help. The passage expresses a point that the successful fight against neo-colonialism were in danger and the newly independent nations like Ghana may be re-colonized via financial power of the developed countries. The best answer is E.",
        "topic": "history",
    },
    {
        "question": "What is the approximate mean cranial capacity of Homo erectus?",
        "options": "A. 1200 cc\nB. under 650 cc\nC. 1700 cc\nD. 1350 cc\nE. just under 1000 cc\nF. 1500 cc\nG. under 500 cc\nH. about 800 cc\nI. just over 1100 cc\nJ. about 900 cc",
        "solution": "We refer to Wikipedia articles on prehistory for help. The average cranium capacity of Homo erectus is less than 1000 cubic cm. The best answer is E.",
        "topic": "history",
    },
    {
        "question": "Recent research on hominid species dating from the Middle Pliocene indicates there was (as of 2020):",
        "options": "A. multiple hominid species but with limited diversity.\nB. a single species with no diversity.\nC. decreased species diversity but increased numbers of hammerstones and flakes, indicating stone tool manufacture.\nD. a single dominant species that outcompeted all others, leading to decreased diversity.\nE. increased species diversity due to a prolonged ice age followed by a severe drought.\nF. decreased species diversity due to a prolonged ice age followed by a severe drought.\nG. a great amount of species diversity, or a single species that exhibited a lot of diversity.\nH. increased species diversity but with decreased population numbers due to harsh climate conditions.\nI. increased species diversity but decreased numbers of hammerstones and flakes, indicating less stone tool manufacture.\nJ. very little species diversity during this period and very few hominids.",
        "solution": "We refer to Wikipedia articles on prehistory for help. Recent research has recognized multiple hominid species from the Middle Pliocene, meaning that there is a great amount of species diversity or diversity in a single species. The best answer is G.",
        "topic": "history",
    },
]

mmlu_pro_few_shot_llama_psychology = [
    {
        "question": "Pascale is interested in the processing strategies children use to learn new information. Pascale would best be classified as what type of psychologist?",
        "options": "A. social\nB. school\nC. sociocultural\nD. forensic\nE. behaviorist\nF. health\nG. clinical\nH. cognitive\nI. psychoanalytic\nJ. developmental",
        "solution": "We refer to Wikipedia articles on psychology for help. Sociocultural psychologist focuses on the effect of societal factors on people. Clinical psychologist focuses on people with mental issues. Cognitive psychologist focuses on how people think and learn, including the processing strategies. Behaviorist focuses more on the environment and experience effect on people. The best answer is H.",
        "topic": "psychology",
    },
    {
        "question": "According to Caplan's model of consultee-centered case consultation, the consultant is primarily interested in",
        "options": "A. identifying the causes and solutions of the client's presenting problems\nB. establishing a hierarchy of authority to enable effective decision making\nC. ensuring the consultee adheres strictly to a predetermined action plan\nD. proposing multiple alternative solutions for the consultee to choose from\nE. identifying the strengths and weaknesses of the consultee's current approach\nF. presenting a single, well-defined and unambiguous course of action for the consultant to overcome skills deficits\nG. developing a comprehensive treatment plan for the client\nH. identifying and eliminating the causes of the consultee's difficulties in handling a problem\nI. focusing on the consultant's personal growth and development\nJ. focusing on the relationship between the client and the consultee",
        "solution": "We refer to Wikipedia articles on psychology for help. Caplan defines two type of consultation. Client-centered case consultation aims to handle client's problems, while consultee-centered case consultation aims to identify the reason of client's difficulty to solve problems. The best answer is H.",
        "topic": "psychology",
    },
    {
        "question": "According to the Individuals with Disabilities Education Improvement Act, which of the following must an educational agency do before it changes the educational placement of a student with a disability?",
        "options": "A. Notify the parents in writing\nB. Obtain the child's consent\nC. Obtain a court order\nD. Conduct a new evaluation of the child's disability\nE. Discuss with the child's psychologist\nF. Give the child a trial period in the new environment\nG. Obtain parental consent\nH. Notify the local education authority\nI. Arrange a meeting with all teachers and administrators\nJ. Obtain school board approval",
        "solution": "We refer to Wikipedia articles on psychology for help. When the decision to change the educational placement of a student with a disability is made, the educational agency must notify the parents in writing on that date. The best answer is A.",
        "topic": "psychology",
    },
    {
        "question": "Ani believes that her attitudes and behavior play a central role in what happens to her. Such a belief is likely to be associated with",
        "options": "A. low self-esteem.\nB. a strong id.\nC. a high level of anxiety.\nD. a strong superego.\nE. high levels of self-consciousness.\nF. an external locus of control.\nG. an inferiority complex.\nH. a low level of self-awareness.\nI. low self-efficacy.\nJ. an internal locus of control.",
        "solution": "We refer to Wikipedia articles on psychology for help. People with an external locus of control believes fate and luck play an important role in their lives, while people with an internal locus of control believes they control their lives. The best answer is J.",
        "topic": "psychology",
    },
    {
        "question": "In terms of Hofstede’s (1980) five cultural dimensions, the United States scores at the top of the scale on:",
        "options": "A. individualism and long-term orientation.\nB. individualism and power distance.\nC. uncertainty avoidance.\nD. long-term orientation.\nE. individualism.\nF. individualism and masculinity.\nG. long-term orientation and uncertainty avoidance.\nH. power distance.\nI. power distance and masculinity.\nJ. N/A",
        "solution": "We refer to Wikipedia articles on psychology for help. US scores highest on individualism among the five cultural dimensions. The best answer is E.",
        "topic": "psychology",
    },
]

mmlu_pro_few_shot_llama_law = [
    {
        "question": "What is the judge ad hoc?",
        "options": "A. Judge ad hoc is the president of the ICJ\nB. Judge ad hoc is a temporary judge appointed for a specific period of time\nC. Judge ad hoc is the judge that each party will always nominate in every contentious case\nD. Judge ad hoc is the member of the bench of the ICJ with a casting vote\nE. Judge ad hoc is a judge who is nominated by the parties involved in a contentious case, irrespective of their nationality\nF. Judge ad hoc is a judge who decides on the admissibility of cases before the ICJ\nG. Judge ad hoc is a judge appointed by the Security Council of the United Nations\nH. Judge ad hoc is a surrogate judge, in case a judge is disqualified or passes away\nI. If a party to a contentious case before the ICJ does not have a national sitting as judge, it is entitled to nominate someone as a judge solely for that case, with the title of judge ad hoc\nJ. N/A",
        "solution": 'We refer to Wikipedia articles on international law for help. As "ad hoc" implies, a judge ad hoc is appointed only for a specific case or period, when a party to a contentious case before the International Court of Justice does not have a regular national sitting as judge. The best answer is I.',
        "topic": "law",
    },
    {
        "question": "Functions of the law include all but which of the following?",
        "options": "A. defining the limits of government power\nB. regulating the use of public spaces\nC. keeping the peace\nD. maximizing individual freedom\nE. maintaining order and stability\nF. preventing environmental degradation\nG. providing a basis for compromise\nH. promoting social justice\nI. promoting the principles of the free enterprise system\nJ. encouraging economic growth",
        "solution": "We refer to Wikipedia articles on jurisprudence for help. Laws are fundamentally about helping resolve disputes between individuals, and therefore essential for maximizing individual freedom, providing a basis for compromise, and keeping the peace. The best answer is I.",
        "topic": "law",
    },
    {
        "question": 'The ________ School of jurisprudence postulates that the law is based on what is "correct."',
        "options": "A. Legal Pragmatism\nB. Legal Formalism\nC. Comparative\nD. Analytical\nE. Sociological\nF. Historical\nG. Critical Legal Studies\nH. Realist\nI. Positivist\nJ. Natural Law",
        "solution": 'We refer to Wikipedia articles on jurisprudence for help. Natural Law School of jurisprudence focuses on the laws of nature, and states that the law should be based on ethics, morals, and what is "correct". Analytical deals with the law as it already exists, Historical postulates that the law was found and not made, and Sociological studies how the law and society impact each other. The best answer is J.',
        "topic": "law",
    },
    {
        "question": "Which word best summarizes Weber's explanation of the development of formally rational law?",
        "options": "A. Socialism.\nB. Legitimacy.\nC. Authority.\nD. Democracy.\nE. Bureaucracy.\nF. Conflict.\nG. Capitalism.\nH. Charisma.\nI. Co-operation.\nJ. Tradition.",
        "solution": "We refer to Wikipedia articles on jurisprudence for help. Weber explained the development of formal rationality in laws as how the modern society moved from tradition to rationality, where people decide actions based less on how they were culturally done and more on expected utilities. How rational individuals optimize efficiency of accomplishing tasks for higher rewards is a core principle of Capitalism. The best answer is G.",
        "topic": "law",
    },
    {
        "question": "A state has recently enacted a statute prohibiting the disposal of any nuclear wastes within the state. This law does not contravene or conflict with any federal statutes. A man operates a company in the state that is engaged in the disposal of nuclear wastes. Subsequent to the passage of the state statute, the man, not yet aware of the new law, entered into contracts with many out-of-state firms to dispose of their nuclear wastes in the state. On account of this new law, however, the man will be unable to perform these contracts. Assume that the man has standing to challenge this state law. Which of the following presents his strongest constitutional grounds to challenge the state law prohibiting the disposal of nuclear wastes within the state?",
        "options": "A. The second amendment - the right to bear arms.\nB. The due process clause of the Fourteenth Amendment.\nC. The tenth amendment - powers not delegated to the United States by the Constitution.\nD. The first amendment - freedom of speech.\nE. The privileges and immunities clause of Article IV, Section 2.\nF. The commerce clause.\nG. The sixth amendment - right to a fair trial.\nH. The eighth amendment - prohibition of cruel and unusual punishment.\nI. The equal protection clause of the Fourteenth Amendment.\nJ. N/A",
        "solution": "We refer to Wikipedia articles on law for help. The commerce clause states that Congress shall have the power to regulate commerce with foreign Nations, and among the several States, and with the Indian Tribes. The statute affects inter-state commerce which puts it into question. Hence the man's strongest argument should be the commerce clause. The best answer is F.",
        "topic": "law",
    },
]

mmlu_pro_few_shot_tigerlab_math = [
    {
        "question": "The symmetric group $S_n$ has $factorial{n}$ elements, hence it is not true that $S_{10}$ has 10 elements.\nFind the characteristic of the ring 2Z.",
        "options": "A. 0\nB. 30\nC. 3\nD. 10\nE. 12\nF. 50\nG. 2\nH. 100\nI. 20\nJ. 5",
        "solution": "A characteristic of a ring is R is $n$ if the statement $ka = 0$ for all $a\\in 2Z$ implies that $k$ is a multiple of $n$. Assume that $ka = 0$ for all $a\\in 2Z$ for some $k$. In particular $2k = 0$. Hence $k=0$ and $n=0$. The answer is (A).",
        "topic": "math",
    },
    {
        "question": "Let V be the set of all real polynomials p(x). Let transformations T, S be defined on V by T:p(x) -> xp(x) and S:p(x) -> p'(x) = d/dx p(x), and interpret (ST)(p(x)) as S(T(p(x))). Which of the following is true?",
        "options": "A. ST + TS is the identity map of V onto itself.\nB. TS = 0\nC. ST = 1\nD. ST - TS = 0\nE. ST = T\nF. ST = 0\nG. ST = TS\nH. ST - TS is the identity map of V onto itself.\nI. TS = T\nJ. ST = S",
        "solution": "For a given polynomial $p$ we have\n\\[ST(p) = (xp(x))’ = p(x) + xp’(x)\\]\nand\n\\[TS(p) = xp’(x).\\]\nHence \\[ST(p) - TS(p) = p(x) + xp’(x) - xp’(x).\\] The answer is (H).",
        "topic": "math",
    },
    {
        "question": "Let A be the set of all ordered pairs of integers (m, n) such that 7m + 12n = 22. What is the greatest negative number in the set B = {m + n : (m, n) \\in A}?",
        "options": "A. -5\nB. 0\nC. -3\nD. -7\nE. -4\nF. -6\nG. -1\nH. -2\nI. -9",
        "solution": "We have 12n = 22 - 7m and one of the solutions is $m = -2$, $n = 3$. Then $m + n = 1$, hence we need to look for smaller $m$ in order to make $m + n$ negative. The next solution is $m = -14$ and $n = 10$. For smaller $m$ we have $m + n$ smaller than $-4$. The answer is (E).",
        "topic": "math",
    },
    {
        "question": "A tank initially contains a salt solution of 3 grams of salt dissolved in 100 liters of water. A salt solution containing 0.02 grams of salt per liter of water is sprayed into the tank at a rate of 4 liters per minute. The sprayed solution is continually mixed with the salt solution in the tank, and the mixture flows out of the tank at a rate of 4 liters per minute. If the mixing is instantaneous, how many grams of salt are in the tank after 100 minutes have elapsed?",
        "options": "A. 3 + e^-2\nB. 2 - e^-4\nC. 2 - e^-2\nD. 3 + e^-4\nE. 2 + e^-3\nF. 2 - e^-3\nG. 3 - e^-2\nH. 2 + e^-2\nI. 2 + e^-4\nJ. 2",
        "solution": "For all $t \\in \\mathbb{R}$, let $s(t)$ denote the number grams of salt in the tank at the $t$ minute mark. Then $s(0) = 3$.\nWe use $s$ and $s(t)$ interchangeably. We also use $s^{\\prime}$ and $s^{\\prime}(t)$ interchangeably. The solution sprayed into the tank adds $(0.02) 4=2 / 25$ grams of salt per minute. There are always 100 liters of liquid in the tank, containing $s$ grams of salt. So the density of salt in the tank is $s / 100$ grams per liter. The flow of water out of the tank therefore subtracts $4(s / 100)=s / 25$ grams of salt per minute. Then, for all $t \\in \\mathbb{R}$, we have $s^{\\prime}(t)=(2 / 25)-(s / 25)=(2-s) / 25$, and so $[s(t)=2] \\Rightarrow\\left[s^{\\prime}(t)=0ight]$. For all $t \\in \\mathbb{R}$,\n$$\n\x0crac{d}{d t}[\\ln (s-2)]=\x0crac{s^{\\prime}}{s-2}=\x0crac{-1}{25}=\x0crac{d}{d t}\\left[-\x0crac{t}{25}ight] .\n$$\nChoose $C \\in \\mathbb{R}$ such that, for all $t \\in \\mathbb{R}, \\ln ((s(t)-2))=-[t / 25]+C$. Let $K:=e^{C}$. Then, for all $t \\in \\mathbb{R}$, we have $(s(t))-2=K e^{-t / 25}$, and so $s(t)=2+K e^{-t / 25}$. Then $3=s(0)=2+K e^{0}=2+K$, so $K=1$. Then $s(100)=2+K e^{-100 / 25}=2+1 \\cdot e^{-4}=2+e^{-4}$. The answer is (I).",
        "topic": "math",
    },
    {
        "question": "A total of 30 players will play basketball at a park. There will be exactly 5 players on each team. Which statement correctly explains how to find the number of teams needed?",
        "options": "A. Multiply 5 by 5 to find 25 teams.\nB. Divide 30 by 5 to find 6 teams.\nC. Add 5 to 30 to find 35 teams.\nD. Subtract 30 from 5 to find -25 teams.\nE. Divide 5 by 30 to find 0.1667 teams.\nF. Add 5 to 30 then divide by 2 to find 17.5 teams.",
        "solution": "We want to find the number of teams. We know that there are 5 players/team, and 30 players. Thus to get the number of teams we divide players by players/team, so 30 players / 5 players/team = 6 teams. The answer is (B).",
        "topic": "math",
    },
]


mmlu_pro_few_shot_tigerlab_health = [
    {
        "question": "Which of the following is the body cavity that contains the pituitary gland?",
        "options": "A. Ventral\nB. Dorsal\nC. Buccal\nD. Thoracic\nE. Pericardial\nF. Abdominal\nG. Spinal\nH. Pelvic\nI. Pleural\nJ. Cranial",
        "solution": "We refer to Wikipedia articles on anatomy for help. Let’s solve this problem step by step. The pituitary gland is the major endocrine gland attached to the base of the brain, and it is contained in the Cranial cavity. The answer is (J).",
        "topic": "health",
    },
    {
        "question": "What is the embryological origin of the hyoid bone?",
        "options": "A. The third and fourth pharyngeal arches\nB. The fourth pharyngeal arch\nC. The third pharyngeal arch\nD. The second pharyngeal arch\nE. The second, third and fourth pharyngeal arches\nF. The first pharyngeal arch\nG. The second and third pharyngeal arches\nH. The first and third pharyngeal arches\nI. The first, second and third pharyngeal arches\nJ. The first and second pharyngeal arches",
        "solution": "We refer to Wikipedia articles on anatomy for help. Let’s solve this problem step by step. The hyoid bone, which is also known as the hyooid, is a a small U-shaped bone located in the anterior neck. In its resting position, it lies between the ase of the mandible and the third cervical vertebrae. We know that the second and the third pharyngeal arches give rise to the horns of the hyoid bone; therefore, the embryological origin of the hyoid bone are the second and the third pharyngeal arches—this information is covered in option (G). Therefore, we conclude that (G) must be the correct answer. The answer is (G).",
        "topic": "health",
    },
    {
        "question": "What is the difference between a male and a female catheter?",
        "options": "A. Female catheters are used more frequently than male catheters.\nB. Male catheters are bigger than female catheters.\nC. Male catheters are more flexible than female catheters.\nD. Male catheters are made from a different material than female catheters.\nE. Female catheters are longer than male catheters.\nF. Male catheters are longer than female catheters.\nG. Female catheters are bigger than male catheters.\nH. Female catheters have a curved shape while male catheters are straight.\nI. Male and female catheters are different colours.\nJ. Male catheters have a smaller diameter than female catheters.",
        "solution": "We refer to Wikipedia articles on clinical knowledge for help. The difference between a male and female catheter is that male catheters tend to be longer than female catheters. The answer is (F).",
        "topic": "health",
    },
    {
        "question": "How many attempts should you make to cannulate a patient before passing the job on to a senior colleague, according to the medical knowledge of 2020?",
        "options": "A. 1\nB. Unlimited attempts\nC. 5\nD. 0, always pass on to a senior colleague\nE. 7\nF. 2\nG. 4\nH. 6\nI. 3\nJ. 8",
        "solution": "We refer to Wikipedia articles on clinical knowledge for help. According to the medical protocol as of 2020, you should make two attempts to cannulate a patient before passing the job on to a more-senior practitioner. The answer is (F).",
        "topic": "health",
    },
    {
        "question": "Why are parvoviruses a highly impactful parasite?",
        "options": "A. They are able to alter the host's DNA\nB. Because they have no nucleic acid\nC. They can survive in extreme temperatures\nD. Only replicate in dividing cells\nE. They can infect multiple species\nF. They don't require a host to survive\nG. Can integrate into host chromosomes",
        "solution": "We refer to Wikipedia articles on virology for help. Paroviruses are highly impactful because they do not have nucleic acid. The answer is (B).",
        "topic": "health",
    },
]


mmlu_pro_few_shot_tigerlab_physics = [
    {
        "question": "Say the pupil of your eye has a diameter of 5 mm and you have a telescope with an aperture of 50 cm. How much more light can the telescope gather than your eye?",
        "options": "A. 1000 times more\nB. 50 times more\nC. 5000 times more\nD. 500 times more\nE. 10000 times more\nF. 20000 times more\nG. 2000 times more\nH. 100 times more\nI. 10 times more",
        "solution": "The amount of light is proportional to the aperture area $A = \\pi D^2/4$ for a lens with diameter $D$, so the relative amounts of light between the eye with diameter 5mm and the telescope with diameter 50mm is $(50 cm)^2/(5mm)^2 = 10000$. The answer is (E).",
        "topic": "physics",
    },
    {
        "question": "Where do most short-period comets come from and how do we know?",
        "options": "A. The Kuiper belt; short period comets tend to be in the plane of the solar system just like the Kuiper belt.\nB. The asteroid belt; short period comets tend to come from random directions indicating a spherical distribution of comets called the asteroid belt.\nC. The asteroid belt; short period comets tend to be in the plane of the solar system just like the asteroid belt.\nD. The Oort cloud; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the Oort cloud.\nE. The Oort Cloud; short period comets tend to come from random directions indicating a spherical distribution of comets called the Oort Cloud.\nF. The Oort cloud; short period comets tend to be in the plane of the solar system just like the Oort cloud.\nG. The asteroid belt; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the asteroid belt.",
        "solution": "Most short-period comets come from the Kuiper belt, and we know because short period coments tend to be in the plane of the solar system, just like the Kuiper belt is. The answer is (A).",
        "topic": "physics",
    },
    {
        "question": "A refracting telescope consists of two converging lenses separated by 100 cm. The eye-piece lens has a focal length of 20 cm. The angular magnification of the telescope is",
        "options": "A. 10\nB. 40\nC. 6\nD. 25\nE. 15\nF. 50\nG. 30\nH. 4\nI. 5\nJ. 20",
        "solution": "In a refracting telescope, if both lenses are converging, the focus of both lenses must be between the two lenses, and thus the focal lengths of the two lenses must add up to their separation. Since the focal length of one lens is 20 cm, the focal length of the other must be 80 cm. The magnification is the ratio of these two focal lengths, or 4. The answer is (H).",
        "topic": "physics",
    },
    {
        "question": "Colors in a soap bubble result from light",
        "options": "A. dispersion\nB. deflection\nC. refraction\nD. reflection\nE. interference\nF. converted to a different frequency\nG. polarization\nH. absorption\nI. diffraction\nJ. transmission",
        "solution": "In a soap bubble film, the light bounces between the two soap-air interfaces many times, interfering with itself constructively or destructively depending on the width of the film. This results in different colors being visible. The answer is (E).",
        "topic": "physics",
    },
    {
        "question": "A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?",
        "options": "A. 240 W\nB. 120 W\nC. 10 W\nD. 480 W\nE. 360 W\nF. 200 W\nG. 30 W\nH. 150 W\nI. 60 W\nJ. 300 W",
        "solution": "Rate of energy usage is known as power; in an dissipative electrical circuit, power is given by voltage times current. So in our case, the power is 120 V times 2 amps, or 240 W. The answer is (A).",
        "topic": "physics",
    },
]

mmlu_pro_few_shot_tigerlab_business = [
    {
        "question": "In contrast to _______, _______ aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of ___________, which allow campaigns to facilitate the company in achieving _________ .",
        "options": "A. Boycotts, Buyalls, Blockchain technology, Increased Sales\nB. Buycotts, Boycotts, Digital technology, Decreased Sales\nC. Boycotts, Buycotts, Digital technology, Decreased Sales\nD. Buycotts, Boycotts, Blockchain technology, Charitable donations\nE. Boycotts, Buyalls, Blockchain technology, Charitable donations\nF. Boycotts, Buycotts, Digital technology, Increased Sales\nG. Buycotts, Boycotts, Digital technology, Increased Sales\nH. Boycotts, Buycotts, Physical technology, Increased Sales\nI. Buycotts, Buyalls, Blockchain technology, Charitable donations\nJ. Boycotts, Buycotts, Blockchain technology, Decreased Sales",
        "solution": "We refer to Wikipedia articles on business ethics for help. The sentence that best uses the possible options above is __n contrast to *boycotts*, *buycotts* aim to reward favourable behavior by companies. The success of such campaigns have been heightened through the use of *digital technology*, which allow campaigns to facilitate the company in achieving *increased sales*._ The answer is (F).",
        "topic": "business",
    },
    {
        "question": "_______ is the direct attempt to formally or informally manage ethical issues or problems, through specific policies, practices and programmes.",
        "options": "A. Operational management\nB. Corporate governance\nC. Environmental management\nD. Business ethics management\nE. Sustainability\nF. Stakeholder management\nG. Social marketing\nH. Human resource management",
        "solution": "We refer to Wikipedia articles on business ethics for help. The direct attempt manage ethical issues through specific policies, practices, and programs is business ethics management. The answer is (D).",
        "topic": "business",
    },
    {
        "question": "How can organisational structures that are characterised by democratic and inclusive styles of management be described?",
        "options": "A. Flat\nB. Bureaucratic\nC. Autocratic\nD. Hierarchical\nE. Functional\nF. Decentralized\nG. Matrix\nH. Network\nI. Divisional\nJ. Centralized",
        "solution": "We refer to Wikipedia articles on management for help. Flat organizational structures are characterized by democratic and inclusive styles of management, and have few (if any) levels of management between the workers and managers.  The answer is (A).",
        "topic": "business",
    },
    {
        "question": "Although the content and quality can be as controlled as direct mail, response rates of this medium are lower because of the lack of a personal address mechanism. This media format is known as:",
        "options": "A. Online banners.\nB. Television advertising.\nC. Email marketing.\nD. Care lines.\nE. Direct mail.\nF. Inserts.\nG. Door to door.\nH. Radio advertising.\nI. Billboards.\nJ. Social media advertising.",
        "solution": "We refer to Wikipedia articles on marketing for help. Door to door marketing delivers non-addressed items within all buildings within a geographic area. While it can control the content and quality as well as direct mail marketing, its response rate is lower because of the lack of a personal address mechanism. The answer is (G).",
        "topic": "business",
    },
    {
        "question": "In an organization, the group of people tasked with buying decisions is referred to as the _______________.",
        "options": "A. Procurement centre.\nB. Chief executive unit.\nC. Resources allocation group.\nD. Marketing department.\nE. Purchasing department.\nF. Supply chain management team.\nG. Outsourcing unit.\nH. Decision-making unit.\nI. Operations unit.\nJ. Financial management team.",
        "solution": "We refer to Wikipedia articles on marketing for help. In an organization, the group of the people tasked with buying decision is referred to as the decision-making unit. The answer is (H).",
        "topic": "business",
    },
]

mmlu_pro_few_shot_tigerlab_biology = [
    {
        "question": "Which of the following represents an accurate statement concerning arthropods?",
        "options": "A. They possess an exoskeleton composed primarily of peptidoglycan.\nB. They possess an open circulatory system with a dorsal heart.\nC. They are members of a biologically unsuccessful phylum incapable of exploiting diverse habitats and nutrition sources.\nD. They lack paired, jointed appendages.",
        "solution": "Peptidoglycan is known to comprise the plasma membrane of most bacteria, rather than the exoskeleton of arthropods, which is made of chitin, which rules out (A). The answer (C) is false because arthropods are a highly successful phylum. Likewise, arthropods have paired, jointed appendages, which rules out (D). The only remaining option is (B), as arthropods have an open circulatory system with a dorsal tubular heart. The answer is (B).",
        "topic": "biology",
    },
    {
        "question": "In a given population, 1 out of every 400 people has a cancer caused by a completely recessive allele, b. Assuming the population is in Hardy-Weinberg equilibrium, which of the following is the expected proportion of individuals who carry the b allele but are not expected to develop the cancer?",
        "options": "A. 19/400\nB. 1/400\nC. 40/400\nD. 38/400\nE. 2/400\nF. 1/200\nG. 20/400\nH. 50/400",
        "solution": "According to the Hardy Weinberg Law, $p^2 + 2 p q + q^2 = 1$, and $p + q = 1$ where $p$ is the frequency of the dominant allele, $q$ is the frequency of the recessive allele, and $p^2$, $q^2$, and $2pq$ are the frequencies of dominant homozygous, recessive homozygous, and heterozygous individuals, respectively. \u200bThe frequency of the recessive allele (q) is $\\sqrt{\x0crac{1}{400}} = 0.05$. We have $p = 1 - q = 0.95$. The frequency of heterozygous individuals is $2pq = 2 \\cdot 0.05 \\cdot 0.95 = 0.095$. The number of heterozygous individuals is equal to the frequency of heterozygous individuals times the size of the population, or $0.095 * 400 = 38$. So we end up with 38/400. The answer is (D).",
        "topic": "biology",
    },
    {
        "question": "A mutation in a bacterial enzyme changed a previously polar amino acid into a nonpolar amino acid. This amino acid was located at a site distant from the enzyme’s active site. How might this mutation alter the enzyme’s substrate specificity?",
        "options": "A. By changing the enzyme’s pH optimum\nB. By changing the enzyme's molecular weight\nC. An amino acid change away from the active site increases the enzyme's substrate specificity.\nD. By changing the shape of the protein\nE. By changing the enzyme's temperature optimum\nF. By altering the enzyme's ability to be denatured\nG. By changing the enzyme’s location in the cell\nH. By changing the enzyme's color\nI. An amino acid change away from the active site cannot alter the enzyme’s substrate specificity.\nJ. By altering the enzyme's rate of reaction",
        "solution": "A change in an amino acid leads to a change in the primary structure of the protein. A change in the primary structure may lead to a change in the secondary and the tertiary structure of the protein. A change in the tertiary structure means a change in the shape of the protein, so (C) has to be correct. Since the change does not affect the active site of the enzyme, we do not expect the activity of the enzyme to be affected. The answer is (D).",
        "topic": "biology",
    },
    {
        "question": "Which of the following is not a way to form recombinant DNA?",
        "options": "A. Translation\nB. Conjugation\nC. Specialized transduction\nD. Transformation",
        "solution": "The introduction of foreign DNA or RNA into bacteria or eukaryotic cells is a common technique in molecular biology and scientific research. There are multiple ways foreign DNA can be introduced into cells including transformation, transduction, conjugation, and transfection. In contrast, (A) is not a way to form DNA: during translation the ribosomes synthesize proteins from RNA. The answer is (A).",
        "topic": "biology",
    },
    {
        "question": "Which of the following is not known to be involved in the control of cell division?",
        "options": "A. Microtubules\nB. Checkpoints\nC. DNA polymerase\nD. Centrosomes\nE. Cyclins\nF. Mitochondria\nG. Protein kinases\nH. Fibroblast cells",
        "solution": "Normal cells move through the cell cycle in a regulated way. At the checkpoint stage, they use information about their own internal state and cues from the environment around them to decide whether to proceed with cell division. Cues like these act by changing the activity of core cell cycle regulators inside the cell. The most common regulators are cyclins and cyclin-dependent kinases. Fibroblast cells do not play any role in cell division. The answer is (H).",
        "topic": "biology",
    },
]

mmlu_pro_few_shot_tigerlab_chemistry = [
    {
        "question": "Predict the number of lines in the EPR spectrum of a solution of 13C-labelled methyl radical (13CH3•), assuming the lines do not overlap.",
        "options": "A. 10\nB. 8\nC. 4\nD. 20\nE. 12\nF. 3\nG. 16\nH. 5\nI. 24\nJ. 6",
        "solution": "The electron paramagnetic resonance spectrum will be split by two forms of interactions. The first is the hyperfine interaction with the 13C (nuclear spin $I = \n\x0crac{1}{2}$) which will split the spectrum into 2 lines. This will be further split into 4 lines by the interaction with three equivalent 1H nuclei. The total number of lines is therefore $2 \\cdot 4 = 8$. The answer is (B).",
        "topic": "chemistry",
    },
    {
        "question": "Which of the following lists the hydrides of group-14 elements in order of thermal stability, from lowest to highest?",
        "options": "A. GeH4 < SnH4 < PbH4 < SiH4 < CH4\nB. SiH4 < GeH4 < SnH4 < PbH4 < CH4\nC. PbH4 < CH4 < SnH4 < GeH4 < SiH4\nD. PbH4 < SnH4 < CH4 < GeH4 < SiH4\nE. SnH4 < GeH4 < SiH4 < PbH4 < CH4\nF. CH4 < GeH4 < SnH4 < PbH4 < SiH4\nG. SiH4 < SnH4 < PbH4 < GeH4 < CH4\nH. CH4 < SiH4 < GeH4 < SnH4 < PbH4\nI. CH4 < PbH4 < GeH4 < SnH4 < SiH4\nJ. PbH4 < SnH4 < GeH4 < SiH4 < CH4",
        "solution": "The thermal stability of group-14 hydrides decreases as we move from the top of group 14 to the bottom. The order of elements in the group from top to bottom is C, Si, Ge, Sn, Pb. Therefore in order of increasing thermal stability we have PbH4, SnH4, GeH4, SiH4, and CH4, or answer (J). The answer is (J).",
        "topic": "chemistry",
    },
    {
        "question": "Which of the following is considered an acid anhydride?",
        "options": "A. H2SO3\nB. NaCl\nC. CH4\nD. HNO3\nE. SO2\nF. Al(NO3)3\nG. CaCO3\nH. H2O\nI. HCl",
        "solution": "An acid anhydride is a compound that is derived by removing water from an acid. The chemical formula for water is H2O, which means that we need to determine which of these options, when combined with H2O, forms an acid. SO2, or Sulfur dioxide, when combined with H2O, makes H2SO4, or sulfuric acid. The answer is (E).",
        "topic": "chemistry",
    },
    {
        "question": "A new compound is synthesized and found to be a monoprotic acid with a molar mass of 248 g/mol. When 0.0050 mol of this acid are dissolved in 0.500 L of water, the pH is measured as 3.89. What is the pKa of this acid?",
        "options": "A. 5.78\nB. 4.78\nC. 4.56\nD. 6.89\nE. 7.78\nF. 3.89\nG. 1.23\nH. 2.89\nI. 2.33\nJ. 5.33",
        "solution": "Recall that $[A] = [H^{+}]$. Here, this is equal to $$10^{-3.89}$. Then we have $K_{a} = $\n\x0crac{[H^{+}][A^{-}]}{[HA]} = \n\x0crac{10^{-3.89} \\cdot 10^{-3.89}}{10^{-2}}. The resulting exponent is $-3.89 + (-3.89) - (-2) = 5.78$, therefore $K_a = 10^{-5.78}$. The $pK_a$ is the negative log of $K_a$, which is equal to $5.78$. The answer is (A).",
        "topic": "chemistry",
    },
    {
        "question": "A solution contains 2.00 mole of acetic acid, CH3COOH, and 1.00 mole of calcium acetate, Ca(CH3COO)2. The solution is able to resist the addition of a small amount of strong acid or strong base with only minor changes in the pH of the solution. Larger quantities of strong acid or strong base can cause a significant change in pH. How many moles of nitric acid, HNO3, may be added before the pH begins to change significantly?",
        "options": "A. 0.250 mole\nB. 0.500 mole\nC. 3.00 mole\nD. 1.00 mole\nE. 3.50 mole\nF. 1.50 mole\nG. 2.50 mole\nH. 4.00 mole\nI. 0.750 mole\nJ. 2.00 mole",
        "solution": "We would like to compute the buffer capacity of this solution. First we write the equation for the ionization of the weak acid, in this case of acetic acid. $CH_{3}COOH (aq) + H_{2}O \nightarrow H_{3}O^{+} + CH3COO^{-}$. The conjugate base is therefore the acetate ion. The added strong acid, Nitric acid, will react with the conjugate base. Therefore the maximum amount of acid that can be added will be equal to the amount of acetate ion, or 2 moles. The answer is (J).",
        "topic": "chemistry",
    },
]

mmlu_pro_few_shot_tigerlab_computer_science = [
    {
        "question": "A certain pipelined RISC machine has 8 general-purpose registers R0, R1, . . . , R7 and supports the following operations:\nADD Rs1, Rs2, Rd (Add Rs1 to Rs2 and put the sum in Rd)\nMUL Rs1, Rs2, Rd (Multiply Rs1 by Rs2 and put the product in Rd)\nAn operation normally takes one cycle; however, an operation takes two cycles if it produces a result required by the immediately following operation in an operation sequence.\nConsider the expression AB + ABC + BC, where variables A, B, C are located in registers R0, R1, R2. If the contents of these three registers must not be modified, what is the minimum number of clock cycles required for an operation sequence that computes the value of AB + ABC + BC?",
        "options": "A. 10\nB. 6\nC. 9\nD. 5\nE. 12\nF. 4\nG. 7\nH. 8",
        "solution": "First, we are given that A is in R0, B is in R1, and C is in R2.\nNext, we can see that we must compute three multiplies (AB, BC, and ABC) and two adds (AB + ABC, (AB + ABC) + BC) to compute our final answer, resulting in a minimum of five clock cycles.\nNext, we can see that there is no way to avoid at least one pipeline stall when computing our final answer, because to compute our final sum we must wait at least one cycle for the results from the previous stage to be ready. Thus, our minimum number of cycles must be 6.\nWe can verify that we can create a solution that requires only six cycles as follows:\ncompute AB: MUL R0, R1, R3\ncompute BC: MUL R1, R2, R4\ncompute ABC: MUL R3, R4, R5\ncompute AB + BC: ADD R3, R4, R6\nSTALL\ncompute AB + ABC + BC: ADD R5, R6, R7\nSo there are 6 cycles. The answer is (B).",
        "topic": "computer_science",
    },
    {
        "question": "A compiler generates code for the following assignment statement.\nG := (A + B) * C - (D + E) * F\nThe target machine has a single accumulator and a single-address instruction set consisting of instructions load, store, add, subtract, and multiply. For the arithmetic operations, the left operand is taken from the accumulator and the result appears in the accumulator. The smallest possible number of instructions in the resulting code is",
        "options": "A. 7\nB. 5\nC. 12\nD. 4\nE. 11\nF. 6\nG. 8\nH. 10\nI. 9\nJ. 3",
        "solution": "We can compute the final answer with the following sequence of operations:\n1. LOAD D  (accumulator = D)\n2. ADD E  (accumulator = D+E)\n3. MUL F  (accumulator = (D+E)*F)\n4. STORE X (X = (D+E)*F)\n5. LOAD A  (accumulator = A)\n6. ADD B  (accumulator = A+B)\n7. MUL C  (accumulator = (A+B)*C)\n8. SUB X  (accumulator = (A+B)*C - (D+E)*F)\n9. STORE G (G = (A+B)*C - (D+E)*F)\nThis sequence takes 9 instructions. The answer is (I).",
        "topic": "computer_science",
    },
    {
        "question": "SHA-1 has a message digest of",
        "options": "A. 628 bits\nB. 780 bits\nC. 160 bits\nD. 256 bits\nE. 2048 bits\nF. 820 bits\nG. 128 bits\nH. 512 bits\nI. 1024 bits\nJ. 64 bits",
        "solution": "Since SHA-1 is a hash function which takes an input and produces a 160-bit (20-byte) hash value, its message digest is 160 bits. The answer is (C).",
        "topic": "computer_science",
    },
    {
        "question": "_____________ can modify data on your system – so that your system doesn’t run correctly or you can no longer access specific data, or it may even ask for ransom in order to give your access.",
        "options": "A. Email Trojans\nB. Backdoor Trojans\nC. Rootkit Trojans\nD. Banking Trojans\nE. Trojan-Downloader\nF. Spyware Trojans\nG. Network Trojans\nH. Worm Trojans\nI. Ransom Trojan\nJ. IM – Trojans",
        "solution": "The system is asking for trojans, which are for ransom, which means ransom trojan. The answer is (I).",
        "topic": "computer_science",
    },
    {
        "question": 'In the program below, the initial value of X is 5 and the initial value of Y is 10.\nIF (X < 0){\n DISPLAY ("Foxtrot")\n} ELSE {\n IF (X > Y){\n  DISPLAY ("Hotel")\n } ELSE {\n  IF (Y > 0){\n   DISPLAY ("November")\n  } ELSE {\n   DISPLAY ("Yankee")\n  }\n }\n}\nWhat is displayed as a result of running the program?',
        "options": "A. November\nB. Foxtrot\nC. Zulu\nD. Alpha\nE. Charlie\nF. Bravo\nG. Yankee\nH. Echo\nI. Hotel\nJ. Delta",
        "solution": 'Because X has the value 5, the first conditional IF (X < 0) is false, so we move to the first ELSE clause. Because X is 5 and Y is 10, the second conditional IF (X > Y) is false, so we move to the following ELSE clause. Since Y is 10, the conditional IF (Y > 0) is true, so the command DISPLAY ("November") is executed. The answer is (A).',
        "topic": "computer_science",
    },
]

mmlu_pro_few_shot_tigerlab_economics = [
    {
        "question": "For a stationary autoregressive process, shocks will",
        "options": "A. Never occur\nB. Persist indefinitely\nC. Cause a permanent change in the trend\nD. Grow exponentially\nE. Only affect the next period\nF. Eventually die away\nG. Cause the process to become non-stationary\nH. Immediately disappear\nI. Lead to a cyclical pattern\nJ. Be balanced out by subsequent shocks",
        "solution": "We refer to Wikipedia articles on econometrics for help. This is a formal logic problem about stationally process. For a stationary autoregressive process, shocks will eventually die away. The answer is (F).",
        "topic": "economics",
    },
    {
        "question": "Consider the following AR(1) model with the disturbances having zero mean and unit variance\nyt = 0.2 + 0.4 yt-1 + ut\nThe (unconditional) mean of y will be given by",
        "options": "A. 0.45\nB. 0.2\nC. 0.4\nD. 0.1\nE. 0.3\nF. 0.55\nG. 0.25\nH. 0.33\nI. 0.6\nJ. 0.5",
        "solution": "We refer to Wikipedia articles on econometrics for help. Let’s solve this problem step by step. If we have a an AR(1) model with the disturbances having zero mean and unit variance, then the unconditional mean of y is equal to the following:\nunconditional mean of y = (the intercept term) / (1 - autoregressive coefficient)\nWe know that the intercept term is 0.2 and the autoregressive coefficient is 0.4; thus, we have:\nunconditional mean of y = (0.2) / (1 - 0.4) = (0.2) / (0.6) = 2 / 6 = 1 / 3, which is approximately 0.33. That means that the answer should be (H) 0.33. The answer is (H).",
        "topic": "economics",
    },
    {
        "question": "Suppose that a test statistic has associated with it a p-value of 0.08. Which one of the following statements is true?\n(i) If the size of the test were exactly 8%, we would be indifferent between rejecting and not rejecting the null hypothesis\n(ii) The null would be rejected if a 10% size of test were used\n(iii) The null would not be rejected if a 1% size of test were used\n(iv) The null would be rejected if a 5% size of test were used.",
        "options": "A. (iii) and (iv) only\nB. (i) and (iv) only\nC. (i), (ii), and (iii) only\nD. (i) and (ii) only\nE. (i) and (iii) only\nF. (i), (ii), (iii), and (iv).\nG. (ii), (iii), and (iv) only\nH. (ii) and (iii) only\nI. (ii) and (iv) only\nJ. (i), (iii), and (iv) only",
        "solution": "We refer to Wikipedia articles on econometrics for help. Let’s reason about each of the options.\n(i) is a true statement.\n(ii) is a true statement.\n(iii) is a true statement.\n(iv) is not a true statement. Thus, (i), (ii), and (iii) are true. The answer is (C).",
        "topic": "economics",
    },
    {
        "question": "Which of the following policies best describes supply-side fiscal policy?",
        "options": "A. Higher taxes on household income\nB. Increased government spending\nC. Increased taxes on corporate profits\nD. Increased import tariffs\nE. Decreased interest rates\nF. Lower taxes on consumer goods\nG. Lower taxes on research and development of new technology\nH. Reduced government spending\nI. Higher taxes on capital gains\nJ. An increase in the money supply",
        "solution": "We refer to Wikipedia articles on macroeconomics for help. Supply-side fiscal policy stimulates the economy by encouraging more production of goods and services through reduction in taxes and deregulation. The answer is (G).",
        "topic": "economics",
    },
    {
        "question": "The concentration ratio for a monopoly is",
        "options": "A. 50\nB. 5\nC. 10\nD. 90\nE. 15\nF. 100\nG. 0\nH. 25\nI. 75",
        "solution": "We refer to Wikipedia articles on microeconomics for help. The concentration ratio is calculated as the sum of market share of a specific number of largest companies. Monopoly means one company or entity controls the entire market, therefore, the concentration ratio is 100 percent. The answer is (F).",
        "topic": "economics",
    },
]

mmlu_pro_few_shot_tigerlab_engineering = [
    {
        "question": "A point pole has a strength of 4π * 10^-4 weber. The force in newtons on a point pole of 4π * 1.5 * 10^-4 weber placed at a distance of 10 cm from it will be",
        "options": "A. 25 N.\nB. 12.5 N.\nC. 15 N.\nD. 10 N.\nE. 7.5 N.\nF. 20 N.\nG. 5 N.\nH. 3.75 N.",
        "solution": "The force between two point poles is given by m_1m_2/(mu_0 4 \\pi r^2), in analogy to Coulomb’s law. Plugging in the values given in the question, we calculate that the force is approximately 15 N. The answer is (C).",
        "topic": "engineering",
    },
    {
        "question": "The coil of a moving coil meter has 100 turns, is 40 mm long and 30 mm wide. The control torque is 240*10-6 N-m on full scale. If magnetic flux density is 1Wb/m2 range of meter is",
        "options": "A. 2 mA.\nB. 5 mA.\nC. 1.5 mA.\nD. 0.5 mA.\nE. 6 mA.\nF. 4 mA.\nG. 3 mA.\nH. 1 mA.\nI. 2.5 mA.\nJ. 3.5 mA.",
        "solution": "The torque on a coil in a uniform magnetic field is given by BANI, where B is the magnetic flux density, A is the area of the coil, N is the number of turns, and I is the current. So we have that I = (Torque)/(BAN), or 240e-6/(1200e-6 * 100 * 1) = 2e-3. The answer is (A).",
        "topic": "engineering",
    },
    {
        "question": "In an SR latch built from NOR gates, which condition is not allowed",
        "options": "A. S=0, R=2\nB. S=2, R=2\nC. S=1, R=1\nD. S=1, R=-1\nE. S=1, R=2\nF. S=0, R=0\nG. S=2, R=0\nH. S=1, R=0\nI. S=2, R=1\nJ. S=0, R=1",
        "solution": "An SR latch is a set-reset latch; in the case where S=1 and R=1, the circuit has no stable state; instead a race condition will be produced within the circuit, so the device will be in an undefined state. So S=1, R=1 is an illegal input. The answer is (C).",
        "topic": "engineering",
    },
    {
        "question": "Two long parallel conductors carry 100 A. If the conductors are separated by 20 mm, the force per meter of length of each conductor will be",
        "options": "A. 100 N.\nB. 0.01 N.\nC. 10 N.\nD. 0.5 N.\nE. 0.001 N.\nF. 5 N.\nG. 1 N.\nH. 50 N.\nI. 0.1 N.\nJ. 0.0001 N.",
        "solution": "The magnetic force-per-length between two current-carrying conductors is given by \\mu_0 I_1 I_2 / (2 \\pi r), where $r$ is the separation distance and I_1 and I_2 are the currents. Plugging in 100 A for I_1 and I_2, and 20 mm for r, gives 0.1 N. The answer is (I).",
        "topic": "engineering",
    },
    {
        "question": "In a 2 pole lap winding dc machine , the resistance of one conductor is 2Ω and total number of conductors is 100. Find the total resistance",
        "options": "A. 50Ω\nB. 1Ω\nC. 25Ω\nD. 200Ω\nE. 10Ω\nF. 100Ω\nG. 500Ω\nH. 150Ω\nI. 75Ω\nJ. 20Ω",
        "solution": "In lap winding, effectively two resistors are connected in parallel, so the actual resistance of each pair is 1 Ohm. Since we have 50 pairs, we get a total resistance of 50 Ohms. The answer is (A).",
        "topic": "engineering",
    },
]

mmlu_pro_few_shot_tigerlab_philosophy = [
    {
        "question": "Which of the given formulas of PL is the best symbolization of the following sentence?\nTurtles live long lives and are happy creatures, unless they are injured.",
        "options": "A. (L • H) ≡ I\nB. (L • H) ∨ I\nC. L • (H ∨ I)\nD. L • (H ⊃ R).",
        "solution": "We refer to Wikipedia articles on formal logic for help. Let’s solve this step by step. Let “L” denote “living long”, H “being happy”, and “I” “being injured”. Now, consider each choice:\n(A) means (living long AND being happy) is equivalent to (being injured). \n(B) means (living long AND being happy) OR (being injured). \n(C) means (living long) AND (being happy OR being injured). \n(D) means (living long) AND (being happy implies being R), but what R denotes is not clear.\nObviously, (B) is the best symbolization of the original sentence. The answer is (B).",
        "topic": "philosophy",
    },
    {
        "question": "Select the best translation into predicate logic.George borrows Hector's lawnmower. (g: George; h: Hector; l: Hector's lawnmower; Bxyx: x borrows y from z).",
        "options": "A. Bhgh\nB. Bggh\nC. Bhlh\nD. Bghl\nE. Bhlg\nF. Blhg\nG. Bllg\nH. Blgh\nI. Bhhg\nJ. Bglh",
        "solution": "We refer to Wikipedia articles on formal logic for help. Let’s solve this step by step. We are told that “Bxyx” means “x borrows y from z”. We can rewrite “George borrows Hector's lawnmower” as “George borrows a lawnmower from Hector”, which can then be translated into predicate logic as “Bglh”. The answer “Bglh” appears in (J); therefore, (J) must be the correct answer. The answer is (J).",
        "topic": "philosophy",
    },
    {
        "question": "The study of reality in the broadest sense, an inquiry into the elemental nature of the universe and the things in it, is known as _____.",
        "options": "A. quantum physics\nB. metaphysics\nC. ontology\nD. phenomenology\nE. cosmology\nF. teleology\nG. epistemology\nH. axiology",
        "solution": "We refer to Wikipedia articles on philosophy for help. Among the options, only metaphysics studies the nature of reality and existence. The answer is (B).",
        "topic": "philosophy",
    },
    {
        "question": "According to Moore’s “ideal utilitarianism,” the right action is the one that brings about the greatest amount of:",
        "options": "A. wealth.\nB. virtue.\nC. fairness.\nD. pleasure.\nE. peace.\nF. justice.\nG. happiness.\nH. power.\nI. good.\nJ. knowledge.",
        "solution": "We refer to Wikipedia articles on philosophy for help. Moore's \"ideal utilitarianism\" states that one's actions should maximize intrinsic goods. The answer is (I).",
        "topic": "philosophy",
    },
    {
        "question": "What is the sign of the covenant for Jewish males?",
        "options": "A. Fasting on Yom Kippur\nB. Lighting Shabbat candles\nC. The rainbow\nD. Circumcision\nE. The Torah\nF. Bar mitzvah\nG. Keeping kosher\nH. Wearing a kippah\nI. A son\nJ. The Star of David",
        "solution": "We refer to Wikipedia articles on world religions for help. In Judaism, the most distinctive sign of the covenant is circumcision (brit milah). The answer is (D).",
        "topic": "philosophy",
    },
]

mmlu_pro_few_shot_tigerlab_other = [
    {
        "question": "As of 2017, how many of the world’s 1-year-old children today have been vaccinated against some disease? *",
        "options": "A. 30%\nB. 60%\nC. 10%\nD. 90%\nE. 80%\nF. 40%\nG. 100%\nH. 50%",
        "solution": "We refer to Wikipedia articles on global facts for help. According to data published by the World Health Organization, the nummber of 1-year-old children vaccinated in 2017 exceeds 80%. The answer is (E).",
        "topic": "other",
    },
    {
        "question": "Which one of the following items is an example of nonmaterial culture?",
        "options": "A. A dove feather\nB. Dove symbol\nC. Dove body lotion\nD. Dove deodorant\nE. Dove soap\nF. Dove candy bar\nG. Dove conditioner\nH. A dove (bird).\nI. Dove chocolate\nJ. Dove shampoo",
        "solution": "We refer to Wikipedia articles on geography for help. Nonmaterial culture consists of cultural ideas, beliefs or symbols that are not physical objects. The answer is (B).",
        "topic": "other",
    },
    {
        "question": "Which of the following cases established the precedent that a defendant must be informed of the right to remain silent, the right to a lawyer, and protection from self-incrimination?",
        "options": "A. Brown v. Board of Education\nB. Miranda v. Arizona\nC. Roe v. Wade\nD. Betts v. Brady\nE. Plessy v. Ferguson\nF. Dred Scott v. Sandford\nG. Weeks v. United States\nH. Gideon v. Wainwright\nI. Marbury v. Madison\nJ. Mapp v. Ohio",
        "solution": "We refer to Wikipedia articles on government and politics for help. In the landmark Miranda v. Arizona in 1966, the US Supreme Court, based on the Fifth and Sixth Amendment of the US Constitution, guaranteed a defendant's right to an attorney and protection from self-incrimination. The answer is (B).",
        "topic": "other",
    },
    {
        "question": "A woman who knows she has active herpes and untreated syphilis but continues to have sex without informing her partners of her condition has, in psychoanalytic terms:",
        "options": "A. a weak conscious mind\nB. a strong conscious mind\nC. a weak id\nD. a weak ego\nE. a weak unconscious mind\nF. a strong id\nG. a strong ego\nH. a strong superego\nI. a strong preconscious mind\nJ. a weak superego",
        "solution": "We refer to Wikipedia articles on human sexuality for help. A person with weak superego tends to be delinquent, criminal or have antisocial personality. The action of the woman who knows she has active venereal disease but still have sex with her partners indicate she may has antisocial personality. The answer is (J).",
        "topic": "other",
    },
    {
        "question": "What place is named in the title of the 1979 live album by rock legends Cheap Trick?",
        "options": "A. Brooklyn\nB. Beijing\nC. Budapest\nD. Boston\nE. Bhutan\nF. Barcelona\nG. Britain\nH. Brisbane\nI. Bruges\nJ. Budokan",
        "solution": "We refer to Wikipedia for help. Nippon Budokan is an indoor arena in Tokyo, Japan renowned for hosting rock music concerts including Cheap Trick in 1978. 'Cheap Trick at Budokan' became the name of their album. The answer is (J).",
        "topic": "other",
    },
]

mmlu_pro_few_shot_tigerlab_history = [
    {
        "question": 'This question refers to the following information.\nIn Russia there was nothing going on well, and [Souvarine] was in despair over the news he had received. His old companions were all turning to the politicians; the famous Nihilists who made Europe tremble-sons of village priests, of the lower middle class, of tradesmen-could not rise above the idea of national liberation, and seemed to believe that the world would be delivered-when they had killed their despot&…\n"Foolery! They\'ll never get out of it with their foolery."\nThen, lowering his voice still more, in a few bitter words he described his old dream of fraternity. He had renounced his rank and his fortune; he had gone among workmen, only in the hope of seeing at last the foundation of a new society of labour in common. All the sous in his pockets had long gone to the urchins of the settlement; he had been as tender as a brother with the colliers, smiling at their suspicion, winning them over by his quiet workmanlike ways and his dislike of chattering. But decidedly the fusion had not taken place.\nHis voice changed, his eyes grew bright, he fixed them on étienne, directly addressing him:\n"Now, do you understand that? These hatworkers at Marseilles who have won the great lottery prize of a hundred thousand francs have gone off at once and invested it, declaring that they are going to live without doing anything! Yes, that is your idea, all of you French workmen; you want to unearth a treasure in order to devour it alone afterwards in some lazy, selfish corner. You may cry out as much as you like against the rich, you haven\'t got courage enough to give back to the poor the money that luck brings you. You will never be worthy of happiness as long as you own anything, and your hatred of the bourgeois proceeds solely from an angry desire to be bourgeois yourselves in their place."\némile Zola, French writer, Germinal, 1885\nThe passage displays the direct concern for the welfare of the working classes that was typically a part of which movement?',
        "options": "A. Communist\nB. Anarchist\nC. Feminist\nD. Fascist\nE. Imperialist\nF. Nationalist\nG. Enlightenment\nH. Existentialist",
        "solution": "We refer to Wikipedia articles on european history for help. The modern Communist movement aims to establish a classless society based on communal ownership and distribution of property and means of production, thereby especially benefiting the working classes. The answer is (A).",
        "topic": "history",
    },
    {
        "question": "This question refers to the following information.\nThe following excerpt is from a pamphlet.\nYou will do me the justice to remember, that I have always strenuously supported the Right of every man to his own opinion, however different that opinion might be to mine. He who denies to another this right, makes a slave of himself to his present opinion, because he precludes himself the right of changing it.\nThe most formidable weapon against errors of every kind is Reason. I have never used any other, and I trust I never shall.\nThe circumstance that has now taken place in France of the total abolition of the whole national order of priesthood, and of everything appertaining to compulsive systems of religion, and compulsive articles of faith, has not only precipitated my intention, but rendered a work of this kind exceedingly necessary, lest in the general wreck of superstition, of false systems of government, and false theology, we lose sight of morality, of humanity, and of the theology that is true.\nI believe in one God, and no more; and I hope for happiness beyond this life.\nI believe in the equality of man; and I believe that religious duties consist in doing justice, loving mercy, and endeavoring to make our fellow-creatures happy.\nI do not believe in the creed professed by the Jewish church, by the Roman church, by the Greek church, by the Turkish church, by the Protestant church, nor by any church that I know of. My own mind is my own church.\nAll national institutions of churches, whether Jewish, Christian or Turkish, appear to me no other than human inventions, set up to terrify and enslave mankind, and monopolize power and profit.\nI do not mean by this declaration to condemn those who believe otherwise; they have the same right to their belief as I have to mine.\n—Thomas Paine, The Age of Reason, 1794–1795\nWhich of the following Enlightenment philosophes designed a system of checks and balances for government to avoid abuses of power?",
        "options": "A. Thomas Hobbes\nB. Jean Jacques Rousseau\nC. Baron Montesquieu\nD. Voltaire\nE. Denis Diderot\nF. Immanuel Kant\nG. David Hume\nH. Mary Wollstonecraft\nI. Adam Smith\nJ. John Locke",
        "solution": "We refer to Wikipedia articles on european history for help. Baron Montesquieu was a 18th centrury French philsopher who wrote extensively against the monoplization of power and advocated for a system of checks and balances in government to prevent the rise of despotism. The answer is (C).",
        "topic": "history",
    },
    {
        "question": 'This question refers to the following information.\n"The struggle against neo-colonialism is not aimed at excluding the capital of the developed world from operating in less developed countries. It is aimed at preventing the financial power of the developed countries being used in such a way as to impoverish the less developed.\nNon-alignment, as practiced by Ghana and many other countries, is based on co-operation with all States whether they be capitalist, socialist or have a mixed economy. Such a policy, therefore, involves foreign investment from capitalist countries, but it must be invested in accordance with a national plan drawn up by the government of the non-aligned State with its own interests in mind. The issue is not what return the foreign investor receives on his investments…The question is one of power. A State in the grip of neo-colonialism is not master of its own destiny."\nKwame Nkrumah, Neo-Colonialism, 1965\nWhich of the following provides the best context for Nkrumah\'s writings?',
        "options": "A. The spread of communism\nB. The Industrial Revolution\nC. The French Revolution\nD. The Cold War\nE. Decolonization\nF. World War II\nG. Autarky\nH. The rise of multinational corporations\nI. The American Civil Rights Movement\nJ. Regional Free Trade Associations",
        "solution": "We refer to Wikipedia articles on world history for help. The passage expresses a point that the successful fight against neo-colonialism were in danger and the newly independent nations like Ghana may be re-colonized via financial power of the developed countries. The answer is (E).",
        "topic": "history",
    },
    {
        "question": "What is the approximate mean cranial capacity of Homo erectus?",
        "options": "A. 1200 cc\nB. under 650 cc\nC. 1700 cc\nD. 1350 cc\nE. just under 1000 cc\nF. 1500 cc\nG. under 500 cc\nH. about 800 cc\nI. just over 1100 cc\nJ. about 900 cc",
        "solution": "We refer to Wikipedia articles on prehistory for help. The average cranium capacity of Homo erectus is less than 1000 cubic cm. The answer is (E).",
        "topic": "history",
    },
    {
        "question": "Recent research on hominid species dating from the Middle Pliocene indicates there was (as of 2020):",
        "options": "A. multiple hominid species but with limited diversity.\nB. a single species with no diversity.\nC. decreased species diversity but increased numbers of hammerstones and flakes, indicating stone tool manufacture.\nD. a single dominant species that outcompeted all others, leading to decreased diversity.\nE. increased species diversity due to a prolonged ice age followed by a severe drought.\nF. decreased species diversity due to a prolonged ice age followed by a severe drought.\nG. a great amount of species diversity, or a single species that exhibited a lot of diversity.\nH. increased species diversity but with decreased population numbers due to harsh climate conditions.\nI. increased species diversity but decreased numbers of hammerstones and flakes, indicating less stone tool manufacture.\nJ. very little species diversity during this period and very few hominids.",
        "solution": "We refer to Wikipedia articles on prehistory for help. Recent research has recognized multiple hominid species from the Middle Pliocene, meaning that there is a great amount of species diversity or diversity in a single species. The answer is (G).",
        "topic": "history",
    },
]

mmlu_pro_few_shot_tigerlab_psychology = [
    {
        "question": "Pascale is interested in the processing strategies children use to learn new information. Pascale would best be classified as what type of psychologist?",
        "options": "A. social\nB. school\nC. sociocultural\nD. forensic\nE. behaviorist\nF. health\nG. clinical\nH. cognitive\nI. psychoanalytic\nJ. developmental",
        "solution": "We refer to Wikipedia articles on psychology for help. Sociocultural psychologist focuses on the effect of societal factors on people. Clinical psychologist focuses on people with mental issues. Cognitive psychologist focuses on how people think and learn, including the processing strategies. Behaviorist focuses more on the environment and experience effect on people. The answer is (H).",
        "topic": "psychology",
    },
    {
        "question": "According to Caplan's model of consultee-centered case consultation, the consultant is primarily interested in",
        "options": "A. identifying the causes and solutions of the client's presenting problems\nB. establishing a hierarchy of authority to enable effective decision making\nC. ensuring the consultee adheres strictly to a predetermined action plan\nD. proposing multiple alternative solutions for the consultee to choose from\nE. identifying the strengths and weaknesses of the consultee's current approach\nF. presenting a single, well-defined and unambiguous course of action for the consultant to overcome skills deficits\nG. developing a comprehensive treatment plan for the client\nH. identifying and eliminating the causes of the consultee's difficulties in handling a problem\nI. focusing on the consultant's personal growth and development\nJ. focusing on the relationship between the client and the consultee",
        "solution": "We refer to Wikipedia articles on psychology for help. Caplan defines two type of consultation. Client-centered case consultation aims to handle client's problems, while consultee-centered case consultation aims to identify the reason of client's difficulty to solve problems. The answer is (H).",
        "topic": "psychology",
    },
    {
        "question": "According to the Individuals with Disabilities Education Improvement Act, which of the following must an educational agency do before it changes the educational placement of a student with a disability?",
        "options": "A. Notify the parents in writing\nB. Obtain the child's consent\nC. Obtain a court order\nD. Conduct a new evaluation of the child's disability\nE. Discuss with the child's psychologist\nF. Give the child a trial period in the new environment\nG. Obtain parental consent\nH. Notify the local education authority\nI. Arrange a meeting with all teachers and administrators\nJ. Obtain school board approval",
        "solution": "We refer to Wikipedia articles on psychology for help. When the decision to change the educational placement of a student with a disability is made, the educational agency must notify the parents in writing on that date. The answer is (A).",
        "topic": "psychology",
    },
    {
        "question": "Ani believes that her attitudes and behavior play a central role in what happens to her. Such a belief is likely to be associated with",
        "options": "A. low self-esteem.\nB. a strong id.\nC. a high level of anxiety.\nD. a strong superego.\nE. high levels of self-consciousness.\nF. an external locus of control.\nG. an inferiority complex.\nH. a low level of self-awareness.\nI. low self-efficacy.\nJ. an internal locus of control.",
        "solution": "We refer to Wikipedia articles on psychology for help. People with an external locus of control believes fate and luck play an important role in their lives, while people with an internal locus of control believes they control their lives. The answer is (J).",
        "topic": "psychology",
    },
    {
        "question": "In terms of Hofstede’s (1980) five cultural dimensions, the United States scores at the top of the scale on:",
        "options": "A. individualism and long-term orientation.\nB. individualism and power distance.\nC. uncertainty avoidance.\nD. long-term orientation.\nE. individualism.\nF. individualism and masculinity.\nG. long-term orientation and uncertainty avoidance.\nH. power distance.\nI. power distance and masculinity.",
        "solution": "We refer to Wikipedia articles on psychology for help. US scores highest on individualism among the five cultural dimensions. The answer is (E).",
        "topic": "psychology",
    },
]

mmlu_pro_few_shot_tigerlab_law = [
    {
        "question": "What is the judge ad hoc?",
        "options": "A. Judge ad hoc is the president of the ICJ\nB. Judge ad hoc is a temporary judge appointed for a specific period of time\nC. Judge ad hoc is the judge that each party will always nominate in every contentious case\nD. Judge ad hoc is the member of the bench of the ICJ with a casting vote\nE. Judge ad hoc is a judge who is nominated by the parties involved in a contentious case, irrespective of their nationality\nF. Judge ad hoc is a judge who decides on the admissibility of cases before the ICJ\nG. Judge ad hoc is a judge appointed by the Security Council of the United Nations\nH. Judge ad hoc is a surrogate judge, in case a judge is disqualified or passes away\nI. If a party to a contentious case before the ICJ does not have a national sitting as judge, it is entitled to nominate someone as a judge solely for that case, with the title of judge ad hoc",
        "solution": 'We refer to Wikipedia articles on international law for help. As "ad hoc" implies, a judge ad hoc is appointed only for a specific case or period, when a party to a contentious case before the International Court of Justice does not have a regular national sitting as judge. The answer is (I).',
        "topic": "law",
    },
    {
        "question": "Functions of the law include all but which of the following?",
        "options": "A. defining the limits of government power\nB. regulating the use of public spaces\nC. keeping the peace\nD. maximizing individual freedom\nE. maintaining order and stability\nF. preventing environmental degradation\nG. providing a basis for compromise\nH. promoting social justice\nI. promoting the principles of the free enterprise system\nJ. encouraging economic growth",
        "solution": "We refer to Wikipedia articles on jurisprudence for help. Laws are fundamentally about helping resolve disputes between individuals, and therefore essential for maximizing individual freedom, providing a basis for compromise, and keeping the peace. The answer is (I).",
        "topic": "law",
    },
    {
        "question": 'The ________ School of jurisprudence postulates that the law is based on what is "correct."',
        "options": "A. Legal Pragmatism\nB. Legal Formalism\nC. Comparative\nD. Analytical\nE. Sociological\nF. Historical\nG. Critical Legal Studies\nH. Realist\nI. Positivist\nJ. Natural Law",
        "solution": 'We refer to Wikipedia articles on jurisprudence for help. Natural Law School of jurisprudence focuses on the laws of nature, and states that the law should be based on ethics, morals, and what is "correct". Analytical deals with the law as it already exists, Historical postulates that the law was found and not made, and Sociological studies how the law and society impact each other. The answer is (J).',
        "topic": "law",
    },
    {
        "question": "Which word best summarizes Weber's explanation of the development of formally rational law?",
        "options": "A. Socialism.\nB. Legitimacy.\nC. Authority.\nD. Democracy.\nE. Bureaucracy.\nF. Conflict.\nG. Capitalism.\nH. Charisma.\nI. Co-operation.\nJ. Tradition.",
        "solution": "We refer to Wikipedia articles on jurisprudence for help. Weber explained the development of formal rationality in laws as how the modern society moved from tradition to rationality, where people decide actions based less on how they were culturally done and more on expected utilities. How rational individuals optimize efficiency of accomplishing tasks for higher rewards is a core principle of Capitalism. The answer is (G).",
        "topic": "law",
    },
    {
        "question": "A state has recently enacted a statute prohibiting the disposal of any nuclear wastes within the state. This law does not contravene or conflict with any federal statutes. A man operates a company in the state that is engaged in the disposal of nuclear wastes. Subsequent to the passage of the state statute, the man, not yet aware of the new law, entered into contracts with many out-of-state firms to dispose of their nuclear wastes in the state. On account of this new law, however, the man will be unable to perform these contracts. Assume that the man has standing to challenge this state law. Which of the following presents his strongest constitutional grounds to challenge the state law prohibiting the disposal of nuclear wastes within the state?",
        "options": "A. The second amendment - the right to bear arms.\nB. The due process clause of the Fourteenth Amendment.\nC. The tenth amendment - powers not delegated to the United States by the Constitution.\nD. The first amendment - freedom of speech.\nE. The privileges and immunities clause of Article IV, Section 2.\nF. The commerce clause.\nG. The sixth amendment - right to a fair trial.\nH. The eighth amendment - prohibition of cruel and unusual punishment.\nI. The equal protection clause of the Fourteenth Amendment.",
        "solution": "We refer to Wikipedia articles on law for help. The commerce clause states that Congress shall have the power to regulate commerce with foreign Nations, and among the several States, and with the Indian Tribes. The statute affects inter-state commerce which puts it into question. Hence the man's strongest argument should be the commerce clause. The answer is (F).",
        "topic": "law",
    },
]


examples_map = {
    'mmlu_pro_few_shot_llama_math': mmlu_pro_few_shot_llama_math,
    'mmlu_pro_few_shot_llama_health': mmlu_pro_few_shot_llama_health,
    'mmlu_pro_few_shot_llama_physics': mmlu_pro_few_shot_llama_physics,
    'mmlu_pro_few_shot_llama_business': mmlu_pro_few_shot_llama_business,
    'mmlu_pro_few_shot_llama_biology': mmlu_pro_few_shot_llama_biology,
    'mmlu_pro_few_shot_llama_chemistry': mmlu_pro_few_shot_llama_chemistry,
    'mmlu_pro_few_shot_llama_computer_science': mmlu_pro_few_shot_llama_computer_science,
    'mmlu_pro_few_shot_llama_economics': mmlu_pro_few_shot_llama_economics,
    'mmlu_pro_few_shot_llama_engineering': mmlu_pro_few_shot_llama_engineering,
    'mmlu_pro_few_shot_llama_philosophy': mmlu_pro_few_shot_llama_philosophy,
    'mmlu_pro_few_shot_llama_other': mmlu_pro_few_shot_llama_other,
    'mmlu_pro_few_shot_llama_history': mmlu_pro_few_shot_llama_history,
    'mmlu_pro_few_shot_llama_psychology': mmlu_pro_few_shot_llama_psychology,
    'mmlu_pro_few_shot_llama_law': mmlu_pro_few_shot_llama_law,
    'mmlu_pro_few_shot_tigerlab_math': mmlu_pro_few_shot_tigerlab_math,
    'mmlu_pro_few_shot_tigerlab_health': mmlu_pro_few_shot_tigerlab_health,
    'mmlu_pro_few_shot_tigerlab_physics': mmlu_pro_few_shot_tigerlab_physics,
    'mmlu_pro_few_shot_tigerlab_business': mmlu_pro_few_shot_tigerlab_business,
    'mmlu_pro_few_shot_tigerlab_biology': mmlu_pro_few_shot_tigerlab_biology,
    'mmlu_pro_few_shot_tigerlab_chemistry': mmlu_pro_few_shot_tigerlab_chemistry,
    'mmlu_pro_few_shot_tigerlab_computer_science': mmlu_pro_few_shot_tigerlab_computer_science,
    'mmlu_pro_few_shot_tigerlab_economics': mmlu_pro_few_shot_tigerlab_economics,
    'mmlu_pro_few_shot_tigerlab_engineering': mmlu_pro_few_shot_tigerlab_engineering,
    'mmlu_pro_few_shot_tigerlab_philosophy': mmlu_pro_few_shot_tigerlab_philosophy,
    'mmlu_pro_few_shot_tigerlab_other': mmlu_pro_few_shot_tigerlab_other,
    'mmlu_pro_few_shot_tigerlab_history': mmlu_pro_few_shot_tigerlab_history,
    'mmlu_pro_few_shot_tigerlab_psychology': mmlu_pro_few_shot_tigerlab_psychology,
    'mmlu_pro_few_shot_tigerlab_law': mmlu_pro_few_shot_tigerlab_law,
}
