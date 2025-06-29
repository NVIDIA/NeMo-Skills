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

mmlu_few_shot_abstract_algebra = [
    {
        "problem": 'Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3',
        "options": 'A. 0\nB. 1\nC. 2\nD. 3',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'abstract_algebra',
    },
    {
        "problem": 'Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.\nA. True, True\nB. False, False\nC. True, False\nD. False, True',
        "options": 'A. True, True\nB. False, False\nC. True, False\nD. False, True',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'abstract_algebra',
    },
    {
        "problem": 'Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.\nA. True, True\nB. False, False\nC. True, False\nD. False, True',
        "options": 'A. True, True\nB. False, False\nC. True, False\nD. False, True',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'abstract_algebra',
    },
    {
        "problem": 'Statement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.\nA. True, True\nB. False, False\nC. True, False\nD. False, True',
        "options": 'A. True, True\nB. False, False\nC. True, False\nD. False, True',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'abstract_algebra',
    },
    {
        "problem": 'Find the characteristic of the ring 2Z.\nA. 0\nB. 3\nC. 12\nD. 30',
        "options": 'A. 0\nB. 3\nC. 12\nD. 30',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'abstract_algebra',
    },
]

mmlu_few_shot_anatomy = [
    {
        "problem": 'What is the embryological origin of the hyoid bone?\nA. The first pharyngeal arch\nB. The first and second pharyngeal arches\nC. The second pharyngeal arch\nD. The second and third pharyngeal arches',
        "options": 'A. The first pharyngeal arch\nB. The first and second pharyngeal arches\nC. The second pharyngeal arch\nD. The second and third pharyngeal arches',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'anatomy',
    },
    {
        "problem": 'Which of these branches of the trigeminal nerve contain somatic motor processes?\nA. The supraorbital nerve\nB. The infraorbital nerve\nC. The mental nerve\nD. None of the above',
        "options": 'A. The supraorbital nerve\nB. The infraorbital nerve\nC. The mental nerve\nD. None of the above',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'anatomy',
    },
    {
        "problem": 'The pleura\nA. have no sensory innervation.\nB. are separated by a 2 mm space.\nC. extend into the neck.\nD. are composed of respiratory epithelium.',
        "options": 'A. have no sensory innervation.\nB. are separated by a 2 mm space.\nC. extend into the neck.\nD. are composed of respiratory epithelium.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'anatomy',
    },
    {
        "problem": "In Angle's Class II Div 2 occlusion there is\nA. excess overbite of the upper lateral incisors.\nB. negative overjet of the upper central incisors.\nC. excess overjet of the upper lateral incisors.\nD. excess overjet of the upper central incisors.",
        "options": 'A. excess overbite of the upper lateral incisors.\nB. negative overjet of the upper central incisors.\nC. excess overjet of the upper lateral incisors.\nD. excess overjet of the upper central incisors.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'anatomy',
    },
    {
        "problem": 'Which of the following is the body cavity that contains the pituitary gland?\nA. Abdominal\nB. Cranial\nC. Pleural\nD. Spinal',
        "options": 'A. Abdominal\nB. Cranial\nC. Pleural\nD. Spinal',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'anatomy',
    },
]

mmlu_few_shot_astronomy = [
    {
        "problem": 'You are pushing a truck along a road. Would it be easier to accelerate this truck on Mars? Why? (Assume there is no friction)\nA. It would be harder since the truck is heavier on Mars.\nB. It would be easier since the truck is lighter on Mars.\nC. It would be harder since the truck is lighter on Mars.\nD. It would be the same no matter where you are.',
        "options": 'A. It would be harder since the truck is heavier on Mars.\nB. It would be easier since the truck is lighter on Mars.\nC. It would be harder since the truck is lighter on Mars.\nD. It would be the same no matter where you are.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'astronomy',
    },
    {
        "problem": 'Where do most short-period comets come from and how do we know?\nA. The Kuiper belt; short period comets tend to be in the plane of the solar system just like the Kuiper belt.\nB. The Kuiper belt; short period comets tend to come from random directions indicating a spherical distribution of comets called the Kuiper belt.\nC. The asteroid belt; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the asteroid belt.\nD. The Oort cloud; short period comets tend to be in the plane of the solar system just like the Oort cloud.',
        "options": 'A. The Kuiper belt; short period comets tend to be in the plane of the solar system just like the Kuiper belt.\nB. The Kuiper belt; short period comets tend to come from random directions indicating a spherical distribution of comets called the Kuiper belt.\nC. The asteroid belt; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the asteroid belt.\nD. The Oort cloud; short period comets tend to be in the plane of the solar system just like the Oort cloud.',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'astronomy',
    },
    {
        "problem": 'Say the pupil of your eye has a diameter of 5 mm and you have a telescope with an aperture of 50 cm. How much more light can the telescope gather than your eye?\nA. 10000 times more\nB. 100 times more\nC. 1000 times more\nD. 10 times more',
        "options": 'A. 10000 times more\nB. 100 times more\nC. 1000 times more\nD. 10 times more',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'astronomy',
    },
    {
        "problem": "Why isn't there a planet where the asteroid belt is located?\nA. A planet once formed here but it was broken apart by a catastrophic collision.\nB. There was not enough material in this part of the solar nebula to form a planet.\nC. There was too much rocky material to form a terrestrial planet but not enough gaseous material to form a jovian planet.\nD. Resonance with Jupiter prevented material from collecting together to form a planet.",
        "options": 'A. A planet once formed here but it was broken apart by a catastrophic collision.\nB. There was not enough material in this part of the solar nebula to form a planet.\nC. There was too much rocky material to form a terrestrial planet but not enough gaseous material to form a jovian planet.\nD. Resonance with Jupiter prevented material from collecting together to form a planet.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'astronomy',
    },
    {
        "problem": 'Why is Mars red?\nA. Because the surface is covered with heavily oxidized ("rusted") minerals.\nB. Because the atmosphere scatters more light at bluer wavelengths transmitting mostly red light.\nC. Because Mars is covered with ancient lava flows which are red in color.\nD. Because flowing water on Mars\'s surface altered the surface minerals several billion years ago.',
        "options": 'A. Because the surface is covered with heavily oxidized ("rusted") minerals.\nB. Because the atmosphere scatters more light at bluer wavelengths transmitting mostly red light.\nC. Because Mars is covered with ancient lava flows which are red in color.\nD. Because flowing water on Mars\'s surface altered the surface minerals several billion years ago.',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'astronomy',
    },
]

mmlu_few_shot_business_ethics = [
    {
        "problem": 'Beyond the business case for engaging in CSR there are a number of moral arguments relating to: negative _______, the _______that corporations possess and the ________ of business and society.\nA. Externalities, Power, Independence\nB. Publicity, Insubstantial resources, Mutual dependence\nC. Publicity, Power, Independence\nD. Externalities, Power, Mutual dependence',
        "options": 'A. Externalities, Power, Independence\nB. Publicity, Insubstantial resources, Mutual dependence\nC. Publicity, Power, Independence\nD. Externalities, Power, Mutual dependence',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'business_ethics',
    },
    {
        "problem": '_______ is the direct attempt to formally or informally manage ethical issues or problems, through specific policies, practices and programmes.\nA. Corporate social responsibility\nB. Business ethics management\nC. Sustainability\nD. Environmental management',
        "options": 'A. Corporate social responsibility\nB. Business ethics management\nC. Sustainability\nD. Environmental management',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'business_ethics',
    },
    {
        "problem": 'To ensure the independence of the non-executive board members, they are a number of steps which can be taken, which include non-executives being drawn from _______ the company, being appointed for a _________ time period as well as being appointed _________.\nA. Outside, Limited, Independently\nB. Inside, Limited, Intermittently\nC. Outside, Unlimited, Intermittently\nD. Inside, Unlimited, Independently',
        "options": 'A. Outside, Limited, Independently\nB. Inside, Limited, Intermittently\nC. Outside, Unlimited, Intermittently\nD. Inside, Unlimited, Independently',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'business_ethics',
    },
    {
        "problem": "Three contrasting tactics that CSO's can engage in to meet their aims are ________ which typically involves research and communication, ________, which may involve physically attacking a company's operations or ________, often involving some form of _______.\nA. Non-violent direct action, Violent direct action, Indirect action, Boycott\nB. Indirect action, Instrumental action, Non-violent direct action, Information campaign\nC. Indirect action, Violent direct action, Non-violent direct-action Boycott\nD. Non-violent direct action, Instrumental action, Indirect action, Information campaign",
        "options": 'A. Non-violent direct action, Violent direct action, Indirect action, Boycott\nB. Indirect action, Instrumental action, Non-violent direct action, Information campaign\nC. Indirect action, Violent direct action, Non-violent direct-action Boycott\nD. Non-violent direct action, Instrumental action, Indirect action, Information campaign',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'business_ethics',
    },
    {
        "problem": 'In contrast to _______, _______ aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of ___________, which allow campaigns to facilitate the company in achieving _________ .\nA. Buycotts, Boycotts, Blockchain technology, Charitable donations\nB. Buycotts, Boycotts, Digital technology, Increased Sales\nC. Boycotts, Buyalls, Blockchain technology, Charitable donations\nD. Boycotts, Buycotts, Digital technology, Increased Sales',
        "options": 'A. Buycotts, Boycotts, Blockchain technology, Charitable donations\nB. Buycotts, Boycotts, Digital technology, Increased Sales\nC. Boycotts, Buyalls, Blockchain technology, Charitable donations\nD. Boycotts, Buycotts, Digital technology, Increased Sales',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'business_ethics',
    },
]

mmlu_few_shot_clinical_knowledge = [
    {
        "problem": 'The energy for all forms of muscle contraction is provided by:\nA. ATP.\nB. ADP.\nC. phosphocreatine.\nD. oxidative phosphorylation.',
        "options": 'A. ATP.\nB. ADP.\nC. phosphocreatine.\nD. oxidative phosphorylation.',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'clinical_knowledge',
    },
    {
        "problem": 'What is the difference between a male and a female catheter?\nA. Male and female catheters are different colours.\nB. Male catheters are longer than female catheters.\nC. Male catheters are bigger than female catheters.\nD. Female catheters are longer than male catheters.',
        "options": 'A. Male and female catheters are different colours.\nB. Male catheters are longer than female catheters.\nC. Male catheters are bigger than female catheters.\nD. Female catheters are longer than male catheters.',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'clinical_knowledge',
    },
    {
        "problem": 'In the assessment of the hand function which of the following is true?\nA. Abduction of the thumb is supplied by spinal root T2\nB. Opposition of the thumb by opponens policis is supplied by spinal root T1\nC. Finger adduction is supplied by the median nerve\nD. Finger abduction is mediated by the palmar interossei',
        "options": 'A. Abduction of the thumb is supplied by spinal root T2\nB. Opposition of the thumb by opponens policis is supplied by spinal root T1\nC. Finger adduction is supplied by the median nerve\nD. Finger abduction is mediated by the palmar interossei',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'clinical_knowledge',
    },
    {
        "problem": 'How many attempts should you make to cannulate a patient before passing the job on to a senior colleague, according to the medical knowledge of 2020?\nA. 4\nB. 3\nC. 2\nD. 1',
        "options": 'A. 4\nB. 3\nC. 2\nD. 1',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'clinical_knowledge',
    },
    {
        "problem": 'Glycolysis is the name given to the pathway involving the conversion of:\nA. glycogen to glucose-1-phosphate.\nB. glycogen or glucose to fructose.\nC. glycogen or glucose to pyruvate or lactate.\nD. glycogen or glucose to pyruvate or acetyl CoA.',
        "options": 'A. glycogen to glucose-1-phosphate.\nB. glycogen or glucose to fructose.\nC. glycogen or glucose to pyruvate or lactate.\nD. glycogen or glucose to pyruvate or acetyl CoA.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'clinical_knowledge',
    },
]

mmlu_few_shot_college_biology = [
    {
        "problem": 'Which of the following represents an accurate statement concerning arthropods?\nA. They possess an exoskeleton composed primarily of peptidoglycan.\nB. They possess an open circulatory system with a dorsal heart.\nC. They are members of a biologically unsuccessful phylum incapable of exploiting diverse habitats and nutrition sources.\nD. They lack paired, jointed appendages.',
        "options": 'A. They possess an exoskeleton composed primarily of peptidoglycan.\nB. They possess an open circulatory system with a dorsal heart.\nC. They are members of a biologically unsuccessful phylum incapable of exploiting diverse habitats and nutrition sources.\nD. They lack paired, jointed appendages.',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'college_biology',
    },
    {
        "problem": 'In a given population, 1 out of every 400 people has a cancer caused by a completely recessive allele, b. Assuming the population is in Hardy-Weinberg equilibrium, which of the following is the expected proportion of individuals who carry the b allele but are not expected to develop the cancer?\nA. 1/400\nB. 19/400\nC. 20/400\nD. 38/400',
        "options": 'A. 1/400\nB. 19/400\nC. 20/400\nD. 38/400',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'college_biology',
    },
    {
        "problem": "The presence of homologous structures in two different organisms, such as the humerus in the front limb of a human and a bird, indicates that\nA. the human and bird are polyphyletic species\nB. a human's and bird's evolution is convergent\nC. the human and bird belong to a clade\nD. the human and bird developed by analogy",
        "options": "A. the human and bird are polyphyletic species\nB. a human's and bird's evolution is convergent\nC. the human and bird belong to a clade\nD. the human and bird developed by analogy",
        "solution": "The answer is \\boxed{C}.",
        "topic": "college_biology",
    },
    {
        "problem": 'According to the pressure-flow model of movement of phloem contents, photosynthate movement from source to sink is driven by\nA. an ATP-dependent pressure-flow pump\nB. a water-pressure potential gradient\nC. transpiration\nD. apoplastic diffusion',
        "options": 'A. an ATP-dependent pressure-flow pump\nB. a water-pressure potential gradient\nC. transpiration\nD. apoplastic diffusion',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'college_biology',
    },
    {
        "problem": 'Which of the following contain DNA sequences required for the segregation of chromosomes in mitosis and meiosis?\nA. Telomeres\nB. Centromeres\nC. Nucleosomes\nD. Spliceosomes',
        "options": 'A. Telomeres\nB. Centromeres\nC. Nucleosomes\nD. Spliceosomes',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'college_biology',
    },
]

mmlu_few_shot_college_chemistry = [
    {
        "problem": 'Which of the following statements about the lanthanide elements is NOT true?\nA. The most common oxidation state for the lanthanide elements is +3.\nB. Lanthanide complexes often have high coordination numbers (> 6).\nC. All of the lanthanide elements react with aqueous acid to liberate hydrogen.\nD. The atomic radii of the lanthanide elements increase across the period from La to Lu.',
        "options": 'A. The most common oxidation state for the lanthanide elements is +3.\nB. Lanthanide complexes often have high coordination numbers (> 6).\nC. All of the lanthanide elements react with aqueous acid to liberate hydrogen.\nD. The atomic radii of the lanthanide elements increase across the period from La to Lu.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'college_chemistry',
    },
    {
        "problem": 'A 0.217 g sample of HgO (molar mass = 217 g) reacts with excess iodide ions according to the reaction shown above. Titration of the resulting solution requires how many mL of 0.10 M HCl to reach equivalence point?\nA. 1.0 mL\nB. 10 mL\nC. 20 mL\nD. 50 mL',
        "options": 'A. 1.0 mL\nB. 10 mL\nC. 20 mL\nD. 50 mL',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'college_chemistry',
    },
    {
        "problem": 'Predict the number of lines in the EPR spectrum of a solution of 13C-labelled methyl radical (13CH3•), assuming the lines do not overlap.\nA. 4\nB. 3\nC. 6\nD. 24',
        "options": 'A. 4\nB. 3\nC. 6\nD. 24',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'college_chemistry',
    },
    {
        "problem": '3 Cl−(aq) + 4 CrO_4^2−(aq) + 23 H+(aq) → 3 HClO2(aq) + 4 Cr3+(aq) + 10 H2O(l). In the reaction shown above, Cl−(aq) behaves as\nA. an acid\nB. a base\nC. a catalyst\nD. a reducing agent',
        "options": 'A. an acid\nB. a base\nC. a catalyst\nD. a reducing agent',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'college_chemistry',
    },
    {
        "problem": 'Which of the following lists the hydrides of group-14 elements in order of thermal stability, from lowest to highest?\nA. PbH4 < SnH4 < GeH4 < SiH4 < CH4\nB. PbH4 < SnH4 < CH4 < GeH4 < SiH4\nC. CH4 < SiH4 < GeH4 < SnH4 < PbH4\nD. CH4 < PbH4 < GeH4 < SnH4 < SiH4',
        "options": 'A. PbH4 < SnH4 < GeH4 < SiH4 < CH4\nB. PbH4 < SnH4 < CH4 < GeH4 < SiH4\nC. CH4 < SiH4 < GeH4 < SnH4 < PbH4\nD. CH4 < PbH4 < GeH4 < SnH4 < SiH4',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'college_chemistry',
    },
]

mmlu_few_shot_college_computer_science = [
    {
        "problem": 'Which of the following regular expressions is equivalent to (describes the same set of strings as) (a* + b)*(c + d)?\nA. a*(c + d)+ b(c + d)\nB. a*(c + d)* + b(c + d)*\nC. a*(c + d)+ b*(c + d)\nD. (a + b)*c +(a + b)*d',
        "options": 'A. a*(c + d)+ b(c + d)\nB. a*(c + d)* + b(c + d)*\nC. a*(c + d)+ b*(c + d)\nD. (a + b)*c +(a + b)*d',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'college_computer_science',
    },
    {
        "problem": 'A certain pipelined RISC machine has 8 general-purpose registers R0, R1, . . . , R7 and supports the following operations. ADD Rs1, Rs2, Rd Add Rs1 to Rs2 and put the sum in Rd MUL Rs1, Rs2, Rd Multiply Rs1 by Rs2 and put the product in Rd An operation normally takes one cycle; however, an operation takes two cycles if it produces a result required by the immediately following operation in an operation sequence. Consider the expression AB + ABC + BC, where variables A, B, C are located in registers R0, R1, R2. If the contents of these three registers must not be modified, what is the minimum number of clock cycles required for an operation sequence that computes the value of AB + ABC + BC?\nA. 5\nB. 6\nC. 7\nD. 8',
        "options": 'A. 5\nB. 6\nC. 7\nD. 8',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'college_computer_science',
    },
    {
        "problem": 'The Singleton design pattern is used to guarantee that only a single instance of a class may be instantiated. Which of the following is (are) true of this design pattern? I. The Singleton class has a static factory method to provide its instance. II. The Singleton class can be a subclass of another class. III. The Singleton class has a private constructor.\nA. I only\nB. II only\nC. III only\nD. I, II, and III',
        "options": 'A. I only\nB. II only\nC. III only\nD. I, II, and III',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'college_computer_science',
    },
    {
        "problem": 'A compiler generates code for the following assignment statement. G := (A + B) * C - (D + E) * F The target machine has a single accumulator and a single-address instruction set consisting of instructions load, store, add, subtract, and multiply. For the arithmetic operations, the left operand is taken from the accumulator and the result appears in the accumulator. The smallest possible number of instructions in the resulting code is\nA. 5\nB. 6\nC. 7\nD. 9',
        "options": 'A. 5\nB. 6\nC. 7\nD. 9',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'college_computer_science',
    },
    {
        "problem": 'Consider a computer design in which multiple processors, each with a private cache memory, share global memory using a single bus. This bus is the critical system resource. Each processor can execute one instruction every 500 nanoseconds as long as memory references are satisfied by its local cache. When a cache miss occurs, the processor is delayed for an additional 2,000 nanoseconds. During half of this additional delay, the bus is dedicated to serving the cache miss. During the other half, the processor cannot continue, but the bus is free to service requests from other processors. On average, each instruction requires 2 memory references. On average, cache misses occur on 1 percent of references. What proportion of the capacity of the bus would a single processor consume, ignoring delays due to competition from other processors?\nA. 1/50\nB. 1/27\nC. 1/25\nD. 2/27',
        "options": 'A. 1/50\nB. 1/27\nC. 1/25\nD. 2/27',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'college_computer_science',
    },
]

mmlu_few_shot_college_mathematics = [
    {
        "problem": "Let V be the set of all real polynomials p(x). Let transformations T, S be defined on V by T:p(x) -> xp(x) and S:p(x) -> p'(x) = d/dx p(x), and interpret (ST)(p(x)) as S(T(p(x))). Which of the following is true?\nA. ST = 0\nB. ST = T\nC. ST = TS\nD. ST - TS is the identity map of V onto itself.",
        "options": 'A. ST = 0\nB. ST = T\nC. ST = TS\nD. ST - TS is the identity map of V onto itself.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'college_mathematics',
    },
    {
        "problem": 'A tank initially contains a salt solution of 3 grams of salt dissolved in 100 liters of water. A salt solution containing 0.02 grams of salt per liter of water is sprayed into the tank at a rate of 4 liters per minute. The sprayed solution is continually mixed with the salt solution in the tank, and the mixture flows out of the tank at a rate of 4 liters per minute. If the mixing is instantaneous, how many grams of salt are in the tank after 100 minutes have elapsed?\nA. 2\nB. 2 - e^-2\nC. 2 + e^-2\nD. 2 + e^-4',
        "options": 'A. 2\nB. 2 - e^-2\nC. 2 + e^-2\nD. 2 + e^-4',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'college_mathematics',
    },
    {
        "problem": 'Let A be a real 2x2 matrix. Which of the following statements must be true?\r I. All of the entries of A^2 are nonnegative.\r II. The determinant of A^2 is nonnegative.\r III. If A has two distinct eigenvalues, then A^2 has two distinct eigenvalues.\nA. I only\nB. II only\nC. III only\nD. II and III only',
        "options": 'A. I only\nB. II only\nC. III only\nD. II and III only',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'college_mathematics',
    },
    {
        "problem": 'Suppose that f(1 + x) = f(x) for all real x. If f is a polynomial and f(5) = 11, then f(15/2)\nA. -11\nB. 0\nC. 11\nD. 33/2',
        "options": 'A. -11\nB. 0\nC. 11\nD. 33/2',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'college_mathematics',
    },
    {
        "problem": 'Let A be the set of all ordered pairs of integers (m, n) such that 7m + 12n = 22. What is the greatest negative number in the set B = {m + n : (m, n) \\in A}?\nA. -5\nB. -4\nC. -3\nD. -2',
        "options": 'A. -5\nB. -4\nC. -3\nD. -2',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'college_mathematics',
    },
]

mmlu_few_shot_college_medicine = [
    {
        "problem": 'Glucose is transported into the muscle cell:\nA. via protein transporters called GLUT4.\nB. only in the presence of insulin.\nC. via hexokinase.\nD. via monocarbylic acid transporters.',
        "options": 'A. via protein transporters called GLUT4.\nB. only in the presence of insulin.\nC. via hexokinase.\nD. via monocarbylic acid transporters.',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'college_medicine',
    },
    {
        "problem": 'Which of the following is not a true statement?\nA. Muscle glycogen is broken down enzymatically to glucose-1-phosphate\nB. Elite endurance runners have a high proportion of Type I fibres in their leg muscles\nC. Liver glycogen is important in the maintenance of the blood glucose concentration\nD. Insulin promotes glucose uptake by all tissues in the body',
        "options": 'A. Muscle glycogen is broken down enzymatically to glucose-1-phosphate\nB. Elite endurance runners have a high proportion of Type I fibres in their leg muscles\nC. Liver glycogen is important in the maintenance of the blood glucose concentration\nD. Insulin promotes glucose uptake by all tissues in the body',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'college_medicine',
    },
    {
        "problem": 'In a genetic test of a newborn, a rare genetic disorder is found that has X-linked recessive transmission. Which of the following statements is likely true regarding the pedigree of this disorder?\nA. All descendants on the maternal side will have the disorder.\nB. Females will be approximately twice as affected as males in this family.\nC. All daughters of an affected male will be affected.\nD. There will be equal distribution of males and females affected.',
        "options": 'A. All descendants on the maternal side will have the disorder.\nB. Females will be approximately twice as affected as males in this family.\nC. All daughters of an affected male will be affected.\nD. There will be equal distribution of males and females affected.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'college_medicine',
    },
    {
        "problem": 'A high school science teacher fills a 1 liter bottle with pure nitrogen and seals the lid. The pressure is 1.70 atm, and the room temperature is 25°C. Which two variables will both increase the pressure of the system, if all other variables are held constant?\nA. Increasing temperature, increasing moles of gas\nB. Increasing temperature, increasing volume\nC. Decreasing volume, decreasing temperature\nD. Decreasing moles of gas, increasing volume',
        "options": 'A. Increasing temperature, increasing moles of gas\nB. Increasing temperature, increasing volume\nC. Decreasing volume, decreasing temperature\nD. Decreasing moles of gas, increasing volume',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'college_medicine',
    },
    {
        "problem": 'An expected side effect of creatine supplementation is:\nA. muscle weakness.\nB. gain in body mass.\nC. muscle cramps.\nD. loss of electrolytes.',
        "options": 'A. muscle weakness.\nB. gain in body mass.\nC. muscle cramps.\nD. loss of electrolytes.',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'college_medicine',
    },
]

mmlu_few_shot_college_physics = [
    {
        "problem": 'A refracting telescope consists of two converging lenses separated by 100 cm. The eye-piece lens has a focal length of 20 cm. The angular magnification of the telescope is\nA. 4\nB. 5\nC. 6\nD. 20',
        "options": 'A. 4\nB. 5\nC. 6\nD. 20',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'college_physics',
    },
    {
        "problem": 'For which of the following thermodynamic processes is the increase in the internal energy of an ideal gas equal to the heat added to the gas?\nA. Constant temperature\nB. Constant volume\nC. Constant pressure\nD. Adiabatic',
        "options": 'A. Constant temperature\nB. Constant volume\nC. Constant pressure\nD. Adiabatic',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'college_physics',
    },
    {
        "problem": 'One end of a Nichrome wire of length 2L and cross-sectional area A is attached to an end of another Nichrome wire of length L and cross- sectional area 2A. If the free end of the longer wire is at an electric potential of 8.0 volts, and the free end of the shorter wire is at an electric potential of 1.0 volt, the potential at the junction of the two wires is most nearly equal to\nA. 2.4 V\nB. 3.3 V\nC. 4.5 V\nD. 5.7 V',
        "options": 'A. 2.4 V\nB. 3.3 V\nC. 4.5 V\nD. 5.7 V',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'college_physics',
    },
    {
        "problem": 'A refracting telescope consists of two converging lenses separated by 100 cm. The eye-piece lens has a focal length of 20 cm. The angular magnification of the telescope is\nA. 4\nB. 5\nC. 6\nD. 20',
        "options": 'A. 4\nB. 5\nC. 6\nD. 20',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'college_physics',
    },
    {
        "problem": 'The muon decays with a characteristic lifetime of about 10^-6 second into an electron, a muon neutrino, and an electron antineutrino. The muon is forbidden from decaying into an electron and just a single neutrino by the law of conservation of\nA. charge\nB. mass\nC. energy and momentum\nD. lepton number',
        "options": 'A. charge\nB. mass\nC. energy and momentum\nD. lepton number',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'college_physics',
    },
]

mmlu_few_shot_computer_security = [
    {
        "problem": 'SHA-1 has a message digest of\nA. 160 bits\nB. 512 bits\nC. 628 bits\nD. 820 bits',
        "options": 'A. 160 bits\nB. 512 bits\nC. 628 bits\nD. 820 bits',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'computer_security',
    },
    {
        "problem": '_____________ can modify data on your system – so that your system doesn’t run correctly or you can no longer access specific data, or it may even ask for ransom in order to give your access.\nA. IM – Trojans\nB. Backdoor Trojans\nC. Trojan-Downloader\nD. Ransom Trojan',
        "options": 'A. IM – Trojans\nB. Backdoor Trojans\nC. Trojan-Downloader\nD. Ransom Trojan',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'computer_security',
    },
    {
        "problem": 'What is ethical hacking?\nA. "Hacking" ethics so they justify unintended selfish behavior\nB. Hacking systems (e.g., during penetration testing) to expose vulnerabilities so they can be fixed, rather than exploited\nC. Hacking into systems run by those whose ethics you disagree with\nD. A slang term for rapid software development, e.g., as part of hackathons',
        "options": 'A. "Hacking" ethics so they justify unintended selfish behavior\nB. Hacking systems (e.g., during penetration testing) to expose vulnerabilities so they can be fixed, rather than exploited\nC. Hacking into systems run by those whose ethics you disagree with\nD. A slang term for rapid software development, e.g., as part of hackathons',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'computer_security',
    },
    {
        "problem": 'Exploitation of the Heartbleed bug permits\nA. overwriting cryptographic keys in memory\nB. a kind of code injection\nC. a read outside bounds of a buffer\nD. a format string attack',
        "options": 'A. overwriting cryptographic keys in memory\nB. a kind of code injection\nC. a read outside bounds of a buffer\nD. a format string attack',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'computer_security',
    },
    {
        "problem": 'The ____________ is anything which your search engine cannot search.\nA. Haunted web\nB. World Wide Web\nC. Surface web\nD. Deep Web',
        "options": 'A. Haunted web\nB. World Wide Web\nC. Surface web\nD. Deep Web',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'computer_security',
    },
]

mmlu_few_shot_conceptual_physics = [
    {
        "problem": 'Compared with the mass of a uranium atom undergoing fission, the combined masses of the products after fission are\nA. less\nB. more\nC. the same\nD. zero',
        "options": 'A. less\nB. more\nC. the same\nD. zero',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'conceptual_physics',
    },
    {
        "problem": 'Things that are equivalent according to the equivalence principle are\nA. space and time.\nB. a traveling twin and a stay-at-home twin.\nC. gravity and acceleration.\nD. mass and energy.',
        "options": 'A. space and time.\nB. a traveling twin and a stay-at-home twin.\nC. gravity and acceleration.\nD. mass and energy.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'conceptual_physics',
    },
    {
        "problem": 'Colors in a soap bubble result from light\nA. converted to a different frequency\nB. deflection\nC. interference\nD. polarization',
        "options": 'A. converted to a different frequency\nB. deflection\nC. interference\nD. polarization',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'conceptual_physics',
    },
    {
        "problem": 'A model airplane flies slower when flying into the wind and faster with wind at its back. When launched at right angles to the wind a cross wind its groundspeed compared with flying in still air is\nA. the same\nB. greater\nC. less\nD. either greater or less depending on wind speed',
        "options": 'A. the same\nB. greater\nC. less\nD. either greater or less depending on wind speed',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'conceptual_physics',
    },
    {
        "problem": 'Which of these three elements has the most mass per nucleon?\nA. Hydrogen\nB. Iron\nC. Uranium\nD. Same in each',
        "options": 'A. Hydrogen\nB. Iron\nC. Uranium\nD. Same in each',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'conceptual_physics',
    },
]

mmlu_few_shot_econometrics = [
    {
        "problem": 'For a stationary autoregressive process, shocks will\nA. Eventually die away\nB. Persist indefinitely\nC. Grow exponentially\nD. Never occur',
        "options": 'A. Eventually die away\nB. Persist indefinitely\nC. Grow exponentially\nD. Never occur',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'econometrics',
    },
    {
        "problem": 'Consider the following AR(1) model with the disturbances having zero mean and unit variance  yt = 0.2 + 0.4 yt-1 + ut  The (unconditional) mean of y will be given by\nA. 0.2\nB. 0.4\nC. 0.5\nD. 0.33',
        "options": 'A. 0.2\nB. 0.4\nC. 0.5\nD. 0.33',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'econometrics',
    },
    {
        "problem": 'Suppose that a test statistic has associated with it a p-value of 0.08. Which one of the following statements is true?  (i) If the size of the test were exactly 8%, we would be indifferent between rejecting and not rejecting the null hypothesis  (ii) The null would be rejected if a 10% size of test were used  (iii) The null would not be rejected if a 1% size of test were used  (iv) The null would be rejected if a 5% size of test were used.\nA. (ii) and (iv) only\nB. (i) and (iii) only\nC. (i), (ii), and (iii) only\nD. (i), (ii), (iii), and (iv)',
        "options": 'A. (ii) and (iv) only\nB. (i) and (iii) only\nC. (i), (ii), and (iii) only\nD. (i), (ii), (iii), and (iv)',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'econometrics',
    },
    {
        "problem": 'What would be then consequences for the OLS estimator if heteroscedasticity is present in a regression model but ignored?\nA. It will be biased\nB. It will be inconsistent\nC. It will be inefficient\nD. All of (a), (b) and (c) will be true.',
        "options": 'A. It will be biased\nB. It will be inconsistent\nC. It will be inefficient\nD. All of (a), (b) and (c) will be true.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'econometrics',
    },
    {
        "problem": "Suppose now that a researcher wishes to use information criteria to determine the optimal lag length for a VAR. 500 observations are available for the bi-variate VAR, and the values of the determinant of the variance-covariance matrix of residuals are 0.0336, 0.0169, 0.0084, and 0.0062 for 1, 2, 3, and 4 lags respectively. What is the optimal model order according to Akaike's information criterion?\nA. 1 lag\nB. 2 lags\nC. 3 lags\nD. 4 lags",
        "options": 'A. 1 lag\nB. 2 lags\nC. 3 lags\nD. 4 lags',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'econometrics',
    },
]

mmlu_few_shot_electrical_engineering = [
    {
        "problem": 'In an SR latch built from NOR gates, which condition is not allowed\nA. S=0, R=0\nB. S=0, R=1\nC. S=1, R=0\nD. S=1, R=1',
        "options": 'A. S=0, R=0\nB. S=0, R=1\nC. S=1, R=0\nD. S=1, R=1',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'electrical_engineering',
    },
    {
        "problem": 'In a 2 pole lap winding dc machine , the resistance of one conductor is 2Ω and total number of conductors is 100. Find the total resistance\nA. 200Ω\nB. 100Ω\nC. 50Ω\nD. 10Ω',
        "options": 'A. 200Ω\nB. 100Ω\nC. 50Ω\nD. 10Ω',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'electrical_engineering',
    },
    {
        "problem": 'The coil of a moving coil meter has 100 turns, is 40 mm long and 30 mm wide. The control torque is 240*10-6 N-m on full scale. If magnetic flux density is 1Wb/m2 range of meter is\nA. 1 mA.\nB. 2 mA.\nC. 3 mA.\nD. 4 mA.',
        "options": 'A. 1 mA.\nB. 2 mA.\nC. 3 mA.\nD. 4 mA.',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'electrical_engineering',
    },
    {
        "problem": 'Two long parallel conductors carry 100 A. If the conductors are separated by 20 mm, the force per meter of length of each conductor will be\nA. 100 N.\nB. 0.1 N.\nC. 1 N.\nD. 0.01 N.',
        "options": 'A. 100 N.\nB. 0.1 N.\nC. 1 N.\nD. 0.01 N.',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'electrical_engineering',
    },
    {
        "problem": 'A point pole has a strength of 4π * 10^-4 weber. The force in newtons on a point pole of 4π * 1.5 * 10^-4 weber placed at a distance of 10 cm from it will be\nA. 15 N.\nB. 20 N.\nC. 7.5 N.\nD. 3.75 N.',
        "options": 'A. 15 N.\nB. 20 N.\nC. 7.5 N.\nD. 3.75 N.',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'electrical_engineering',
    },
]

mmlu_few_shot_elementary_mathematics = [
    {
        "problem": 'The population of the city where Michelle was born is 145,826. What is the value of the 5 in the number 145,826?\nA. 5 thousands\nB. 5 hundreds\nC. 5 tens\nD. 5 ones',
        "options": 'A. 5 thousands\nB. 5 hundreds\nC. 5 tens\nD. 5 ones',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'elementary_mathematics',
    },
    {
        "problem": 'Olivia used the rule "Add 11" to create the number pattern shown below. 10, 21, 32, 43, 54 Which statement about the number pattern is true?\nA. The 10th number in the pattern will be an even number.\nB. The number pattern will never have two even numbers next to each other.\nC. The next two numbers in the pattern will be an even number then an odd number.\nD. If the number pattern started with an odd number then the pattern would have only odd numbers in it.',
        "options": 'A. The 10th number in the pattern will be an even number.\nB. The number pattern will never have two even numbers next to each other.\nC. The next two numbers in the pattern will be an even number then an odd number.\nD. If the number pattern started with an odd number then the pattern would have only odd numbers in it.',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'elementary_mathematics',
    },
    {
        "problem": 'A total of 30 players will play basketball at a park. There will be exactly 5 players on each team. Which statement correctly explains how to find the number of teams needed?\nA. Add 5 to 30 to find 35 teams.\nB. Divide 30 by 5 to find 6 teams.\nC. Multiply 30 and 5 to find 150 teams.\nD. Subtract 5 from 30 to find 25 teams.',
        "options": 'A. Add 5 to 30 to find 35 teams.\nB. Divide 30 by 5 to find 6 teams.\nC. Multiply 30 and 5 to find 150 teams.\nD. Subtract 5 from 30 to find 25 teams.',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'elementary_mathematics',
    },
    {
        "problem": 'A store sells 107 different colors of paint. They have 25 cans of each color in storage. The number of cans of paint the store has in storage can be found using the expression below. 107 × 25. How many cans of paint does the store have in storage?\nA. 749\nB. 2,675\nC. 2,945\nD. 4,250',
        "options": 'A. 749\nB. 2,675\nC. 2,945\nD. 4,250',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'elementary_mathematics',
    },
    {
        "problem": 'Which expression is equivalent to 5 x 9?\nA. (5 x 4) x (6 x 5)\nB. (5 x 5) + (5 x 4)\nC. (5 x 5) + (5 x 9)\nD. (5 x 9) x (6 x 9)',
        "options": 'A. (5 x 4) x (6 x 5)\nB. (5 x 5) + (5 x 4)\nC. (5 x 5) + (5 x 9)\nD. (5 x 9) x (6 x 9)',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'elementary_mathematics',
    },
]

mmlu_few_shot_formal_logic = [
    {
        "problem": 'Select the best translation into predicate logic: No people drive on Mars.\nA. ~Pd\nB. (∀x)(Px ∨ ~Dx)\nC. (∀x)(Px ⊃ ~Dx)\nD. ~Dp',
        "options": 'A. ~Pd\nB. (∀x)(Px ∨ ~Dx)\nC. (∀x)(Px ⊃ ~Dx)\nD. ~Dp',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'formal_logic',
    },
    {
        "problem": "Select the best translation into predicate logic.George borrows Hector's lawnmower. (g: George; h: Hector; l: Hector's lawnmower; Bxyx: x borrows y from z)\nA. Blgh\nB. Bhlg\nC. Bglh\nD. Bghl",
        "options": 'A. Blgh\nB. Bhlg\nC. Bglh\nD. Bghl',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'formal_logic',
    },
    {
        "problem": 'Select the best English interpretation of the given arguments in predicate logic. Dm (∀x)(Wx ⊃ ~Dx) (∀x)Wx ∨ Ag\t/ (∃x)Ax\nA. Marina is a dancer. Some weaklings are not dancers. Either everything is a weakling or Georgia plays volleyball. So something plays volleyball.\nB. Marina is a dancer. No weakling is a dancer. Everything is either a weakling or plays volleyball. So something plays volleyball.\nC. Marina is a dancer. Some weaklings are not dancers. Everything is either a weakling or plays volleyball. So something plays volleyball.\nD. Marina is a dancer. No weakling is a dancer. Either everything is a weakling or Georgia plays volleyball. So something plays volleyball.',
        "options": 'A. Marina is a dancer. Some weaklings are not dancers. Either everything is a weakling or Georgia plays volleyball. So something plays volleyball.\nB. Marina is a dancer. No weakling is a dancer. Everything is either a weakling or plays volleyball. So something plays volleyball.\nC. Marina is a dancer. Some weaklings are not dancers. Everything is either a weakling or plays volleyball. So something plays volleyball.\nD. Marina is a dancer. No weakling is a dancer. Either everything is a weakling or Georgia plays volleyball. So something plays volleyball.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'formal_logic',
    },
    {
        "problem": 'Construct a complete truth table for the following pairs of propositions. Then, using the truth tables, determine whether the statements are logically equivalent or contradictory. If neither, determine whether they are consistent or inconsistent. Justify your answers. E ⊃ (F · E) and ~E · F\nA. Logically equivalent\nB. Contradictory\nC. Neither logically equivalent nor contradictory, but consistent\nD. Inconsistent',
        "options": 'A. Logically equivalent\nB. Contradictory\nC. Neither logically equivalent nor contradictory, but consistent\nD. Inconsistent',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'formal_logic',
    },
    {
        "problem": 'Which of the given formulas of PL is the best symbolization of the following sentence? Turtles live long lives and are happy creatures, unless they are injured.\nA. (L • H) ≡ I\nB. (L • H) ∨ I\nC. L • (H ∨ I)\nD. L • (H ⊃ R)',
        "options": 'A. (L • H) ≡ I\nB. (L • H) ∨ I\nC. L • (H ∨ I)\nD. L • (H ⊃ R)',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'formal_logic',
    },
]

mmlu_few_shot_global_facts = [
    {
        "problem": 'Which of the following pairs of statements are both true (as of 2019)?\nA. People tend to be optimistic about their own future and the future of their nation or the world.\nB. People tend to be optimistic about their own future but pessimistic about the future of their nation or the world.\nC. People tend to be pessimistic about their own future but optimistic about the future of their nation or the world.\nD. People tend to be pessimistic about their own future and the future of their nation or the world.',
        "options": 'A. People tend to be optimistic about their own future and the future of their nation or the world.\nB. People tend to be optimistic about their own future but pessimistic about the future of their nation or the world.\nC. People tend to be pessimistic about their own future but optimistic about the future of their nation or the world.\nD. People tend to be pessimistic about their own future and the future of their nation or the world.',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'global_facts',
    },
    {
        "problem": 'As of 2019, about what percentage of Americans agree that the state is run for the benefit of all the people?\nA. 31%\nB. 46%\nC. 61%\nD. 76%',
        "options": 'A. 31%\nB. 46%\nC. 61%\nD. 76%',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'global_facts',
    },
    {
        "problem": 'As of 2015, since 1990 forests have ____ in Europe and have ____ in Africa and the Americas.\nA. increased, increased\nB. increased, decreased\nC. decreased, increased\nD. decreased, decreased',
        "options": 'A. increased, increased\nB. increased, decreased\nC. decreased, increased\nD. decreased, decreased',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'global_facts',
    },
    {
        "problem": 'As of 2019, about what percentage of Russians say it is very important to have free media in our country without government/state censorship?\nA. 38%\nB. 53%\nC. 68%\nD. 83%',
        "options": 'A. 38%\nB. 53%\nC. 68%\nD. 83%',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'global_facts',
    },
    {
        "problem": 'As of 2017, how many of the world’s 1-year-old children today have been vaccinated against some disease? *\nA. 80%\nB. 60%\nC. 40%\nD. 20%',
        "options": 'A. 80%\nB. 60%\nC. 40%\nD. 20%',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'global_facts',
    },
]

mmlu_few_shot_high_school_biology = [
    {
        "problem": 'Which of the following is not a way to form recombinant DNA?\nA. Translation\nB. Conjugation\nC. Specialized transduction\nD. Transformation',
        "options": 'A. Translation\nB. Conjugation\nC. Specialized transduction\nD. Transformation',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'high_school_biology',
    },
    {
        "problem": 'A mutation in a bacterial enzyme changed a previously polar amino acid into a nonpolar amino acid. This amino acid was located at a site distant from the enzyme’s active site. How might this mutation alter the enzyme’s substrate specificity?\nA. By changing the enzyme’s pH optimum\nB. By changing the enzyme’s location in the cell\nC. By changing the shape of the protein\nD. An amino acid change away from the active site cannot alter the enzyme’s substrate specificity.',
        "options": 'A. By changing the enzyme’s pH optimum\nB. By changing the enzyme’s location in the cell\nC. By changing the shape of the protein\nD. An amino acid change away from the active site cannot alter the enzyme’s substrate specificity.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'high_school_biology',
    },
    {
        "problem": 'In animal cells, which of the following represents the most likely pathway that a secretory protein takes as it is synthesized in a cell?\nA. Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER\nB. Ribosome–Golgi apparatus–rough ER–secretory vesicle–plasma membrane\nC. Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER\nD. Ribosome–rough ER–Golgi apparatus–secretory vesicle–plasma membrane',
        "options": 'A. Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER\nB. Ribosome–Golgi apparatus–rough ER–secretory vesicle–plasma membrane\nC. Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER\nD. Ribosome–rough ER–Golgi apparatus–secretory vesicle–plasma membrane',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_biology',
    },
    {
        "problem": 'Which of the following is not known to be involved in the control of cell division?\nA. Cyclins\nB. Protein kinases\nC. Checkpoints\nD. Fibroblast cells',
        "options": 'A. Cyclins\nB. Protein kinases\nC. Checkpoints\nD. Fibroblast cells',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_biology',
    },
    {
        "problem": 'Homologous structures are often cited as evidence for the process of natural selection. All of the following are examples of homologous structures EXCEPT\nA. the wings of a bird and the wings of a bat\nB. the flippers of a whale and the arms of a man\nC. the pectoral fins of a porpoise and the flippers of a seal\nD. the forelegs of an insect and the forelimbs of a dog',
        "options": 'A. the wings of a bird and the wings of a bat\nB. the flippers of a whale and the arms of a man\nC. the pectoral fins of a porpoise and the flippers of a seal\nD. the forelegs of an insect and the forelimbs of a dog',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_biology',
    },
]

mmlu_few_shot_high_school_chemistry = [
    {
        "problem": 'Which of the following is considered an acid anhydride?\nA. HCl\nB. H2SO3\nC. SO2\nD. Al(NO3)3',
        "options": 'A. HCl\nB. H2SO3\nC. SO2\nD. Al(NO3)3',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'high_school_chemistry',
    },
    {
        "problem": 'Which of the following is expected to be a polar molecule?\nA. PCl4F\nB. BF3\nC. CO2\nD. Si(CH3)4',
        "options": 'A. PCl4F\nB. BF3\nC. CO2\nD. Si(CH3)4',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'high_school_chemistry',
    },
    {
        "problem": 'From the solubility rules, which of the following is true?\nA. All chlorides, bromides, and iodides are soluble\nB. All sulfates are soluble\nC. All hydroxides are soluble\nD. All ammonium-containing compounds are soluble',
        "options": 'A. All chlorides, bromides, and iodides are soluble\nB. All sulfates are soluble\nC. All hydroxides are soluble\nD. All ammonium-containing compounds are soluble',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_chemistry',
    },
    {
        "problem": 'A new compound is synthesized and found to be a monoprotic acid with a molar mass of 248 g/mol. When 0.0050 mol of this acid are dissolved in 0.500 L of water, the pH is measured as 3.89. What is the pKa of this acid?\nA. 3.89\nB. 7.78\nC. 5.78\nD. 2.33',
        "options": 'A. 3.89\nB. 7.78\nC. 5.78\nD. 2.33',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'high_school_chemistry',
    },
    {
        "problem": 'A solution contains 2.00 mole of acetic acid, CH3COOH, and 1.00 mole of calcium acetate, Ca(CH3COO)2. The solution is able to resist the addition of a small amount of strong acid or strong base with only minor changes in the pH of the solution. Larger quantities of strong acid or strong base can cause a significant change in pH. How many moles of nitric acid, HNO3, may be added before the pH begins to change significantly?\nA. 0.500 mole\nB. 1.00 mole\nC. 2.00 mole\nD. 3.00 mole',
        "options": 'A. 0.500 mole\nB. 1.00 mole\nC. 2.00 mole\nD. 3.00 mole',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'high_school_chemistry',
    },
]

mmlu_few_shot_high_school_computer_science = [
    {
        "problem": 'Which of the following is an example of the use of a device on the Internet of Things (IoT) ?\nA. A car alerts a driver that it is about to hit an object.\nB. A hiker uses a G P S watch to keep track of her position.\nC. A refrigerator orders milk from an online delivery service when the milk in the refrigerator is almost gone.\nD. A runner uses a watch with optical sensors to monitor his heart rate.',
        "options": 'A. A car alerts a driver that it is about to hit an object.\nB. A hiker uses a G P S watch to keep track of her position.\nC. A refrigerator orders milk from an online delivery service when the milk in the refrigerator is almost gone.\nD. A runner uses a watch with optical sensors to monitor his heart rate.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'high_school_computer_science',
    },
    {
        "problem": "Many Web browsers allow users to open anonymous windows. During a browsing session in an anonymous window, the browser does not record a browsing history or a list of downloaded files. When the anonymous window is exited, cookies created during the session are deleted. Which of the following statements about browsing sessions in an anonymous window is true?\nA. The activities of a user browsing in an anonymous window will not be visible to people who monitor the user's network, such as the system administrator.\nB. Items placed in a Web store's shopping cart for future purchase during the anonymous browsing session will not be saved on the user's computer.\nC. A user will not be able to log in to e-mail or social media accounts during the anonymous browsing session.\nD. A user browsing in an anonymous window will be protected from viruses launched from any web sites visited or files downloaded.",
        "options": "A. The activities of a user browsing in an anonymous window will not be visible to people who monitor the user's network, such as the system administrator.\nB. Items placed in a Web store's shopping cart for future purchase during the anonymous browsing session will not be saved on the user's computer.\nC. A user will not be able to log in to e-mail or social media accounts during the anonymous browsing session.\nD. A user browsing in an anonymous window will be protected from viruses launched from any web sites visited or files downloaded.",
        "solution": "The answer is \\boxed{B}.",
        "topic": "high_school_computer_science",
    },
    {
        "problem": 'What is the output of "abc"[::-1] in Python 3?\nA. Error\nB. abc\nC. cba\nD. c',
        "options": "A. Error\nB. abc\nC. cba\nD. c",
        "solution": "The answer is \\boxed{C}.",
        "topic": "high_school_computer_science",
    },
    {
        "problem": 'In the program below, the initial value of x is 5 and the initial value of y is 10.  IF (X < O)  {  DISPLAY ("Foxtrot")  }  ELSE  {  IF (X > y)  {    DISPLAY ("Hotel")  }  ELSE   {    IF (y > O)    {    DISPLAY ("November")    }    ELSE    {    DISPLAY ("Yankee")    }  }  }    What is displayed as a result of running the program?\nA. Foxtrot\nB. Hotel\nC. November\nD. Yankee',
        "options": 'A. Foxtrot\nB. Hotel\nC. November\nD. Yankee',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'high_school_computer_science',
    },
    {
        "problem": 'A list of numbers has n elements, indexed from 1 to n. The following algorithm is intended to display the number of elements in the list that have a value greater than 100. The algorithm uses the variables count and position. Steps 3 and 4 are missing.  Step 1: Set count to 0 and position to 1.  Step 2: If the value of the element at index position is greater  than 100, increase the value of count by 1.  Step 3: (missing step)  Step 4: (missing step)  Step 5: Display the value of count.  Which of the following could be used to replace steps 3 and 4 so that the algorithm works as intended?\nA. Step 3: Increase the value of position by 1.\n Step 4: Repeat steps 2 and 3 until the value of count is greater than 100.\nB. Step 3: Increase the value of position by 1.\n Step 4: Repeat steps 2 and 3 until t he value of position is greater than n.\nC. Step 3: Repeat step 2 until the value of count is greater than 100.\n Step 4: Increase the value of position by 1.\nD. Step 3: Repeat step 2 until the value of position is greater than n.\n Step 4: Increase the value of count by 1.',
        "options": 'A. Step 3: Increase the value of position by 1.\n Step 4: Repeat steps 2 and 3 until the value of count is greater than 100.\nB. Step 3: Increase the value of position by 1.\n Step 4: Repeat steps 2 and 3 until t he value of position is greater than n.\nC. Step 3: Repeat step 2 until the value of count is greater than 100.\n Step 4: Increase the value of position by 1.\nD. Step 3: Repeat step 2 until the value of position is greater than n.\n Step 4: Increase the value of count by 1.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_computer_science',
    },
]

mmlu_few_shot_high_school_european_history = [
    {
        "problem": "This question refers to the following information. The following excerpt is from a pamphlet. You will do me the justice to remember, that I have always strenuously supported the Right of every man to his own opinion, however different that opinion might be to mine. He who denies to another this right, makes a slave of himself to his present opinion, because he precludes himself the right of changing it. The most formidable weapon against errors of every kind is Reason. I have never used any other, and I trust I never shall. The circumstance that has now taken place in France of the total abolition of the whole national order of priesthood, and of everything appertaining to compulsive systems of religion, and compulsive articles of faith, has not only precipitated my intention, but rendered a work of this kind exceedingly necessary, lest in the general wreck of superstition, of false systems of government, and false theology, we lose sight of morality, of humanity, and of the theology that is true. I believe in one God, and no more; and I hope for happiness beyond this life. I believe in the equality of man; and I believe that religious duties consist in doing justice, loving mercy, and endeavoring to make our fellow-creatures happy. I do not believe in the creed professed by the Jewish church, by the Roman church, by the Greek church, by the Turkish church, by the Protestant church, nor by any church that I know of. My own mind is my own church. All national institutions of churches, whether Jewish, Christian or Turkish, appear to me no other than human inventions, set up to terrify and enslave mankind, and monopolize power and profit. I do not mean by this declaration to condemn those who believe otherwise; they have the same right to their belief as I have to mine. —Thomas Paine, The Age of Reason, 1794–1795 Which of the following Enlightenment philosophes designed a system of checks and balances for government to avoid abuses of power?\nA. Jean Jacques Rousseau\nB. Baron Montesquieu\nC. Mary Wollstonecraft\nD. Adam Smith",
        "options": "A. Jean Jacques Rousseau\nB. Baron Montesquieu\nC. Mary Wollstonecraft\nD. Adam Smith",
        "solution": "The answer is \\boxed{B}.",
        "topic": "high_school_european_history",
    },
    {
        "problem": "This question refers to the following information. Read the following excerpt. The revolutionary seed had penetrated into every country and spread more or less. It was greatly developed under the régime of the military despotism of Bonaparte. His conquests displaced a number of laws, institutions, and customs; broke through bonds sacred among all nations, strong enough to resist time itself; which is more than can be said of certain benefits conferred by these innovators. The monarchs will fulfil the duties imposed upon them by Him who, by entrusting them with power, has charged them to watch over the maintenance of justice, and the rights of all, to avoid the paths of error, and tread firmly in the way of truth. Placed beyond the passions which agitate society, it is in days of trial chiefly that they are called upon to despoil realities of their false appearances, and to show themselves as they are, fathers invested with the authority belonging by right to the heads of families, to prove that, in days of mourning, they know how to be just, wise, and therefore strong, and that they will not abandon the people whom they ought to govern to be the sport of factions, to error and its consequences, which must involve the loss of society. Union between the monarchs is the basis of the policy which must now be followed to save society from total ruin. . . . Let them not confound concessions made to parties with the good they ought to do for their people, in modifying, according to their recognized needs, such branches of the administration as require it. Let them be just, but strong; beneficent, but strict. Let them maintain religious principles in all their purity, and not allow the faith to be attacked and morality interpreted according to the social contract or the visions of foolish sectarians. Let them suppress Secret Societies; that gangrene of society. —Klemens von Metternich, Political Confession of Faith, 1820 Which of the following was the greatest cause of the fears expressed by Metternich in the document above?\nA. The ideas of personal liberty and nationalism conceived during the Enlightenment resulted in radical revolutions that could spread throughout Europe.\nB. The conquest of Europe by Napoleon led to the creation of new factions and shifted the European balance of power.\nC. The power of monarchs had grown to the point where it needed to be checked by other powers within each nation or domination of civilians would occur.\nD. The rising and falling economic cycle of the newly emerging capitalist economy could lead to civilian unrest that must be suppressed.",
        "options": "A. The ideas of personal liberty and nationalism conceived during the Enlightenment resulted in radical revolutions that could spread throughout Europe.\nB. The conquest of Europe by Napoleon led to the creation of new factions and shifted the European balance of power.\nC. The power of monarchs had grown to the point where it needed to be checked by other powers within each nation or domination of civilians would occur.\nD. The rising and falling economic cycle of the newly emerging capitalist economy could lead to civilian unrest that must be suppressed.",
        "solution": "The answer is \\boxed{A}.",
        "topic": "high_school_european_history",
    },
    {
        "problem": 'This question refers to the following information. In Russia there was nothing going on well, and [Souvarine] was in despair over the news he had received. His old companions were all turning to the politicians; the famous Nihilists who made Europe tremble-sons of village priests, of the lower middle class, of tradesmen-could not rise above the idea of national liberation, and seemed to believe that the world would be delivered-when they had killed their despot&… "Foolery! They\'ll never get out of it with their foolery." Then, lowering his voice still more, in a few bitter words he described his old dream of fraternity. He had renounced his rank and his fortune; he had gone among workmen, only in the hope of seeing at last the foundation of a new society of labour in common. All the sous in his pockets had long gone to the urchins of the settlement; he had been as tender as a brother with the colliers, smiling at their suspicion, winning them over by his quiet workmanlike ways and his dislike of chattering. But decidedly the fusion had not taken place. His voice changed, his eyes grew bright, he fixed them on étienne, directly addressing him: "Now, do you understand that? These hatworkers at Marseilles who have won the great lottery prize of a hundred thousand francs have gone off at once and invested it, declaring that they are going to live without doing anything! Yes, that is your idea, all of you French workmen; you want to unearth a treasure in order to devour it alone afterwards in some lazy, selfish corner. You may cry out as much as you like against the rich, you haven\'t got courage enough to give back to the poor the money that luck brings you. You will never be worthy of happiness as long as you own anything, and your hatred of the bourgeois proceeds solely from an angry desire to be bourgeois yourselves in their place." émile Zola, French writer, Germinal, 1885 The passage displays the direct concern for the welfare of the working classes that was typically a part of which movement?\nA. Capitalist\nB. Scientific\nC. Communist\nD. Existentialist',
        "options": "A. Capitalist\nB. Scientific\nC. Communist\nD. Existentialist",
        "solution": "The answer is \\boxed{C}.",
        "topic": "high_school_european_history",
    },
    {
        "problem": "This question refers to the following information. The excerpts below are from the Navigation Acts of 1651. [A]fter the first day of December, one thousand six hundred fifty and one, and from thence forwards, no goods or commodities whatsoever of the growth, production or manufacture of Asia, Africa or America, or of any part thereof; or of any islands belonging to them, or which are described or laid down in the usual maps or cards of those places, as well of the English plantations as others, shall be imported or brought into this Commonwealth of England, or into Ireland, or any other lands, islands, plantations, or territories to this Commonwealth belonging, or in their possession, in any other ship or ships, vessel or vessels whatsoever, but only in such as do truly and without fraud belong only to the people of this Commonwealth, or the plantations thereof, as the proprietors or right owners thereof; and whereof the master and mariners are also of the people of this Commonwealth, under the penalty of the forfeiture and loss of all the goods that shall be imported contrary to this act, , , , [N]o goods or commodities of the growth, production, or manufacture of Europe, or of any part thereof, shall after the first day of December, one thousand six hundred fifty and one, be imported or brought into this Commonwealth of England, or any other lands or territories to this Commonwealth belonging, or in their possession, in any ship or ships, vessel or vessels whatsoever, but in such as do truly and without fraud belong only to the people of this Commonwealth, and in no other, except only such foreign ships and vessels as do truly and properly belong to the people of that country or place, of which the said goods are the growth, production or manufacture. Which of the following best describes the outcome of the Navigation Acts of 1651?\nA. They served as a catalyst for the growth of English shipping and overseas trade, but did little to limit the prospects of the Dutch in the seventeenth century.\nB. They brought about almost immediate hardships for the Dutch economy as their dominance of overseas trade quickly ended.\nC. They were rescinded during the restoration of the Stuarts as they sought normal diplomatic relations with the Dutch so not as to need Parliament's financial support for war.\nD. They led to nearly a century of recurrent war between England and the Netherlands, which would not end until after American independence.",
        "options": "A. They served as a catalyst for the growth of English shipping and overseas trade, but did little to limit the prospects of the Dutch in the seventeenth century.\nB. They brought about almost immediate hardships for the Dutch economy as their dominance of overseas trade quickly ended.\nC. They were rescinded during the restoration of the Stuarts as they sought normal diplomatic relations with the Dutch so not as to need Parliament's financial support for war.\nD. They led to nearly a century of recurrent war between England and the Netherlands, which would not end until after American independence.",
        "solution": "The answer is \\boxed{A}.",
        "topic": "high_school_european_history",
    },
    {
        "problem": "This question refers to the following information. Albeit the king's Majesty justly and rightfully is and ought to be the supreme head of the Church of England, and so is recognized by the clergy of this realm in their convocations, yet nevertheless, for corroboration and confirmation thereof, and for increase of virtue in Christ's religion within this realm of England, and to repress and extirpate all errors, heresies, and other enormities and abuses heretofore used in the same, be it enacted, by authority of this present Parliament, that the king, our sovereign lord, his heirs and successors, kings of this realm, shall be taken, accepted, and reputed the only supreme head in earth of the Church of England, called Anglicans Ecclesia; and shall have and enjoy, annexed and united to the imperial crown of this realm, as well the title and style thereof, as all honors, dignities, preeminences, jurisdictions, privileges, authorities, immunities, profits, and commodities to the said dignity of the supreme head of the same Church belonging and appertaining; and that our said sovereign lord, his heirs and successors, kings of this realm, shall have full power and authority from time to time to visit, repress, redress, record, order, correct, restrain, and amend all such errors, heresies, abuses, offenses, contempts, and enormities, whatsoever they be, which by any manner of spiritual authority or jurisdiction ought or may lawfully be reformed, repressed, ordered, redressed, corrected, restrained, or amended, most to the pleasure of Almighty God, the increase of virtue in Christ's religion, and for the conservation of the peace, unity, and tranquility of this realm; any usage, foreign land, foreign authority, prescription, or any other thing or things to the contrary hereof notwithstanding. English Parliament, Act of Supremacy, 1534 From the passage, one may infer that the English Parliament wished to argue that the Act of Supremacy would\nA. give the English king a new position of authority\nB. give the position of head of the Church of England to Henry VIII alone and exclude his heirs\nC. establish Calvinism as the one true theology in England\nD. end various forms of corruption plaguing the Church in England",
        "options": "A. give the English king a new position of authority\nB. give the position of head of the Church of England to Henry VIII alone and exclude his heirs\nC. establish Calvinism as the one true theology in England\nD. end various forms of corruption plaguing the Church in England",
        "solution": "The answer is \\boxed{D}.",
        "topic": "high_school_european_history",
    },
]

mmlu_few_shot_high_school_geography = [
    {
        "problem": "The rate of natural increase of a population is found by subtracting the\nA. crude death rate from the crude birth date.\nB. crude birth rate from the crude death rate.\nC. doubling time from the crude birth rate.\nD. fertility rate from the crude death rate.",
        "options": "A. crude death rate from the crude birth date.\nB. crude birth rate from the crude death rate.\nC. doubling time from the crude birth rate.\nD. fertility rate from the crude death rate.",
        "solution": "The answer is \\boxed{A}.",
        "topic": "high_school_geography",
    },
    {
        "problem": "During the third stage of the demographic transition model, which of the following is true?\nA. Birth rates increase and population growth rate is less rapid.\nB. Birth rates decline and population growth rate is less rapid.\nC. Birth rates increase and population growth rate increases.\nD. Birth rates decrease and population growth rate increases.",
        "options": "A. Birth rates increase and population growth rate is less rapid.\nB. Birth rates decline and population growth rate is less rapid.\nC. Birth rates increase and population growth rate increases.\nD. Birth rates decrease and population growth rate increases.",
        "solution": "The answer is \\boxed{B}.",
        "topic": "high_school_geography",
    },
    {
        "problem": "Which of the following statements is NOT accurate regarding the services provided by local governments in the United States?\nA. Duplication of efforts occurs often.\nB. Social problems of the central city spill over into the surrounding residential suburbs.\nC. Inefficiency in providing services occurs often.\nD. One neighborhood's efforts to reduce pollution are always supported by neighboring communities.",
        "options": "A. Duplication of efforts occurs often.\nB. Social problems of the central city spill over into the surrounding residential suburbs.\nC. Inefficiency in providing services occurs often.\nD. One neighborhood's efforts to reduce pollution are always supported by neighboring communities.",
        "solution": "The answer is \\boxed{D}.",
        "topic": "high_school_geography",
    },
    {
        "problem": "The practice of hiring a foreign third-party service provider to run an operation is called\nA. outsourcing.\nB. offshoring.\nC. maquiladoras.\nD. locational interdependence.",
        "options": "A. outsourcing.\nB. offshoring.\nC. maquiladoras.\nD. locational interdependence.",
        "solution": "The answer is \\boxed{B}.",
        "topic": "high_school_geography",
    },
    {
        "problem": "Which one of the following items is an example of nonmaterial culture?\nA. Dove soap\nB. Dove candy bar\nC. Dove symbol\nD. A dove (bird)",
        "options": "A. Dove soap\nB. Dove candy bar\nC. Dove symbol\nD. A dove (bird)",
        "solution": "The answer is \\boxed{C}.",
        "topic": "high_school_geography",
    },
]

mmlu_few_shot_high_school_government_and_politics = [
    {
        "problem": 'Uncertainty over the limits to presidential power is caused primarily by the fact that\nA. the constitutional definition of those powers is broad and unspecific\nB. most people agree that the Constitution places too many limits on presidential power\nC. the Supreme Court consistently refuses to rule on cases concerning presidential powers\nD. constitutional amendments have greatly increased presidential powers',
        "options": 'A. the constitutional definition of those powers is broad and unspecific\nB. most people agree that the Constitution places too many limits on presidential power\nC. the Supreme Court consistently refuses to rule on cases concerning presidential powers\nD. constitutional amendments have greatly increased presidential powers',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'high_school_government_and_politics',
    },
    {
        "problem": 'The term "budget deficit" refers to the\nA. annual increase in federal spending on the military\nB. amount of interest on the national debt\nC. difference between the initial budget proposals made by the president and Congress\nD. amount the government spends in excess of its revenues',
        "options": 'A. annual increase in federal spending on the military\nB. amount of interest on the national debt\nC. difference between the initial budget proposals made by the president and Congress\nD. amount the government spends in excess of its revenues',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_government_and_politics',
    },
    {
        "problem": 'Which of the following cases established the precedent that a defendant must be informed of the right to remain silent, the right to a lawyer, and protection from self-incrimination?\nA. Weeks v. United States\nB. Betts v. Brady\nC. Mapp v. Ohio\nD. Miranda v. Arizona',
        "options": 'A. Weeks v. United States\nB. Betts v. Brady\nC. Mapp v. Ohio\nD. Miranda v. Arizona',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_government_and_politics',
    },
    {
        "problem": "Which of the following statements about cabinet departments is FALSE?\nA. They are established by the legislative branch.\nB. Their members often don't have much influence over presidential decisions.\nC. They cannot all be run by leaders who belong to the same political party the president does.\nD. Not every federal agency is a cabinet department.",
        "options": "A. They are established by the legislative branch.\nB. Their members often don't have much influence over presidential decisions.\nC. They cannot all be run by leaders who belong to the same political party the president does.\nD. Not every federal agency is a cabinet department.",
        "solution": "The answer is \\boxed{C}.",
        "topic": "high_school_government_and_politics",
    },
    {
        "problem": "Which of the following best states an argument made by James Madison in The Federalist number 10?\nA. Honest politicians can prevent factions from developing.\nB. Factions are more likely to occur in large republics than in small ones.\nC. The negative effects of factionalism can be reduced by a republican government.\nD. Free elections are the people's best defense against factionalism.",
        "options": "A. Honest politicians can prevent factions from developing.\nB. Factions are more likely to occur in large republics than in small ones.\nC. The negative effects of factionalism can be reduced by a republican government.\nD. Free elections are the people's best defense against factionalism.",
        "solution": "The answer is \\boxed{C}.",
        "topic": "high_school_government_and_politics",
    },
]

mmlu_few_shot_high_school_macroeconomics = [
    {
        "problem": 'Which of the following is not included in the U.S. GDP?\nA. The U.S. military opens a new base in a foreign country with 1000 U.S. personnel.\nB. Japanese consumers buy thousands of CDs produced in the United States.\nC. An American pop singer performs a sold-out concert in Paris.\nD. A French theatrical production tours dozens of American cities.',
        "options": 'A. The U.S. military opens a new base in a foreign country with 1000 U.S. personnel.\nB. Japanese consumers buy thousands of CDs produced in the United States.\nC. An American pop singer performs a sold-out concert in Paris.\nD. A French theatrical production tours dozens of American cities.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'high_school_macroeconomics',
    },
    {
        "problem": 'The short-run Phillips curve indicates a\nA. direct relation between unemployment and inflation\nB. direct relation between price and quantity demanded\nC. inverse relation between price and quantity demanded\nD. inverse relation between unemployment and inflation',
        "options": 'A. direct relation between unemployment and inflation\nB. direct relation between price and quantity demanded\nC. inverse relation between price and quantity demanded\nD. inverse relation between unemployment and inflation',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_macroeconomics',
    },
    {
        "problem": 'A federal deficit occurs when\nA. exports exceed imports.\nB. imports exceed exports.\nC. federal tax collections exceed spending.\nD. federal spending exceeds federal tax revenues.',
        "options": 'A. exports exceed imports.\nB. imports exceed exports.\nC. federal tax collections exceed spending.\nD. federal spending exceeds federal tax revenues.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_macroeconomics',
    },
    {
        "problem": 'Holding all else equal which of the following monetary policies would be used to boost U.S. exports?\nA. Increasing the discount rate\nB. Increasing the reserve ratio\nC. Buying government securities\nD. Lowering tariffs',
        "options": 'A. Increasing the discount rate\nB. Increasing the reserve ratio\nC. Buying government securities\nD. Lowering tariffs',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'high_school_macroeconomics',
    },
    {
        "problem": 'Which of the following policies best describes supply-side fiscal policy?\nA. An increase in the money supply\nB. Increased government spending\nC. Lower taxes on research and development of new technology\nD. Higher taxes on household income',
        "options": 'A. An increase in the money supply\nB. Increased government spending\nC. Lower taxes on research and development of new technology\nD. Higher taxes on household income',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'high_school_macroeconomics',
    },
]

mmlu_few_shot_high_school_mathematics = [
    {
        "problem": 'Joe was in charge of lights for a dance. The red light blinks every two seconds, the yellow light every three seconds, and the blue light every five seconds. If we include the very beginning and very end of the dance, how many times during a seven minute dance will all the lights come on at the same time? (Assume that all three lights blink simultaneously at the very beginning of the dance.)\nA. 3\nB. 15\nC. 6\nD. 5',
        "options": 'A. 3\nB. 15\nC. 6\nD. 5',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'high_school_mathematics',
    },
    {
        "problem": 'Five thousand dollars compounded annually at an $x\\%$ interest rate takes six years to double. At the same interest rate, how many years will it take $\\$300$ to grow to $\\$9600$?\nA. 12\nB. 1\nC. 30\nD. 5',
        "options": 'A. 12\nB. 1\nC. 30\nD. 5',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'high_school_mathematics',
    },
    {
        "problem": 'The variable $x$ varies directly as the square of $y$, and $y$ varies directly as the cube of $z$. If $x$ equals $-16$ when $z$ equals 2, what is the value of $x$ when $z$ equals $\\frac{1}{2}$?\nA. -1\nB. 16\nC. -\\frac{1}{256}\nD. \\frac{1}{16}',
        "options": 'A. -1\nB. 16\nC. -\\frac{1}{256}\nD. \\frac{1}{16}',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'high_school_mathematics',
    },
    {
        "problem": 'Simplify and write the result with a rational denominator: $$\\sqrt{\\sqrt[3]{\\sqrt{\\frac{1}{729}}}}$$\nA. \\frac{3\\sqrt{3}}{3}\nB. \\frac{1}{3}\nC. \\sqrt{3}\nD. \\frac{\\sqrt{3}}{3}',
        "options": 'A. \\frac{3\\sqrt{3}}{3}\nB. \\frac{1}{3}\nC. \\sqrt{3}\nD. \\frac{\\sqrt{3}}{3}',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_mathematics',
    },
    {
        "problem": 'Ten students take a biology test and receive the following scores: 45, 55, 50, 70, 65, 80, 40, 90, 70, 85. What is the mean of the students’ test scores?\nA. 55\nB. 60\nC. 62\nD. 65',
        "options": 'A. 55\nB. 60\nC. 62\nD. 65',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_mathematics',
    },
]

mmlu_few_shot_high_school_microeconomics = [
    {
        "problem": 'In a competitive labor market for housepainters, which of the following would increase the demand for housepainters?\nA. An effective minimum wage imposed on this labor market.\nB. An increase in the price of gallons of paint.\nC. An increase in the construction of new houses.\nD. An increase in the price of mechanical painters so long as the output effect exceeds the substitution effect.',
        "options": 'A. An effective minimum wage imposed on this labor market.\nB. An increase in the price of gallons of paint.\nC. An increase in the construction of new houses.\nD. An increase in the price of mechanical painters so long as the output effect exceeds the substitution effect.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'high_school_microeconomics',
    },
    {
        "problem": 'If the government subsidizes producers in a perfectly competitive market, then\nA. the demand for the product will increase\nB. the demand for the product will decrease\nC. the consumer surplus will increase\nD. the consumer surplus will decrease',
        "options": 'A. the demand for the product will increase\nB. the demand for the product will decrease\nC. the consumer surplus will increase\nD. the consumer surplus will decrease',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'high_school_microeconomics',
    },
    {
        "problem": 'The concentration ratio for a monopoly is\nA. 0\nB. 5\nC. 10\nD. 100',
        "options": 'A. 0\nB. 5\nC. 10\nD. 100',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_microeconomics',
    },
    {
        "problem": 'Which of the following is true of a price floor?\nA. The price floor shifts the demand curve to the left.\nB. An effective floor creates a shortage of the good.\nC. The price floor shifts the supply curve of the good to the right.\nD. To be an effective floor, it must be set above the equilibrium price.',
        "options": 'A. The price floor shifts the demand curve to the left.\nB. An effective floor creates a shortage of the good.\nC. The price floor shifts the supply curve of the good to the right.\nD. To be an effective floor, it must be set above the equilibrium price.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_microeconomics',
    },
    {
        "problem": 'Which of the following is necessarily a characteristic of oligopoly?\nA. Free entry into and exit from the market\nB. A few large producers\nC. One producer of a good with no close substitutes\nD. A homogenous product',
        "options": 'A. Free entry into and exit from the market\nB. A few large producers\nC. One producer of a good with no close substitutes\nD. A homogenous product',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'high_school_microeconomics',
    },
]

mmlu_few_shot_high_school_physics = [
    {
        "problem": 'Which of the following conditions will ensure that angular momentum is conserved? I. Conservation of linear momentum II. Zero net external force III. Zero net external torque\nA. I and II only\nB. I and III only\nC. II and III only\nD. III only',
        "options": 'A. I and II only\nB. I and III only\nC. II and III only\nD. III only',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_physics',
    },
    {
        "problem": 'A pipe full of air is closed at one end. A standing wave is produced in the pipe, causing the pipe to sound a note. Which of the following is a correct statement about the wave’s properties at the closed end of the pipe?\nA. The pressure is at a node, but the particle displacement is at an antinode.\nB. The pressure is at an antinode, but the particle displacement is at a node.\nC. The pressure and the particle displacement are both at nodes.\nD. The pressure and the particle displacement are both at antinodes.',
        "options": 'A. The pressure is at a node, but the particle displacement is at an antinode.\nB. The pressure is at an antinode, but the particle displacement is at a node.\nC. The pressure and the particle displacement are both at nodes.\nD. The pressure and the particle displacement are both at antinodes.',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'high_school_physics',
    },
    {
        "problem": 'A photocell of work function ϕ = 2eV is connected to a resistor in series. Light of frequency f = 1 × 10^15 Hz hits a metal plate of the photocell. If the power of the light is P = 100 W, what is the current through the resistor?\nA. 2:00 AM\nB. 6:00 AM\nC. 12:00 AM\nD. 24 A',
        "options": 'A. 2:00 AM\nB. 6:00 AM\nC. 12:00 AM\nD. 24 A',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_physics',
    },
    {
        "problem": 'A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?\nA. 10 W\nB. 30 W\nC. 60 W\nD. 240 W',
        "options": 'A. 10 W\nB. 30 W\nC. 60 W\nD. 240 W',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_physics',
    },
    {
        "problem": 'A point charge, Q = +1 mC, is fixed at the origin. How much work is required to move a charge, Q = +8 µC, from the point (0, 4 meters) to the point (3 meters, 0)?\nA. 3.5 J\nB. 6.0 J\nC. 22.5 J\nD. 40 J',
        "options": 'A. 3.5 J\nB. 6.0 J\nC. 22.5 J\nD. 40 J',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'high_school_physics',
    },
]

mmlu_few_shot_high_school_psychology = [
    {
        "problem": 'Ani believes that her attitudes and behavior play a central role in what happens to her. Such a belief is likely to be associated with\nA. a strong superego.\nB. low self-esteem.\nC. low self-efficacy.\nD. an internal locus of control.',
        "options": 'A. a strong superego.\nB. low self-esteem.\nC. low self-efficacy.\nD. an internal locus of control.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_psychology',
    },
    {
        "problem": "According to Caplan's model of consultee-centered case consultation, the consultant is primarily interested in\nA. identifying the causes and solutions of the client's presenting problems\nB. identifying and eliminating the causes of the consultee's difficulties in handling a problem\nC. establishing a hierarchy of authority to enable effective decision making\nD. presenting a single, well-defined and unambiguous course of action for the consultant to overcome skills deficits",
        "options": "A. identifying the causes and solutions of the client's presenting problems\nB. identifying and eliminating the causes of the consultee's difficulties in handling a problem\nC. establishing a hierarchy of authority to enable effective decision making\nD. presenting a single, well-defined and unambiguous course of action for the consultant to overcome skills deficits",
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'high_school_psychology',
    },
    {
        "problem": 'While swimming in the ocean, Ivan is frightened by a dark shadow in the water even before he has the chance to identify what the shadow is. The synaptic connections taking place during this incident of fright are best described by which of the following?\nA. Messages are sent from the thalamus directly to the amygdala.\nB. Messages are sent from the thalamus to the "what" and "where" pathways.\nC. Messages are sent from the parasympathetic nervous system to the cerebral cortex.\nD. Messages are sent from the frontal lobes to the pituitary gland.',
        "options": 'A. Messages are sent from the thalamus directly to the amygdala.\nB. Messages are sent from the thalamus to the "what" and "where" pathways.\nC. Messages are sent from the parasympathetic nervous system to the cerebral cortex.\nD. Messages are sent from the frontal lobes to the pituitary gland.',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'high_school_psychology',
    },
    {
        "problem": 'According to the Individuals with Disabilities Education Improvement Act, which of the following must an educational agency do before it changes the educational placement of a student with a disability?\nA. Give the child a trial period in the new environment\nB. Notify the parents in writing\nC. Obtain school board approval\nD. Obtain parental consent',
        "options": 'A. Give the child a trial period in the new environment\nB. Notify the parents in writing\nC. Obtain school board approval\nD. Obtain parental consent',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'high_school_psychology',
    },
    {
        "problem": 'Pascale is interested in the processing strategies children use to learn new information. Pascale would best be classified as what type of psychologist?\nA. sociocultural\nB. clinical\nC. cognitive\nD. behaviorist',
        "options": 'A. sociocultural\nB. clinical\nC. cognitive\nD. behaviorist',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'high_school_psychology',
    },
]

mmlu_few_shot_high_school_statistics = [
    {
        "problem": 'Which of the following is a correct statement about correlation?\nA. If the slope of the regression line is exactly 1, then the correlation is exactly 1.\nB. If the correlation is 0, then the slope of the regression line is undefined.\nC. Switching which variable is called x and which is called y changes the sign of the correlation.\nD. The correlation r is equal to the slope of the regression line when z-scores for the y-variable are plotted against z-scores for the x-variable.',
        "options": 'A. If the slope of the regression line is exactly 1, then the correlation is exactly 1.\nB. If the correlation is 0, then the slope of the regression line is undefined.\nC. Switching which variable is called x and which is called y changes the sign of the correlation.\nD. The correlation r is equal to the slope of the regression line when z-scores for the y-variable are plotted against z-scores for the x-variable.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_statistics',
    },
    {
        "problem": 'Suppose X and Y are random variables with E(X) = 37, var(X) = 5, E(Y) = 62, and var(Y) = 12. What are the expected value and variance of the random variable X + Y?\nA. E(X + Y) = 99, var(X + Y) = 8.5\nB. E(X + Y) = 99, var(X + Y) = 13\nC. E(X + Y) = 99, var(X + Y) = 17\nD. There is insufficient information to answer this question.',
        "options": 'A. E(X + Y) = 99, var(X + Y) = 8.5\nB. E(X + Y) = 99, var(X + Y) = 13\nC. E(X + Y) = 99, var(X + Y) = 17\nD. There is insufficient information to answer this question.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_statistics',
    },
    {
        "problem": 'After a frost warning was issued, the owner of a large orange grove asked his workers to spray all his trees with water. The water was supposed to freeze and form a protective covering of ice around the orange blossom. Nevertheless, the owner suspected that some trees suffered considerable damage due to the frost. To estimate the proportion of trees that suffered more than 50 percent damage due to the frost, he took a random sample of 100 trees from his grove. What is the response variable in this experiment?\nA. The proportion of trees that suffered more than 50 percent damage due to frost.\nB. The number of trees affected by the frost.\nC. The number of trees sampled from the grove.\nD. For each sampled tree, whether it suffered more than 50 percent damage or at most 50 percent damage.',
        "options": 'A. The proportion of trees that suffered more than 50 percent damage due to frost.\nB. The number of trees affected by the frost.\nC. The number of trees sampled from the grove.\nD. For each sampled tree, whether it suffered more than 50 percent damage or at most 50 percent damage.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_statistics',
    },
    {
        "problem": "A new smartwatch is manufactured in one part of a factory, then secured for shipping in another, independent part of the factory. The weight of the smartwatch has a mean of 62 grams and a standard deviation of 1.0 grams. The weight of the packaging (box, user's guide, bubble wrap, etc.) has a mean of 456 grams and a standard deviation of 6 grams. Together, the distribution of the weight of the smartwatch and its packaging would have the following mean and standard deviation:\nA. Mean 518 grams; standard deviation 7.0 grams\nB. Mean 518 grams; standard deviation 3.5 grams\nC. Mean 518 grams; standard deviation 6.1 grams\nD. Mean 394 grams; standard deviation 6.1 grams",
        "options": 'A. Mean 518 grams; standard deviation 7.0 grams\nB. Mean 518 grams; standard deviation 3.5 grams\nC. Mean 518 grams; standard deviation 6.1 grams\nD. Mean 394 grams; standard deviation 6.1 grams',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'high_school_statistics',
    },
    {
        "problem": 'Which of the following sets has the smallest standard deviation? Which has the largest? I: {1,2,3} II: {-10,10} III: {100}\nA. I, II\nB. II, III\nC. III, I\nD. III, II',
        "options": 'A. I, II\nB. II, III\nC. III, I\nD. III, II',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_statistics',
    },
]

mmlu_few_shot_high_school_us_history = [
    {
        "problem": 'This question refers to the following information. "Society in every state is a blessing, but government even in its best state is but a necessary evil; in its worst state an intolerable one; for when we suffer, or are exposed to the same miseries by a government, which we might expect in a country without government, our calamity is heightened by reflecting that we furnish the means by which we suffer. Government, like dress, is the badge of lost innocence; the palaces of kings are built on the ruins of the bowers of paradise. For were the impulses of conscience clear, uniform, and irresistibly obeyed, man would need no other lawgiver; but that not being the case, he finds it necessary to surrender up a part of his property to furnish means for the protection of the rest; and this he is induced to do by the same prudence which in every other case advises him out of two evils to choose the least. Wherefore, security being the true design and end of government, it unanswerably follows that whatever form thereof appears most likely to ensure it to us, with the least expense and greatest benefit, is preferable to all others." Thomas Paine, Common Sense, 1776 Which of the following "miseries" alluded to above were most condemned by Anti-Federalists of the post-Revolutionary era?\nA. Organized response to Bacon\'s Rebellion\nB. Federal response to Shays\'s Rebellion\nC. Federal response to the Whiskey Rebellion\nD. Federal response to Pontiac\'s Rebellion',
        "options": "A. Organized response to Bacon's Rebellion\nB. Federal response to Shays's Rebellion\nC. Federal response to the Whiskey Rebellion\nD. Federal response to Pontiac's Rebellion",
        "solution": "The answer is \\boxed{C}.",
        "topic": "high_school_us_history",
    },
    {
        "problem": 'This question refers to the following information. "As our late Conduct at the Conestoga Manor and Lancaster have occasioned much Speculation & a great diversity of Sentiments in this and neighboring Governments; some vindicating & others condemning it; some charitably alleviating the Crime, & others maliciously painting it in the most odious & detestable Colours, we think it our duty to lay before the Publick, the whole Matter as it appeared, & still appears, to us. . . . "If these things are not sufficient to prove an unjustifiable Attachment in the Quakers to the Indians Savages, a fixed Resolution to befriend them & an utter insensibility to human Distresses, let us consider a few more recent Facts. When we found the last Summer that we were likely to get no Assistance from the Government, some Volunteers went out at our own Expense, determined to drive our Enemies from our Borders; & when we came near to the great Island, we understood that a Number of their Warriors had gone out against our Frontiers. Upon this we returned and came up with them and fought with them at the Munfey Hill where we lost some of our Men & killed some of their Warriors & thereby saved our Frontiers from this Story in another Expedition. But no sooner had we destroyed their Provisions on the great Island, & ruined their trade with the good People at Bethlehem, but these very Indians, who were justly suspected of having murdered our Friends in Northampton County, were by the Influence of some Quakers taken under the Protection of the Government to screen them from the Resentments of the Friends and Relations of the Murdered, & to support them thro the Winter." —"Apology of the Paxton Boys" (pamphlet), 1764 (Note: "apology" in this context should be read as an explanation, not an admission of guilt or regret.) The sentiments expressed in the explanation above reflect which of the ongoing tensions during the colonial period of American history?\nA. Tensions between British policies and the aspirations of North American colonists.\nB. Tensions between American Indians allied with the French and those allied with the British.\nC. Tensions between freed African Americans and white planters.\nD. Tensions between backcountry settlers and elites within colonial America.',
        "options": 'A. Tensions between British policies and the aspirations of North American colonists.\nB. Tensions between American Indians allied with the French and those allied with the British.\nC. Tensions between freed African Americans and white planters.\nD. Tensions between backcountry settlers and elites within colonial America.',
        "solution": "The answer is \\boxed{D}.",
        "topic": "high_school_us_history",
    },
    {
        "problem": 'This question refers to the following information. "In the new Code of Laws which I suppose it will be necessary for you to make I desire you would Remember the Ladies, and be more generous and favorable to them than your ancestors. Do not put such unlimited power into the hands of the Husbands. Remember all Men would be tyrants if they could. If particular care and attention is not paid to the Ladies we are determined to foment a Rebellion, and will not hold ourselves bound by any Laws in which we have no voice, or Representation." Abigail Adams, in a letter to John Adams, 1776 "Special legislation for woman has placed us in a most anomalous position. Women invested with the rights of citizens in one section—voters, jurors, office-holders—crossing an imaginary line, are subjects in the next. In some States, a married woman may hold property and transact business in her own name; in others, her earnings belong to her husband. In some States, a woman may testify against her husband, sue and be sued in the courts; in others, she has no redress in case of damage to person, property, or character. In case of divorce on account of adultery in the husband, the innocent wife is held to possess no right to children or property, unless by special decree of the court. But in no State of the Union has the wife the right to her own person, or to any part of the joint earnings of the co-partnership during the life of her husband. In some States women may enter the law schools and practice in the courts; in others they are forbidden. In some universities girls enjoy equal educational advantages with boys, while many of the proudest institutions in the land deny them admittance, though the sons of China, Japan and Africa are welcomed there. But the privileges already granted in the several States are by no means secure." Susan B. Anthony, "Declaration of Rights for Women," July 4, 1876 The sentiments expressed in the second excerpt by Susan B. Anthony are most likely in support of\nA. the Equal Rights Amendment\nB. universal suffrage\nC. states\' rights\nD. prohibition',
        "options": "A. the Equal Rights Amendment\nB. universal suffrage\nC. states' rights\nD. prohibition",
        "solution": "The answer is \\boxed{B}.",
        "topic": "high_school_us_history",
    },
    {
        "problem": 'This question refers to the following information. Our leaders talk about stopping aggression from the north, but this was a struggle among groups of Vietnamese until we intervened. We seem bent upon saving the Vietnamese from Ho Chi Minh even if we have to kill them and demolish their country to do it. As the native people survey bombed-out villages, women and children burned by napalm, rice crops destroyed and cities overrun with our military personnel, they are doubtless saying secretly of the Vietcong guerillas and of the American forces, "A plague on both your houses." … Stop the bombing, north and south, end search and destroy offensive sweeps, and confine our military action to holding operations on the ground. Bombing the north has failed to halt or seriously check the flow of troops to the south and may, in fact, have prompted a much greater war effort by Hanoi. —Senator George McGovern, "The Lessons of Vietnam," April 25, 1967 Which of the following opinions from the 1960s most directly reflects the perspective of George McGovern\'s speech?\nA. Americans must maximize their technological edge in Vietnam.\nB. American bombing in Vietnam is step by step leading to progress in the war.\nC. American bombing in Vietnam is a failure.\nD. America must not give in to defeatism about the war in Vietnam.',
        "options": 'A. Americans must maximize their technological edge in Vietnam.\nB. American bombing in Vietnam is step by step leading to progress in the war.\nC. American bombing in Vietnam is a failure.\nD. America must not give in to defeatism about the war in Vietnam.',
        "solution": "The answer is \\boxed{C}.",
        "topic": "high_school_us_history",
    },
    {
        "problem": 'This question refers to the following information. I come not to urge personal claims, nor to seek individual benefits; I appear as the advocate of those who cannot plead their own cause; I come as the friend of those who are deserted, oppressed, and desolate. In the Providence of God, I am the voice of the maniac whose piercing cries from the dreary dungeons of your jails penetrate not your Halls of Legislation. I am the Hope of the poor crazed beings who pine in the cells, and stalls, and cages, and waste rooms of your poor-houses. I am the Revelation of hundreds of wailing, suffering creatures, hidden in your private dwellings, and in pens and cabins—shut out, cut off from all healing influences, from all mind-restoring cares.… Could their melancholy histories be spread before you as revealed to my grieved spirit during the last three months, how promptly, how earnestly would you search out the most approved means of relief; how trifling, how insignificant, by comparison, would appear the sacrifices you are asked to make; how would a few dimes and dollars, gathered from each citizen, diminish in value as a possession, compared with the certain benefits and vast good to be secured for the suffering insane...by the consecration and application of a sufficient fund to the construction of a suitable hospital.… —Dorothea Dix, Memorial Soliciting a State Hospital for the Protection and Cure of the Insane, Submitted to the General Assembly of North Carolina, November 1848 Dorothea Dix can best be compared to whom?\nA. Abigail Adams\nB. Clara Barton\nC. Shirley Temple\nD. Hillary Clinton',
        "options": 'A. Abigail Adams\nB. Clara Barton\nC. Shirley Temple\nD. Hillary Clinton',
        "solution": "The answer is \\boxed{B}.",
        "topic": "high_school_us_history",
    },
]

mmlu_few_shot_high_school_world_history = [
    {
        "problem": 'This question refers to the following information. He contains all works and desires and all perfumes and all tastes. He enfolds the whole universe and in silence is loving to all. This is the Spirit that is in my heart, this is Brahman. To him I shall come when I go beyond this life, and to him will come he who has faith and doubts not. —The Upanishads, India, c. 1000 BCE To which religion does the speaker most likely belong?\nA. Hinduism\nB. Buddhism\nC. Shintoism\nD. Zoroastrianism',
        "options": 'A. Hinduism\nB. Buddhism\nC. Shintoism\nD. Zoroastrianism',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'high_school_world_history',
    },
    {
        "problem": 'This question refers to the following information. "The struggle against neo-colonialism is not aimed at excluding the capital of the developed world from operating in less developed countries. It is aimed at preventing the financial power of the developed countries being used in such a way as to impoverish the less developed. Non-alignment, as practiced by Ghana and many other countries, is based on co-operation with all States whether they be capitalist, socialist or have a mixed economy. Such a policy, therefore, involves foreign investment from capitalist countries, but it must be invested in accordance with a national plan drawn up by the government of the non-aligned State with its own interests in mind. The issue is not what return the foreign investor receives on his investments…The question is one of power. A State in the grip of neo-colonialism is not master of its own destiny." Kwame Nkrumah, Neo-Colonialism, 1965 Which of the following provides the best context for Nkrumah\'s writings?\nA. The Industrial Revolution\nB. Decolonization\nC. Regional Free Trade Associations\nD. Autarky',
        "options": 'A. The Industrial Revolution\nB. Decolonization\nC. Regional Free Trade Associations\nD. Autarky',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'high_school_world_history',
    },
    {
        "problem": 'This question refers to the following information. "The real grievance of the worker is the insecurity of his existence; he is not sure that he will always have work, he is not sure that he will always be healthy, and he foresees that he will one day be old and unfit to work. If he falls into poverty, even if only through a prolonged illness, he is then completely helpless, exam_ins to his own devices, and society does not currently recognize any real obligation towards him beyond the usual help for the poor, even if he has been working all the time ever so faithfully and diligently. The usual help for the poor, however, leaves a lot to be desired, especially in large cities, where it is very much worse than in the country." Otto von Bismarck, 1884 Otto von Bismarck likely made this speech in reaction to which of the following issues?\nA. Social acceptance of child labor\nB. Declining life expectancy in Germany\nC. Criticisms of German trade tariffs\nD. Negative effects attributed to industrial capitalism',
        "options": 'A. Social acceptance of child labor\nB. Declining life expectancy in Germany\nC. Criticisms of German trade tariffs\nD. Negative effects attributed to industrial capitalism',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'high_school_world_history',
    },
    {
        "problem": 'This question refers to the following information. "Indeed, as both the fatwas of distinguished [scholars] who base their opinion on reason and tradition alike and the consensus of the Sunni community agree that the ancient obligation of extirpation, extermination, and expulsion of evil innovation must be the aim of our exalted aspiration, for "Religious zeal is a victory for the Faith of God the Beneficent"; then, in accordance with the words of the Prophet (Peace upon him!) "Whosoever introduces evil innovation into our order must be expelled" and "Whosoever does aught against our order must be expelled," action has become necessary and exigent…" Letter from Ottoman Sultan Selim I to Safavid Shah Ismail I, 1514 The letter from Selim I is most clearly an example of which of the following?\nA. The maintenance of military supremacy at all costs\nB. Expanding tensions between religious sects\nC. Factors that brought about the collapse of the Ottoman Empire\nD. Peacemaking efforts among the Islamic empires',
        "options": 'A. The maintenance of military supremacy at all costs\nB. Expanding tensions between religious sects\nC. Factors that brought about the collapse of the Ottoman Empire\nD. Peacemaking efforts among the Islamic empires',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'high_school_world_history',
    },
    {
        "problem": 'This question refers to the following information. "At least one of the [world\'s] societies would have to somehow enormously increase its productivity [in order to achieve global hegemony]. That quantum jump would have to be made before the various scientific, technological, agricultural, and industrial revolutions on which our post-quantum-leap world rests. It could only be accomplished by exploiting the ecosystems, mineral resources, and human assets of whole continents outside the lands of the society making the jump. Western Europe did just that by means of its brutality and guns and, more important, by geographical and ecological luck." Copyright © 2015 Cambridge University Press. Alfred Crosby, historian, Ecological Imperialism, 2004 The "quantum jump" mentioned in the passage most directly contributed to which of the following developments in the period 1450–1750 C.E.?\nA. A breakdown in trade routes through the collapse of the established state structure\nB. An increase in the population of the world through more plentiful supplies of food\nC. The spread of Chinese and Indian belief systems across the world\nD. An increase in social unrest',
        "options": 'A. A breakdown in trade routes through the collapse of the established state structure\nB. An increase in the population of the world through more plentiful supplies of food\nC. The spread of Chinese and Indian belief systems across the world\nD. An increase in social unrest',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'high_school_world_history',
    },
]

mmlu_few_shot_human_aging = [
    {
        "problem": 'Which of the following persons is more likely to remain at home alone, as of 2019?\nA. An Asian man or woman\nB. A Hispanic man\nC. An African American woman\nD. A white man or woman',
        "options": 'A. An Asian man or woman\nB. A Hispanic man\nC. An African American woman\nD. A white man or woman',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'human_aging',
    },
    {
        "problem": 'The finding that adults tend to remember events from their adolescence better than from other periods in their lives is referred to as the\nA. Adolescence advantage\nB. Reminiscence bump\nC. Memorial memorial\nD. Quadratic retrieval spike',
        "options": 'A. Adolescence advantage\nB. Reminiscence bump\nC. Memorial memorial\nD. Quadratic retrieval spike',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'human_aging',
    },
    {
        "problem": 'When older adults move to a new state after retirement, which of the following is the more likely destination?\nA. Texas\nB. California\nC. Hawaii\nD. Vermont',
        "options": 'A. Texas\nB. California\nC. Hawaii\nD. Vermont',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'human_aging',
    },
    {
        "problem": 'Which element in tobacco smoke is responsible for cancers?\nA. Nicotine\nB. Tar\nC. Carbon monoxide\nD. Smoke particles',
        "options": 'A. Nicotine\nB. Tar\nC. Carbon monoxide\nD. Smoke particles',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'human_aging',
    },
    {
        "problem": 'All other things being equal, which of the following persons is more likely to show osteoporosis?\nA. An older Hispanic American woman\nB. An older African American woman\nC. An older Asian American woman\nD. An older Native American woman',
        "options": 'A. An older Hispanic American woman\nB. An older African American woman\nC. An older Asian American woman\nD. An older Native American woman',
        "solution": "The answer is \\boxed{C}.",
        "topic": "human_aging",
    },
]

mmlu_few_shot_human_sexuality = [
    {
        "problem": 'Morning sickness is typically a problem:\nA. during the first trimester\nB. during the second trimester\nC. during the third trimester\nD. all through the pregnancy',
        "options": 'A. during the first trimester\nB. during the second trimester\nC. during the third trimester\nD. all through the pregnancy',
        "solution": "The answer is \\boxed{A}.",
        "topic": "human_sexuality",
    },
    {
        "problem": 'A woman who knows she has active herpes and untreated syphilis but continues to have sex without informing her partners of her condition has, in psychoanalytic terms:\nA. a strong ego\nB. a weak superego\nC. a weak id\nD. a strong superego',
        "options": 'A. a strong ego\nB. a weak superego\nC. a weak id\nD. a strong superego',
        "solution": "The answer is \\boxed{B}.",
        "topic": "human_sexuality",
    },
    {
        "problem": "Women's ability to have multiple orgasms is primarily due to:\nA. the fact that they do not have a refractory period.\nB. the response of the inner layers of the vagina.\nC. having alternating orgasms in different locations.\nD. the G-Spot.",
        "options": 'A. the fact that they do not have a refractory period.\nB. the response of the inner layers of the vagina.\nC. having alternating orgasms in different locations.\nD. the G-Spot.',
        "solution": "The answer is \\boxed{A}.",
        "topic": "human_sexuality",
    },
    {
        "problem": "The nature of homosexual activities that occur during preadolescence include all but which of the following?\nA. sexual intercourse\nB. circle jerks\nC. exhibitionism\nD. touching each other's genitals",
        "options": "A. sexual intercourse\nB. circle jerks\nC. exhibitionism\nD. touching each other's genitals",
        "solution": "The answer is \\boxed{A}.",
        "topic": "human_sexuality",
    },
    {
        "problem": 'The most common disorder among men who seek sexual therapy is:\nA. premature ejaculation\nB. inhibited ejaculation\nC. erectile disorder\nD. ejaculatory disorder',
        "options": 'A. premature ejaculation\nB. inhibited ejaculation\nC. erectile disorder\nD. ejaculatory disorder',
        "solution": "The answer is \\boxed{C}.",
        "topic": "human_sexuality",
    },
]

mmlu_few_shot_international_law = [
    {
        "problem": "What types of force does Article 2(4) of the UN Charter prohibit?\nA. Article 2(4) encompasses only armed force\nB. Article 2(4) encompasses all types of force, including sanctions\nC. Article 2(4) encompasses all interference in the domestic affairs of States\nD. Article 2(4) encompasses force directed only against a State's territorial integrity",
        "options": "A. Article 2(4) encompasses only armed force\nB. Article 2(4) encompasses all types of force, including sanctions\nC. Article 2(4) encompasses all interference in the domestic affairs of States\nD. Article 2(4) encompasses force directed only against a State's territorial integrity",
        "solution": "The answer is \\boxed{A}.",
        "topic": "international_law",
    },
    {
        "problem": 'What is the judge ad hoc?\nA. If a party to a contentious case before the ICJ does not have a national sitting as judge, it is entitled to nominate someone as a judge solely for that case, with the title of judge ad hoc\nB. Judge ad hoc is the member of the bench of the ICJ with a casting vote\nC. Judge ad hoc is a surrogate judge, in case a judge is disqualified or passes away\nD. Judge ad hoc is the judge that each party will always nominate in every contentious case',
        "options": 'A. If a party to a contentious case before the ICJ does not have a national sitting as judge, it is entitled to nominate someone as a judge solely for that case, with the title of judge ad hoc\nB. Judge ad hoc is the member of the bench of the ICJ with a casting vote\nC. Judge ad hoc is a surrogate judge, in case a judge is disqualified or passes away\nD. Judge ad hoc is the judge that each party will always nominate in every contentious case',
        "solution": "The answer is \\boxed{A}.",
        "topic": "international_law",
    },
    {
        "problem": "Would a reservation to the definition of torture in the ICCPR be acceptable in contemporary practice?\nA. This is an acceptable reservation if the reserving country's legislation employs a different definition\nB. This is an unacceptable reservation because it contravenes the object and purpose of the ICCPR\nC. This is an unacceptable reservation because the definition of torture in the ICCPR is consistent with customary international law\nD. This is an acceptable reservation because under general international law States have the right to enter reservations to treaties",
        "options": "A. This is an acceptable reservation if the reserving country's legislation employs a different definition\nB. This is an unacceptable reservation because it contravenes the object and purpose of the ICCPR\nC. This is an unacceptable reservation because the definition of torture in the ICCPR is consistent with customary international law\nD. This is an acceptable reservation because under general international law States have the right to enter reservations to treaties",
        "solution": "The answer is \\boxed{B}.",
        "topic": "international_law",
    },
    {
        "problem": "When 'consent' can serve as a circumstance precluding the wrongfulness of a State conduct?\nA. Consent can serve as a circumstance precluding the wrongfulness whenever it is given\nB. Consent can never serve as a circumstance precluding wrongfulness\nC. Consent can serve as a circumstance precluding wrongfulness, provided the consent is valid and to the extent that the conduct remains within the limits of the consent given\nD. Consent can always serve as a circumstance precluding wrongfulness, no matter which organ of the State gives it",
        "options": 'A. Consent can serve as a circumstance precluding the wrongfulness whenever it is given\nB. Consent can never serve as a circumstance precluding wrongfulness\nC. Consent can serve as a circumstance precluding wrongfulness, provided the consent is valid and to the extent that the conduct remains within the limits of the consent given\nD. Consent can always serve as a circumstance precluding wrongfulness, no matter which organ of the State gives it',
        "solution": "The answer is \\boxed{C}.",
        "topic": "international_law",
    },
    {
        "problem": 'How the consent to be bound of a State may be expressed?\nA. The consent of a State to be bound is expressed only by ratification\nB. The consent of a state to be bound by a treaty may be expressed by signature, ratification, acceptance, approval or accession\nC. The consent of a State to be bound is expressed by signature\nD. The consent of a State to be bound is expressed by whatever means they choose',
        "options": 'A. The consent of a State to be bound is expressed only by ratification\nB. The consent of a state to be bound by a treaty may be expressed by signature, ratification, acceptance, approval or accession\nC. The consent of a State to be bound is expressed by signature\nD. The consent of a State to be bound is expressed by whatever means they choose',
        "solution": "The answer is \\boxed{B}.",
        "topic": "international_law",
    },
]

mmlu_few_shot_jurisprudence = [
    {
        "problem": "Which position does Rawls claim is the least likely to be adopted by the POP (people in the original position)?\nA. The POP would choose equality above liberty.\nB. The POP would opt for the 'maximin' strategy.\nC. The POP would opt for the 'difference principle'.\nD. The POP would reject the 'system of natural liberty.'",
        "options": "A. The POP would choose equality above liberty.\nB. The POP would opt for the 'maximin' strategy.\nC. The POP would opt for the 'difference principle'.\nD. The POP would reject the 'system of natural liberty.'",
        "solution": "The answer is \\boxed{A}.",
        "topic": "jurisprudence",
    },
    {
        "problem": 'Functions of the law include all but which of the following?\nA. maximizing individual freedom\nB. providing a basis for compromise\nC. keeping the peace\nD. promoting the principles of the free enterprise system',
        "options": 'A. maximizing individual freedom\nB. providing a basis for compromise\nC. keeping the peace\nD. promoting the principles of the free enterprise system',
        "solution": "The answer is \\boxed{D}.",
        "topic": "jurisprudence",
    },
    {
        "problem": "Which word best summarizes Weber's explanation of the development of formally rational law?\nA. Authority.\nB. Charisma.\nC. Co-operation.\nD. Capitalism.",
        "options": 'A. Authority.\nB. Charisma.\nC. Co-operation.\nD. Capitalism.',
        "solution": "The answer is \\boxed{D}.",
        "topic": "jurisprudence",
    },
    {
        "problem": 'The ________ School of jurisprudence postulates that the law is based on what is "correct."\nA. Natural Law\nB. Analytical\nC. Historical\nD. Sociological',
        "options": 'A. Natural Law\nB. Analytical\nC. Historical\nD. Sociological',
        "solution": "The answer is \\boxed{A}.",
        "topic": "jurisprudence",
    },
    {
        "problem": "Iverson Jewelers wrote a letter to Miller, 'We have received an exceptionally fine self winding Rolox watch which we will sell to you at a very favorable price.'\nA. The letter is an offer to sell\nB. A valid offer cannot be made by letter.\nC. The letter contains a valid offer which will terminate within a reasonable time.\nD. The letter lacks one of the essential elements of an offer.",
        "options": 'A. The letter is an offer to sell\nB. A valid offer cannot be made by letter.\nC. The letter contains a valid offer which will terminate within a reasonable time.\nD. The letter lacks one of the essential elements of an offer.',
        "solution": "The answer is \\boxed{D}.",
        "topic": "jurisprudence",
    },
]

mmlu_few_shot_logical_fallacies = [
    {
        "problem": "If someone attacks the character of an opposing arguer, instead of responding to that opponent's arguments, the first person has probably committed which of the following fallacies?\nA. tu quoque\nB. horse laugh\nC. argument against the person\nD. ignoratio elenchi",
        "options": 'A. tu quoque\nB. horse laugh\nC. argument against the person\nD. ignoratio elenchi',
        "solution": "The answer is \\boxed{C}.",
        "topic": "logical_fallacies",
    },
    {
        "problem": "The complex question fallacy consists of\nA. arguing something is inferior just because it doesn't do something it was never intended to do.\nB. including more than one claim in the proposition and treating proof for one claim as proof for all the claims.\nC. drawing a conclusion before examining the evidence, and only considering evidence that supports that conclusion.\nD. asking a question that includes either an unproven assumption or more than one question, thus making a straightforward yes or no answer meaningless.",
        "options": "A. arguing something is inferior just because it doesn't do something it was never intended to do.\nB. including more than one claim in the proposition and treating proof for one claim as proof for all the claims.\nC. drawing a conclusion before examining the evidence, and only considering evidence that supports that conclusion.\nD. asking a question that includes either an unproven assumption or more than one question, thus making a straightforward yes or no answer meaningless.",
        "solution": "The answer is \\boxed{D}.",
        "topic": "logical_fallacies",
    },
    {
        "problem": 'Which of the following is true of a valid categorical syllogism?\nA. The minor premise must deny the antecedent\nB. The major premise must affirm the consequent\nC. The middle term must be used in at least one premise in a universal or unqualified sense\nD. All of the above',
        "options": 'A. The minor premise must deny the antecedent\nB. The major premise must affirm the consequent\nC. The middle term must be used in at least one premise in a universal or unqualified sense\nD. All of the above',
        "solution": "The answer is \\boxed{C}.",
        "topic": "logical_fallacies",
    },
    {
        "problem": 'Arguing that what is true of the parts must be true of the whole is the fallacy of...\nA. Division\nB. Composition\nC. Appeal to the person\nD. Appeal to ignorance',
        "options": 'A. Division\nB. Composition\nC. Appeal to the person\nD. Appeal to ignorance',
        "solution": "The answer is \\boxed{B}.",
        "topic": "logical_fallacies",
    },
    {
        "problem": 'When an arguer causes confusion during refutation because of real or feigned lack of an ability to engage in refutation, that arguer may have committed the fallacy of\nA. poor sportsmanship\nB. appeal to compassion\nC. argument against the person\nD. ignorance of refutation',
        "options": 'A. poor sportsmanship\nB. appeal to compassion\nC. argument against the person\nD. ignorance of refutation',
        "solution": "The answer is \\boxed{D}.",
        "topic": "logical_fallacies",
    },
]

mmlu_few_shot_machine_learning = [
    {
        "problem": 'A 6-sided die is rolled 15 times and the results are: side 1 comes up 0 times; side 2: 1 time; side 3: 2 times; side 4: 3 times; side 5: 4 times; side 6: 5 times. Based on these results, what is the probability of side 3 coming up when using Add-1 Smoothing?\nA. 2.0/15\nB. 1.0/7\nC. 3.0/16\nD. 1.0/5',
        "options": 'A. 2.0/15\nB. 1.0/7\nC. 3.0/16\nD. 1.0/5',
        "solution": "The answer is \\boxed{B}.",
        "topic": "machine_learning",
    },
    {
        "problem": 'Which image data augmentation is most common for natural images?\nA. random crop and horizontal flip\nB. random crop and vertical flip\nC. posterization\nD. dithering',
        "options": 'A. random crop and horizontal flip\nB. random crop and vertical flip\nC. posterization\nD. dithering',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'machine_learning',
    },
    {
        "problem": 'You are reviewing papers for the World’s Fanciest Machine Learning Conference, and you see submissions with the following claims. Which ones would you consider accepting?\nA. My method achieves a training error lower than all previous methods!\nB. My method achieves a test error lower than all previous methods! (Footnote: When regularisation parameter λ is chosen so as to minimise test error.)\nC. My method achieves a test error lower than all previous methods! (Footnote: When regularisation parameter λ is chosen so as to minimise cross-validaton error.)\nD. My method achieves a cross-validation error lower than all previous methods! (Footnote: When regularisation parameter λ is chosen so as to minimise cross-validaton error.)',
        "options": 'A. My method achieves a training error lower than all previous methods!\nB. My method achieves a test error lower than all previous methods! (Footnote: When regularisation parameter λ is chosen so as to minimise test error.)\nC. My method achieves a test error lower than all previous methods! (Footnote: When regularisation parameter λ is chosen so as to minimise cross-validaton error.)\nD. My method achieves a cross-validation error lower than all previous methods! (Footnote: When regularisation parameter λ is chosen so as to minimise cross-validaton error.)',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'machine_learning',
    },
    {
        "problem": "To achieve an 0/1 loss estimate that is less than 1 percent of the true 0/1 loss (with probability 95%), according to Hoeffding's inequality the IID test set must have how many examples?\nA. around 10 examples\nB. around 100 examples\nC. between 100 and 500 examples\nD. more than 1000 examples",
        "options": 'A. around 10 examples\nB. around 100 examples\nC. between 100 and 500 examples\nD. more than 1000 examples',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'machine_learning',
    },
    {
        "problem": 'Traditionally, when we have a real-valued input attribute during decision-tree learning we consider a binary split according to whether the attribute is above or below some threshold. Pat suggests that instead we should just have a multiway split with one branch for each of the distinct values of the attribute. From the list below choose the single biggest problem with Pat’s suggestion:\nA. It is too computationally expensive.\nB. It would probably result in a decision tree that scores badly on the training set and a testset.\nC. It would probably result in a decision tree that scores well on the training set but badly on a testset.\nD. It would probably result in a decision tree that scores well on a testset but badly on a training set.',
        "options": 'A. It is too computationally expensive.\nB. It would probably result in a decision tree that scores badly on the training set and a testset.\nC. It would probably result in a decision tree that scores well on the training set but badly on a testset.\nD. It would probably result in a decision tree that scores well on a testset but badly on a training set.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'machine_learning',
    },
]

mmlu_few_shot_management = [
    {
        "problem": 'What are the two main dimensions of the Ohio Studies into leadership?\nA. Starting position and end position\nB. Initial environment and changed environment\nC. Organisational structure and conditioning\nD. Initiating structure and considerations',
        "options": 'A. Starting position and end position\nB. Initial environment and changed environment\nC. Organisational structure and conditioning\nD. Initiating structure and considerations',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'management',
    },
    {
        "problem": 'Hygiene factors are associated with which writer?\nA. Frederick Hertzberg\nB. D.C. McClelland\nC. Abraham Maslow\nD. Douglas McGregor',
        "options": 'A. Frederick Hertzberg\nB. D.C. McClelland\nC. Abraham Maslow\nD. Douglas McGregor',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'management',
    },
    {
        "problem": 'Which element of the cultural web forms regalia?\nA. Symbols\nB. Rituals and routines\nC. Power structures\nD. Control systems',
        "options": 'A. Symbols\nB. Rituals and routines\nC. Power structures\nD. Control systems',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'management',
    },
    {
        "problem": "What characteristic is not a key feature of the 'open systems' model of management?\nA. Morale\nB. Innovation\nC. Growth resource\nD. Adaptation",
        "options": 'A. Morale\nB. Innovation\nC. Growth resource\nD. Adaptation',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'management',
    },
    {
        "problem": 'How can organisational structures that are characterised by democratic and inclusive styles of management be described?\nA. Hierarchical\nB. Bureaucratic\nC. Flat\nD. Functional',
        "options": 'A. Hierarchical\nB. Bureaucratic\nC. Flat\nD. Functional',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'management',
    },
]

mmlu_few_shot_marketing = [
    {
        "problem": '_____________ is a natural outcome when combining demographic and geographic variables.\nA. Geodemographics\nB. Product differentiation.\nC. ANSOFF matrix.\nD. Brand management.',
        "options": 'A. Geodemographics\nB. Product differentiation.\nC. ANSOFF matrix.\nD. Brand management.',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'marketing',
    },
    {
        "problem": 'In an organization, the group of people tasked with buying decisions is referred to as the _______________.\nA. Outsourcing unit.\nB. Procurement centre.\nC. Chief executive unit.\nD. Decision-making unit.',
        "options": 'A. Outsourcing unit.\nB. Procurement centre.\nC. Chief executive unit.\nD. Decision-making unit.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'marketing',
    },
    {
        "problem": "Which of the following is an assumption in Maslow's hierarchy of needs?\nA. Needs are dependent on culture and also on social class.\nB. Lower-level needs must be at least partially satisfied before higher needs can affect behaviour.\nC. Needs are not prioritized or arranged in any particular order.\nD. Satisfied needs are motivators, and new needs emerge when current needs remain unmet.",
        "options": 'A. Needs are dependent on culture and also on social class.\nB. Lower-level needs must be at least partially satisfied before higher needs can affect behaviour.\nC. Needs are not prioritized or arranged in any particular order.\nD. Satisfied needs are motivators, and new needs emerge when current needs remain unmet.',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'marketing',
    },
    {
        "problem": 'The single group within society that is most vulnerable to reference group influence is:\nA. The older consumer who feels somewhat left out of things.\nB. The married women, many of whom feel a need for stability in their lives.\nC. New immigrants who really want to assimilate into their new culture.\nD. Children, who base most of their buying decisions on outside influences.',
        "options": 'A. The older consumer who feels somewhat left out of things.\nB. The married women, many of whom feel a need for stability in their lives.\nC. New immigrants who really want to assimilate into their new culture.\nD. Children, who base most of their buying decisions on outside influences.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'marketing',
    },
    {
        "problem": 'Although the content and quality can be as controlled as direct mail, response rates of this medium are lower because of the lack of a personal address mechanism. This media format is known as:\nA. Care lines.\nB. Direct mail.\nC. Inserts.\nD. Door to door.',
        "options": 'A. Care lines.\nB. Direct mail.\nC. Inserts.\nD. Door to door.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'marketing',
    },
]

mmlu_few_shot_medical_genetics = [
    {
        "problem": 'Large triplet repeat expansions can be detected by:\nA. polymerase chain reaction.\nB. single strand conformational polymorphism analysis.\nC. Southern blotting.\nD. Western blotting.',
        "options": 'A. polymerase chain reaction.\nB. single strand conformational polymorphism analysis.\nC. Southern blotting.\nD. Western blotting.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'medical_genetics',
    },
    {
        "problem": 'DNA ligase is\nA. an enzyme that joins fragments in normal DNA replication\nB. an enzyme of bacterial origin which cuts DNA at defined base sequences\nC. an enzyme that facilitates transcription of specific genes\nD. an enzyme which limits the level to which a particular nutrient reaches',
        "options": 'A. an enzyme that joins fragments in normal DNA replication\nB. an enzyme of bacterial origin which cuts DNA at defined base sequences\nC. an enzyme that facilitates transcription of specific genes\nD. an enzyme which limits the level to which a particular nutrient reaches',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'medical_genetics',
    },
    {
        "problem": 'A gene showing codominance\nA. has both alleles independently expressed in the heterozygote\nB. has one allele dominant to the other\nC. has alleles tightly linked on the same chromosome\nD. has alleles expressed at the same time in development',
        "options": 'A. has both alleles independently expressed in the heterozygote\nB. has one allele dominant to the other\nC. has alleles tightly linked on the same chromosome\nD. has alleles expressed at the same time in development',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'medical_genetics',
    },
    {
        "problem": 'Which of the following conditions does not show multifactorial inheritance?\nA. Pyloric stenosis\nB. Schizophrenia\nC. Spina bifida (neural tube defects)\nD. Marfan syndrome',
        "options": 'A. Pyloric stenosis\nB. Schizophrenia\nC. Spina bifida (neural tube defects)\nD. Marfan syndrome',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'medical_genetics',
    },
    {
        "problem": 'The stage of meiosis in which chromosomes pair and cross over is:\nA. prophase I\nB. metaphase I\nC. prophase II\nD. metaphase II',
        "options": 'A. prophase I\nB. metaphase I\nC. prophase II\nD. metaphase II',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'medical_genetics',
    },
]

mmlu_few_shot_miscellaneous = [
    {
        "problem": 'How many axles does a standard automobile have?\nA. one\nB. two\nC. four\nD. eight',
        "options": 'A. one\nB. two\nC. four\nD. eight',
        "solution": "The answer is \\boxed{B}.",
        "topic": "miscellaneous",
    },
    {
        "problem": 'What place is named in the title of the 1979 live album by rock legends Cheap Trick?\nA. Budapest\nB. Budokan\nC. Bhutan\nD. Britain',
        "options": 'A. Budapest\nB. Budokan\nC. Bhutan\nD. Britain',
        "solution": "The answer is \\boxed{B}.",
        "topic": "miscellaneous",
    },
    {
        "problem": "Who is the shortest man to ever win an NBA slam dunk competition?\nA. Anthony 'Spud' Webb\nB. Michael 'Air' Jordan\nC. Tyrone 'Muggsy' Bogues\nD. Julius 'Dr J' Erving",
        "options": "A. Anthony 'Spud' Webb\nB. Michael 'Air' Jordan\nC. Tyrone 'Muggsy' Bogues\nD. Julius 'Dr J' Erving",
        "solution": "The answer is \\boxed{A}.",
        "topic": "miscellaneous",
    },
    {
        "problem": 'What is produced during photosynthesis?\nA. hydrogen\nB. nylon\nC. oxygen\nD. light',
        "options": 'A. hydrogen\nB. nylon\nC. oxygen\nD. light',
        "solution": "The answer is \\boxed{C}.",
        "topic": "miscellaneous",
    },
    {
        "problem": "Which of these songs was a Top 10 hit for the rock band The Police?\nA. 'Radio Ga-Ga'\nB. 'Ob-la-di Ob-la-da'\nC. 'De Do Do Do De Da Da Da'\nD. 'In-a-Gadda-Da-Vida'",
        "options": "A. 'Radio Ga-Ga'\nB. 'Ob-la-di Ob-la-da'\nC. 'De Do Do Do De Da Da Da'\nD. 'In-a-Gadda-Da-Vida'",
        "solution": "The answer is \\boxed{C}.",
        "topic": "miscellaneous",
    },
]

mmlu_few_shot_moral_disputes = [
    {
        "problem": 'According to Metz, what is wrong with consequentialist arguments against capital punishment based on African values?\nA. It is unclear as of yet whether or not capital punishment deters harm to the community.\nB. It is unclear as of yet whether or not capital punishment deters harm to any individuals.\nC. Consequentialism is not supported by African values.\nD. Even though consequentialism is supported by African values, no consequentialist arguments framed in terms of African values have been offered.',
        "options": 'A. It is unclear as of yet whether or not capital punishment deters harm to the community.\nB. It is unclear as of yet whether or not capital punishment deters harm to any individuals.\nC. Consequentialism is not supported by African values.\nD. Even though consequentialism is supported by African values, no consequentialist arguments framed in terms of African values have been offered.',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'moral_disputes',
    },
    {
        "problem": 'A fertilized ovum is also known as\nA. a zygote.\nB. an embryo.\nC. a viability.\nD. a blastocyst.',
        "options": 'A. a zygote.\nB. an embryo.\nC. a viability.\nD. a blastocyst.',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'moral_disputes',
    },
    {
        "problem": 'Which of the following is an intrinsic good, according to Singer?\nA. being in the upper class\nB. the ability to learn and solve problems\nC. being more beautiful than most of the population\nD. being shorter than average',
        "options": 'A. being in the upper class\nB. the ability to learn and solve problems\nC. being more beautiful than most of the population\nD. being shorter than average',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'moral_disputes',
    },
    {
        "problem": 'Pence compares six different cases of reproduction, from natural twinning to SCNT. What conclusion does he draw from this comparison?\nA. SCNT is not a different kind of reproduction because there are no morally relevant differences between it and other permissible means of reproduction.\nB. Because there is a low risk of harm for natural twinning, there will be a low risk of harm for SCNT.\nC. both A and B\nD. neither A nor B',
        "options": 'A. SCNT is not a different kind of reproduction because there are no morally relevant differences between it and other permissible means of reproduction.\nB. Because there is a low risk of harm for natural twinning, there will be a low risk of harm for SCNT.\nC. both A and B\nD. neither A nor B',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'moral_disputes',
    },
    {
        "problem": 'Baron admits that the versions of the ticking bomb hypothetical she discusses are "stunningly stupid," but she claims this is actually evidence of\nA. the stupidity of most traditional philosophical examples.\nB. a general lack of intelligence among people with advanced degrees.\nC. the wrongness of torture.\nD. the readiness on the part of many intelligent people to see torture as the best solution to deal with terrorism.',
        "options": 'A. the stupidity of most traditional philosophical examples.\nB. a general lack of intelligence among people with advanced degrees.\nC. the wrongness of torture.\nD. the readiness on the part of many intelligent people to see torture as the best solution to deal with terrorism.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'moral_disputes',
    },
]

mmlu_few_shot_moral_scenarios = [
    {
        "problem": 'For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I really loved my newborn baby boy. Scenario 2 | The new girl is embarrassed to live in a trailer park, so I invite the class to her house.\nA. Wrong, Wrong\nB. Wrong, Not wrong\nC. Not wrong, Wrong\nD. Not wrong, Not wrong',
        "options": 'A. Wrong, Wrong\nB. Wrong, Not wrong\nC. Not wrong, Wrong\nD. Not wrong, Not wrong',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'moral_scenarios',
    },
    {
        "problem": 'For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I asked for a cigarette from the guy outside the store. Scenario 2 | When the security guard left his post, I pocketed a rare coin.\nA. Wrong, Wrong\nB. Wrong, Not wrong\nC. Not wrong, Wrong\nD. Not wrong, Not wrong',
        "options": 'A. Wrong, Wrong\nB. Wrong, Not wrong\nC. Not wrong, Wrong\nD. Not wrong, Not wrong',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'moral_scenarios',
    },
    {
        "problem": 'For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I bought an old pick-up truck for work. Scenario 2 | I looked up at the gorgeous waterfall and threw a rock at it.\nA. Wrong, Wrong\nB. Wrong, Not wrong\nC. Not wrong, Wrong\nD. Not wrong, Not wrong',
        "options": 'A. Wrong, Wrong\nB. Wrong, Not wrong\nC. Not wrong, Wrong\nD. Not wrong, Not wrong',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'moral_scenarios',
    },
    {
        "problem": "For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | During the house's demolition, I pushed him into a wall. Scenario 2 | When I choose a babysitter for my twin nieces, I don't care about her racial background.\nA. Wrong, Wrong\nB. Wrong, Not wrong\nC. Not wrong, Wrong\nD. Not wrong, Not wrong",
        "options": 'A. Wrong, Wrong\nB. Wrong, Not wrong\nC. Not wrong, Wrong\nD. Not wrong, Not wrong',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'moral_scenarios',
    },
    {
        "problem": 'For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I took his wallet after he harassed me at the bar. Scenario 2 | I made love to my newborn baby boy.\nA. Wrong, Wrong\nB. Wrong, Not wrong\nC. Not wrong, Wrong\nD. Not wrong, Not wrong',
        "options": 'A. Wrong, Wrong\nB. Wrong, Not wrong\nC. Not wrong, Wrong\nD. Not wrong, Not wrong',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'moral_scenarios',
    },
]

mmlu_few_shot_nutrition = [
    {
        "problem": 'In a cohort study, the risk ratio of developing diabetes was 0.86 when comparing consumers of tea (the exposed) to those who did not drink tea (the unexposed). Which one statement is correct (according to knowledge in 2020)?\nA. The tea drinkers have lower risk of developing diabetes.\nB. The tea drinkers have higher risk of developing diabetes.\nC. Based on the information given we cannot tell if the observed difference in disease risk is the result of chance.\nD. The risk ratio is close to the value one, so there is no difference in disease risk between the two groups.',
        "options": 'A. The tea drinkers have lower risk of developing diabetes.\nB. The tea drinkers have higher risk of developing diabetes.\nC. Based on the information given we cannot tell if the observed difference in disease risk is the result of chance.\nD. The risk ratio is close to the value one, so there is no difference in disease risk between the two groups.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'nutrition',
    },
    {
        "problem": 'Which of the following statements is correct (according to knowledge in 2020)?\nA. Consumers with phenylketonuria must avoid the consumption of the sweetener aspartame\nB. Consumers with phenylketonuria must avoid the consumption of the sweetener saccharin\nC. Consumers with phenylketonuria must avoid the consumption of the sweetener sucralose\nD. Consumers with phenylketonuria must avoid the consumption of the sweetener acesulfame K',
        "options": 'A. Consumers with phenylketonuria must avoid the consumption of the sweetener aspartame\nB. Consumers with phenylketonuria must avoid the consumption of the sweetener saccharin\nC. Consumers with phenylketonuria must avoid the consumption of the sweetener sucralose\nD. Consumers with phenylketonuria must avoid the consumption of the sweetener acesulfame K',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'nutrition',
    },
    {
        "problem": 'Which of the following is the most plausible explanation for the protective effect of dietary fibre against cancer of the colon, as of 2020?\nA. Propionic acid, formed during colonic fibre fermentation inhibits liver fatty acid synthesis\nB. Butyric acid, formed during colonic fibre fermentation stimulates "silencing" of the SLC5A8 tumour suppressor gene\nC. None of these options are correct\nD. Butyric acid, formed during colonic fibre fermentation stimulates anti-oxidant defences in the colon',
        "options": 'A. Propionic acid, formed during colonic fibre fermentation inhibits liver fatty acid synthesis\nB. Butyric acid, formed during colonic fibre fermentation stimulates "silencing" of the SLC5A8 tumour suppressor gene\nC. None of these options are correct\nD. Butyric acid, formed during colonic fibre fermentation stimulates anti-oxidant defences in the colon',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'nutrition',
    },
    {
        "problem": 'Which of the following statements about iodine is correct, as of 2020?\nA. 50% of adults consume iodine at levels below the RNI\nB. Dairy products are a poor source of iodine\nC. The iodine content of organic milk is generally lower that the level in non-organic milk\nD. UK dietary reference values recommend an increase in iodine intake in pregnancy',
        "options": 'A. 50% of adults consume iodine at levels below the RNI\nB. Dairy products are a poor source of iodine\nC. The iodine content of organic milk is generally lower that the level in non-organic milk\nD. UK dietary reference values recommend an increase in iodine intake in pregnancy',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'nutrition',
    },
    {
        "problem": 'What is the first-line drug for patients with type 2 diabetes and obesity, as of 2020?\nA. Acarbose\nB. Metformin\nC. Sulphonylureas\nD. Insulin',
        "options": 'A. Acarbose\nB. Metformin\nC. Sulphonylureas\nD. Insulin',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'nutrition',
    },
]

mmlu_few_shot_philosophy = [
    {
        "problem": 'Psychological egoism is:\nA. an ethical theory about how we ought to behave.\nB. a generalization concerning the way people tend to behave.\nC. a claim about human nature and the ways people are capable of behaving.\nD. none of the above.',
        "options": 'A. an ethical theory about how we ought to behave.\nB. a generalization concerning the way people tend to behave.\nC. a claim about human nature and the ways people are capable of behaving.\nD. none of the above.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'philosophy',
    },
    {
        "problem": 'According to Moore’s “ideal utilitarianism,” the right action is the one that brings about the greatest amount of:\nA. pleasure.\nB. happiness.\nC. good.\nD. virtue.',
        "options": 'A. pleasure.\nB. happiness.\nC. good.\nD. virtue.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'philosophy',
    },
    {
        "problem": "According to d'Holbach, people always act according to _____.\nA. free choices\nB. dictates of the soul\nC. necessary natural laws\nD. undetermined will",
        "options": 'A. free choices\nB. dictates of the soul\nC. necessary natural laws\nD. undetermined will',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'philosophy',
    },
    {
        "problem": "Before Tolstoy's Christian conversion, what was his perspective on the meaning of life?\nA. optimist\nB. satisfied\nC. nominally religious\nD. pessimist",
        "options": 'A. optimist\nB. satisfied\nC. nominally religious\nD. pessimist',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'philosophy',
    },
    {
        "problem": 'The study of reality in the broadest sense, an inquiry into the elemental nature of the universe and the things in it, is known as _____.\nA. metaphysics\nB. epistemology\nC. quantum physics\nD. axiology',
        "options": 'A. metaphysics\nB. epistemology\nC. quantum physics\nD. axiology',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'philosophy',
    },
]

mmlu_few_shot_prehistory = [
    {
        "problem": 'The great Mayan king Pacal built temples in the city of Palenque in order to:\nA. satisfy the powerful Mayan astronomer priests.\nB. display his generosity to the common people, since they were allowed to live in the temples.\nC. frighten away enemies, in particular the Spaniards.\nD. legitimize his kingship, since his father was not royal.',
        "options": 'A. satisfy the powerful Mayan astronomer priests.\nB. display his generosity to the common people, since they were allowed to live in the temples.\nC. frighten away enemies, in particular the Spaniards.\nD. legitimize his kingship, since his father was not royal.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'prehistory',
    },
    {
        "problem": 'According to Timothy Pauketat, the evidence for social stratification and political power at Cahokia suggests:\nA. a center of Mississippian civilization with conditions similar to the rise of early states.\nB. the limitations of authority in a Native American society of egalitarian foragers.\nC. a simple chiefdom or perhaps a complex chiefdom had evolved by A.D. 1500.\nD. a center of Mississippian civilization with conditions similar to societies on the Northwest Coast of North America.',
        "options": 'A. a center of Mississippian civilization with conditions similar to the rise of early states.\nB. the limitations of authority in a Native American society of egalitarian foragers.\nC. a simple chiefdom or perhaps a complex chiefdom had evolved by A.D. 1500.\nD. a center of Mississippian civilization with conditions similar to societies on the Northwest Coast of North America.',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'prehistory',
    },
    {
        "problem": 'Researchers now believe that the decline of the Maya was caused chiefly by:\nA. a cataclysm of some kind, such as an earthquake, volcano, or tsunami.\nB. ecological degradation resulting from slash-and-burn farming techniques.\nC. endless wars between neighboring Mayan city-states.\nD. practices of interbreeding that led to a steep rise in congenital disorders.',
        "options": 'A. a cataclysm of some kind, such as an earthquake, volcano, or tsunami.\nB. ecological degradation resulting from slash-and-burn farming techniques.\nC. endless wars between neighboring Mayan city-states.\nD. practices of interbreeding that led to a steep rise in congenital disorders.',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'prehistory',
    },
    {
        "problem": 'Recent research on hominid species dating from the Middle Pliocene indicates there was (as of 2020):\nA. a great amount of species diversity, or a single species that exhibited a lot of diversity.\nB. very little species diversity during this period and very few hominids.\nC. decreased species diversity due to a prolonged ice age followed by a severe drought.\nD. decreased species diversity but increased numbers of hammerstones and flakes, indicating stone tool manufacture.',
        "options": 'A. a great amount of species diversity, or a single species that exhibited a lot of diversity.\nB. very little species diversity during this period and very few hominids.\nC. decreased species diversity due to a prolonged ice age followed by a severe drought.\nD. decreased species diversity but increased numbers of hammerstones and flakes, indicating stone tool manufacture.',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'prehistory',
    },
    {
        "problem": 'What is the approximate mean cranial capacity of Homo erectus?\nA. under 650 cc\nB. about 800 cc\nC. just under 1000 cc\nD. 1200 cc',
        "options": 'A. under 650 cc\nB. about 800 cc\nC. just under 1000 cc\nD. 1200 cc',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'prehistory',
    },
]

mmlu_few_shot_professional_accounting = [
    {
        "problem": "Box a nongovernmental not-for-profit organization had the following transactions during the year: Proceeds from sale of investments $80000 Purchase of property plant and equipment $10000 Proceeds from long-term debt $100000 Loss on sale of investment $5000 What amount should be reported as net cash provided by financing activities in Box's statement of cash flows?\nA. $70,000\nB. $75,000\nC. $80,000\nD. 100000",
        "options": 'A. $70,000\nB. $75,000\nC. $80,000\nD. 100000',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'professional_accounting',
    },
    {
        "problem": 'One hundred years ago, your great-great-grandmother invested $100 at 5% yearly interest. What is the investment worth today?\nA. $13,000\nB. $600\nC. $15,000\nD. $28,000',
        "options": 'A. $13,000\nB. $600\nC. $15,000\nD. $28,000',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'professional_accounting',
    },
    {
        "problem": "Krete is an unmarried taxpayer with income exclusively from wages. By December 31, year 1, Krete's employer has withheld $16,000 in federal income taxes and Krete has made no estimated tax payments. On April 15, year 2, Krete timely filed for an extension request to file her individual tax return, and paid $300 of additional taxes. Krete's year 1 tax liability was $16,500 when she timely filed her return on April 30, year 2, and paid the remaining tax liability balance. What amount would be subject to the penalty for underpayment of estimated taxes?\nA. $0\nB. $500\nC. $1,650\nD. $16,500",
        "options": 'A. $0\nB. $500\nC. $1,650\nD. $16,500',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'professional_accounting',
    },
    {
        "problem": 'On January 1, year 1, Alpha Co. signed an annual maintenance agreement with a software provider for $15,000 and the maintenance period begins on March 1, year 2. Alpha also incurred $5,000 of costs on January 1, year 1, related to software modification requests that will increase the functionality of the software. Alpha depreciates and amortizes its computer and software assets over five years using the straight-line method. What amount is the total expense that Alpha should recognize related to the maintenance agreement and the software modifications for the year ended December 31, year 1?\nA. $5,000\nB. $13,500\nC. $16,000\nD. $20,000',
        "options": 'A. $5,000\nB. $13,500\nC. $16,000\nD. $20,000',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'professional_accounting',
    },
    {
        "problem": 'An auditor traces the serial numbers on equipment to a nonissuer’s subledger. Which of the following management assertions is supported by this test?\nA. Valuation and allocation\nB. Completeness\nC. Rights and obligations\nD. Presentation and disclosure',
        "options": 'A. Valuation and allocation\nB. Completeness\nC. Rights and obligations\nD. Presentation and disclosure',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'professional_accounting',
    },
]

mmlu_few_shot_professional_law = [
    {
        "problem": 'A state legislature has recently enacted a statute making it a misdemeanor to curse or revile or use obscene or opprobrious language toward or in reference to a police officer perfonning his duties. A student at a state university organized a demonstration on campus to protest the war. The rally was attended by a group of 50 students who shouted anti-war messages at cars passing by. To show his contempt for the United States, the student sewed the American flag to the rear of his jeans. When a police officer saw the flag sown on the student\'s jeans, he approached and told him to remove the flag or he would be placed under arrest. The student became angered and shouted at the police officer, "Listen, you bastard, I\'ll wear this rag anywhere I please. " The student was subsequently placed under arrest and charged with violating the state statute. The student subsequently brings suit in state court challenging the constitutionality of the statute. The strongest constitutional argument for the student is that\nA. the statute is void for vagueness under the Fourteenth Amendment\'s due process clause.\nB. the statute is invalid because it violates the petitioner\'s freedom of speech under the First Amendment.\nC. the statute is an abridgment of freedom of speech under the First Amendment because less restrictive means are available for achieving the same purpose.\nD. the statute is overbroad and consequently invalid under the First and FourteenthAmendments.',
        "options": "A. the statute is void for vagueness under the Fourteenth Amendment's due process clause.\nB. the statute is invalid because it violates the petitioner's freedom of speech under the First Amendment.\nC. the statute is an abridgment of freedom of speech under the First Amendment because less restrictive means are available for achieving the same purpose.\nD. the statute is overbroad and consequently invalid under the First and FourteenthAmendments.",
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'professional_law',
    },
    {
        "problem": 'A state has recently enacted a statute prohibiting the disposal of any nuclear wastes within the state. This law does not contravene or conflict with any federal statutes. A man operates a company in the state that is engaged in the disposal of nuclear wastes. Subsequent to the passage of the state statute, the man, not yet aware of the new law, entered into contracts with many out-of-state firms to dispose of their nuclear wastes in the state. On account of this new law, however, the man will be unable to perform these contracts. Assume that the man has standing to challenge this state law. Which of the following presents his strongest constitutional grounds to challenge the state law prohibiting the disposal of nuclear wastes within the state?\nA. The commerce clause.\nB. The equal protection clause of the Fourteenth Amendment.\nC. The privileges and immunities clause of Article IV, Section 2. \nD. The contract clause.',
        "options": 'A. The commerce clause.\nB. The equal protection clause of the Fourteenth Amendment.\nC. The privileges and immunities clause of Article IV, Section 2. \nD. The contract clause.',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'professional_law',
    },
    {
        "problem": 'Judge took judicial notice of some facts at the beginning of the trial. Which of the following is not an appropriate kind of fact for judicial notice?\nA. Indisputable facts.\nB. Facts that have been asserted by individual political organizations.\nC. Facts recognized to be true by common knowledge.\nD. Facts capable of scientific verification.',
        "options": 'A. Indisputable facts.\nB. Facts that have been asserted by individual political organizations.\nC. Facts recognized to be true by common knowledge.\nD. Facts capable of scientific verification.',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'professional_law',
    },
    {
        "problem": "On October 1, 1980, a developer, owner of several hundred acres in a rural county, drafted a general development plan for the area. The duly recorded plan imposed elaborate limitations and restrictions upon the land in the plan, which was to be developed as a residential district. The restrictions were to extend to all persons acquiring any of the lots and to their heirs, assigns, and lessees. It was further provided that all subsequent owners would be charged with due notice of the restrictions. Among those restrictions in the general plan were the following:(22) A franchise right is created in a strip of land 10 feet in width along the rear of each lot for the use of public utility companies with right of ingress and egress. (23) No house or structure of any kind shall be built on the aforementioned strip of land running through the said blocks. In 2000, a retiree purchased one of the lots, built a house, and erected a fence in the rear of his property within the restricted area. In 2004, a teacher purchased a lot adjacent to the retiree's property and built a new house. Two years later, a librarian purchased the lot that adjoined the teacher's property. The three deeds to those properties each contained references to the deed book where the general plan was recorded. In 2008, the librarian began the construction of a seven-foot post-and-rail fence along the line dividing his lot with the teacher's, and along the center of the area subject to the franchise right. Although the teacher objected to its construction, the fence was completed. If the teacher seeks a mandatory injunction to compel removal of the librarian's fence, the court will most likely\nA. grant relief, because the fence was in violation of the easement restriction. \nB. grant relief, because the encroachment of the fence violated the restriction in the original plan. \nC. deny relief, because the teacher failed to enforce the restriction against the retiree. \nD. deny relief, because the fence would not be construed as \"a structure\" within the terms of the restriction. ",
        "options": 'A. grant relief, because the fence was in violation of the easement restriction. \nB. grant relief, because the encroachment of the fence violated the restriction in the original plan. \nC. deny relief, because the teacher failed to enforce the restriction against the retiree. \nD. deny relief, because the fence would not be construed as "a structure" within the terms of the restriction. ',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'professional_law',
    },
    {
        "problem": "A son owed a creditor $5,000. The son's father contacted the creditor and told him that he wanted to pay the son's debt. The father signed a document that stated the father would pay the son's debt at a rate of $500 a month for 10 months. The creditor made no written or oral commitment to forbear to sue the son to collect the $5,000 debt, and the father made no oral or written request for any such forbearance. For the next five months, the father made and the creditor accepted the $500 monthly payments as agreed. During that period, the creditor, in fact, did forbear to take any legal action against the son. However, the father then informed the creditor that he would make no further payments on the debt. Which of the following is the most persuasive argument that the father is liable to the creditor under the terms of their agreement?\nA. The father's promise and the creditor's reliance thereon, if proved, gave rise to a valid claim by the creditor against the father based on the doctrine of promissory estoppel. \nB. Because it was foreseeable that the father's promise would induce the creditor to forbear taking any action against the son, such forbearance was, as a matter of law, a bargained-for consideration for the father's promise. \nC. The father's five payments to the creditor totaling $2,500 manifested a serious intent on the father's part to be contractually bound, and such manifestation is generally recognized as an effective substitute for consideration. \nD. By assuming the antecedent debt obligation that the son owed to the creditor, the father became a surety whose promise to the creditor was enforceable, since it was in writing and supported by adequate consideration. ",
        "options": "A. The father's promise and the creditor's reliance thereon, if proved, gave rise to a valid claim by the creditor against the father based on the doctrine of promissory estoppel. \nB. Because it was foreseeable that the father's promise would induce the creditor to forbear taking any action against the son, such forbearance was, as a matter of law, a bargained-for consideration for the father's promise. \nC. The father's five payments to the creditor totaling $2,500 manifested a serious intent on the father's part to be contractually bound, and such manifestation is generally recognized as an effective substitute for consideration. \nD. By assuming the antecedent debt obligation that the son owed to the creditor, the father became a surety whose promise to the creditor was enforceable, since it was in writing and supported by adequate consideration. ",
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'professional_law',
    },
]

mmlu_few_shot_professional_medicine = [
    {
        "problem": 'A 42-year-old man comes to the office for preoperative evaluation prior to undergoing adrenalectomy scheduled in 2 weeks. One month ago, he received care in the emergency department for pain over his right flank following a motor vehicle collision. At that time, blood pressure was 160/100 mm Hg and CT scan of the abdomen showed an incidental 10-cm left adrenal mass. Results of laboratory studies, including complete blood count, serum electrolyte concentrations, and liver function tests, were within the reference ranges. The patient otherwise had been healthy and had never been told that he had elevated blood pressure. He takes no medications. A follow-up visit in the office 2 weeks ago disclosed elevated urinary normetanephrine and metanephrine and plasma aldosterone concentrations. The patient was referred to a surgeon, who recommended the adrenalectomy. Today, vital signs are temperature 36.6°C (97.9°F), pulse 100/min, respirations 14/min, and blood pressure 170/95 mm Hg. Physical examination discloses no significant findings. Initial preoperative preparation should include treatment with which of the following?\nA. Labetalol\nB. A loading dose of potassium chloride\nC. Nifedipine\nD. Phenoxybenzamine',
        "options": 'A. Labetalol\nB. A loading dose of potassium chloride\nC. Nifedipine\nD. Phenoxybenzamine',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'professional_medicine',
    },
    {
        "problem": 'A 36-year-old male presents to the office with a\xa03-week\xa0history of low back pain. He denies any recent trauma but says that he climbs in and out of his truck numerous times a day for his job. Examination of the patient in the prone position reveals a deep sacral sulcus on the left, a posterior inferior lateral angle on the right, and a lumbosacral junction that springs freely on compression. The most likely diagnosis is\nA. left-on-left sacral torsion\nB. left-on-right sacral torsion\nC. right unilateral sacral flexion\nD. right-on-right sacral torsion',
        "options": 'A. left-on-left sacral torsion\nB. left-on-right sacral torsion\nC. right unilateral sacral flexion\nD. right-on-right sacral torsion',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'professional_medicine',
    },
    {
        "problem": 'A previously healthy 32-year-old woman comes to the physician 8 months after her husband was killed in a car crash. Since that time, she has had a decreased appetite and difficulty falling asleep. She states that she is often sad and cries frequently. She has been rechecking the door lock five times before leaving her house and has to count exactly five pieces of toilet paper before she uses it. She says that she has always been a perfectionist but these urges and rituals are new. Pharmacotherapy should be targeted to which of the following neurotransmitters?\nA. Dopamine\nB. Glutamate\nC. Norepinephrine\nD. Serotonin',
        "options": 'A. Dopamine\nB. Glutamate\nC. Norepinephrine\nD. Serotonin',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'professional_medicine',
    },
    {
        "problem": "A 44-year-old man comes to the office because of a 3-day history of sore throat, nonproductive cough, runny nose, and frontal headache. He says the headache is worse in the morning and ibuprofen does provide some relief. He has not had shortness of breath. Medical history is unremarkable. He takes no medications other than the ibuprofen for pain. Vital signs are temperature 37.4°C (99.4°F), pulse 88/min, respirations 18/min, and blood pressure 120/84 mm Hg. Examination of the nares shows erythematous mucous membranes. Examination of the throat shows erythema and follicular lymphoid hyperplasia on the posterior oropharynx. There is no palpable cervical adenopathy. Lungs are clear to auscultation. Which of the following is the most likely cause of this patient's symptoms?\nA. Allergic rhinitis\nB. Epstein-Barr virus\nC. Mycoplasma pneumoniae\nD. Rhinovirus",
        "options": 'A. Allergic rhinitis\nB. Epstein-Barr virus\nC. Mycoplasma pneumoniae\nD. Rhinovirus',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'professional_medicine',
    },
    {
        "problem": 'A 22-year-old male marathon runner presents to the office with the complaint of right-sided rib pain when he runs long distances. Physical examination reveals normal heart and lung findings and an exhalation dysfunction at ribs\xa04-5 on the right. Which of the following muscles or muscle groups will be most useful in correcting this dysfunction utilizing a direct method?\nA. anterior scalene\nB. latissimus dorsi\nC. pectoralis minor\nD. quadratus lumborum',
        "options": 'A. anterior scalene\nB. latissimus dorsi\nC. pectoralis minor\nD. quadratus lumborum',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'professional_medicine',
    },
]

mmlu_few_shot_professional_psychology = [
    {
        "problem": "One of your therapy clients asks your advice about a good weight- reduction program. You have investigated the programs in the community and are enrolled in the one you consider the best. This program offers a $50 bonus to its patrons for each new person they bring into the program. Under these circumstances, your most appropriate response would be to\nA. tell your client the pros and cons of each program you know about except for the one in which you are enrolled\nB. recommend to your client the program in which you are enrolled and explain the $50 bonus you will receive\nC. recommend to your client the program in which you are enrolled and offer to have the $50 bonus credited to your client's account in the program\nD. tell your client the pros and cons of each program you know about, but do not claim the $50 bonus if your client enrolls in your program",
        "options": "A. tell your client the pros and cons of each program you know about except for the one in which you are enrolled\nB. recommend to your client the program in which you are enrolled and explain the $50 bonus you will receive\nC. recommend to your client the program in which you are enrolled and offer to have the $50 bonus credited to your client's account in the program\nD. tell your client the pros and cons of each program you know about, but do not claim the $50 bonus if your client enrolls in your program",
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'professional_psychology',
    },
    {
        "problem": 'There are three ways to measure the Central Tendency: the Mean, the Median and the Mode. From your knowledge about them, what is the mode?\nA. less sensitive to extreme scores than the mean\nB. more useful for skewed distributions\nC. sensitive to extreme values and highly skewed distributions\nD. the most frequently occurring number',
        "options": 'A. less sensitive to extreme scores than the mean\nB. more useful for skewed distributions\nC. sensitive to extreme values and highly skewed distributions\nD. the most frequently occurring number',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'professional_psychology',
    },
    {
        "problem": 'In terms of Hofstede’s (1980) five cultural dimensions, the United States scores at the top of the scale on:\nA. individualism.\nB. individualism and power distance.\nC. power distance and masculinity.\nD. uncertainty avoidance.',
        "options": 'A. individualism.\nB. individualism and power distance.\nC. power distance and masculinity.\nD. uncertainty avoidance.',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'professional_psychology',
    },
    {
        "problem": "Carl Jung believed that a client's transference:\nA. is a fantasy that distracts the client from reality.\nB. represents “mixed feelings” toward the therapist. \nC. \"is a form of \"\"acting out.\"\"\"\nD. reflects the client’s personal and collective unconscious.",
        "options": "A. is a fantasy that distracts the client from reality.\nB. represents “mixed feelings” toward the therapist. \nC. \"is a form of \"\"acting out.\"\"\"\nD. reflects the client’s personal and collective unconscious.",
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'professional_psychology',
    },
    {
        "problem": 'In the construction of a multiple regression equation for purposes of prediction, the optimal combination of measures is one in which the predictors\nA. are uncorrelated with each other but are moderately correlated with the criterion\nB. have low correlations with each other and low correlations with the criterion\nC. are highly intercorrelated with each other and moderately correlated with the criterion\nD. have low correlations with the criterion bur are moderately correlated with each other',
        "options": 'A. are uncorrelated with each other but are moderately correlated with the criterion\nB. have low correlations with each other and low correlations with the criterion\nC. are highly intercorrelated with each other and moderately correlated with the criterion\nD. have low correlations with the criterion bur are moderately correlated with each other',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'professional_psychology',
    },
]

mmlu_few_shot_public_relations = [
    {
        "problem": "What should a public relations media practitioner do if she does not know the answer to a reporter's question?\nA. Give the reporter other information she is certain is correct.\nB. Say that the information is 'off the record' and will be disseminated later.\nC. Say 'I don't know' and promise to provide the information later.\nD. Say 'no comment,' rather than appear uninformed.",
        "options": "A. Give the reporter other information she is certain is correct.\nB. Say that the information is 'off the record' and will be disseminated later.\nC. Say 'I don't know' and promise to provide the information later.\nD. Say 'no comment,' rather than appear uninformed.",
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'public_relations',
    },
    {
        "problem": 'In issues management, what is the most proactive approach to addressing negative or misleading information posted online about your organization?\nA. Buy domain names that could be used by opposition groups.\nB. Post anonymous comments on blogs to combat this information.\nC. Prepare a news release that discredits the inaccurate information.\nD. Make policy changes to address complaints highlighted on these sites.',
        "options": 'A. Buy domain names that could be used by opposition groups.\nB. Post anonymous comments on blogs to combat this information.\nC. Prepare a news release that discredits the inaccurate information.\nD. Make policy changes to address complaints highlighted on these sites.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'public_relations',
    },
    {
        "problem": 'Which of these statements is true of the Vatican in 2010 at the time of the accusations of child abuse cover-ups?\nA. There was a coordinated media response.\nB. Consistent messages were communicated.\nC. Criticisms were taken as attacks on the Catholic Church.\nD. The credibility of the Vatican was upheld.',
        "options": 'A. There was a coordinated media response.\nB. Consistent messages were communicated.\nC. Criticisms were taken as attacks on the Catholic Church.\nD. The credibility of the Vatican was upheld.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'public_relations',
    },
    {
        "problem": 'At which stage in the planning process would a situation analysis be carried out?\nA. Defining the program\nB. Planning the program\nC. Taking action and implementing ideas\nD. Evaluation of the program',
        "options": 'A. Defining the program\nB. Planning the program\nC. Taking action and implementing ideas\nD. Evaluation of the program',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'public_relations',
    },
    {
        "problem": 'Earth Hour was a campaign launched by which organization?\nA. Greenpeace\nB. The UN\nC. Oxfam\nD. World Wildlife Fund',
        "options": 'A. Greenpeace\nB. The UN\nC. Oxfam\nD. World Wildlife Fund',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'public_relations',
    },
]

mmlu_few_shot_security_studies = [
    {
        "problem": "What distinguishes coercive diplomacy from military force?\nA. Compellence is another term for coercive diplomacy, but covering a narrower set of criteria; compellence covers those threats aimed at initiating adversary action. A threat to coerce a state to give up part of its territory would count as coercive diplomacy, as long as that threat proactively initiates action before reactive diplomacy is taken.\nB. Coercive diplomacy constitutes the threats of limited force to induce adversary's incentive to comply with the coercer's demands. It is an influence strategy that is intended to obtain compliance: the use of force to defeat an opponent first does not count. It leaves an element of choice with the target to comply, or to continue.\nC. Military force, or the threat of military force, utilises fear to achieve strategic objectives. Coercive diplomacy is differentiated from this approach, because it does not use fear as a tool for coercing an adversary.\nD. Coercive diplomacy is employed to use force but to limit its effects on the international community. Coercive diplomacy is an aggressive strategy that is intended to obtain compliance through defeat. It does not leave an element of choice with the target, the target either being forced to comply or engage in conflict. It seeks to control by imposing compliance by removing any opportunity for negotiation or concession.",
        "options": "A. Compellence is another term for coercive diplomacy, but covering a narrower set of criteria; compellence covers those threats aimed at initiating adversary action. A threat to coerce a state to give up part of its territory would count as coercive diplomacy, as long as that threat proactively initiates action before reactive diplomacy is taken.\nB. Coercive diplomacy constitutes the threats of limited force to induce adversary's incentive to comply with the coercer's demands. It is an influence strategy that is intended to obtain compliance: the use of force to defeat an opponent first does not count. It leaves an element of choice with the target to comply, or to continue.\nC. Military force, or the threat of military force, utilises fear to achieve strategic objectives. Coercive diplomacy is differentiated from this approach, because it does not use fear as a tool for coercing an adversary.\nD. Coercive diplomacy is employed to use force but to limit its effects on the international community. Coercive diplomacy is an aggressive strategy that is intended to obtain compliance through defeat. It does not leave an element of choice with the target, the target either being forced to comply or engage in conflict. It seeks to control by imposing compliance by removing any opportunity for negotiation or concession.",
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'security_studies',
    },
    {
        "problem": 'Which of the following is the best lens through which to investigate the role of child soldiers?\nA. Child soldiers are victims of combat that need re-education and rehabilitation.\nB. Children and their mothers are not active subjects in warfare and are best considered as subjects in the private sphere.\nC. Children are most often innocent bystanders in war and are best used as signifiers of peace.\nD. Children have political subjecthood that is missed when they are considered as passive victims of warfare.',
        "options": 'A. Child soldiers are victims of combat that need re-education and rehabilitation.\nB. Children and their mothers are not active subjects in warfare and are best considered as subjects in the private sphere.\nC. Children are most often innocent bystanders in war and are best used as signifiers of peace.\nD. Children have political subjecthood that is missed when they are considered as passive victims of warfare.',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'security_studies',
    },
    {
        "problem": 'In order to become securitized, a threat must be presented in which of these ways?\nA. As an existential threat that requires immediate and extraordinary action, posing a threat to the survival of the state or to societal security.\nB. As requiring immediate and extraordinary action by the state, threatening the survival of a referent object and therefore warranting the use of measures not normally employed in the political realm.\nC. As an urgent threat to the survival of the referent object, so serious that it legitimises the employment of extraordinary action in response.\nD. As an urgent threat to the survival of the audience that requires extraordinary or emergency measures.',
        "options": 'A. As an existential threat that requires immediate and extraordinary action, posing a threat to the survival of the state or to societal security.\nB. As requiring immediate and extraordinary action by the state, threatening the survival of a referent object and therefore warranting the use of measures not normally employed in the political realm.\nC. As an urgent threat to the survival of the referent object, so serious that it legitimises the employment of extraordinary action in response.\nD. As an urgent threat to the survival of the audience that requires extraordinary or emergency measures.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'security_studies',
    },
    {
        "problem": 'How can we best describe the relationship between the state-centric approach and the concept of human security?\nA. There are such wide divisions within the human security framework regarding the nature of threats and referent objects that no widely applicable comparisons between state-centric approaches and human security can be drawn.\nB. By adopting the framework of human security, the limitations of the realist state-centric approach become evident. Whilst human security defines the referent object as the person or population, state-centric approaches prioritise the security of the state, de-prioritizing the pursuit of human security.\nC. The state-centric approach to security is a faction of human security, usually defined within the broad school of human security. By being state-centric this approach prioritises the individual as the referent object in security studies.\nD. Both the state-centric and human-centric approaches to security are mutually exclusive and offer a sufficient analytic framework with which to understand the international security system. It is therefore the role of security analysts to determine which of these substantial concepts is correct, and which should be discarded.',
        "options": 'A. There are such wide divisions within the human security framework regarding the nature of threats and referent objects that no widely applicable comparisons between state-centric approaches and human security can be drawn.\nB. By adopting the framework of human security, the limitations of the realist state-centric approach become evident. Whilst human security defines the referent object as the person or population, state-centric approaches prioritise the security of the state, de-prioritizing the pursuit of human security.\nC. The state-centric approach to security is a faction of human security, usually defined within the broad school of human security. By being state-centric this approach prioritises the individual as the referent object in security studies.\nD. Both the state-centric and human-centric approaches to security are mutually exclusive and offer a sufficient analytic framework with which to understand the international security system. It is therefore the role of security analysts to determine which of these substantial concepts is correct, and which should be discarded.',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'security_studies',
    },
    {
        "problem": 'What are the frameworks of analysis within which terrorism has been considered (as of 2020)?\nA. Competition between larger nations has resulted in some countries actively supporting terrorist groups to undermine the strength of rival states. Terrorist networks are extended patronage clubs maintained and paid for by their donor states and are conceptualised as being like state actors, to be dealt with using military force.\nB. Globalization has enabled the internationalization of terrorist activities by opening up their operational space, although coordination is still managed from a geographical base. This suggests that terrorist groups are nationally structured which means that terrorism cannot be considered in terms of a war to be defeated militarily without having serious implications on the indigenous population.\nC. Terrorism can be viewed as a problem to be resolved by military means (war on terrorism), by normal police techniques (terrorism as crime), or as a medical problem with underlying causes and symptoms (terrorism as disease).\nD. Terrorism is viewed as a criminal problem. The criminalization of terrorism has two important implications. Firstly, it suggests that terrorism can be eradicated - terrorists can be caught and brought to trial by normal judicial proceedings thereby removing the threat from society - and secondly, it suggests that preventative crime techniques are applicable to prevent its development.',
        "options": 'A. Competition between larger nations has resulted in some countries actively supporting terrorist groups to undermine the strength of rival states. Terrorist networks are extended patronage clubs maintained and paid for by their donor states and are conceptualised as being like state actors, to be dealt with using military force.\nB. Globalization has enabled the internationalization of terrorist activities by opening up their operational space, although coordination is still managed from a geographical base. This suggests that terrorist groups are nationally structured which means that terrorism cannot be considered in terms of a war to be defeated militarily without having serious implications on the indigenous population.\nC. Terrorism can be viewed as a problem to be resolved by military means (war on terrorism), by normal police techniques (terrorism as crime), or as a medical problem with underlying causes and symptoms (terrorism as disease).\nD. Terrorism is viewed as a criminal problem. The criminalization of terrorism has two important implications. Firstly, it suggests that terrorism can be eradicated - terrorists can be caught and brought to trial by normal judicial proceedings thereby removing the threat from society - and secondly, it suggests that preventative crime techniques are applicable to prevent its development.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'security_studies',
    },
]

mmlu_few_shot_sociology = [
    {
        "problem": 'Which of the following did the post-war welfare state of 1948 not aim to provide:\nA. free health care and education for all\nB. a minimum wage\nC. full employment\nD. universal welfare',
        "options": 'A. free health care and education for all\nB. a minimum wage\nC. full employment\nD. universal welfare',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'sociology',
    },
    {
        "problem": 'What does Berger (1963) describe as a metaphor for social reality?\nA. a fairground ride\nB. a circus\nC. a puppet theatre\nD. a ballet',
        "options": 'A. a fairground ride\nB. a circus\nC. a puppet theatre\nD. a ballet',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'sociology',
    },
    {
        "problem": "The shift from 'civil religion' to 'common religion' means that:\nA. the increasing bureaucracy of the state has made religion only a marginal part of our lives\nB. despite the weakening of traditional authority, our everyday lives and 'common sense' remain shaped by religious beliefs and values\nC. religious participation in collective worship may have declined, but people still practise their faiths in private\nD. people are much more likely to discuss their religious beliefs in public, informal settings",
        "options": "A. the increasing bureaucracy of the state has made religion only a marginal part of our lives\nB. despite the weakening of traditional authority, our everyday lives and 'common sense' remain shaped by religious beliefs and values\nC. religious participation in collective worship may have declined, but people still practise their faiths in private\nD. people are much more likely to discuss their religious beliefs in public, informal settings",
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'sociology',
    },
    {
        "problem": "The term 'hegemony' refers to:\nA. the tendency for the working class not to realize their own interests\nB. a dominant ideology that legitimates economic, political and cultural power\nC. a form of dual consciousness based on ideology and everyday experiences\nD. a mode of payment given for outstanding topiary",
        "options": 'A. the tendency for the working class not to realize their own interests\nB. a dominant ideology that legitimates economic, political and cultural power\nC. a form of dual consciousness based on ideology and everyday experiences\nD. a mode of payment given for outstanding topiary',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'sociology',
    },
    {
        "problem": 'Which of the following is not a problem associated with official statistics on strike action?\nA. most strikes go unnoticed by employers and the mass media\nB. not all industrial disputes will be reported by the employer\nC. the definition of strikes excludes those that involve fewer than ten workers or last less than one day\nD. it is hard to compare strikes that were measured in different ways',
        "options": 'A. most strikes go unnoticed by employers and the mass media\nB. not all industrial disputes will be reported by the employer\nC. the definition of strikes excludes those that involve fewer than ten workers or last less than one day\nD. it is hard to compare strikes that were measured in different ways',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'sociology',
    },
]

mmlu_few_shot_us_foreign_policy = [
    {
        "problem": "How did the 2008 financial crisis affect America's international reputation?\nA. It damaged support for the US model of political economy and capitalism\nB. It created anger at the United States for exaggerating the crisis\nC. It increased support for American global leadership under President Obama\nD. It reduced global use of the US dollar",
        "options": 'A. It damaged support for the US model of political economy and capitalism\nB. It created anger at the United States for exaggerating the crisis\nC. It increased support for American global leadership under President Obama\nD. It reduced global use of the US dollar',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'us_foreign_policy',
    },
    {
        "problem": 'How did NSC-68 change U.S. strategy?\nA. It globalized containment.\nB. It militarized containment.\nC. It called for the development of the hydrogen bomb.\nD. All of the above',
        "options": 'A. It globalized containment.\nB. It militarized containment.\nC. It called for the development of the hydrogen bomb.\nD. All of the above',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'us_foreign_policy',
    },
    {
        "problem": 'The realm of policy decisions concerned primarily with relations between the United States and the rest of the world is known as\nA. terrorism policy.\nB. economic policy.\nC. foreign policy.\nD. international policy.',
        "options": 'A. terrorism policy.\nB. economic policy.\nC. foreign policy.\nD. international policy.',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'us_foreign_policy',
    },
    {
        "problem": 'How do Defensive Realism and Offensive Realism differ in their explanation of state behaviour?\nA. Defensive realists place greater emphasis on the role of international institutions\nB. Defensive realists place less emphasis on geographical factors\nC. Offensive realists give more priority to the national interest than Defensive realists.\nD. Defensive realists believe states are security maximizers, while Offensive realists believe states to be power maximizers',
        "options": 'A. Defensive realists place greater emphasis on the role of international institutions\nB. Defensive realists place less emphasis on geographical factors\nC. Offensive realists give more priority to the national interest than Defensive realists.\nD. Defensive realists believe states are security maximizers, while Offensive realists believe states to be power maximizers',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'us_foreign_policy',
    },
    {
        "problem": "How did Donald Trump attack globalization in the 2016 campaign?\nA. Globalization had made men like him too rich\nB. Globalization only benefited certain American states, such as New York\nC. Liberal elites had encouraged globalization, while 'ordinary Americans' lost jobs because of it\nD. Globalization encouraged damaging trade wars",
        "options": "A. Globalization had made men like him too rich\nB. Globalization only benefited certain American states, such as New York\nC. Liberal elites had encouraged globalization, while 'ordinary Americans' lost jobs because of it\nD. Globalization encouraged damaging trade wars",
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'us_foreign_policy',
    },
]

mmlu_few_shot_virology = [
    {
        "problem": 'Why are parvoviruses a highly impactful parasite?\nA. Because they have no nucleic acid\nB. They require a helper virus\nC. Only replicate in dividing cells\nD. Can integrate into host chromosomes',
        "options": 'A. Because they have no nucleic acid\nB. They require a helper virus\nC. Only replicate in dividing cells\nD. Can integrate into host chromosomes',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'virology',
    },
    {
        "problem": 'Which of the following is a morphological characteristic of the paramyxoviruses.\nA. Fragile viruses often visualised with RNA spewing from the inside\nB. Elongate viruses\nC. Icosahedral viruses with envelope\nD. Very large viruses',
        "options": 'A. Fragile viruses often visualised with RNA spewing from the inside\nB. Elongate viruses\nC. Icosahedral viruses with envelope\nD. Very large viruses',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'virology',
    },
    {
        "problem": 'A key factor facilitating the application of nested case-control studies from the MACS was:\nA. Data collection\nB. Establishment of a repository of biologic specimens\nC. Participant interest\nD. Administration of the questionnaire by staff',
        "options": 'A. Data collection\nB. Establishment of a repository of biologic specimens\nC. Participant interest\nD. Administration of the questionnaire by staff',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'virology',
    },
    {
        "problem": 'The most important goal of a behavioral intervention is:\nA. Change in behavior\nB. Comprehensive coverage\nC. Effective use of behavioral theory\nD. Sustained behavior change',
        "options": 'A. Change in behavior\nB. Comprehensive coverage\nC. Effective use of behavioral theory\nD. Sustained behavior change',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'virology',
    },
    {
        "problem": 'The median survival time to AIDS and death was established by following:\nA. Seroprevalent HIV-infected individuals\nB. Seronegatives\nC. Seroconverters\nD. High-risk seronegatives',
        "options": 'A. Seroprevalent HIV-infected individuals\nB. Seronegatives\nC. Seroconverters\nD. High-risk seronegatives',
        "solution": 'The answer is \\boxed{C}.',
        "topic": 'virology',
    },
]

mmlu_few_shot_world_religions = [
    {
        "problem": 'What is the sign of the covenant for Jewish males?\nA. The rainbow\nB. Circumcision\nC. A son\nD. Bar mitzvah',
        "options": 'A. The rainbow\nB. Circumcision\nC. A son\nD. Bar mitzvah',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'world_religions',
    },
    {
        "problem": 'What is the Second Gem in Buddhism?\nA. The Dharma\nB. The Sangha\nC. The Buddha\nD. The Bodhisattva',
        "options": 'A. The Dharma\nB. The Sangha\nC. The Buddha\nD. The Bodhisattva',
        "solution": 'The answer is \\boxed{A}.',
        "topic": 'world_religions',
    },
    {
        "problem": 'In which dynasty was the "Mandate of Heaven" developed to legitimatize the new rulers?\nA. Shang\nB. Zhou\nC. Han\nD. Xia',
        "options": 'A. Shang\nB. Zhou\nC. Han\nD. Xia',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'world_religions',
    },
    {
        "problem": 'Which Japanese government promoted a kind of national cult based on the emperor and his associations with kami?\nA. Honen\nB. Tanaka\nC. Tokugawa\nD. Meiji',
        "options": 'A. Honen\nB. Tanaka\nC. Tokugawa\nD. Meiji',
        "solution": 'The answer is \\boxed{D}.',
        "topic": 'world_religions',
    },
    {
        "problem": 'How can the Upanishads be characterized?\nA. Ritual texts\nB. Philosophical texts\nC. Hymns\nD. Origin stories',
        "options": 'A. Ritual texts\nB. Philosophical texts\nC. Hymns\nD. Origin stories',
        "solution": 'The answer is \\boxed{B}.',
        "topic": 'world_religions',
    },
]

examples_map = {
    'mmlu_few_shot_abstract_algebra': mmlu_few_shot_abstract_algebra,
    'mmlu_few_shot_anatomy': mmlu_few_shot_anatomy,
    'mmlu_few_shot_astronomy': mmlu_few_shot_astronomy,
    'mmlu_few_shot_business_ethics': mmlu_few_shot_business_ethics,
    'mmlu_few_shot_clinical_knowledge': mmlu_few_shot_clinical_knowledge,
    'mmlu_few_shot_college_biology': mmlu_few_shot_college_biology,
    'mmlu_few_shot_college_chemistry': mmlu_few_shot_college_chemistry,
    'mmlu_few_shot_college_computer_science': mmlu_few_shot_college_computer_science,
    'mmlu_few_shot_college_mathematics': mmlu_few_shot_college_mathematics,
    'mmlu_few_shot_college_medicine': mmlu_few_shot_college_medicine,
    'mmlu_few_shot_college_physics': mmlu_few_shot_college_physics,
    'mmlu_few_shot_computer_security': mmlu_few_shot_computer_security,
    'mmlu_few_shot_conceptual_physics': mmlu_few_shot_conceptual_physics,
    'mmlu_few_shot_econometrics': mmlu_few_shot_econometrics,
    'mmlu_few_shot_electrical_engineering': mmlu_few_shot_electrical_engineering,
    'mmlu_few_shot_elementary_mathematics': mmlu_few_shot_elementary_mathematics,
    'mmlu_few_shot_formal_logic': mmlu_few_shot_formal_logic,
    'mmlu_few_shot_global_facts': mmlu_few_shot_global_facts,
    'mmlu_few_shot_high_school_biology': mmlu_few_shot_high_school_biology,
    'mmlu_few_shot_high_school_chemistry': mmlu_few_shot_high_school_chemistry,
    'mmlu_few_shot_high_school_computer_science': mmlu_few_shot_high_school_computer_science,
    'mmlu_few_shot_high_school_european_history': mmlu_few_shot_high_school_european_history,
    'mmlu_few_shot_high_school_geography': mmlu_few_shot_high_school_geography,
    'mmlu_few_shot_high_school_government_and_politics': mmlu_few_shot_high_school_government_and_politics,
    'mmlu_few_shot_high_school_macroeconomics': mmlu_few_shot_high_school_macroeconomics,
    'mmlu_few_shot_high_school_mathematics': mmlu_few_shot_high_school_mathematics,
    'mmlu_few_shot_high_school_microeconomics': mmlu_few_shot_high_school_microeconomics,
    'mmlu_few_shot_high_school_physics': mmlu_few_shot_high_school_physics,
    'mmlu_few_shot_high_school_psychology': mmlu_few_shot_high_school_psychology,
    'mmlu_few_shot_high_school_statistics': mmlu_few_shot_high_school_statistics,
    'mmlu_few_shot_high_school_us_history': mmlu_few_shot_high_school_us_history,
    'mmlu_few_shot_high_school_world_history': mmlu_few_shot_high_school_world_history,
    'mmlu_few_shot_human_aging': mmlu_few_shot_human_aging,
    'mmlu_few_shot_human_sexuality': mmlu_few_shot_human_sexuality,
    'mmlu_few_shot_international_law': mmlu_few_shot_international_law,
    'mmlu_few_shot_jurisprudence': mmlu_few_shot_jurisprudence,
    'mmlu_few_shot_logical_fallacies': mmlu_few_shot_logical_fallacies,
    'mmlu_few_shot_machine_learning': mmlu_few_shot_machine_learning,
    'mmlu_few_shot_management': mmlu_few_shot_management,
    'mmlu_few_shot_marketing': mmlu_few_shot_marketing,
    'mmlu_few_shot_medical_genetics': mmlu_few_shot_medical_genetics,
    'mmlu_few_shot_miscellaneous': mmlu_few_shot_miscellaneous,
    'mmlu_few_shot_moral_disputes': mmlu_few_shot_moral_disputes,
    'mmlu_few_shot_moral_scenarios': mmlu_few_shot_moral_scenarios,
    'mmlu_few_shot_nutrition': mmlu_few_shot_nutrition,
    'mmlu_few_shot_philosophy': mmlu_few_shot_philosophy,
    'mmlu_few_shot_prehistory': mmlu_few_shot_prehistory,
    'mmlu_few_shot_professional_accounting': mmlu_few_shot_professional_accounting,
    'mmlu_few_shot_professional_law': mmlu_few_shot_professional_law,
    'mmlu_few_shot_professional_medicine': mmlu_few_shot_professional_medicine,
    'mmlu_few_shot_professional_psychology': mmlu_few_shot_professional_psychology,
    'mmlu_few_shot_public_relations': mmlu_few_shot_public_relations,
    'mmlu_few_shot_security_studies': mmlu_few_shot_security_studies,
    'mmlu_few_shot_sociology': mmlu_few_shot_sociology,
    'mmlu_few_shot_us_foreign_policy': mmlu_few_shot_us_foreign_policy,
    'mmlu_few_shot_virology': mmlu_few_shot_virology,
    'mmlu_few_shot_world_religions': mmlu_few_shot_world_religions,
}
