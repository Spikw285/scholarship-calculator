### Introduction
In this guidebook, you'll find essential information and the mathematical model used by the Scholarship Calculator: The goal: provide students (and developers) a clear, reproducible way to compute which scores are necessary in course components to achieve target final grades (e.g., scholarship threshold). *Note that university rules almost always differ. This tool reflects the rules described below; always check your sillabus or instructor for institution-specific policy so you could configure it appropiately*
From now on, the *trimester* academic period of study is used in this document.
## Glossary
#### Scholarship
A scholarship is an opportunity for a student to receive financial support based on academic performance. To receive a scholarship, **all subjects** for the academic period (in this case, trimesters) must have a final grade (called **RegTotal**) greater than or equal to 70%.
- For example, if a student achieves a RegTotal of 69% in a subject, they **will not** receive the scholarship.
- _**However,**_ if a student’s RegTotal is greater than 69.5% (e.g., 69.6%), the student **will** receive the scholarship because the LMS rounds digits to the _closest integer_.
- In the rare case of an exact 69.5%, you should consult your teacher for clarification.
- Also note: even if a student has RegTotal above 70% in 5 out of 6 subjects, if one subject is below 70% (unless it is very close, e.g., 69%, and a consultation with the teacher leads to an adjustment), the student will not gain the scholarship.
#### Increased Scholarship (Incr. Scholarship)
The increased scholarship follows the same principle as the regular scholarship, except that **all subjects** for the academic period must have a RegTotal greater than or equal to 90%. This level is commonly unachievable, as grades from RegMid, RegEnd, and RegFinal are usually lower, but financial benefits are a bit greater compared to regular scholarship, but often it's effort is not worth.
#### RegMid
**RegMid** is the final grade for the first half of the academic period. It is calculated based on the weighted components specified in the syllabus, including the Midterm exam.
#### RegEnd
**RegEnd** is the final grade for the second half of the academic period. It is calculated according to the weights described in the syllabus, including the Endterm exam.
#### RegTerm
RegTerm is a simplified representation of the average of RegMid and RegEnd. In other words,
$\frac{\text{RegMid}+\text{RegEnd}}{2}$ 
This notation is used for ease of calculation, even though syllabi may list these components separately.
#### RegFinal
**RegFinal** is the grade obtained for the final exam. Generally, the grade from the Final exam is taken into account in full. However, if a student scores below 50% on the Final exam but still above 25%, they have the opportunity to retake the exam - the so-called **FX Final exam**.
- If the FX Final exam is passed, the FX score (i.e., the second attempt) replaces the original Final exam score, but only if it is higher.
    - _For example:_ If a student scores 49% on the Final exam and then takes the FX Final exam, scoring 74%, the RegFinal becomes 74% instead of 49%.
- **Important:** If a student wishes to retake the Final exam (i.e., take the FX Final), they **need** to consult with the teacher as soon as possible (ideally at the end of Final exam period, but *before* FX Final exam periods).
#### Summer term (summer semester)
The **Summer Term** is a period during which a student works to “close” a subject in which they received an unsatisfactory grade.
- Typically, if either RegMid or RegEnd is below 25%, the student is automatically assigned to the Summer Term.
- Additionally, if RegMid or RegEnd is above 25% but (somehow) the overall RegTerm is below 50%, the student is also assigned to the Summer Term.
	- *Note: it is still technically possible for a student to continue studying even if one of the components is below threshold, and still be able to close a subject. But in this specific case syllabus says that if one of the said components is less than threshold, then student is sent to summer semester automatically, therefore, RegTerm cannot be less than 50%. For example, if RegTerm is 49.5%, and RegFinal is 51%, it's technically possible for a student to complete the subject, but in syllabus it's stated that one of the components of RegTerm cannot be less than 25%, therefore, student is sent to summer semester*
- For students with RegTerm of 50% or higher, passing the Final exam (which is counted immediately as RegFinal) is required. However, if a student fails two consecutive exam attempts (both the Final and FX Final are below 50%), they are assigned to the Summer Term
**Note:** 
- If you earn more than 27 credits in one academic year, you will not be allowed to take the Summer Term for subjects exceeding that 27-credit limit.
- If you cannot “close” a subject in the Summer Term (or overall cannot close all subjects), you will automatically be placed in an overcourse, meaning you must pay for the next course and, if applicable, lose your scholarship eligibility.
- Additionally, if you are taking the Summer Term, you must pay fees per credit - rates being 20k/28k/45k/10597 (38.5\$/54\$/87\$/20.5\$) tenge per credit for Bachelor/Master/Doctoral/Foreign National, respectively .
    - For instance, if one subject is worth 5 credits, a Bachelor student would need to pay 100k tenge for that subject regardless of the course.
#### Credit
**Credit** represents the “weight” or importance of a subject component.
- For example, if a component is worth 2 credits, it may be considered less critical compared to a component worth 5 credits, which is deemed more important.
#### RegTotal
RegTotal is the student's final grade for the academic period, which is usually calculated as follows: $\text{RegTerm}\times60\%+\text{RegFinal}\times40\%$ .
Typically, RegTotal is determined at the end of the exam period, though preliminary calculations are often provided to guide your target grades.
#### Attendance
**Attendance** is the percentage of classes a student attended, whether physically or remotely.
- Each course has its own minimum attendance threshold as stated in the syllabus, but it is usually at least 70%.
- If a student attends fewer classes than the minimum requirement, they are automatically assigned to the Summer Term.
- The same rounding rules applied to grades (rounding to the closest integer) typically apply to attendance as well, but for clarity, please reconfirm with your teacher.
- Attendance is generally classified into four states:
    - **Not Assessed:** Not yet recorded (placeholder).
    - **Attended/Present:** Present before the teacher arrives.
    - **Late:** Arrived 2–5 minutes late (depending on the teacher).
    - **Absent:** Arrived more than 5–10 minutes late (depending on the teacher).
#### Syllabus
**Syllabus** is the document that outlines all the details of a subject, including topics, resources, and the methods for calculating grades and attendance.
- **It is highly recommended to read the syllabus for each subject at the start of every academic period** to avoid unnecessary questions.
- Syllabi are typically provided in the LMS (usually in PDF format).
#### Standart Grade System
Standart Grade System (hereinafter SGS) Is system of grading made by developer to simplify calculations. It's fully customizeable:
if you input the data according to your syllabus, the system can calculate grades differently to match your course’s requirements
### A prime example
Let's say that student A has course B and this course has the following grading conditions:
- 1 attestation (RegMid): 
	- Sets of problems - 6% for one problem, 5 equations, 30% overall
	- Quiz - 30%
	- Mid-term exam - 40%
	Overall grade for 1 assessment - 100%
- 2 attestation (EndTerm): 
	- Sets of problems - 6% for one problem, 5 equations, 30% overall
	- Quiz - 30%
	- End-term exam - 40%
	Overall grade for 2 assessment - 100%.
- Final Exam - 100%
*Cumulative overall grade for the course* = 
$0.3\times{1^{\text{st}}}\text{Att}+0.3\times{2^{\text{nd}}}\text{Att}+0.4\times\text{Final}=100$ 
Then, student has passed RegMid with 70% of 100, and then passes RegEnd with 85% out of 100. Then, RegTerm is calculated
$\text{RegTerm}=\frac{\text{RegMid}+\text{RegEnd}}{2}=\frac{70+85}{2}=\frac{155}{2}=77.5$
And then student passes Final exam at 60%, whereas RegTotal is calculated
$\text{RegTotal}=\text{RegTerm}\times0.6+\text{Final}\times.4=77.5*.6+60*.4=46.5+24=70.5$
And student A is capable of gaining *scholarship* because one of his subjects has RegTotal more than 70%
## Problem statement
Given:
- a target threshold $T$ (e.g., 70% for scholarship),
- current component scores $c_i$ for $i=1..n$,
- component weights $w_i$ (inside RegTerm, normalized to sum=1),
- final exam maximum allowed $F_{max}$ (Student may cap the final at a realistic target),
we need to compute target component scores $s_i$ (and optionally a target Final $F$) such that
$$
\text{RegTotal}=w_{\text{term}}\times(\sum^n_{i=1}\tilde{w_i}s_i)+w_{\text{final}}\times F\geq T,
$$
subject to $0\leq s_i\leq100$ and $s_i\geq c_i$ (by default we do not lower existing scores).
We return candidate solutions that minimize a chosen **effort** metric (how "hard" student must work to achieve the result)
### Notation
$n$ - number of RegTerm components.
$\tilde{w_i}$ - normalized weight of component $i$ (so $\sum_i\tilde{w_i}=1$).
$c_i\in[0,100]$ - current scores
$s_i\in[0,100]$ - target scores (decision variables).
$w_{\text{term}},w_{\text{final}}$ - weights of term and final (sum = 1).
$F\in[0,100]$ - chosen RegFinal (often $F\leq F_{\text{max}})$.
$T$ - target RegTotal (e.g., 70).
Required RegTerm for given final $F$:
$$
R(F)=\frac{T-w_{\text{final}}\times F}{w_{\text{term}}}.
$$
Current RegTerm contribution:
$$
C=\sum^n_{i=1}\tilde{w_i}c_i.
$$
Need to increase RegTerm by $\Delta=\text{max}(0,R(F)-C).$

## Effort metrics
There are given several metrics for different calculations:
1. Sum increase (L1)
$\text{effort}=\sum^n_{i=1}\text{max}(0,s_i-c_i)$ - minimal total percent you must add across components. Greedy-optimal: invests in the components with largest $\tilde{w_i}$.
2. Weighted L1
$\text{effort}=\sum^n_{i=1}\alpha_i\text{max}(0,s_i-c_i)$ - $\alpha_i$ is per-component cost (time/complexity). Useful when 1% in different components costs differently.
3. Minimize max single increase (min-max)
 Minimize $t$ s.t. $\forall i:s_i-c_i\leq t$
4. L2 smoothing
Minimize $sum_i(s_i-c_i)^2$ - "balanced" solution, spreads increases evenly. Requires QP
## Algorithms
### Greedy
Optimal for minimizing L1 when per-1% cost is equal across components.
Steps:
- Compute $\Delta$.
- Sort components by $\tilde{w_i}$ descending.
- For each component in that order, increase $s_i$ as much as needed (or up to 100) to cover $\Delta$. Stop when $\Delta\leq0$.
- Complexity: $O(n\log n)$.
Behavior: concentrates increases in high-weight components $\rightarrow$ minimal sum increase, possibly unbalanced.
- - -
### Linear Programming (LP)
Used when minimization objective is linear (weighted L1, min-max via auxiliary variable).
Formulation:
- variables: $s_i$ (target scores), $u_i\geq0$ (positive increases).
- objective: minimize $\sum\alpha_i u_i$
- constraints:
	- $s_i-c_i\leq u_i$,
	- $s_i-c_i\geq0$ (if disallow increasing),
	- $0\leq s_i\leq100$,
	- $\sum\tilde{w_i}s_i\geq R(F)$.
- LP solves weighted L1 and min-max modifications.
Behavior: can incorporate per-component costs and fairness constraints.
- - -
### Quadratic Programming
Objective: minimize $\sum(s_i-c_i)^2$ subject to the same constraints.
Used for smoothness, balances increases (L2)
Behavior: spreads the load, reduces variance between increases, may increase total sum against L1.
## Miscellaneous
#### Teacher-Related
If your teacher calculates grades differently from the syllabus, discuss the discrepancies with your teacher (ideally with the involvement of most of your group).  
*Note: The Standard Grading System (SGS) is based on predefined calculations and does not deviate unless adjustments are made to match your syllabus data.*
#### RegTotal-Related
While you can manually input values to calculate a desired RegTotal, a function is under development to automatically compute the minimum scores needed in each component to reach your target RegTotal.
#### Technical Questions
You can view your grades, syllabus, and all other related information on the LMS Moodle at ~~[moodle.astanait.edu.kz](https://moodle.astanait.edu.kz)~~(URL is deprecated! Use [lms.astanait.edu.kz](lms.astanait.edu.kz) instead). 
To avoid issues:
- Ensure that **RegMid** and **RegEnd** are at least 25%.
- **RegFinal** must be at least 50%.
- **Attendance** should be at least 70%.  
    Adhering to these minimum requirements will help you avoid FX exams and Summer Term assignments.
## You can also go to [Frequently asked questions](FAQ.md)