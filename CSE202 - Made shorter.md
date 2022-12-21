<details>
<summary><b>Lecture 1 - Introduction</b></summary>

## Binary powering
Input: $(x,n ) \in A \times \N$<br>
Output: $y \in \N : y=x^{n}$


Idea:
$$
x^{n} = \begin{cases}
1 & \text{if } n=0 \\
(x^{n/2})^{2} & \text{if } n \text{ is even} \\
(x^{n-1/2})^{2}\cdot x & \text{if } n \text{ is odd}
\end{cases}
$$

```python
def binpow(x,n):
    if n==0: return 1
    tmp=binpow(x,n//2)
    tmp*=tmp
    if n%2==0: return tmp
    else: return tmp*x
```

## Correctness
An algorithm is **correct** if:
1. It terminates
2. It computes what its specification claims

## Complexity
#### The scientific approach:
1. Experiment for various sizes
2. Model
3. Analyze
4. Validate with experiments
5. If necessary, go to 2

### Binary powering - Analysis
- Lemma: For $n \ge 1, C(n) = \lfloor log_{2}n \rfloor - 1 + \lambda(n)$ where $\lambda(n)$ is the number of 1's in the binary representation of $n$.

---
# IV. Lower bounds
## Complexity of a problem
- **Def - complexity of a problem**: that of the most efficient algorithm that solves it.

### Simple lower bounds
<center>size(Input) + size(Output) $\le$ complexity </center>

---
# V. Reductions
Problem X reduces to problem Y if there is an algorithm that solves X by solving Y.<br>

<center>Complexity of solving X = complexity of solving Y + cost of the reduction</center>

</details>


<details >
<summary><b>Lecture 2 - Divide and conquer</b></summary>

# 1. Polynomials
- Polynomials behave like integers, without carries.

#### Polynomials of Degree 1
$F= f_0+f_1T$<br> $G= g_0+g_1T$<br>
$H:= FG = h_0 + h_1T+ h_2T^2$

Naive algorithm:
$H = (f_{0}g_{0}) + (f_{0}g_{1} + f_1g_0)T + f_1g_1T^2$


## Karatsuba's Algorithm
- **Idea**: Evaluate $FG = h_0 + (\tilde h_1-h_0-h_2)T + h_2T^2$ at $T=x^k$

Algorithm: 
1. if $n$ is small, use naive multiplication
2. Let $k:= \lceil \frac{n}{2} \rceil$
3. Split $F=F_0 + x^kF_1, G= G_0+x^kG_1$ <br>
    $F_0,F_1,G_0,G_1$ of degree $<k$
4. compute **recursively** <br>
  $H_0 := F_0G_0, H_2 := F_1G_1, \tilde H_1 :=(F_0+F_1)(G_0+G_1)$
5. return $H_0+x^k(\tilde H_1 - H_0 -H_2) + x^{2k}H_2$

Complexity: $C(n) = O(n^{log_{2}3})$

---

# 2. Integers
No theorem of complexity equivalence exists, but the
algorithms over polynomials can often be adapted to integers,
with the same complexity

#### Karatsuba's Algorithm for Integers

$F$ and $G$ integers $<2^n \mapsto H:=FG$

Algorithm: 
1. if $n$ is small, use naive multiplication
2. Let $k:= \lceil \frac{n}{2} \rceil$
3. Split $F=F_0 + 2^kF_1, G= G_0+2^kG_1$ <br>
    $F_0,F_1,G_0,G_1<2^k$
4. compute **recursively** <br>
  $H_0 := F_0G_0, H_2 := F_1G_1, \tilde H_1 :=(F_0+F_1)(G_0+G_1)$
5. return $H_0+2^k(\tilde H_1 - H-0 -H_2) + 2^{2k}H_2$

---

# 3. Matrix Multiplication
- **Idea**: Split the matrices into 4 submatrices and compute the product recursively.

Input: two $n \times n$ matrices $A$,$X$ with $n=2^k$<br>
Output: $AX$

Strassen's algorithm:
1. if $n=1$, return $AX$
2. Split $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}, X = \begin{pmatrix} x & y \\ z & t \end{pmatrix}$ with $(n/2) \times (n/2)$ blocks
3. Compute recursively the $7$ products <br>
$q_1 = a(x+z), q_2 = d(y+t), q_3 = (d-a)(z-y),$ <br>
$q_4 = (b-d)(z+t), q_5 = (b-a)z, q_6 = (c-a)(x+y), q_7 = (c-d)y$
4. Return $\begin{pmatrix}
q_1+q_5 & q_2+q_3+q_4-q_5 \\
q_1 + q_3 + q_6 - q_7 & q_2 + q_7
 \end{pmatrix}$

 ## Application: Graph Transitive Closure
- **Idea**: Represent a graph as an adjacency matrix and compute the transitive closure by matrix multiplication.

Let $G = (V,E)$ be a graph with n vertices.<br>
If $A$ is the adjacency matrix of $G$, then $(A ⋁ I)^{n-1}$ is the
adjacency matrix of $G^*$<br>
The matrix $(A ⋁ I)^{n-1}$ can be computed by log n squaring
operations/multiplications 
</details>

<details >
<summary><b>Lecture 3 - Divide & Conquer 2</b></summary>

# 1. Comparing Rankings
- *Similarity metric (kendall-tau distance)*: number of inversions between two rankings

## Counting inversions
Input: An array A<br>
Output: Numbers of pairs $i<j$ such that $A[i]>A[j]$<br>
(Divide and conquer: $O(nlogn)$)

## Counting inversions: DAC 
Variation of **merge-sort**
1. assume each half is sorted
2. count inversions where A[i] and A[j] are in different halves
3. merge two sorted halves into sorted whole
**Merge-and-Count:** count inversions while merging the two sorted lists 

### Sort and Count Algorithm
```
Sort-and-Count(A):
  if A has one element
    return (0,A)

  Divide A into two halves A1,A2
  (r1,A1) <- Sort-and-Count(A1)
  (r2,A2) <- Sort-and-Count(A2)

  (rC,A) <- Merge-and-Count(A1,A2)
  return (r1+r2+rC,A)
```
```
Merge-and-Count(A1,A2):
  initialize an empty array B
  Inv <- 0

  if A1 or A2 is empty 
    return (0,nonempty list)

  Compare first elems of A1, A2
  If the smallest is in A1:
    move it at the end of B
  Else
    move it at the end of B
    Inv += |A1|

  return (Inv,B)
```

# 2. Selction: Linear Time with DAC
## Complexity of DAC algorithms
$O(log n)$: binary powering<br>
$O(n log n)$: merge sort, counting inversions<br>
$O(nlog23 ≈ n 1.58)$: Karatsuba multiplication (integers, polynomials)<br>
$O(nlog27 ≈ n 2.80)$: Strassen’s matrix multiplication<br>

## Statement of the problem
_Select:_ $(A:= \{a_1, \dots,a_n \},k) \mapsto x \in A$ s.t. $| \{a \in A| a \leq x \}| = k$

**Algorithm:** 
```
Select(A,k):
  If |A| = 1, return A[0]
  Choose a good pivot p 
  q := Partition(A,p)
  If q=k return q
  If q>k return Select(A[:q],k)
  If q<k return Select(A[q:],k-q)
```
Worst case: $C(n) \leq C(n) + O(n) \rightarrow O(n^2)$

### Selection in worst-case linear time

_Goal._ Find pivot element p that divides list of n elements into two pieces so that each piece is guaranteed to have $\leq 7/10 n$ elements.

### Median-of-medians selection algorithm
```
MOM-Select(A,k):
  n <- |A|
  if n < 50:
    return k-th smallest element of A via mergesort
  Group A into n/5 groups of 5 elements each (ignore leftovers)
  B <- median of each group of 5
  p <- MOM-Select(B,n/10)

  (L,R) <- Partition(A,p)
  if (k<|L|) return MOM-select(L,k)
  else if (k>|L|) return MOM-select(R,k-|L|)
  else return p 
```
</details>

<details open>
<summary><b>Lecture 4 - Divide and conquer 3</b></summary>
## Master Theorem

Divide and conquer has a recurrence given by: 
$$C(n) \leq mC(\lceil n/p \rceil) + f(n) \text{\qquad for } n \geq p$$


> **Master Theorem - Version 1**
>
> Assume $C(n) \leq mC(n \lceil n/p \rceil)+f(n)$ if $n \geq p$,  
> with $f(n) = cn^\alpha$ ($\alpha \geq 0$). Let $q = p^\alpha$.  
> Then, as $n \to \infty$,  
> $$C(n) = \begin{cases} O(n^\alpha), &\text{if } q>m, \\ O \left(n^\alpha \log n \right), &\text{if } q = m, \\ O \left(n^{\log_p m} \right), &\text{if } q<m. \end{cases}$$
q/m governs which part of the recursion tree dominates

> **Master Theorem - More general**
>
> Assume $C(n) \leq mC(n \lceil n/p \rceil)+f(n)$ if $n \geq p$, with $f(n)$ _increasing_   
> and there exist $(q,r)$ s.t. $q \leq f(pn)/f(n) \leq r$ for large enough $n$.    
> Then, as $n \to \infty$,  
> $$C(n) = \begin{cases} O(f(n)), &\text{if } q>m, \\ O \left(f(n) \log n \right), &\text{if } q = m, \\ O \left(f(n) n^{\log_p (m/q)} \right), &\text{if } q<m. \end{cases}$$
**Note 1.** The previous theorem is a special case of this.  
**Note 2.** A tighter value of $q$ gives a better complexity bound.

# 2. Closest Pair of Points
A case when merging sub-results is not so easy
## Problem: Given points in the plane, find the closest pair.

_Naive method:_ compute all $O(n^2)$ pairwise distances, return pair with smallest one.  
_Divide and Conquer:_ split points into left and right, solve both subproblems (ok), **recombine** (hard).

**1D Approach.** Given $n$ points on a line, find the _closest_ pair 

_Solution:_ sort them in $O(n \cdot \log n)$ and traverse the list computing the distance from each point to the next.

_Sorting Solution **for 2D**:_
- sort by x-coordinate and consider nearby points
- sort by y-coordinate and consider nearby points

## Comparisons within a Strip

> Each point has to be compared with _at most 7_  
> of the next ones for the $y$-coordinate

_Def._ Let $s_i$ be the point in the 3d-strip with the $i$-th smallest y-coordinate  
_Claim._ If $|j-i|>7$, then the distance between $s_i$ and $s_j$ is _at least d_
</details>