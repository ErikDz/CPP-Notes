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

<details>
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

<details>
<summary><b>Lecture 5 - Randomisation 1</b></summary>
<br>
<br>

Two types of randomisation:
1. **Las Vegas**: always correct
2. **Monte Carlo**: probabilistic correct

# 1. Toy Monte Carlo Example: Freivalds' Algorithm 
## Problem: Given $A,B,C \in \mathbb{R}^{n \times n}$, decide if $AB=C$.

**Direct approach:** Compute $D = C - AB$ and test whether $D=0$.
**Cost:** $O(n^2.38)$

Freivalds' algorithm:
```
1. Pick a random v uniformly from {0,1}^n
2. Compute w := Cv - A(Bv)
3. Return (w=0)
```

Repeating the algorithm $k$ times, the probability of error is at most $2^{-k}$.
> $Pr(\text{k errors}) \le 1/2^k$

# 2. Another Monte Carlo Example: Min-Cut in a Graph 

Given a graph $G=(V,E)$, find a cut $(S,S')$ of minimum size.

**Naive approach:** Enumerate all cuts and find the minimum one.
**Cost:** $O(2^n)$

**Randomised approach:** Pick a random cut $(S,S')$ and return it.
**Cost:** $O(n)$

Contraction algorithm [Karger'95]:
> 1. Pick a random edge $(u,v)$
> 2. Contract the edge $e$, i.e. $u$ absorbs $v$ 
>    * delete edges between $u$ and $v$ 
>    * redirect all edges incident to $v$ to $u$
> 3. Repeat until there are only two vertices left
> 4. Return the cut $(S,S')$ where $S'$ is the set of nodes absorbed by $u$

**Cost:** $O(n^4ln(n))$

Improved Contraction algorithm [Karger-Stein'96]:
> $n$ is the number of nodes
> if $n \le 6$ then, brute force enumeration
> $t = \lceil 1 + n/\sqrt{2} \rceil$
> Perform two independent runs of the contraction algorithm to obrain $H_1$ and $H_2$ each with $t$ vertices
> Recursively compute min-cuts in each $H_1$ and $H_2$
> Return the smaller of the two min-cuts

**Cost:** $O(n^2log(n))$

# 3. Randomised QuickSort

Recall: Quicstort partitioning
```python
def partition(A, l, r):
  #Runs in place
    x = A[r]
    i = l-1
    for j in range(l, r):
        if A[j] <= x:
            i += 1
            A[i], A[j] = A[j], A[i]
    A[i+1], A[r] = A[r], A[i+1]
    return i+1
```

Recall: Quicksort
```python
def quicksort(A, l, r):
    if l < r:
        q = partition(A, l, r)
        quicksort(A, l, q-1)
        quicksort(A, q+1, r)
```

**Randomised Quicksort**
```python
import random
def sort(A):
    random.shuffle(A) # Randomize the input (in O(n) ops)
    quicksort(A, 0, len(A))
```
<center>

For an **arbitrarily bad input** the **expected** number of comparisons is $\approx 2nlogn - 2.85n$

</center>


| Algorithm | Running time | In place | Extra space | Deterministic |
|:---------:|:------------:|:--------:|:-----------:|:-------------:|
| Quicksort | $O(nlogn)$   | Yes      | $logn$          | No            |
| Mergesort | $O(nlogn)$   | No       | $n$         | Yes           |


# 4. QuickSelect

```python
def select(A, k):
  random.shuffle(A)
  retrun quickselect(A, 0, len(A), k)

def quickselect(A, l, r, k):
  q = partition(A, l, r)
  if q==k: retrun A[q]
  if q<k: return quickselect(A, q+1, r, k)
  return quickselect(A, l, q-1, k)
```

Sorting gives an algorithm $O(nlogn)$ comparisons

</details>

<details>
<summary><b>Lecture 6 - Randomisation 2</b></summary>

# 1. Hash Functions

**Definition:** A hash function $h: A \rightarrow \mathbb{Z}$ is a function that maps objects form a given universe *(int, floats, strings, files etc.)* to integers.

Applications:
- Hash tables: This lecture
- Fingerprinting: check that a file ahs not been corrupted / modified; detect duplicate data; avoid backup of unchanged portions of a file; search pattern in a text (next tutorial)

# 2. Hash Tables
**Collisions do occur!** Hash tables need to detect and handle them.

**Time for insertion**: 
* m = table size
* n = number of elements

When $\alpha = n/m < 1 \text{, } \mathbb{E}( \text{nº probes}) = O(1)$


## Simple dictionaries via Hash Tables with Separate Chaining
**Def - Separate chaining:** Each table entry is a linked list of key-value pairs.

When collision occurs with separate chaining, the new element is added to the end of the list.

```python
def FindInList(key,L):
  for i, (k,v) in enumerate(L):
    if k == key: return i
    return -1

def FindInTable(key,T):
  L = T[hash(key)]
  return L, FindInList(key,L)
```
## Simple dictionaries via Hash tables with Linear Probing

**Def - Linear probing:**: Each table entry is either empty or contains a key-value pair. If a collision occurs, the next empty entry is used.

```python
def FindInTable(key,T):
  v = hash(key)
  while T[v] != None and T[v][0] != key:
    v = (v+1) % m
  return v
```

# 3. Application to Sparse Matrices
**Def - Sparse matrix:** a matrix with many zero entries.
**Ex.** Adjacency matrix of the graph of the web

**Data-structure:** array of dictionaries, where only the nonzero entries are stored

</details>

<details> 
<summary><b>Lecture 7 - Randomisation 3</b></summary>

# 1. Random walk in a maze

## Probabilistic Algorithm
**Input**: $u$ initial vertex, $v$ target vertex.
```
While u != v:
  Pick a random neighbor w of u
  u = w
Return
```
Random variable $X_k$ = vertex visited at $k$th step ($X_0 = u$)

## Exiting the maze
**Lemma:**  $\sum_{v|(u,v) \in G} T(v,u) = 2m - d(u)$
$\implies$ for any edge $(u,v)$, $T(v,u) \le 2m - 1$

Where $T(u,v)$ is the transition probability from $u$ to $v$.
And $m$ is the number of edges in the graph.

**Proposition 1:** $T(u,v) \le (2m-1)\Delta(u,v)$ for any edge $(u,v)$
Where $\Delta(u,v)$ is the number of edges connecting $u$ to $v$.

**Proposition 2:** Expected time to visit all nodes: $T(u,.) \le 2m(n-1)$

# 2. Satisfiability

**Def - Satisfiability:** Given a boolean formula $\phi$ in conjunctive normal form, is there a truth assignment that makes $\phi$ true?

**Example:** $\phi = (x_1 \lor x_2) \land (x_1 \lor \neg x_2) \land (\neg x_1 \lor x_2) \land (\neg x_1 \lor \neg x_2)$

**Def - Clause:** A disjunction of variables.
> $x \vee y \vee z$

**Def - Conjunctive normal form (CNF):** A conjunction of clauses.
> $(x_1 \lor x_2) \land (x_1 \lor \neg x_2)$

## k-SAT
**Def - k-SAT:** A boolean formula $\phi$ is k-SAT if every clause contains at most $k$ variables.

# 3. WalkSat
**Input:** A boolean formula $\phi$ in CNF in $n$ variables.
**Output:** an assignment or FAIL
> 1. Pick an assignemnt $B \in {0,1}^n$ uniformly at random  
> 2. Repeat N times:
>    - If the formula is satisfied, return $B$
>    - Pick a clause $C$ at random
>    - Pick a variable $x$ in $C$ at random
>    - Flip $x$ in $B$
> 3. Return FAIL

N is to be determined by the analysis.

## Analysis of Walksat when $k=2$
$$\mathbb{P}(success) \ge 1/2$$

WalkSat gives a Monte Carlo algorithm in time **$O(n^2)$**

### Analysis for Larger $k$

Same worst-case reasoning gives $\mathbb{P}(\Delta d= -1) \geq 1/k$.  
With $\Delta d$ the change in the number of unsatisfied clauses when flipping a variable.
Probability $p(d)$ of reaching $0$ starting from $d$ when $\mathbb{P}(\Delta d= -1) = 1/k$ (worst-case).

**Lemma.**
$$p(d) = (k-1)^{-d}$$  

Probability that WalkSat succeeds (with $N= \infty$):
$$\mathbb{P}(success) \geq \left(\frac{k}{2(k-1)}\right)^n$$  

### Stopping after $3n$ Steps for 3-SAT

$$\mathbb{P}(success) \geq \dfrac{(3/4)^n}{3n+1}$$


</details>


<details open>
<summary><b>Lecture 8 - Amortization</b></summary>

<br>

**Def - Amortized analysis:** average the wors-case over a sequence of operations. 
**Def - Average-case:** average complexity over random inputs or random executions.

# 1. Dynamic Tables
### Tables in Low-Level Languages
```python
A = []
for i in range(N): A.append(i)
# Would have quadratic complexitiy with naive implementation
```

Increasing the size of the table requires:
&nbsp;&nbsp;&nbsp;&nbsp; Allocating a new array of memory
&nbsp;&nbsp;&nbsp;&nbsp; **copying** the old array into the new one
*Expensive!*

### Dynamic tables
**Def - Dynamic table:** a table that can grow and shrink.

They use three fields:
1. Size
2. Capacity
3. Pointer to the array

Worst-Case cost of append: $O(size)$.

### Amortized Cost of a Sequence of Append
 *For example, the amortized cost of a sequence of append operations to a list data structure is the average cost per append operation, over the entire sequence of operations.*

Sequence of capacities:
$$t_{k+1} = \lfloor \alpha(t_k+1)\rfloor, \quad t_0=0$$

$t_k+1$ describes the capacity of the table after $k$ appends.

Total cost of $N$ apped: $C_N \leq N + \sum_{t_k\leq N}t_k$

> **Thm.** Amortized cost bounded by
> $$\dfrac{C_N}{N} \leq 1+ \dfrac{\alpha}{\alpha-1}$$

### Deletion
Retrieve memory when the `size` of the table decreases
* When the size of the table decreases, it is possible to retrieve memory

Dangerous scenario:
- increase by a factor $\alpha$ when full;
- decrease by a factor $1/\alpha$ when possible.
 <sup>can be problematic because it may lead to frequent resizing of the table


_Solution:_ leave space to prepay for the next growth. 

```py
def pop(self):
  if self.size==0: raise IndexError
  res = self.table[self.size]
  self.resize(self.size-1)
  return res

def resize(self,newsize):
  if newsize> self.capacity or newsize< self.capacity/beta:
    self.realloc((int)(alpha*newsize))
  self.size = newsize
```
#### Application to Hash Tables

Hash tables with linear probing require a filling ratio (number of elements stored in the data structure to its capacity) bounded away from 1. _Implemented with dynamic tables._

Resizing the table requires to rehash all the entries.

In Python, the hash function is computed once as a 64-bit integer, and stored with the object. Only its value mod the new size is recomputed.

# 2. Union find

Abstract Data Type for *Equivalence Classes*.
Main operations:
1. Find$(p)$: identifier for the equivalence class of $p$
2. Union$(p,q)$: add the relation p $\sim$ q (Combines two subsets)

## Forests in arrays
$p[i] := parent(i)$
$[2,3,2,3,10,6,6,6,10,6,2,11]$
```
        6           3        11        2
      / | \         |                 / \
     5  7  9        1                0  10
                                        / \
                                       4   8
```

First version:
```python
def find(p,a):
  while p[a]!=a: a=p[a]
  return a

def union(p,a,b):
  link(p, find(p,a), find(p,b))

def link(p,a,b):
  p[a]=b
```

Worst case:
```python	
for i in range(N): union(p,0,i)
# Uses O(N^2) array accesses
```

### Union by Rank
Maintain rank (=height) of each tree.
```python
def link(p,a,b):
  if a==b: return
  if rk[b]>rk[a]: p[a]=b
  else: p[b]=a
  if rk[a]==rk[b]: rk[a]+=1
```
Worst case for find: $O(\log N)$

### Path Compression
```python
def find(p,a):
  if p[a]!=a: p[a]=find(p,p[a])
  return p[a]
```
Preserves the properties of rank (becomes an upper bound on height). Worst-case for find unchanged.
<center>

 **Theorem.** A sequence of $m \ge n$ union or find operation uses $O(mlog^{\star}n)$ array accesses.
</center>

$\log^{\star}n$: number of iterations of $\log_2$ before reaching $\leq1$. 

## Link & Compress
```python
def compress(p,a,b):
  # b ancestor of a
  if a!=b:
    compress(p,p[a],b)
    p[a]=p[b]
```

</details>