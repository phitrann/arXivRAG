## Exact Community Recovery in the Geometric SBM


#### Julia Gaudio[∗] Xiaochun Niu[†] Ermin Wei[‡]


We study the problem of exact community recovery in the Geometric Stochastic Block
Model (GSBM), where each vertex has an unknown community label as well as a known position,
generated according to a Poisson point process in R[d]. Edges are formed independently conditioned
on the community labels and positions, where vertices may only be connected by an edge if
they are within a prescribed distance of each other. The GSBM thus favors the formation of
dense local subgraphs, which commonly occur in real-world networks, a property that makes
the GSBM qualitatively very different from the standard Stochastic Block Model (SBM). We
propose a linear-time algorithm for exact community recovery, which succeeds down to the
information-theoretic threshold, confirming a conjecture of Abbe, Baccelli, and Sankararaman.
The algorithm involves two phases. The first phase exploits the density of local subgraphs to
propagate estimated community labels among sufficiently occupied subregions, and produces an
almost-exact vertex labeling. The second phase then refines the initial labels using a Poisson
testing procedure. Thus, the GSBM enjoys local to global amplification just as the SBM, with
the advantage of admitting an information-theoretically optimal, linear-time algorithm.


### 1 Introduction


Community detection is the problem of identifying latent community structure in a network.
In 1983, Holland, Laskey, and Leinhardt [20] introduced the Stochastic Block Model (SBM), a
probabilistic model which generates graphs with community structure, where edges are generated
independently conditioned on community labels. Since then, the SBM has been intensively studied in
the probability, statistics, machine learning, and information theory communities. Many community
recovery problems are now well-understood; for example, the fundamental limits of the exact recovery
problem are known, and there is a corresponding efficient algorithm that achieves those limits [5].
For an overview of theoretical developments and open questions, please see the survey of Abbe [1].


While the SBM is a powerful model, its simplicity fails to capture certain properties that occur
in real-world networks. In particular, social networks typically contain many triangles; a given pair
of people are more likely to be friends if they already have a friend in common [27]. The SBM
by its very nature does not capture this transitive behavior, since edges are formed independently,
conditioned on the community assignments. To address this shortcoming, Baccelli and Sankararaman

[29] introduced a spatial random graph model, which we refer to as the Geometric Stochastic Block
Model (GSBM). In the GSBM, vertices are generated according to a Poisson point process in a
bounded region of R[d]. Each vertex is randomly assigned one of two community labels, with equal


**Abstract**


-----

probability. A given pair of vertices (u, v) is connected by an edge with a probability that depends
on both the community labels of u and v as well as their distance. Edges are formed independently,
conditioned on the community assignments and locations. The geometric embedding thus governs
the transitive edge behavior. The goal is to determine the communities of the vertices, observing the
edges and the locations. In a follow-up work, Abbe, Sankararaman, and Baccelli [2] studied both
partial recovery in sparse graphs, as well as exact recovery in logarithmic-degree graphs. Their work
established a phase transition for both partial and exact recovery, in terms of the Poisson intensity
parameter λ. The critical value of λ was identified in some special cases of the sparse model, but a
precise characterization of the information-theoretic threshold for exact recovery in the logarithmic
regime was left open.
Our work resolves this gap, by identifying the information-theoretic threshold for exact recovery
in the logarithmic degree regime (and confirming a conjecture of Abbe et al [2]). Additionally, we
propose a polynomial-time algorithm achieving the information-theoretic threshold. The algorithm
consists of two phases: the first phase produces a preliminary almost-exact labeling through a
local label propagation scheme, while the second phase refines the initial labels to achieve exact
recovery. At a high level, the algorithm bears some similarity to prior works on the SBM using a
two-phase approach [5, 25]. Our work therefore shows that just like the SBM, the GSBM exhibits
the so-called local to global amplification phenomenon [1], meaning that exact recovery is achievable
whenever the probability of misclassifying an individual vertex, given the labels of the remaining
_n −_ 1 vertices, is o(1/n). However, the GSBM is qualitatively very different from the SBM, and
it is not apparent at the outset that it should exhibit local to global amplification. In particular,
the GSBM is not a low-rank model, suggesting that approaches such as spectral methods [2] and
semidefinite programming [17], which exploit the low-rank structure of the SBM, may fail in the
GSBM. In order to achieve almost exact recovery in the GSBM, we instead use the density of local
subgraphs to propagate labels. Our propagation scheme allows us to achieve almost exact recovery,
and also ensures that no local region has too many misclassified vertices. The dispersion of errors is
crucial to showing that labels can be correctly refined in the second phase.
Notably, our algorithm runs in linear time (where the input size is the number of edges). This is
in contrast with the SBM, for which no statistically optimal linear-time algorithm for exact recovery
has been proposed. To our knowledge, the best-known runtime for the SBM in the logarithmic
degree regime is achieved by the spectral algorithm of Abbe et al [4], which runs in O(n log[2] _n) time,_
while the number of edges is Θ(n log n). More recent work of Cohen–Addad et al [11] proposed a
linear-time algorithm for the SBM, but the algorithm was not shown to achieve the informationtheoretic threshold for exact recovery. Intuitively, the strong local interactions in the GSBM enable
more efficient algorithms than what seems to be possible in the SBM.


**Notation and organization.** We write [n] = {1, · · ·, n}. We use Bachmann–Landau notation
with respect to the parameter n; i.e. o(1) means on(1). Bin denotes the binomial distribution. For
_µ ∈_ R[m], Poisson(µ) denotes the m-type Poisson distribution.


The rest of the paper is organized as follows. Section 2 describes the exact recovery problem as
well as our main result (Theorem 2.2). The exact recovery algorithm is given in Section 3, along
with an outline of the proof of exact recovery. Sections 4 and 5 include the proofs of the two phases
of the algorithm. Section 6 contains the proof of impossibility (Theorem 2.3) (a slight generalization
of [2, Theorem 3.7] to cover the disassortative case). Section 7 includes additional related work. We
conclude with future directions in Section 8.


-----

### 2 Model and main results


We now describe the GSBM in the logarithmic degree regime, where edges are formed only between
sufficiently close vertices, as proposed in [2, 29].


**Definition 2.1. Let λ > 0, a, b ∈** [0, 1], and a ̸= b be constants, and let d ∈ N. A graph G is
sampled from GSBM(λ, n, a, b, d) according to the following steps:


1. The locations of vertices are determined according to a homogeneous Poisson point process[1]

with intensity λ in the region Sd,n := [−n[1][/d]/2, n[1][/d]/2][d] _⊂_ R[d]. Let V ⊂Sd,n denote the vertex
set.

2. Community labels are generated independently. The ground truth label of vertex u ∈ _V is_
given by σ0(u) ∈{−1, 1}, with P(σ0(u) = 1) = P(σ0(u) = −1) = 1/2.

3. Conditioned on the locations and community labels, edges are formed independently. Letting
_E denote the edge set, for u, v ∈_ _V and u ̸= v, we have_


The graph does not contain self-loops. Here ∥u − _v∥_ denotes the toroidal metric:


where ∥· ∥2 is the standard Euclidean metric.


In other words, a given pair of vertices can only be connected by an edge if they are within a
distance of (log n)[1][/d]; in that case, we say they are mutually visible. When a pair of vertices are
mutually visible, the probability of being connected by an edge depends on their community labels,
as in the standard SBM. Observe that any unit volume region has Poisson(λ) vertices (and hence λ
vertices in expectation). In particular, the expected number of vertices in the region Sd,n is λn.


Given an estimator _σ =_ _σn, we define A(σ, σ0) = maxs∈{±1}([�]u∈V_ [1][σ][�][(][u][)=][sσ]0[(][u][)][)][/][|][V][ |][ as the]
agreement of _σ and σ0. We define some recovery requirements including exact recovery as follows._
� � �



- Exact recovery: lim

� _n_

_→∞_ [P][(][A][(][σ, σ][�] [0][) = 1) = 1][,]

- Almost exact recovery: lim
_n→∞_ [P][(][A][(][σ, σ][�] [0][)][ ≥] [1][ −] _[ϵ][) = 1][, for all][ ϵ >][ 0][,]_

- Partial recovery: lim
_n→∞_ [P][(][A][(][σ, σ][0][)][ ≥] _[α][) = 1][, for some][ α >][ 1][/][2][.]_


In other words, an exact recovery estimator must recover all labels (up to a global sign flip), with
probability tending to 1 as the graph size goes to infinity. Abbe et al [2] identified an impossibility
regime for the exact recovery problem. Here, νd is the volume of a unit Euclidean ball in d dimensions.


**Theorem 2.1 (Theorem 3.7 in [2]). Let λ > 0, d ∈** N, and 0 ≤ _b < a ≤_ 1 satisfy


_and let G ∼_ _GSBM(λ, n, a, b, d). Then any estimator_ _σ fails to achieve exact recovery._


-----

Abbe et al [2] conjectured that the above result is tight, but only established that exact recovery
is achievable for λ > λ(a, b, d) sufficiently large [2, Theorem 3.9]. In this regime, [2] provided a
polynomial-time algorithm based on the observation that the relative community labels of two nearby
vertices can be determined with high accuracy by counting their common neighbors. By taking
_λ > 0 large enough to drive up the density of points, the failure probability of pairwise classification_
can be taken to be an arbitrarily small inverse polynomial in n.


Our main result is a positive resolution to [2, Conjecture 3.8] (with a slight modification for the
case d = 1, noting that ν1 = 2).


**Theorem 2.2 (Achievability). There exists a polynomial-time algorithm achieving exact recovery in**
_G ∼_ _GSBM(λ, n, a, b, d) whenever_


_1. d = 1, λ > 1, a, b ∈_ [0, 1], and 2λ(1 − _√ab −_ (1 − _a)(1 −_ _b)) > 1; or_

_2. d_ 2, a, b [0, 1], and λνd(1 _√ab_ (1 �a)(1 _b)) > 1._
_≥_ _∈_ _−_ _−_ _−_ _−_

�


We drop the requirement that a > b in Theorem 2.1, thus covering the disassortative case. We
additionally expand the impossible regime for d = 1, compared to Theorem 2.1.


**Theorem 2.3 (Impossibility). Let λ > 0, d ∈** N, and a, b ∈ [0, 1] satisfy (2.1) and let G ∼
_GSBM(λ, n, a, b, d). Then any estimator_ _σ fails to achieve exact recovery. Additionally, if d = 1 and_
_λ < 1, then any estimator_ _σ fails to achieve exact recovery._
�


Putting Theorems 2.2 and 2.3 together establishes the information-theoretic threshold for exact
recovery in the GSBM, and shows that recovery is efficiently achievable above the threshold. We
remark that the condition λνd(1 _√ab_ (1 _a)(1_ _b)) > 1 in Theorem 2.2 is equivalent to_
_−_ _−_ _−_ _−_

_D+(x_ _y) > 1, where D+(x_ _y) is the Chernoff–Hellinger (CH) divergence [5] between the vectors_
_∥_ _∥_ �
_x = λνd[a, 1_ _a, b, 1_ _b]/2 and y = λνd[b, 1_ _b, a, 1_ _a]/2. As we will show, the exact recovery_
_−_ _−_ _−_ _−_
problem can be reduced to a multitype Poisson hypothesis testing problem; the CH-divergence
condition characterizes the parameters for which the hypothesis test is successful.
Abbe et al [2] suggested that the threshold given by Theorem 2.1 might be achieved by a
two-round procedure reminiscent of the exact recovery algorithm for the SBM developed by Abbe
and Sandon [5]. Indeed, our algorithm is a two-round procedure, but the details of the first phase
(achieving almost exact recovery) are qualitatively very different from the strategy employed in the
standard SBM. At a high level, our algorithm spreads vertex label information locally by exploiting
the density of local subgraphs. The information is spread by iteratively labeling “blocks”, labeling a
given block by using a previously labeled block as a reference. To ensure that the algorithm spreads
label information to all (sufficiently dense) blocks, we establish a connectivity property of the dense
blocks that holds with high probability whenever λνd > 1 (λ > 1 if d = 1). This is in contrast to the
Sphere Comparison algorithm [5] for the SBM, where the relative labels of a pair of vertices u, v are
determined by comparing their neighborhoods.


The algorithm in Phase I in fact achieves almost exact recovery for a wider range of parameters
than what is required to achieve exact recovery.


**Theorem 2.4. There is a polynomial-time algorithm achieving almost exact recovery in G ∼**
_GSBM(λ, n, a, b, d) whenever_


_1. d = 1, λ > 1, and a, b ∈_ [0, 1] with a ̸= b; or

_2. d ≥_ 2, λνd > 1, and a, b ∈ [0, 1] with a ̸= b.


-----

### 3 Exact recovery algorithm


This section presents our algorithm, which consists of two phases. In Phase I, our goal is to estimate
an almost-exact labeling _σ_ : V →{−1, 0, 1}, where the label 0 indicates uncertainty. Phase I is
based on the following observation: for any δ > 0, if we know the true labels of some δ log n vertices
visible to a given vertex v �, then by computing edge statistics, we can determine the label of v with
probability 1 − _n[−][c], for some c(δ) > 0. In Phase I, we partition the region into hypercubes of volume_
Θ(log n) (called blocks), and show how to produce an almost exact labeling of all blocks that contain
at least δ log n vertices (called occupied blocks), by an iterative label propagation scheme. Next,
Phase II refines the labeling _σ to_ _σ using Poisson testing. Phase II builds upon a well-established_
approach in the SBM literature [5, 25], to refine an almost-exact labeling with dispersed errors into
an exact labeling. The main novelty of our algorithm therefore lies in Phase I. � �


Before describing the algorithm, we introduce the notion of a degree profile.


**Definition 3.1 (Degree profile). Given G ∼** GSBM(λ, n, a, b, d), the degree profile of a vertex u ∈ _V_
with respect to a reference set S ⊂ _V and a labeling σ_ : S →{−1, 1} is given by the 4-tuple,


where


Note that we only consider v such that ∥u − _v∥≤_ (log n)[1][/d], since we only want to count non-edges
to vertices that are visible to u. For convenience, when V serves as the reference set, we write
_d(u, σ) := d(u, σ, V ) and d(u, σ) := [d[+]1_ [(][u, σ][)][, d]1[−][(][u, σ][)][, d][+]1[(][u, σ][)][, d][−]1[(][u, σ][)]][.]
_−_ _−_


#### 3.1 Exact recovery for d = 1.


We first describe the algorithm specialized to the case d = 1. Several additional ideas are required to
move to the d ≥ 2 case, to ensure uninterrupted propagation of label estimates over all occupied
blocks. We first describe the simplest case where d = 1, λ > 2, and a, b ∈ [0, 1] with a ̸= b.


**Algorithm for λ > 2.** The algorithm is presented in Algorithm 1. In Phase I, we first partition
the interval into blocks of length log n/2 and define Vi as the set of vertices in the ith block for
_i ∈_ [2n/ log n]. In this way, any pair of vertices in adjacent blocks are within a distance of log n.
The density λ > 2 ensures a high probability that all blocks have Ω(log n) vertices, as we later show
in (4.3). Next, we use the Pairwise Classify subroutine to label the first block (Line 3). Here,
we select an arbitrary vertex u0 ∈ _V1 and set_ _σ(u0) = 1. The labels of other vertices u ∈_ _V1 are_
labeled by counting common neighbors with u0, among the vertices in V1. Next, the labeling of V1 is
propagated to other blocks Vi for i 2 utilizing the edges between � _Vi_ 1 and Vi and the estimated
_≥_ _−_
labeling on Vi 1, by thresholding degree profiles with respect to Vi 1 according to Algorithm 3 (Lines
_−_ _−_
4-5). The reference set S in Algorithm 3 plays the role of Vi 1 and S[′] plays the role of Vi. Intuitively,
_−_
if a > b, a vertex tends to exhibit more edges and fewer non-edges within its own community while
having fewer edges and more non-edges with the other community. Conversely, if a < b, the opposite
observation holds. In order to classify the vertices in Vi, we use edges from Vi to the larger set of


-----

_u_ _Vi_ 1 : _σ(u) = 1_ and _u_ _Vi_ 1 : _σ(u) =_ 1, rather than using all edges between Vi and
_{_ _∈_ _−_ _}_ _{_ _∈_ _−_ _−_ _}_
_Vi−1, which simplifies the analysis. In Theorem 4.15, we will demonstrate that Phase I achieves_
almost-exact recovery on � _G under the conditions in Theorem �_ 2.4.


**Algorithm 1 Exact recovery for the GSBM (d = 1 and λ > 2)**


**Input: G ∼** GSBM(λ, n, a, b, 1) where λ > 2.
**Output: An estimated community labeling** _σ : V →{−1, 1}._


1: Phase I:

2: Partition the interval [−n/2, n/2] into 2 �n/ log n blocks[2] of volume log n/2 each. Let Bi be the
_ith block and Vi be the set of vertices in Bi for i ∈_ [2n/ log n].


3: Apply Pairwise Classify (Algorithm 2) on input G, V1, a, b to obtain a labeling _σ of V1._

4: for i = 2, · · ·, 2n/ log n do

�


4: for i = 2, · · ·, 2n/ log n do

5: Apply Propagate (Algorithm 3) on input G, Vi 1, Vi to determine the labeling �σ on Vi.
_−_


6: Phase II:


7: for u ∈ _V do_

8: Apply Refine (Algorithm 4) on input G, _σ, u to obtain_ _σ(u)._


**Algorithm 2 Pairwise Classify**


**Input: Graph G = (V, E), vertex set S ⊂** _V, parameters a, b ∈_ [0, 1] with a ̸= b.

1: Choose an arbitrary vertex u0 _S, and set_ _σ(u0) = 1._
_∈_

2: for u _S_ _u0_ **do**
_∈_ _\ {_ _}_

3: **if** _v_ _S_ _u, u0_ : _u0, v_ _,_ _u, v_ _E_ � > (a + b)[2]( _S_ 2)/4 then
_|{_ _∈_ _\ {_ _}_ _{_ _}_ _{_ _} ∈_ _}|_ _|_ _| −_

4: Set _σ(u) = 1._

5: **else**

6: Set �σ(u) = 1.
_−_


**Algorithm 3 Propagate**


**Input: Graph G = (V, E), mutually visible sets of vertices S, S[′]** _⊂_ _V with S ∩_ _S[′]_ = ∅, where S is
labeled according to _σ._


1: if |{v ∈ _S :_ _σ(v) = 1}| ≥|{v ∈_ _S :_ _σ(v) = −1}| then_

2: **for u** _S[′]_ **do** �
_∈_

3: **if a > b �** and d[+]1 [(][u,][ �][σ, S][)][ ≥] �[(][a][ +][ b][)][ · |{][v][ ∈] _[S][ :][ �][σ][(][v][) = 1][}|][/][2][ then]_

4: Set _σ(u) = 1._

5: **else if a < b and d[+]1** [(][u,][ �][σ, S][)][ <][ (][a][ +][ b][)][ · |{][v][ ∈] _[S][ :][ �][σ][(][v][) = 1][}|][/][2][ then]_

6: Set �σ(u) = 1.

7: **else**

8: Set �σ(u) = 1.
_−_


9: else


10: **for u ∈** _S[′]_ **do**

11: **if a > b and d[+]1[(][u,][ �][σ, S][)][ ≥]** [(][a][ +][ b][)][ · |{][v][ ∈] _[S][ :][ �][σ][(][v][) =][ −][1][}|][/][2][ then]_
_−_

12: Set _σ(u) = −1._

13: **else if a < b and d[+]1[(][u,][ �][σ, S][)][ <][ (][a][ +][ b][)][ · |{][v][ ∈]** _[S][ :][ �][σ][(][v][) =][ −][1][}|][/][2][ then]_
_−_

14: Set �σ(u) = 1.
_−_


-----

**Algorithm 4 Refine**


**Input: Graph G ∼** GSBM(λ, n, a, b, d), vertex u ∈ _V, labeling_ _σ : V →{−1, 0, 1}._
**Output: An estimated labeling** _σ(u) ∈{−1, 1}._
�


_a_ 1 _a_
1: Set _σ(u) = sign_ log _d[+]1_ [(][u,] 1[(][u,] + log _−_ _d[−]1_ [(][u,] 1[(][u,] _._

_b_ _−_ 1 _b_ _−_

�

� _−_
� �� � � �� �[�]

_[σ][)][ −]_ _[d][+]_ _[σ][)]_ _[σ][)][ −]_ _[d][−]_ _[σ][)]_


In Phase II, we refine the almost-exact labeling _σ obtained from Phase I. Our refinement procedure_
mimics the so-called genie-aided estimator [1], which labels a vertex u knowing the labels of all
other vertices (i.e., _σ0(v)_ : v _V_ _u_ ). The degree profile relative to the ground-truth labeling, �
_{_ _∈_ _\ {_ _}}_
_d(u, σ0), is random and depends on realizations of node locations and edges in G and community_
assignment σ0. We use D ∈ R[4] to denote the vector representing the four random variables in
_d(u, σ0)._ Then D is characterized by a multi-type Poisson distribution such that conditioned
on {σ0(u) = 1}, D ∼ Poisson(λνd log n[a, 1 − _a, b, 1 −_ _b]/2) and conditioned on {σ0(u) = −1},_
_D ∼_ Poisson(λνd log n[b, 1 − _b, a, 1 −_ _a]/2). Given a realization D = d(u, σ0), we pick the most likely_
hypothesis to minimize the error probability; that is,


_σgenie(u) = argmaxs∈{1,−1}_ P(D = d(u, σ0) | σ0(u) = s)


For convenience, let


In short, we have σgenie(u) = sign(τ (u, σ0)). The genie-aided estimator motivates the Refine
subroutine (Algorithm 4) in Phase II that assigns _σ(u) = sign(τ_ (u, _σ)) for any u ∈_ _V . Since_ _σ makes_
few errors compared with σ0, for any u _V, its degree profile d(u,_ _σ) is close to d(u, σ0). Thus,_
_∈_
_d(u,_ _σ) is well-approximated by the aforementioned multi-type Poisson distribution. �_ � �
�


**Modified algorithm for general λ > 1.** If 1 < λ < 2, partitioning the interval into blocks of
length log n/2, as done in Line 2 of Algorithm 1, fails. This is because each of the 2n/ log n blocks
is independently empty with probability e[−][λ][ log][ n/][2] = n[−][λ/][2] and −λ/2 > −1, leading to a high
probability of encountering empty blocks, and thus a failure of the propagation scheme. To address
this, we instead adopt smaller blocks of length χ log n, where χ < (1 − 1/λ)/2, for any λ > 1. We
only attempt to label blocks with sufficiently many vertices, according to the following definition.
For the rest of the paper, let V (B) _V denote the set of vertices in a subregion B_ _d,n._
_⊂_ _⊂S_


**Definition 3.2 (Occupied block). Given any δ > 0, a block B ⊂Sd,n is δ-occupied if |V (B)| > δ log n.**
Otherwise, B is δ-unoccupied.


We will show that for sufficiently small δ > 0, all but a negligible fraction of blocks are δ-occupied.
As a result, achieving almost-exact recovery in Phase I only requires labeling the vertices within the
occupied blocks. To ensure successful propagation, we introduce a notion of visibility. Two blocks
_Bi, Bj ∈Sd,n are mutually visible, defined as Bi ∼_ _Bj, if_


-----

Thus, if Bi ∼ _Bj, then any pair of vertices u ∈_ _Bi and v ∈_ _Bj are at a distance at most (log n)[1][/d]_ of
each other. In particular, if Bj is labeled and Bi ∼ _Bj, then we can propagate labels to Bi._


Similar to the case of λ > 2, we propagate labels from left to right. Despite the presence of
unoccupied blocks, we establish that if λ > 1 and χ is chosen as above, each block Bi following the
initial B1 has a corresponding block Bj (j < i) to its left that is occupied and satisfies Bi ∼ _Bj. We_
thus modify Lines 4-5 so that a given block Bi is labeled by one of the visible, occupied blocks to its
left (Figure 1). The modification is formalized in the general algorithm (Algorithm 5) given below.


#### 3.2 Exact recovery for general d.


The propagation scheme becomes more intricate for d ≥ 2. For general d, we divide the region
_Sd,n into hypercubes[3]_ with volume parametrized as χ log n. The underlying intuition for successful
propagation stems from the condition λνd > 1. This condition ensures that the graph formed by
connecting all pairs of mutually visible vertices is connected with high probability, a necessary
condition for exact recovery. Moreover, the condition ensures that every vertex has Ω(log n) vertices
within its visibility radius of (log n)[1][/d]. It turns out that the condition λνd > 1 also ensures that
blocks of volume χ log n for χ > 0 sufficiently small satisfy the same connectivity properties.


To propagate the labels, we need a schedule to visit all occupied blocks. However, the existence
of unoccupied blocks precludes the use of a predefined schedule, such as a lexicographic order scan.
Instead, we employ a data-dependent schedule. The schedule is determined by the set of occupied
blocks, which in turn is determined in Step 1 of Definition 2.1. Crucially, the schedule is thus
independent of the community labels and edges, conditioned on the number of vertices in each block.
We first introduce an auxiliary graph H = (V _[†], E[†]), which records the connectivity relation among_
occupied blocks.


**Definition 3.3 (Visibility graph). Consider a Poisson point process V** _d,n, the (χ log n)-block_
_⊂S_
partition of _d,n,_ _Bi_ _i=1_, corresponding vertex sets _Vi_ _i=1_, and a constant δ > 0. The
_S_ _{_ _}[n/][(][χ][ log][ n][)]_ _{_ _}[n/][(][χ][ log][ n][)]_
(χ, δ)-visibility graph is denoted by H = (V _, E[†]), where the vertex set V_ = _i_ [n/(χ log n)] : _Vi_

_[†]_ _[†]_ _{_ _∈_ _|_ _| ≥_
_δ log n_ consists of all δ-occupied blocks and the edge set is given by E[†] = _i, j_ : i, j _V_ _, Bi_ _Bj_ .
_}_ _{{_ _}_ _∈_ _[†]_ _∼_ _}_


We adopt the standard connectivity definition on the visibility graph. Lemma 4.8 shows that the
visibility graph of the Poisson point process underlying the GSBM is connected with high probability.
Based on this connectivity property, we establish a propagation schedule as follows. We construct
a spanning tree of the visibility graph and designate a root block as the initial block. We specify
an ordering of V = _i1, i2, . . ._ according to a tree traversal (e.g., breadth-first search). Labels are

_[†]_ _{_ _}_
propagated according to this ordering, thus labeling vertex sets Vi1, Vi2, (see Figure 1). Letting
_· · ·_
_p(i) denote the parent of vertex i ∈_ _V_ _[†]_ according to the rooted tree, we label Vij using Vp(ij ) as
reference. Importantly, the visibility graph and thus the propagation schedule is determined only by
the locations of vertices, independent of the labels and edges between mutually visible blocks.


**Algorithm 5 Exact recovery for the GSBM**


**Input: G ∼** GSBM(λ, n, a, b, d).
**Output: An estimated community labeling** _σ : V →{−1, 1}._


**Input: G ∼** GSBM(λ, n, a, b, d).


**Output: An estimated community labeling** _σ : V →{−1, 1}._


1: Phase I:


2: Take small enough χ, δ > 0, satisfying the conditions to be specified in (4.1) and (4.2) respectively.

3: Partition the region Sd,n into n/(χ log n) blocks of volume χ log n each. Let Bi be the ith block
and Vi be the set of vertices in Bi for i ∈ [n/(χ log n)].


-----

4: Form the associated visibility graph H = (V _[†], E[†])._


5: if H is disconnected then

6: Return FAIL.

7: Find a rooted spanning tree of H, ordering V = _i1, i2,_ in breadth-first order.

_[†]_ _{_ _· · · }_

8: Apply Pairwise Classify (Algorithm 2) on input G, Vi1, a, b to obtain a labeling _σ of Vi1._

9: for j = 2, · · ·, |V _[†]| do_

10: Apply Propagate (Algorithm 3) on input G, Vp(ij ), Vij to determine the labeling �σ on Vi.

11: for u ∈ _V \ (∪i∈V †Vi) do_

12: Set _σ(u) = 0._ �


13: Phase II:


14: for u ∈ _V do_


15: Apply Refine (Algorithm 4) on input G, _σ, u to determine_ _σ(u)._


Algorithm 5 presents our algorithm for the general case. We partition the region Sd,n into blocks
with volume χ log n, for a suitably chosen χ > 0. A threshold level of occupancy δ > 0 is specified.
The value of χ is carefully chosen to ensure that the visibility graph H is connected with high
probability in Line 5. In Line 8, we label an initial δ-occupied block, corresponding to the root
of H, using the Pairwise Classify subroutine. In Lines 9-10, we label the occupied blocks in
the tree order determined in Line 7, using the Propagate subroutine. Those vertices appearing in
unoccupied blocks are assigned a label of 0. At the end of Phase I, we obtain a first-stage labeling
_σ_ : V →{−1, 0, 1}, such that with high probability, all occupied blocks are labeled with few mistakes.
Finally, Phase II refines the almost-exact labeling _σ to an exact one_ _σ according to Algorithm 4._
�


To analyze the runtime, note that the number of edges (input size) is Θ(n log n) with high
probability. The visibility graph H = (V _[†], E[†]) can be formed in �_ _O �(n/ log n) time, since |V_ _[†]| =_
_O(n/ log n) and each vertex has at most Θ(1) possible neighbors. If H is connected, a spanning_
tree can be found in O(|E[†]| log(|E[†]|)) time using Kruskal’s algorithm, and |E[†]| = O(n/ log n). The
subsequent Pairwise Classify subroutine goes over all edges of the vertices in V1 to count the
common neighbors, with a runtime of O(log[2] _n). Next, the Propagation subroutine requires counting_
edges and non-edges from any given vertex in an occupied block to the vertices in its reference
block, yielding a runtime of O(n log n). Finally, Refine runs in O(n log n) time, since each visible
neighborhood contains O(log n) vertices. We conclude that Algorithm 5 runs in O(n log n) time,
which is linear in the number of edges.


-----

#### 3.3 Proof outline.


We outline the analysis of Algorithm 5. We begin with Phase I. Our goal is to show that in addition
to achieving almost exact recovery stated in Theorem 2.4, Phase I also satisfies an error dispersion
property. Let N (u) = {v ∈ _V, ∥u −_ _v∥≤_ (log n)[1][/d]} for a vertex u. Namely, for any η > 0, we can
take suitable χ, δ > 0 so that with high probability, every vertex has at most η log n incorrectly
classified vertices in its local neighborhood N (u). Theorem 4.15 will present the formal results.


**Phase I: Connectivity of the visibility graph.** We first establish that the block division
specified in Algorithm 5 ensures that the resulting visibility graph H = (V _[†], E[†]) is connected._
Elementary analysis shows that any fixed subregion of R[d] with volume ν log n contains Ω(log n)
vertices with probability 1 − _o(n[−][1]), whenever ν > 1/λ. A union bound over all vertices then_
implies that all vertices’ neighborhoods have Ω(log n) vertices. In the special case of d = 1, the left
neighborhood of a given vertex has volume log n. The observation with ν = 1 implies that when
_λ > 1, the left neighborhood of every vertex has Ω(log n) points. In fact, we can make a stronger_
claim: if the block lengths are chosen to be sufficiently small (according to (4.1)), then we can ensure
that for a given vertex v ∈ _Vi, there are Ω(log n) vertices among {Vj : Bj ∼_ _Bi, j ̸= i}. In turn, by_
an appropriate choice of δ (according to (4.2)), for a given block Bi, there is at least one δ-occupied,
visible block to its left. Hence, the visibility graph is connected, as shown in Proposition 4.5.


However, the analysis becomes more intricate when d ≥ 2. In particular, while a lexicographic
order propagation schedule succeeds for d = 1, it fails for d ≥ 2. For example, when d = 2, we cannot
say that every vertex has Ω(log n) vertices in the top left quadrant of its neighborhood, since the
volume of the quadrant is only νd log n/4. We therefore establish connectivity of H using the fact
that if H is disconnected, then H must contain an isolated connected component. The key idea is
that if there is an isolated connected component in H, then the corresponding occupied blocks in R[d]

must be surrounded by sufficiently many unoccupied blocks. However, as Lemma 4.7 shows, there
cannot be too many adjacent unoccupied blocks, which prevents the existence of isolated connected
components. As a result, the visibility graph is connected, as shown in Lemma 4.8.


**Phase I: Labeling the initial block.** We show that the Pairwise Classify (Line 8) subroutine
ensures the successful labeling for Vi1. Since we only need to determine community labels up to a
global flip, we are free to set _σ(u0) = 1 for an arbitrary u0_ _Vi1. For any u_ _Vi1_ _u0_, where
_∈_ _∈_ _\ {_ _}_
_|Vi1| = m1, Lemma 4.9 shows that the number of common neighbors of u and u0 follows a binomial_
distribution; in particular, Bin �(m1 2, (a[2] + b[2])/2) if σ0(u) = σ0(u0) and Bin(m1 2, ab) otherwise.
_−_ _−_
We thus threshold the number of common neighbors in order to classify u relative to u0. Lemma 4.10

bounds the probability of misclassifying a given vertex u _Vi1_ _u0_, using Hoeffding’s inequality.
_∈_ _\ {_ _}_
A union bound then implies that all vertices are correctly classified with high probability.


**Phase I: Propagating labels among occupied blocks.** We show that the Propagate subroutine
ensures that _σ makes at most M mistakes in each occupied block, where M is a suitable constant._
Our analysis reduces to bounding the probability that for a given i ∈ _V_ _[†], the estimator_ _σ makes_
more than M � mistakes on Vi, conditioned on making no more than M mistakes on Vp(i). In order
to analyze the probability that a given vertex v _Vi is misclassified, we condition on the �_ _label_
_∈_
_configuration of Vp(i), meaning the number of vertices labeled s according to σ0(_ ) and t according to

_·_
_σ0(u0)σ(_ ), for s, t 1, +1 . We find a uniform upper bound on the probability of misclassifying

_·_ _∈{−_ _}_
an individual vertex v ∈ _Vi when applying the thresholding test given in Algorithm 3, over all label_
configurations of� _Vp(i) with at most M mistakes. To bound the total number of mistakes in Vi,_
observe that the labels of all vertices in Vi are decided based on disjoint subsets of edges between Vi


-----

and Vp(i). Therefore, conditioned on the label configuration of Vp(i), the number of mistakes in Vi
can be stochastically dominated by a binomial random variable. It follows by elementary analysis
that the number of mistakes in Vi is at most M with probability 1 − _o(n[−][1]), as long as M is a_
suitably large constant.


**Phase II: Refining the labels.** Our final step is to refine the initial labeling _σ from Phase I into_
a final labeling _σ. Unfortunately, conditioning on a successful labeling_ _σ destroys the independence_
of edges, making it difficult to bound the error probability of _σ. This issue can be remedied using �_
a technique called � _graph splitting, used in the two-round procedure of [ �_ 5]. Graph splitting is a
procedure to form two graphs, G1 and G2, from the original input graph. A given edge in � _G is_
independently assigned to G1 with probability p, and G2 with probability 1 − _p, for p chosen so that_
almost exact recovery can be achieved on G1, while exact recovery can be achieved on G2. Since
the two graphs are nearly independent, conditioning on the success of almost exact recovery in G1
essentially maintains the independence of edges in G2.
While we believe that our Phase I algorithm, along with graph splitting, would achieve the
information-theoretic threshold in the GSBM, we instead directly analyze the robustness of Poisson
testing. Specifically, we bound the error probability of labeling a given vertex v ∈ _V with respect_
to the worst-case labeling over all labelings that differ from σ0 on at most η log n vertices in the
neighborhood of v. Since _σ makes at most η log n errors with probability 1 −_ _o(1/n) (Theorem 4.15),_
we immediately obtain a bound on the error probability of _σ(v)._
�


The proof in Section 5 bounds the worst-case error probability. We define x = λνd[a, 1 _a, b, 1_
_−_ _−_
_b]/2 and y = λνd[b, 1_ _b, a, 1_ _a]/2, so that D_ _σ0(u) = 1 �_ Poisson(x) and D _σ0(u) =_ 1
_−_ _−_ _| {_ _} ∼_ _| {_ _−_ _} ∼_
Poisson(y). The condition λνd(1 _√ab_ (1 _a)(1_ _b)) > 1 in Theorem 2.2 is equivalent to_
_−_ _−_ _−_ _−_

_D+(x_ _y) > 1, where D+(x_ _y) is the Chernoff–Hellinger divergence of x and y [5]. To provide_
_∥_ _∥_ �
intuition for bounding the error probability at a given vertex u ∈ _V, consider the genie-aided_
estimator σgenie(u), and assume σ0(u) = 1 without loss of generality. Recalling the definition of τ
(3.2), the estimator σgenie(u) makes a mistake when τ (u, σ0) ≤ 0. It can be shown that this occurs
with probability at most n[−][D][+][(][x][∥][y][)]. Viewing the worst-case labeling σ differing from σ0 on at most
_η log n vertices as a perturbation of σ0, we show that τ_ (u, σ) 0 implies τ (u, σ0) _ρη log n for a_
_≤_ _≤_
certain constant ρ. Similarly, the probability of such a mistake is at most n[−][D][+][(][x][∥][y][)+][ρη/][2]. Thus,
for small η > 0, the condition D+(x _y) > 1 and a union bound over all vertices yields an error_
_∥_
probability of o(1).


### 4 Phase I: Proof of almost exact recovery


In this section, we prove Theorem 2.4. We begin by defining sufficiently small constants χ and δ
used in Algorithm 5. We define χ to satisfy the following condition, relying on λ and d:


_d_

The first condition is satisfiable since limχ 0 νd 1 3√dχ[1][/d]/2 = νd and we have νd > (νd +1/λ)/2
_→_ _−_

when λνd > 1. The second one is also satisfiable since 1d=1 + νd 1d 2 = 1 > 1/λ if d = 1 and
otherwise 1d=1 + νd 1d 2 = νd > 1/λ, under the conditions of Theorems� � _·_ _≥_ 2.2 and 2.4. Associated
_·_ _≥_
with the choice of χ, there is a constant δ[′](or _δ for d_ 2) > 0 such that for any block Bi, its visible
_≥_
blocks _j∈V_ _[{][V][j][ :][ B][j][ ∼]_ _[B][i][}][ contain at least][ δ][′][ log][ n][ (or][ �][δ][ log][ n][) vertices with probability][ 1][ −]_ _[o][(][n][−][1][)][.]_
We define Rd = 1 − _√dχ[1][/d]/2. The first condition in[�]_ (4.1) implies that _√dχ[1][/d]/2 < 1/3 and thus_

_Rd > 0[�]. With specific values of δ[′]_ and _δ to be determined in Proposition 4.5 and Lemma 4.6,_


-----

respectively, we define δ such that


Propositions 4.5 and 4.8 will present the connectivity properties of δ-occupied blocks of volume
_χ log n, for χ and δ satisfying the conditions in (4.1) and (4.2), respectively._


We now record some preliminaries (see [9]).


**Lemma 4.1 (Chernoff bound, Poisson). Let X ∼** _Poisson(µ) with µ > 0. For any t > 0,_


_For any 0 < t < µ, we have_


**Lemma 4.2 (Hoeffding’s inequality). Let X1, · · ·, Xn be independent bounded random variables**
_with values Xi ∈_ [0, 1] for all 1 ≤ _i ≤_ _n. Let X =_ _i=1_ _[X][i][ and][ µ][ =][ E][[][X][]][. Then for any][ t][ ≥]_ [0][, it]
_holds that_

[�][n]


**Lemma 4.3 (Chernoff upper bound). Let X1, · · ·, Xn be independent Bernoulli random variables.**
_Let X =_ _i=1_ _[X][i][ and][ µ][ =][ E][(][X][)][. Then for any][ t >][ 0][, we have]_

[�][n]


We also define a homogeneous Poisson point process used to generate locations as described in
Definition 2.1.


**Definition 4.1 ([23]). A homogeneous Poisson point process with intensity λ on S ⊆** R[d] is a random
countable set Φ := _v1, v2,_ _S such that_
_{_ _· · · } ⊂_


1. For any bounded Borel set B ⊂ R[d], the count NΦ(B) := |Φ ∩ _B| = |{i ∈_ N : vi ∈ _B}| has a_
Poisson distribution with mean λvol(B), where vol(B) is the measure (volume) of B.

2. For any k ∈ N and any disjoint Borel sets B1, · · ·, Bk ⊂ R[d], the random variables NΦ(B1),
_, NΦ(Bk) are mutually independent._

_· · ·_


In the GSBM, the set of locations V = _v1, v2,_ are generated by a homogeneous Poisson point
_{_ _· · · }_
process with intensity λ on _n,d. The established properties guarantee that_ _V_ follows Poisson(λn).
_S_ _|_ _|_
Moreover, conditioned on _V_, the locations _vi_ _i_ [ _V_ ] are independently and uniformly distributed
_|_ _|_ _{_ _}_ _∈_ _|_ _|_
in _n,d. This gives a simple construction of a Poisson point process as follows:_
_S_


1. Sample NV Poisson(λn);
_∼_


2. Sample v1, · · ·, vNV independently and uniformly in the region Sn,d.


This procedure ensures that the resulting set _v1,_ _, vNV_ constitutes a Poisson point process as
_{_ _· · ·_ _}_
desired.


-----

#### 4.1 Connectivity of the visibility graph.


In this subsection, we establish the connectivity of the visibility graph H = (V _[†], E[†]) from Line 4 of_
Algorithm 5. The following lemma shows that regions of appropriate volume have Ω(log n) vertices
with high probability.


**Lemma 4.4. For any fixed subset B ⊂Sd,n with a volume ν log n such that λν > 1, there exist**
_constants 0 < γ < λν and ϵ > 0 such that_


_Proof. For a subset B with vol(B) = ν log n, we have |V (B)| ∼_ Poisson(λν log n). To show the
lower bound, we define a function g : (0, λν] → R as g(x) = x(log x − log(λν)) + λν − _x. It is easy_
to check that g is continuous and decreases on (0, λν] with limx 0 g(x) = λν and g(λν) = 0. When
_→_
_λν > 1, it holds that limx_ 0 g(x) = λν > (1 + λν)/2 and thus there exists a constant γ (0, λν)
_→_ _∈_
such that g(γ) > (1 + λν)/2. Thus, the Chernoff bound in Lemma 4.1 yields that


Taking ϵ = (λν − 1)/2 > 0 concludes the proof.


**4.1.1** **The simple case when d = 1 and λ > 1.**


**4.1.1** **The simple case when d = 1 and λ > 1.**


We start with the simple case when d = 1.


**An example when λ > 2.** We first study an example when d = 1 and λ > 2. If λ > 2 and
vol(Bi) = log n/2, we have λvol(Bi)/ log n > 1, and thus Lemma 4.4 ensures the existence of positive
constants γ and ϵ such that P(|Vi| > γ log n) ≥ 1 − _n[−][1][−][ϵ]_ for all i ∈ [2n/ log n]. Thus, the union
bound gives that


Since all blocks are γ-occupied, the (1/2, γ)-visibility graph H = (V _[†], E[†]) is trivially connected._


**General case when λ > 1.** For small density λ, we partition the interval into small blocks and
establish the existence of visible occupied blocks on the left side of each block.


**Proposition 4.5. If d = 1 and λ > 1, with 0 < χ < (1** 1/λ)/2, we consider the blocks _Bi_ _i=1_
_−_ _{_ _}[n/][(][χ][ log][ n][)]_
_obtained from Line 3 in Algorithm 5. Then there exists a constant δ[′]_ _> 0 such that for any 0 < δ < δ[′]χ,_
_it holds that_


_It follows that the (χ, δ)-visibility graph is connected with high probability._


-----

_Proof. For any i ∈_ [n/(χ log n)], we define Ui = _j : j<i,Bj_ _∼Bi_ _[B][j][ as the union of visible blocks on]_
the left-hand side of Bi. We have vol(Ui) = ( 1/χ 1)χ log n (1 2χ) log n and λvol(Ui)/ log n
_⌊_ _⌋−_ _≥_ _−_ _≥_
_λ(1_ 2χ) > 1 when λ > 1 and χ < (1 1/λ)[�]/2. Thus, Lemma 4.4 ensures the existence of
_−_ _−_
positive constants δ[′] and ϵ such that P(| _j : j<i,Bi∼Bj_ _[V][j][| ≤]_ _[δ][′][ log][ n][)][ ≤]_ _[n][−][1][−][ϵ][.]_ We note that
_j : j < i, Bj_ _Bi_ ( 1/χ 1) 1/χ. Thus, we take 0 < δ < δ[′]χ and obtain that
_|{_ _∼_ _}| ≤_ _⌈_ _⌉−_ _≤_

[�]


Therefore, the union bound over all i ∈ [n/(χ log n)] gives


**4.1.2** **General case when d ≥** 2 and λνd > 1.


We now study general cases. We first show that for any block B, the set of surrounding visible blocks
_{length of its longest diagonal is given byB[′]_ : B ∼ _B[′], B[′]_ ≠ _B} contains Ω(log n) vertices. For any block√d(χ log n)[1][/d]. Recall the definition of Bi ⊂Sd,n with vol R(dB = 1i) = − χ√ logdχ n[1][/d], the/2,_

and let Ci be the ball of radius Rd(log n)[1][/d] centered at the center of Bi. Observe that


It follows that if Bj ⊆ _Ci, then Bi ∼_ _Bj. Also, Ci contains all blocks Bj ∼_ _Bi (see Figure 2). We_
define


as the union of all visible blocks to Bi, excluding Bi itself. Observe that as χ → 0, the volume of the
blue region approaches the volume of Ci. The following lemma quantifies this observation, showing
that our conditions on χ guarantee that Ui (and any set with the same volume as Ui) will contain
sufficiently many vertices.


**Lemma 4.6. If χ satisfies the condition in (4.1) and λνd > 1, there exist positive constants** _δ and ϵ,_
_depending on λ and d, such that for any subset S ∈Sd,n with vol(S) = vol(Ui), we have_

[�]


_Proof. We first evaluate the volume of Ui ⊂_ _Ci.[4]_ We define Rd[′] [=][ R][d][ −] _√dχ[1][/d]_ and Ci[′] [as the]

ball centered at the center of Bi with a radius Rd[′] [(][log][ n][)][1][/d][. The condition in][ (][4.1][)][ implies that]


-----

3It follows that vol√dχ[1][/d]/2 < 1 and thus(Ui ∪ _B Ri)d[′] ≥[>]vol[ 0][. Based on geometric observations, we note that](Ci[′][) =][ ν][d][(][R]d[′]_ [)][d][ log][ n][, and thus vol][(][U][i][)][ ≥] [(][ν][d][(][R][ C]d[′] [)][d]i[′][ −][⊂] _[χ][U][) log][i][ ∪]_ _[B][ n][i][.][ ⊂]_ _[C][i][.]_

We now show that when λνd > 1, the conditions in (4.1) imply λ(νd(Rd[′] [)][d][ −] _[χ][)][ >][ 1][ by observing]_
the following relations:


In summary, we have shown that vol(S) = vol(Ui) ≥ (νd(Rd[′] [)][d][ −] _[χ][)][ log][ n][ and][ λ][(][ν][d][(][R]d[′]_ [)][d][ −] _[χ][)][ >][ 1][.]_
Thus, Lemma 4.4 ensures the existence of positive constants _δ and ϵ such that P(|V (S)| >_ _δ log n) >_
1 − _n[−][1][−][ϵ]._

[�] [�]


Henceforth, we use the term “occupied block” to refer to δ-occupied blocks, as well as “unoccupied
block”, with the constant threshold δ = δ(λ, d) defined in (4.2) in the rest of the section. We define
_K =_ _j : Bj_ _Ui_ as the number of blocks in Ui, a constant relying on λ and d. We note that
_|{_ _⊂_ _}|_
_K_ _νd(Rd)[d]/χ_ 1 < _δ/δ since Ui_ _Bi_ _Ci. The key observation in establishing connectivity is_
_≤_ _−_ _∪_ _⊂_
that there cannot be a large cluster of unoccupied blocks.

[�]


**Definition 4.2 (Cluster of blocks). Two blocks are adjacent if they share an edge or a corner. We**
say that a set of blocks B is a cluster if for every B, B[′] _∈B, there is a path of blocks of the form_
(B = Bj1, Bj2, . . ., Bjm = B[′]), where Bjk ∈B for k ∈ [m] and Bjk _, Bjk+1 are adjacent._


The following lemma shows that all clusters of unoccupied blocks have fewer than K blocks,
with high probability. This also implies that Ui contains at least one occupied block for each i.


**Lemma 4.7. Suppose d ≥** 2 and λνd > 1. Let Y be the size of the largest cluster of unoccupied
_blocks produced in Line 3 in Algorithm 5. Then P(Y < K) = 1 −_ _o(1)._


_Proof. We first bound the probability that all K blocks in any given set are unoccupied. For any set_
of K blocks {Bjk _}k[K]=1[, we have]_


-----

where the second inequality holds due to K < _δ/δ and the last inequality follows from Lemma 4.6_
and the fact that vol( _k=1_ _[B][j]k_ [) =][ vol][(][U][i][)][.]

[�]

_[K]_



[�]
Let Z be the number of unoccupied block clusters with a size of K. Then we have P(Y ≥ _K) =_
P(Z ≥ 1). Let S be the set of all possible shapes of clusters of blocks with a size of K. Clearly, |S|
is a constant depending on K and d. For any s S, i [n/(χ log n)], and j [K], we define _s,i,j_
_∈_ _∈_ _∈_ _Z_
as the event that there is a cluster of unoccupied blocks, characterized by shape s with block Bi
occupying the jth position. Due to (4.4), we have P(Zs,i,j) ≤ _n[−][1][−][ϵ]. Thus, the union bound gives_


Finally, we establish the connectivity of the visibility graph.


**Proposition 4.8. Suppose that d ≥** 2 and λνd > 1. Let V ⊂Sd,n be a Poisson point process on Sd,n
_with intensity λ. Then for χ and δ given in (4.1) and (4.2), respectively, the (χ, δ)-visibility graph_
_H on V is connected with probability 1 −_ _o(1)._


_Proof. For a visibility graph H = (V_ _[†], E[†]), we say that S ⊂_ _V_ _[†]_ is a connected component if
the subgraph of H induced on S is connected. Let E be the event that H contains an isolated
connected component. Formally, E is the event that there exists S ⊂ _V_ _[†][ 5]_ such that (1) S ̸= ∅ and
_S ̸= V_ _[†]; (2) S is a connected component; (3) for all i ∈_ _S, j ̸∈_ _S we have {i, j} ̸∈_ _E[†]. Observe that_
_{H is disconnected} = E._


For any S ̸= ∅ and S ⊂ _V_ _[†]_ to be an isolated connected component, it must be completely
surrounded by a cluster of unoccupied blocks. In other words, all blocks in the cluster ([�]i _S_ _[U][i][)][ \]_
_∈_
([�]i _S_ _[B][i][)][ must be unoccupied. We next show that for any isolated, connected component][ S][, we]_
_∈_
have |{j : Bj ⊂ ([�]i∈S _[U][i][)][ \][ (][�]i∈S_ _[B][i][)][}| ≥]_ _[K][; that is, the number of unoccupied blocks visible to an]_
isolated connected component is at least K.


We prove the claim by induction on |S|. In fact, we prove it for S that is isolated, but not
necessarily connected. The claim holds true whenever |S| = 1 by the definition of K. Suppose that
the claim holds for every isolated component with k blocks. Consider an isolated component S, with
_|S| = k + 1. Let F = ([�]i∈S_ _[B][i][)][ �][(][�]i∈S_ _[U][i][)][ be the collective “footprint” of all elements of][ S][ along]_
with the surrounding unoccupied blocks. For each j ∈ _S, let Fj = ([�]i∈S,i≠_ _j_ _[B][i][)][ �][(][�]i∈S,i≠_ _j_ _[U][i][)][ be]_
the footprint of all blocks in S excluding j. Let Gj be the graph formed from G by removing all
vertices from Vj, thus rendering Vj unoccupied. Observe that there must exist some j[⋆] _∈_ _S such that_
_Fj[⋆]_ = F and Fj[⋆] _F_, as the regions _Bi_ _Ui_ _i_ _S are translations of each other. Since S_ _j[⋆]_ is an
_̸_ _⊂_ _{_ _∪_ _}_ _∈_ _\ {_ _}_
isolated component in Gj[⋆], the inductive hypothesis implies that S _j[⋆]_ has at least K surrounding
_\ {_ _}_
unoccupied blocks in Gj[⋆]. Comparing Gj[⋆] to G, there are two cases (see Figure 3 for examples
in 2,n). Case I. In the first case, F _Fj[⋆]_ contains at least one unoccupied block. In that case,
_S_ _\_
the inclusion of Vj[⋆] changes one block from unoccupied to occupied, and increases the number of
surrounding unoccupied blocks by at least one. Thus, S contains at least K surrounding unoccupied
blocks. Case II. In the second case, F _Fj[⋆]_ contains only occupied blocks. Since there are k + 1
_\_
total occupied blocks in F and k of them are in Fj[⋆], we have F _Fj[⋆]_ = Bj[⋆], so that Bj[⋆] _Fj[⋆]_ = .
_\_ _∩_ _∅_
In this case, the set of K surrounding unoccupied blocks in Fj[⋆] remains unoccupied in F .


Thus, E implies {Y ≥ _K}. The result follows from Lemma 4.7._


-----

In summary, Propositions 4.5 and 4.8 establish the connectivity of visibility graphs for cases
when d = 1 and λ > 1, or d ≥ 2 and λνd > 1, ensuring successful label propagation in the algorithm.
For convenience, let H = {H is connected}. We conclude that P(H) = 1 − _o(1)._


#### 4.2 Labeling the initial block.


We now prove that the Pairwise Classify subroutine (Line 8 of Algorithm 5) ensures, with high
probability, the correct labeling of all vertices in the initial block Vi1. Let Nu0,u = |{v ∈ _Vi1 : {v, u0} ∈_
_E, {v, u} ∈_ _E}| be the number of common neighbors of u0 and u within Vi1._

**Lemma 4.9. For any vertex u** _Vi1_ _u0_ _, it holds that_
_∈_ _\ {_ _}_


_1. Conditioned on σ0(u) = σ0(u0) and_ _Vi1_ = mi1, we have Nu0,u Bin(mi1 2, (a[2] + b[2])/2).
_|_ _|_ _∼_ _−_

_2. Conditioned on σ0(u)_ = σ0(u0) and _Vi1_ = mi1, we have Nu0,u Bin(mi1 2, ab).
_̸_ _|_ _|_ _∼_ _−_

_Proof. We first consider the case when σ0(u) = σ0(u0). For any vertex v_ _Vi1_ _u, u0_, we have
_∈_ _\ {_ _}_

P (v, u) ∈ _E, (v, u0) ∈_ _E | σ0(u) = σ0(u0)_
� = P (v, u) _E, (v, u0)_ _E_ _σ0(v) = σ�0(u), σ0(u) = σ0(u0)_ P _σ0(v) = σ0(u)_

_∈_ _∈_ _|_

+� P (v, u) _E, (v, u0)_ _E_ _σ0(v)_ = σ0(u), σ0(u) = σ0(u�0) �P _σ0(v)_ = σ0(u�)
_∈_ _∈_ _|_ _̸_ _̸_

= (a[2] + b[2])/2.

� � � �

The first statement follows from mutual independence of the events (v, u), (v, u0) _E_ over
_{_ _∈_ _}_


_1. Conditioned on σ0(u) = σ0(u0) and_ _Vi1_ = mi1, we have Nu0,u Bin(mi1 2, (a[2] + b[2])/2).
_|_ _|_ _∼_ _−_

_2. Conditioned on σ0(u)_ = σ0(u0) and _Vi1_ = mi1, we have Nu0,u Bin(mi1 2, ab).
_̸_ _|_ _|_ _∼_ _−_

_Proof. We first consider the case when σ0(u) = σ0(u0). For any vertex v_ _Vi1_ _u, u0_, we have
_∈_ _\ {_ _}_


_Proof. We first consider the case when σ0(u) = σ0(u0). For any vertex v_ _Vi1_ _u, u0_, we have
_∈_ _\ {_ _}_


P (v, u) ∈ _E, (v, u0) ∈_ _E | σ0(u) = σ0(u0)_
� = P (v, u) _E, (v, u0)_ _E_ _σ0(v) = σ�0(u), σ0(u) = σ0(u0)_ P _σ0(v) = σ0(u)_

_∈_ _∈_ _|_

+� P (v, u) _E, (v, u0)_ _E_ _σ0(v)_ = σ0(u), σ0(u) = σ0(u�0) �P _σ0(v)_ = σ0(u�)
_∈_ _∈_ _|_ _̸_ _̸_

= (a[2] + b[2])/2.

� � � �


The first statement follows from mutual independence of the events (v, u), (v, u0) _E_ over
_{_ _∈_ _}_
_v_ _Vi1_ _u, u0_, conditioned on _Vi1_ = mi1.
_∈_ _\ {_ _}_ _|_ _|_
Similarly, if σ0(u) = σ0(u0), for any v _Vi1_ _u, u0_, we have
_̸_ _∈_ _\ {_ _}_


P (v, u) ∈ _E, (v, u0) ∈_ _E | σ0(u) ̸= σ0(u0)_
� = P (v, u) _E, (v, u0)_ _E_ _σ0(v) = σ�0(u), σ0(u)_ = σ0(u0) P _σ0(v) = σ0(u)_

_∈_ _∈_ _|_ _̸_

+� P (v, u) _E, (v, u0)_ _E_ _σ0(v)_ = σ0(u), σ0(u) = σ0(u�0) �P _σ0(v)_ = σ0(u�)
_∈_ _∈_ _|_ _̸_ _̸_ _̸_

= ab,

� � � �


implying the second statement.


-----

The following lemma will be used to bound the misclassification probability of u _Vi1_ _u0_
_∈_ _\ {_ _}_
using the thresholding rule given in Algorithm 2, Line 3. Let Tu0,u = {Nu0,u > (a + b)[2](|Vi1| − 2)/4}.
We define constants η1 = exp[(a − _b)[4]/4] and c1 = δ(a −_ _b)[4]/8._


**Lemma 4.10. For any vertex u** _Vi1_ _u0_ _and any mi1_ _δ log n, we have_
_∈_ _\ {_ _}_ _≥_


max P _u[c]0,u_ _σ0(u) = σ0(u0),_ _Vi1_ = mi1 _, P_ _u0,u_ _σ0(u)_ = σ0(u0), _Vi1_ = mi1 _η1n[−][c][1]._
_T_ _|_ _|_ _T_ _̸_ _|_ _|_ _≤_
�
� � � �[�]

_Proof. Fix mi_ �� _δ log n. Lemma 4.9 along with Hoeffding’s inequality (Lemma��_ 4.2) gives that


_Proof. Fix mi1_ _δ log n. Lemma 4.9 along with Hoeffding’s inequality (Lemma 4.2) gives that_
_≥_


P _u[c]0,u_ _σ0(u) = σ0(u0),_ _Vi1_ = mi1
_T_ _|_ _|_

=� P _Nu0,u_ (a[2] + b[2])(mi1 2)/2 � (a _b)[2](mi1_ 2)/4 _σ0(u) = σ0(u0),_ _Vi1_ = mi1

�� _−_ _−_ _≤−_ _−_ _−_ _|_ _|_

exp� (a _b)[4](mi1_ 2)/8 �
_≤_ _−_ _−_ _−_ ��

exp(� (a _b)[4](δ log n_ 2)/8) =� _η1n[−][c][1]._
_≤_ _−_ _−_ _−_


Similarly,


The following proposition ensures the high probability of correct labeling for all vertices in Vi1.


**Proposition 4.11. Suppose that a, b ∈** [0, 1] with a ̸= b. Then Line 8 of Algorithm 5 ensures that
_for any ∆_ _> δ,_


_Proof. For any u_ _Vi1_ _u0_, when mi1 _δ log n, Lemma 4.10 implies_
_∈_ _\ {_ _}_ _≥_


Thus, for any δ log n _mi1_ ∆log n, the union bound yields that
_≤_ _≤_


It follows that


-----

#### 4.3 Propagating labels among occupied blocks.


We now demonstrate that the Propagate subroutine (Lines 9-10 of Algorithm 5) ensures that all
occupied blocks are classified with at most M mistakes, for a suitable constant M .


We introduce a vector m = (m1, _, m(n/(χ log n)))_ Z[(]+[n/][(][χ][ log][ n][))] and define the event
_· · ·_ _∈_


Each V(m) corresponds to a specific (χ, δ)-visibility graph H. Thus, conditioned on an event V(m)
that ensures the connectivity of H, the occupied block set V _[†]_ and the propagation ordering over V _[†]_

are uniquely determined. To simplify the analysis, we fix the vector m in what follows, and condition
on some event V(m) ⊂H, recalling that H = {H is connected}. We write Pm(·) = P (· | V(m)) as a
reminder. Note that conditioned on V(m), the labels of vertices are independent, and the edges are
independent conditioned on the vertex labels.


We denote the configuration of a block as a vector z = (z(1, 1), z(1, −1), z(−1, −1), z(−1, 1)) ∈ _Z+[4]_ [,]
where each entry represents the count of vertices labeled as +1 or −1 by σ0 and _σ. For i ∈_ _V_ _[†], the_
event Ci(z) signifies that the occupied block Vi possesses a configuration z such that
�


Consider i ∈ _V_ _[†]_ _\ {i1} and a configuration z ∈_ Z[4]+[. The key observation is that because the labels]
_σ(u) : u_ _Vi_ are determined using disjoint sets of edges, the labels _σ(u) : u_ _Vi_ are independent
_{_ _∈_ _}_ _{_ _∈_ _}_
conditioned on Cp(i). Thus, the number of mistakes on Vi can be dominated by a binomial random
variable. To formalize this observation, we define constants� _M = 5/[(a�_ _b)[2]δ], c2 = (a_ _b)[2]δ/4, and_
_−_ _−_
_η2 = exp(2(a −_ _b)[2]M_ ). Let Ai be the event that _σ makes at most M mistakes on Vi:_


The following lemma bounds the probability of misclassifying a given vertex using Algorithm 3.


**Lemma 4.12. Suppose that a, b ∈** [0, 1] and a ̸= b, and fix i ∈ _V_ _[†]_ _\ {i1}. Fix z ∈_ Z[4]+ _[such that]_
_z(1, 1)+z(1, −1)+z(−1, −1)+z(−1, 1) = mp(i) and z(1, −1)+z(−1, 1) ≤_ _M (so that Cp(i)(z) ⊂Ap(i))._
_Then for any u_ _Vi, we have_
_∈_


_Proof. We consider the case a > b. Let J+ = {|{v ∈_ _Vp(i) :_ _σ(v) = 1}| ≥|{v ∈_ _Vp(i) :_ _σ(v) = −1}|}._
We first study the case when J+ holds. In this context, Lines 1-8 of Algorithm 3 are executed.
Conditioned on any _p(i)(z), we have_ _v_ _Vp(i) :_ _σ(v) = 1 �_ = z(1, 1) + z( 1, 1). Among these �
_C_ _|{_ _∈_ _}|_ _−_
vertices v ∈ _Vp(i) with_ _σ(v) = 1, z(1, 1) vertices have ground truth label σ0(u0) and z(−1, 1) of_
them have label _σ0(u0). We now bound the probability of making a mistake, meaning that �_
_−_
_σ(u)_ = σ0(u0)σ0(u). �
_̸_


If σ0(u) = σ0(u0), let _Xi_ _i=1_ and _Yi_ _i=1_ be independent random variables with Xi
_{_ _}[z][(1][,][1)]_ _{_ _}[z][(][−][1][,][1)]_ _∼_
Bernoulli� (a) and Yi Bernoulli(b), and Z = _i=1_ _Xi +_ _i=1_ _Yi with mean µZ = z(1, 1)a +_
_∼_
_z(−1, 1)b. For any u ∈_ _Vi, we recall that d[+]1_ [(][u,] [:]

[�][z][(1][,][1)] [�][z][(][−][1][,][1)]

_[σ, V][p][(][i][)][) =][ |{][v][ ∈]_ _[V][p][(][i][)]_ _[σ][(][v][) = 1][,][ {][u, v][} ∈]_ _[E][}|][ and]_


-----

observe that conditioned on {σ0(u) = σ0(u0), Cp(i)(z), the degree profile d[+]1 [(][u,][ �][σ, V][p][(][i][)][)][ has the same]
distribution as Z. Thus, Hoeffding’s inequality yields
�


We recall that J+ implies |{v ∈ _Vp(i) :_ _σ(v) = 1}| ≥|Vp(i)|/2 ≥_ _δ log n/2, and z(1, −1)+z(−1, 1) ≤_ _M_ .
It follows that z(1, 1) + z(−1, 1) ≥ _δ log n/2 and z(1, 1) ≥_ _δ log n/2 −_ _M_ . Thus,
�


where the last two inequalities hold since (z(1, 1) − _M_ )[2]/(z(1, 1) + M ) ≥ _z(1, 1) −_ 3M and z(1, 1) ≥
_δ log n/2 −_ _M_ .


Similarly, when σ0(u) = σ0(u0), let _Xi_ _i=1_ and _Yi_ _i=1_ be independent random variables
_̸_ _{_ _}[z][(][−][1][,][1)]_ _{_ _}[z][(1][,][1)]_
with Xi ∼ Bernoulli(a) and Yi ∼ Bernoulli(b), and _Z =_ _i=1_ _Yi +_ _i=1_ _Xi with mean µZ =_
_z(1, 1)b + z(_ 1, 1)a. For any u _Vi, we observe that d[+]1_ [(][u,][ �][σ, V][p][(][i][)][)][ has the same distribution as][ �][Z][,]
_−_ _∈_ �
conditioned on _σ0(u)_ = σ0(u0), _p(i)(z)_ . By similar steps as the case[�] [�][z][(1][,][1)] [�] σ[z]0[(][−](u[1][,]) =[1)] _σ0(u0), we obtain_
_̸_ _C_
� �


The bounds (4.5) and (4.6) together imply


We can derive symmetric analysis for z such that J+[c] [holds, in which case Algorithm][ 3][ executes]
Lines 9-16. The proof is complete for the case a > b. The analysis for the case b > a is similar.


Before proceeding further and showing the success of the propagation, we state a lemma that,
with high probability, all blocks contain O(log n) vertices.


**Lemma 4.13. For the blocks obtained from Line 3 in Algorithm 5, there exists a constant ∆** _> 0_
_such that_


-----

_Proof. For a block Bi with vol(Bi) = χ log n, we have |Vi| ∼_ Poisson(λχ log n). Thus, the Chernoff
bound in Lemma 4.1 implies that, for ∆ _> (λχ + 1 +_ 2λχ + 1), we have

_[√]_


where the last inequality holds by straightforward calculation. Thus, the union bound gives that


For ∆ _> 0 given by Lemma 4.13, we define I as follows and have P(I) = 1 −_ _o(1)._


The following lemma concludes that Phase I makes few mistakes on occupied blocks during the
propagation.


**Lemma 4.14. Let G ∼** _GSBM(λ, n, a, b, d) with λνd > 1, a, b ∈_ [0, 1], and a ̸= b, and _σ : V →_
_{−1, 0, 1} be the output of Phase I in Algorithm 5 on input G. Suppose m is such that V(m) ⊂I ∩H._
_Lines 9-10 of Algorithm 5 ensure that_ �


_Proof. Consider ij ∈_ _V_ _[†]_ for 2 ≤ _j ≤|V_ _[†]|, and fix z ∈_ Z[4]+ [such that]


Observe that the events that u ∈ _Vij is mislabeled by_ _σ are mutually independent conditioned on_
_Cp(ij_ )(z). Lemma 4.12 shows that each individual vertex in Vij is misclassified with probability at
most η2n[−][c][2], conditioned on _p(ij_ )(z). It follows that conditioned on � _p(ij_ )(z),
_C_ _C_


Let µξ = E[ξ] = η2∆n[−][c][2] log n. Using the Chernoff bound (Lemma 4.3), we obtain


-----

The last inequality holds since c2M = 5/4 by definition and (log n)[M] _n[1][/][8]_ for large enough n.
_≤_
Since Aij is independent of {Aik : k < j, k ̸= p(ij)} conditioned on Cp(ij ), (4.8) implies


Furthermore, since (4.8) is a uniform bound over all z satisfying (4.7), it follows that


Thus, combining Proposition 4.11 with the preceding bound, we have


where we use the fact that there are n/χ log n blocks in total along with Bernoulli’s inequality.


Combining the aforementioned results, we now prove the success of Phase I in Theorem 4.15.
We highlight that since η > 0 is arbitrary, the following equation (4.10) implies Theorem 2.4.


**Theorem 4.15. Given GSBM(λ, n, a, b, d) with a, b ∈** [0, 1], a ̸= b, and d = 1 and λ > 1, or d ≥ 2
_and λνd > 1. Fix any η > 0. Let κ = νd(1 +_ _√dχ[1][/d])[d]/χ. Let_ _σ be the labeling obtained from Phase_

_I with χ > 0 satisfying (4.1) and δ > 0 satisfying (4.2) and δ < η/κ, respectively. Then there exists_
_a constant M such that_ _σ makes at most M mistakes on every occupied block, with high probability, �_


_Moreover, it follows that_


_and_


_Proof. Fixing any η > 0, we consider χ > 0 satisfying (4.1) and δ > 0 satisfying (4.2) and δ < η/κ,_
respectively. Given any m such that V(m) ⊂I ∩H, for occupied blocks, Proposition 4.14 yields the
existence of a constant M > 0 such that


-----

Since the above bound is uniform over all m such that V(m) ⊂I ∩H, we have


where the last step holds by Propositions 4.5 and 4.8, and Lemma 4.13. Thus, we have proven (4.9).
Since δ log n > M for n large enough, it follows that


On the one hand, if _σ makes fewer than δ log n mistakes on Vi for all i ∈_ [n/(χ log n)], then
_σ makes fewer than δn/χ_ _ηn/(χκ) mistakes in_ _d,n. Thus, (4.10) follows from (4.12). On the_
_≤_ _S_
other hand, if _σ makes fewer than �_ _δ log n mistakes on Vi for all i_ [n/(χ log n)], then there will be
_∈_
fewer than� _δκ log n_ _η log n mistakes in all vertices’ neighborhood since each neighborhood_ (u)
_≤_ _N_
intersects at most � _κ blocks. Thus, (4.11) also follows from (4.12)._


### 5 Phase II: Proof of exact recovery


Before proving Theorem 2.2, we first show a concentration bound. We define vectors in R[4],


and random variables _D = [D1[+][, D]1[−][, D][+]1[, D][−]1[]][ ∼]_ [Poisson][(][x][)][, and][ X][ as a linear function of]
_−_ _−_

_[D][,]_


For any t [0, 1], let Dt(x _y) =_ _i_ [4][(][tx][i][ + (1][ −] _[t][)][y][i][ −]_ _[x]i[t][y]i[1][−][t]) be an f_ -divergence. Let D+(x _y) =_
_∈_ _∥_ _∈_ _∥_
maxt [0,1] Dt(x _y) = maxt_ [0,1] Dt(y _x) be the Chernoff-Hellinger divergence, as introduced by [5]._
_∈_ _∥_ _∈_ _∥_
In particular, when x and y are defined in[�] (5.1), the maximum is achieved at t = 1/2 and we have
_D+(x_ _y) = λνd(1_ _√ab_ (1 _a)(1_ _b)) log n._
_∥_ _−_ _−_ _−_ _−_

�


**Lemma 5.1. For any constants ρ > 0 and η > 0, it holds for X defined in (5.2) that**


_Proof. We will apply the Chernoff bound on X. First, we compute its moment-generating function._
For _D = [D1[+][, D]1[−][, D][+]1[, D][−]1[] = (][D][i][)]i[4]=1_
_−_ _−_ _[∼]_ [Poisson][(][x][)][, the definition of][ X][ in][ (][5.2][)][ can be written as]


We recall that for ξ ∼ Poisson(µ) and s ∈ R, we have E[exp(sξ)] = exp[µ(e[s] _−_ 1)]. Thus, we have


-----

Therefore, the Chernoff bound ensures that for any t > 0, we have


It follows that


Now we present the proof of Theorem 2.2, which ensures that Algorithm 5 achieves exact recovery.


_Proof of Theorem 2.2. We first fix a constant c > λ and let E0 = {|V | < cn}._ Since |V | ∼
Poisson(λn), the Chernoff bound in Lemma 4.1 gives that


For η > 0 to be determined, let E1 be the event that _σ makes at most η log n mistakes in the_
neighborhood for all vertices (Phase I succeeds); that is,
�


Theorem 4.15 ensures that P(E1) = 1 − _o(1). Let E2[′]_ [be the event that Algorithm][ 5][ achieves exact]
recovery and E2 be the event that all vertices are labeled correctly relative to σ0(u0); that is,


Then we have P(E2[′] [)][ ≥] [P][(][E][2][)][. Since][ P][(][E][0][)][,][ P][(][E][1][) = 1][ −] _[o][(1)][, it follows that]_


Note that we analyze P( 2 2
distribution. Next, we would like to show that the probability of misclassifying a vertexE _[c]_ _[∩E][1][ ∩E][0][)][ rather than][ P][(][E]_ _[c]_ _[| E][1][,][ E][0][)][, in order to preserve the data] v is o(1/n),_
and conclude that the probability of misclassifying any vertex is o(1). To formalize such an argument,
sample N ∼ Poisson(λn), and generate max{N, cn} points in the region Sd,n uniformly at random.


-----

Note that on the event 0, we have max _N, cn_ = cn. Label the points in order, and set _σ(u0) = 1._
_E_ _{_ _}_
In this way, the first N points form a Poisson point process with intensity λ. We can simulate
Algorithm 5 on the first N points. To bound the failure probability of Phase II, we can assume �
that any v ∈{N + 1, . . ., cn} must also be classified (by thresholding τ (v, σ), computed only using
edge/non-edge observations between v and u ∈ [N ])). For v ∈ [cn], let


Then


so that a union bound yields


Fix v ∈ [cn]. In order to bound P (E2(v)[c] _∩E1), we classify v according to running the Refine_
algorithm with respect to edge/non-edge observations between v and u [N ]. Analyzing 2(v)[c] 1
_∈_ _E_ _∩E_
now reduces to analyzing robust Poisson testing. Let W (v) = {σ : N (v) →{−1, 0, 1}} and dH be
the Hamming distance. We define the set of all estimators that differ from σ0 on at most η log n
vertices in (v), relative to σ0(u0), as
_N_


Let Ev be the event that there exists σ ∈ _W_ _[′](v; η) such that Poisson testing with respect to σ fails_
on vertex v when E2 holds; that is,


We provide some insights into the definition of Ev. Recall that σgenie(v) = sign(τ (v, σ0)) defined in
(3.1) picks the event with the larger likelihood between _σ0(v) = 1_ and _σ0(v) =_ 1 . Thus, for
_{_ _}_ _{_ _−_ _}_
example, suppose that σ0(v) = 1, then σgenie(v) makes a mistake when τ (v, σ0) ≤ 0. We consider
any σ _W_ (v; η). Since σ0(u0)σ(u) = σ0(u) for most u (v), d(v, σ0(u0)σ) and d(v, σ0) and
_∈_ _[′]_ _∈N_
thus τ (v, σ0(u0)σ) and τ (v, σ0) are close. Formalizing the intuition, suppose that σ0(v) = 1. If
_σ0(u0) = 1, then for E2 to hold, we must classify v as +1 to be correct relative to σ0(u0). Thus, v is_
misclassified relative to σ whenever τ (v, σ) 0. If σ0(v) = 1 and σ0(u0) = 1, then we must classify
_≤_ _−_
_v as −1. Then v is misclassified relative to σ whenever τ_ (v, σ) ≥ 0. As a summary, failure in the
case σ0(v) = 1 means τ (v, σ0(u0)σ) 0.
_≤_


It follows that


We aim to show that for η > 0 sufficiently small, P(Ev) = n[−][(1+Ω(1))]. Due to the uniform prior on
_σ0(v), we have_


-----

We now bound the first term in (5.7). Let D ∈ Z[4]+ [represent the ground-truth degree profile of]
vertex v. We consider a realization D = d(v, σ0) and the induced τ (v, σ0). Next, we bound the
distance _τ_ (v, σ0(u0)σ) _τ_ (v, σ0) for any σ _W_ (v; η). We note that the edges and non-edges are
_|_ _−_ _|_ _∈_ _[′]_
fixed in a given graph G; that is, for any σ ∈ _W_ (v), we have


Let α = d[+]1 [(][u, σ][0][(][u][0][)][σ][)][ −] _[d]1[+][(][u, σ][0][) =][ −][(][d][+]1[(][u, σ][0][(][u][0][)][σ][)][ −]_ _[d][+]1[(][u, σ][0][))][ and][ β][ =][ d][−]1_ [(][u, σ][0][(][u][0][)][σ][)][ −]
_−_ _−_
_d[−]1_ [(][u, σ][0][) =][ −][(][d][−]1[(][u, σ][0][(][u][0][)][σ][)][ −] _[d][−]1[(][u, σ][0][))][. It follows that]_
_−_ _−_


For any σ _W_ (v; η), recalling that dH (σ0(u0)σ( ), σ0( )) _η log n, we have_ _α_ _η log n and_
_∈_ _[′]_ _·_ _·_ _≤_ _|_ _| ≤_
_|β| ≤_ _η log n. Thus, we define ρ = 2 · [| log(a/b)| + | log((1 −_ _a)/(1 −_ _b))|] and have_


We define a set Y ⊂ Z[4]+ [as follows:]


Conditioned on _σ0(v) = 1_, Poisson testing fails relative to σ when τ (v, σ0(u0)σ) 0. Thus,
_{_ _}_ _≤_


To bound the above summation, we consider random variables _D ∼_ Poisson(x) with x defined in
(5.1) and X defined in (5.2). Recalling that D _D conditioned on σ0(v) = 1, Lemma 5.1 gives that_
_∼_

[�]


-----

Since λνd(1 _√ab_ (1 _a)(1_ _b)) > 1, we take η = (λνd(1_ _√ab_ (1 _a)(1_ _b))_ 1)/ρ > 0
_−_ _−_ _−_ _−_ _−_ _−_ _−_ _−_ _−_

and conclude that

� �


Similarly, we study the case conditioned on {σ0(v) = −1}. Let Y _[′]_ = {d = (d[+]1 _[, d]1[−][, d][+]−1[, d][−]−1[)][ ∈]_
Z[4]+ [:][ log][(][a/b][)(][d]1[+] 1[) +][ log][((1][ −] _[a][)][/][(1][ −]_ _[b][))(][d][−]1_ 1[)][ ≥−][ρη][ log][ n][}][. The definition of][ E][v][ in][ (][5.5][)]

_[−]_ _[d]−[+]_ _[−]_ _[d]−[−]_
gives that


For the same _D = [D1[+][, D]1[−][, D][+]1[, D][−]1[]][ ∼]_ [Poisson][(][λν][d][ log][ n][[][a,][ 1][ −] _[a, b,][ 1][ −]_ _[b][]][/][2)][, note that condition]_
_−_ _−_
on σ0(v) = −1, we have D ∼ [D−[+]1[, D]−[−]1[, D]1[+][, D]1[−][]][. Thus, with the same][ X][ defined in][ (][5.2][)][, we have]

[�]


Thus, similarly, Lemma 5.1 gives that P(Ev | σ0(v) = −1) ≤ _n[−]_ 2[1] [(][λν][d][(1][−]√ab−[√](1−a)(1−b))+1). There
fore, the above bound together with (5.4), (5.6), and (5.7) implies P( 2
we have P((E2[′] [)][c][)][ ≤] [P][(][E]2[c][) =][ o][(1)][ due to (][5.3][).] _E_ _[c]_ _[∩E][1][ ∩E][0][) =][ o][(1)][. Finally,]_


### 6 Impossibility: Proof of Theorem 2.3


In this section, we prove the impossibility of exact recovery under the given conditions and complete
the proof of Theorem 2.3. Recalling that Theorem 2.1 (Theorem 3.7 in [2]) has already established
the impossibility when λ > 0, d ∈ N, and 0 ≤ _b < a ≤_ 1 satisfying (2.1). Here, we extend the same
result to the case where the requirement a > b is dropped.


**Proposition 6.1. Let λ > 0, d ∈** N, and a, b ∈ [0, 1] satisfy (2.1) and let Gn ∼ _GSBM(λ, n, a, b, d)._
_Then any estimator_ _σ fails to achieve exact recovery._


_Proof. We note that the analysis of Theorem 2.1 builds upon Lemma 8.2 in [2], which itself relies on_

�

Lemma 11 from [5]. Lemma 11 provides the error exponent for hypothesis testing between Poisson
random vectors, forming the basis for the impossibility result. Notably, only the CH-divergence
criterion λνd(1 _√ab_ (1 _a)(1_ _b)) < 1 is needed to ensure the indistinguishability of the two_
_−_ _−_ _−_ _−_

Poisson distributions. Therefore, the impossibility in Theorem 2.1 can be readily extended to the

�

case where the condition a > b is dropped.


Moreover, we show the impossibility of exact recovery for d = 1 and λ < 1.


-----

**Proposition 6.2. When d = 1, let 0 < λ < 1 and a, b** [0, 1] and let Gn _GSBM(λ, n, a, b, d)._
_∈_ _∼_
_Then any estimator_ _σ fails to achieve exact recovery._


_Proof. When d = 1, we partition the interval [_ _n/2, n/2] into n/ log n blocks of length log n each._

� _−_

Notably, if there are k ≥ 2 mutually non-adjacent empty blocks, the interval gets divided into k ≥ 2
disjoint segments that lack mutual visibility. In such scenarios, achieving exact recovery becomes
impossible as we can randomly flip the signs of one segment. Formally, suppose that there are k
segments, where the ith segment contains blocks {Bj : j ∈ seg(i)} for seg(i) ⊂ [n/ log n]. Then for
any s ∈{±1}[k], the labeling σ0 has the same posterior probability as σ(·; s), defined as


It follows that the error probability of the genie-aided estimator is at least 1 − 2/2[k] = 1 − 1/2[k][−][1],
conditioned on there being k segments. Let X be the event of having at least two non-adjacent
empty blocks (and thus two segments). The aforementioned observation means that if X holds, the
error probability is at least 1/2, and thus the exact recovery is unachievable.


We now prove that P(X ) = 1−o(1) if λ < 1. Let Yk be the event of having exactly k empty blocks,
among which at least two of them are non-adjacent. Recalling that each block is independently
empty with probability exp(−λ log n) = n[−][λ], we have


where the second inequality follows by calculating the Binomial series and the geometric series, and
the last inequality holds since 1 − 2n[−][λ] _≥_ 1/2 for large enough n.


In summary, by combining Propositions 6.1 and 6.2, we complete the proof of Theorem 2.3.


### 7 Further related work


Our work contributes to the growing literature on community recovery in random geometric graphs,
beginning with latent space models proposed in the network science and sociology literature (see for
example [18, 19]). There have been several models for community detection in geometric graphs.
The most similar to the one we study is the Soft Geometric Block Model (Soft GBM), proposed by
Avrachenkov et al [7]. The main difference between their model and the GSBM is that the positions
of the vertices are unknown. Avrachenkov et al [7] proposed a spectral algorithm for almost exact
recovery, clustering communities using a higher-order eigenvector of the adjacency matrix. Using


-----

a refinement procedure similar to ours, [7] also achieved exact recovery, though only in the denser
linear average degree regime.


A special case of the Soft GBM is the Geometric Block Model (GBM), proposed by Galhotra et
al [14] with follow-up work including [10, 15]. In the GBM, community assignments are generated
independently, and latent vertex positions are generated uniformly at random on the unit sphere.
Edges are then formed according to parameters _βi,j_, where pair of vertices u, v in communities i, j
_{_ _}_
with locations Zu, Zv are connected if ⟨Zu, Zv⟩≤ _βi,j._


In the previously mentioned models, the vertex positions do not depend on the community
assignments. In contrast, Abbe et al [3] proposed the Gaussian-Mixture Block Model (GMBM),
where (latent) vertex positions are determined according to a mixture of Gaussians, one for each
community. Edges are formed between all pairs of vertices whose distance falls below a threshold.
A similar model was recently studied by Li and Schramm [24] in the high-dimensional setting.
Additionally, Péché and Perchet [26] studied a geometric perturbation of the SBM, where vertices are
generated according to a mixture of Gaussians, and the probability of connecting a pair of vertices is
given by the sum of the SBM parameter and a function of the latent positions.


In addition, some works [6, 13] consider the task of recovering the geometric representation
(locations) of the vertices in random geometric graphs as a form of community detection. Their
setting differs significantly from ours. We refer to the survey [12] for an overview of the recent
developments in non-parametric inference in random geometric graphs.


### 8 Conclusions and future directions


Our work identifies the information-theoretic threshold for exact recovery in the two-community,
balanced, symmetric GSBM. A natural direction for future work is to consider the case of multiple
communities, with general community membership probabilities and general edge probabilities. We
believe that the information-theoretic threshold will again be given by a CH-divergence criterion,
and a variant of our two-phase approach will achieve the threshold.


It would also be interesting to study other spatial network inference problems. For example,
consider Z2-synchronization [4, 8, 22], a signal recovery problem motivated by applications to clock
synchronization [16], robotics [28], and cryogenic electron microscopy [30]. In the standard version
of the problem, each vertex is assigned an unknown label x(v) ∈{±1}. For each pair (u, v), we
observe x(u)x(v) + σWuv, where σ > 0 and Wuv (0, 1). Now suppose that the vertices are
_∼N_
generated according to a Poisson point process, and we observe x(u)x(v) + σWuv only for mutually
visible vertices, which models a signal recovery problem with spatially limited observations. An open
question is then whether our two-phase approach can be adapted to this synchronization problem.


**Acknowledgements.** J.G. was supported in part by NSF CCF-2154100. X.N. and E.W. were
supported in part by NSF ECCS-2030251 and CMMI-2024774.


### References



[1] E. Abbe. Community detection and stochastic block models: recent developments. The Journal
_of Machine Learning Research, 18(1):6446–6531, 2017._

[2] E. Abbe, F. Baccelli, and A. Sankararaman. Community detection on Euclidean random graphs.
_Information and Inference: A Journal of the IMA, 10(1):109–160, 2021._


-----

[3] E. Abbe, E. Boix-Adsera, P. Ralli, and C. Sandon. Graph powering and spectral robustness.
_SIAM Journal on Mathematics of Data Science, 2(1):132–157, 2020._

[4] E. Abbe, J. Fan, K. Wang, and Y. Zhong. Entrywise eigenvector analysis of random matrices
with low expected rank. Annals of Statistics, 48(3):1452, 2020.

[5] E. Abbe and C. Sandon. Community detection in general stochastic block models: Fundamental
limits and efficient algorithms for recovery. In 2015 IEEE 56th Annual Symposium on Foundations
_of Computer Science, pages 670–688. IEEE, 2015._

[6] E. Araya Valdivia and D. C. Yohann. Latent distance estimation for random geometric graphs.
_Advances in Neural Information Processing Systems, 32, 2019._

[7] K. Avrachenkov, A. Bobu, and M. Dreveton. Higher-order spectral clustering for geometric
graphs. Journal of Fourier Analysis and Applications, 27(2):22, 2021.

[8] A. S. Bandeira, N. Boumal, and A. Singer. Tightness of the maximum likelihood semidefinite
relaxation for angular synchronization. Mathematical Programming, 163:145–167, 2017.

[9] S. Boucheron, G. Lugosi, and P. Massart. Concentration inequalities: A nonasymptotic theory
_of independence. Oxford University Press, 2013._

[10] E. Chien, A. Tulino, and J. Llorca. Active learning in the Geometric Block Model. In Proceedings
_of the AAAI Conference on Artificial Intelligence, volume 34, pages 3641–3648, 2020._

[11] V. Cohen-Addad, F. Mallmann-Trenn, and D. Saulpic. Community recovery in the degreeheterogeneous stochastic block model. In Conference on Learning Theory, pages 1662–1692.
PMLR, 2022.

[12] Q. Duchemin and Y. De Castro. Random geometric graph: Some recent developments and
perspectives. High Dimensional Probability IX: The Ethereal Volume, pages 347–392, 2023.

[13] R. Eldan, D. Mikulincer, and H. Pieters. Community detection and percolation of information
in a geometric setting. Combinatorics, Probability and Computing, 31(6):1048–1069, 2022.

[14] S. Galhotra, A. Mazumdar, S. Pal, and B. Saha. The geometric block model. In Proceedings of
_the AAAI Conference on Artificial Intelligence, volume 32, 2018._

[15] S. Galhotra, A. Mazumdar, S. Pal, and B. Saha. Community recovery in the geometric block
model. arXiv preprint arXiv:2206.11303, 2022.

[16] A. Giridhar and P. R. Kumar. Distributed clock synchronization over wireless networks:
Algorithms and analysis. In Proceedings of the 45th IEEE Conference on Decision and Control,
pages 4915–4920. IEEE, 2006.

[17] B. Hajek, Y. Wu, and J. Xu. Achieving exact cluster recovery threshold via semidefinite
programming. IEEE Transactions on Information Theory, 62(5):2788–2797, 2016.

[18] M. S. Handcock, A. E. Raftery, and J. M. Tantrum. Model-based clustering for social networks.
_Journal of the Royal Statistical Society: Series A (Statistics in Society), 170(2):301–354, 2007._

[19] P. D. Hoff, A. E. Raftery, and M. S. Handcock. Latent space approaches to social network
analysis. Journal of the American Statistical Association, 97(460):1090–1098, 2002.


-----

[20] P. W. Holland, K. B. Laskey, and S. Leinhardt. Stochastic blockmodels: First steps. Social
_Networks, 5(2):109–137, 1983._

[21] A. Ivić, E. Krätzel, M. Kühleitner, and W. Nowak. Lattice points in large regions and related
arithmetic functions: recent developments in a very classic topic. Elementare und analytische
_Zahlentheorie, Franz Steiner, pages 89–128, 2006._

[22] A. Javanmard, A. Montanari, and F. Ricci-Tersenghi. Phase transitions in semidefinite relaxations. Proceedings of the National Academy of Sciences, 113(16):E2218–E2223, 2016.

[23] J. F. C. Kingman. Poisson Processes, volume 3. Clarendon Press, 1992.

[24] S. Li and T. Schramm. Spectral clustering in the Gaussian mixture block model. arXiv preprint
_arXiv:2305.00979, 2023._

[25] E. Mossel, J. Neeman, and A. Sly. Consistency thresholds for the planted bisection model.
In Proceedings of the Forty-Seventh Annual ACM Symposium on Theory of Computing, pages
69–75, 2015.

[26] S. Péché and V. Perchet. Robustness of community detection to random geometric perturbations.
_Advances in Neural Information Processing Systems, 33:17827–17837, 2020._

[27] A. Rapoport. Spread of information through a population with socio-structural bias: I. Assumption of transitivity. The Bulletin of Mathematical Biophysics, 15:523–533, 1953.

[28] D. M. Rosen, L. Carlone, A. S. Bandeira, and J. J. Leonard. A certifiably correct algorithm for
synchronization over the special Euclidean group. In Algorithmic Foundations of Robotics XII:
_Proceedings of the Twelfth Workshop on the Algorithmic Foundations of Robotics, pages 64–79._
Springer, 2020.

[29] A. Sankararaman and F. Baccelli. Community detection on Euclidean random graphs. In
_Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms, pages_
2181–2200. SIAM, 2018.

[30] A. Singer. Angular synchronization by eigenvectors and semidefinite programming. Applied and
_Computational Harmonic Analysis, 30(1):20–36, 2011._


-----

