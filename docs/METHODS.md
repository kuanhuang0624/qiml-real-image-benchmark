# Implemented methods

## Shared interface

Frozen ImageNet1K-V1 ResNet-18 produces a pooled 512D feature $f(x)$.
Standardization and randomized PCA are fitted on the current training subset:

$$
z(x)=\operatorname{PCA}_8\left((f(x)-\mu_{\rm train})/s_{\rm train}\right),
\qquad
\alpha_j(x)=\pi\operatorname{clip}\left(z_j(x)/\max(q_{.99,j},10^{-8}),-1,1\right).
$$

The scale $q_{.99,j}$ is the training 99th percentile of $|z_j|$. Validation and
test never fit PCA or scaling. The 512D C-ResNet reference has an unmatched input
width; C-CNN receives original images. Neither should be described as a matched
eight-angle classical control.

## Fixed quantum features

QF-Product applies two repetitions of per-qubit RY then RZ uploads on eight
qubits. QF-Ring additionally applies a CZ ring after each repetition:

$$
|\psi(x)\rangle=\left[U_{\rm ent}
\bigotimes_{j=0}^{7}R_Z(\alpha_j)R_Y(\alpha_j)\right]^2|0\rangle^{\otimes8},
\qquad U_{\rm ent}=I\ \text{or}\ \prod_{(j,k)\in\mathrm{ring}}CZ_{jk}.
$$

Each uses 32 one-qubit uploads and 0 or 16 CZ gates before basis rotations.
The readout has 24 single-qubit X/Y/Z expectations and 24 same-axis ring-pair
expectations, obtained in three global X/Y/Z measurement settings. Qubit zero
is the most significant amplitude bit. X readout uses H; Y uses S-dagger then H.

## PdrQC-Matched

Candidate atomic blocks are RX, RY, RZ, CZ-ring and CNOT-ring. A greedy
depth-at-most-three procedure uses validation accuracy to select the upload
sequence. Training-label discrimination ranks all Pauli strings of weight at
most two: 276 candidates at eight qubits. It keeps 60 observables for ten
classes or 28 for two classes, then greedily groups qubit-wise-compatible
observables. The trained classical readout uses

$$
\phi_k(x)=\langle\psi(x)|P_k|\psi(x)\rangle,
\qquad p(y\mid x)=\operatorname{softmax}((W\phi(x)+b)/T).
$$

Binary classifiers use the corresponding sigmoid. Selection is supervised;
PdrQC is not an unsupervised encoding or a hardware-trained policy. The frozen
sequences and observable groups are in `experiments/selections/pca8/`.

## Finite-shot budgets

For total preparation budget $B$ across $G$ groups, allocate floor$(B/G)$ to
each group and distribute the remainder deterministically by measurement seed.
Thus $\sum_g B_g=B$, **not** $B$ shots in every group. PdrQC requires $B\ge G$.
The QF budget-one case measures only Z and fills unmeasured components with
zero; it is not a complete 48-observable estimate.

The published finite-shot study sampled multinomial outcomes from exact
probabilities, refitted the relevant logistic readouts using noisy training
features, and calibrated using validation outputs. It did not rerun a real
circuit for every shot. Missing PdrQC points below its group count are intentional.

## Other families

- QK-IQP uses two H/RZ/RZZ repetitions with fidelity kernel
  $K(x,x')=|\langle\psi(x)|\psi(x')\rangle|^2$. Kernel shots are per entry;
  inference cost also depends on the support-vector count. Training Gram
  matrices use the recorded symmetry/diagonal/PSD correction.
- VQC-DR uses eight qubits, two reuploading blocks and 48 trainable angles,
  followed by eight Z expectations and a linear classifier. PyTorch reverse-mode
  autodifferentiation and Adam train the circuit and readout.
- Yomo-Matched uses that circuit but aggregates basis-state probabilities by
  contiguous class groups, corrects for group size, and uses the implemented
  cross-entropy/sharpening/entropy objective. It is a matched adaptation, not
  an assertion of exact reproduction of an external paper.
- The PCA-width extension analytically evaluates product-RY Pauli expectations,
  factorizing them into sine/cosine terms; it does not simulate dense 16-qubit
  states for every feature batch.

No QPU, noise model, parameter-shift hardware execution, or hardware speedup
contributed to the completed predictive experiments. See the original
[method provenance](references/METHOD_PROVENANCE.md) for source relationships.
