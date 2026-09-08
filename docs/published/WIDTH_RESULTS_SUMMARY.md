# Results Summary

## MNIST

PdrQC-Matched test accuracy was 0.7934 at PCA8-Q8, 0.8257 at PCA12-Q12, and 0.8018 at PCA16-Q16. The paired width effects versus PCA8-Q8 were +0.0323 (95% CI [0.0256, 0.0390]) at width 12 and +0.0084 ([0.0012, 0.0157]) at width 16.

C-RBF was the best matched classical control at every width: 0.8443, 0.9026, and 0.9173. Consequently, the PdrQC-minus-C-RBF gap widened from -0.0509 at width 8 to -0.0769 at width 12 and -0.1155 at width 16. The unmatched C-ResNet reference was 0.9637 (95% CI [0.9600, 0.9673]).

## Fashion-MNIST

PdrQC-Matched test accuracy was 0.7718 at PCA8-Q8, 0.7633 at PCA12-Q12, and 0.7654 at PCA16-Q16. The paired width effects versus PCA8-Q8 were -0.0085 (95% CI [-0.0139, -0.0031]) at width 12 and -0.0064 ([-0.0122, -0.0007]) at width 16.

C-RBF was the best matched classical control at every width: 0.7937, 0.8154, and 0.8278. The PdrQC-minus-C-RBF gap widened from -0.0219 at width 8 to -0.0521 at width 12 and -0.0624 at width 16. The unmatched C-ResNet reference was 0.8647 (95% CI [0.8582, 0.8710]).

## Exploratory tests

For PCA16-Q16 versus PCA8-Q8, the raw paired-permutation p-values were 0.0272 for MNIST and 0.0331 for Fashion-MNIST. Both Holm-adjusted p-values were 0.0544 across the explicitly separate two-test exploratory family. These results are not merged into the parent confirmatory family and are not equivalence tests.

