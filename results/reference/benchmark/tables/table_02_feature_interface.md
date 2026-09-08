# Table 2: Feature Interface

| dataset | training_scale | backbone | pretraining_source | raw_feature_dimension | PCA_dimension | explained_variance | maximum_clipping_rate | preprocessing_hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnist | D_1k | ResNet18_Weights.IMAGENET1K_V1 | ImageNet-1K external data | 512 | 8 | 0.417611 | 0.01 | 1f18cfa052666786f56b6e0af689b4cd8e25762cbd907b09bd55ab9e37339bd0 |
| mnist | D_5k | ResNet18_Weights.IMAGENET1K_V1 | ImageNet-1K external data | 512 | 8 | 0.418374 | 0.01 | dfb3cb787442779bbc16052f03c5649b8529e26540142a70888bf520a13c1bde |
| mnist | D_10k | ResNet18_Weights.IMAGENET1K_V1 | ImageNet-1K external data | 512 | 8 | 0.41711 | 0.01 | fde80439557c8d457566d45a9453e914f12ebdb1a5ae31bf24021f06f15c80fd |
| mnist | D_full | ResNet18_Weights.IMAGENET1K_V1 | ImageNet-1K external data | 512 | 8 | 0.418176 | 0.01 | 71045d90b589e1e0affb3b342d7307848f95ab1e287a9ed7232ef7f4524af760 |
| fashion_mnist | D_1k | ResNet18_Weights.IMAGENET1K_V1 | ImageNet-1K external data | 512 | 8 | 0.395969 | 0.01 | d7bda14548a4c2d4f6257768e927a06cf00a7ed9c3c0b7a967ed2482927f0200 |
| fashion_mnist | D_5k | ResNet18_Weights.IMAGENET1K_V1 | ImageNet-1K external data | 512 | 8 | 0.387828 | 0.01 | bde6988a5ecc3d1c066ecf1548709d3b78834770be249d4b4aa99866853db471 |
| fashion_mnist | D_10k | ResNet18_Weights.IMAGENET1K_V1 | ImageNet-1K external data | 512 | 8 | 0.387043 | 0.01 | b2be0b3fad284b3567c46848f37c4a0a904a5c9ad7d624f750e46fc6b62cafc9 |
| fashion_mnist | D_full | ResNet18_Weights.IMAGENET1K_V1 | ImageNet-1K external data | 512 | 8 | 0.385269 | 0.01 | 93625740ff935dbd35969b8cc654853069e2c8f9aa2e56a98030afa5b20ae306 |
| cifar10 | D_1k | ResNet18_Weights.IMAGENET1K_V1 | ImageNet-1K external data | 512 | 8 | 0.283872 | 0.01 | 4c6c1955e2ef19befe5d77f969083e983f513aaecb1635c6f946e7d7d823555d |
| cifar10 | D_5k | ResNet18_Weights.IMAGENET1K_V1 | ImageNet-1K external data | 512 | 8 | 0.273516 | 0.01 | 42245b175e7cfe4a970f7546cad4331587098cf6bb1fbef9dde7c59882fc4eae |
| cifar10 | D_10k | ResNet18_Weights.IMAGENET1K_V1 | ImageNet-1K external data | 512 | 8 | 0.272963 | 0.01 | c10b1970d6e007f6d52573eeeadf023495098412a5e8ac615021d29db4812f55 |
| cifar10 | D_full | ResNet18_Weights.IMAGENET1K_V1 | ImageNet-1K external data | 512 | 8 | 0.273082 | 0.01 | d426e672d68573400fd56f5fb6bad00c99d85c7f88b890360fddb4a40083591d |
| breastmnist | D_25pct | ResNet18_Weights.IMAGENET1K_V1 | ImageNet-1K external data | 512 | 8 | 0.419766 | 0.0145985 | cc42ca75802b6a6917d9dee8899bbdd7be135c87a55ff4218dd7a5b1cbe4f4ec |
| breastmnist | D_50pct | ResNet18_Weights.IMAGENET1K_V1 | ImageNet-1K external data | 512 | 8 | 0.4004 | 0.0109489 | b9cc6c0624891264bd69152d2dac862b97115565149cda45ed1d71774d12dc21 |
| breastmnist | D_100pct | ResNet18_Weights.IMAGENET1K_V1 | ImageNet-1K external data | 512 | 8 | 0.375614 | 0.010989 | 5ffa76630ff9c76830817aaac24a9fb6d8f31e595b5830c6440c8e7a74315f3f |
