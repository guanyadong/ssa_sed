# Sparse Self-attention for Semi-supervised sound Event Detection

## Environment

```
pytorch 1.6
```

The sparsemax package needs to be installed first, as follows  

```
pip install sparsemax
```

## Usage
```
self.encoder_layer = TransformerEncoderLayer(d_model=128, nhead=16, normal_func="sparsemax", sparsity=1.3)
```

- The theoretical proof of the excessive sparsity of the Sparsemax is in proof.pdf.
