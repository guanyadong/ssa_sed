# Sparse self-attention for semi-supervised sound event detection

This code is the implementation of the paper 'Sparse self-attention for semi-supervised sound event detection'.
The theoretical proof of the excessive sparsity of the Sparsemax is in proof.pdf.

## Preparation before Training
- Use the environment.yml to install dependencies.
- The audio data needs to be download first, followed by feature extraction.
Download method and feature extraction can refer to [dcase 2020 baseline][dcase2020-baseline].

## Train SED model

- `python main_new.py`

## Test SED model

- `python test1.py`

**Note:** The performance might not be exactly reproducible on a GPU based system.
Therefore you can run test1.py directly to test a trained model. 
If you want to test your own model, please change the path in the test1.py to your model.

### System description
The model is based on [dcase 2019 baseline][dcase2019-baseline] and [dcase 2020 baseline][dcase2020-baseline]. The model is a mean-teacher model [[1]][1][[2]][2].
The implementation of Sparsemax [[3]] is based on [Sparsemax][Sparsemax].

### References
 - [[1]] L. Delphin-Poulat & C. Plapous, technical report, dcase 2019.
 - [[2]] Tarvainen, A. and Valpola, H., 2017.
 Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results.
 In Advances in neural information processing systems (pp. 1195-1204).
 - [[3]] André F. T. Martins, Ramón Fernandez Astudillo, "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"

[1]: http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Delphin_15.pdf
[2]: https://arxiv.org/pdf/1703.01780.pdf
[3]: http://arxiv.org/abs/1602.02068

[dcase2019-baseline]: https://github.com/turpaultn/DCASE2019_task4
[dcase2020-baseline]: https://github.com/turpaultn/dcase20_task4
[sparsemax]: https://github.com/KrisKorrel/sparsemax-pytorch/blob/master/sparsemax.py