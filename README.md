# DVQ - DNA Validations and Quick comparisons 
Abstract
We introduce DVQ (DNA Visualisations and Quick comparisons), an open-source Python library for exploring nucleotide sequences using a variety of methods. Understanding DNA sequences intuitively isimportant for a variety of tasks in biology. DVQ aims to be a one-stop comprehensive library that makes explainable DNA easy for geneticists, researchers, and practitioners who need explanations. For practitioners, the library provides an easy-to-use interface to generate visualisations for their sequencesby only writing a few lines of code. In this report, we demonstrate several example use cases across different types of sequences as well as visualisations. 

Simple early version preprint: http://dx.doi.org/10.13140/RG.2.2.19227.89125

## Methods:
- [x] [Persistant Homological Representations](https://american-cse.org/csci2022-ieee/pdfs/CSCI2022-2lPzsUSRQukMlxf8K2x89I/202800b599/202800b599.pdf)
- [x] [ColorSquare](https://match.pmf.kg.ac.rs/electronic_versions/Match68/n2/match68n2_621-637.pdf)
- [x] [C-Curve](https://pubmed.ncbi.nlm.nih.gov/23246806/) - Removed from development plan due it being redundant vs a 2D Line
- [x] [Spider Representation](https://www.researchgate.net/publication/260971259_Spider_Representation_of_DNA_Sequences) - Removed from development plan due to it being found to be difficult to use for large sequences
- [ ] [KL Divergence](https://pubmed.ncbi.nlm.nih.gov/31981184/)
- [ ] [Perpelxity](https://arxiv.org/pdf/1202.2518.pdf)
- [ ] [Entropy](https://pubmed.ncbi.nlm.nih.gov/9344742/)
- [x] [2D Line](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC162336/)
- [x] [Chaos Game Representation](Fwww.sciencedirect.com%2Fscience%2Farticle%2Fpii%2FS2001037021004736&usg=AOvVaw38odDudWfUCAqbc626rD2e&opi=89978449)
- [x] K-Mer overlap
- [x] [Wen's Method](https://pubmed.ncbi.nlm.nih.gov/29765099/)

## Overview:

* How to use dvq
```python
from dvq import visual

visual.plot_2d_comparison([seqs_1, seqs_2], ['seq_1', 'seq_2'])

```

![An example graphic comparing dna sequences for the same virus ](Untitled.png "2D Comparison - Same virus")

```python
from dvq import statistical

statistical.similarity_wen([seqs_1, seqs_2])
# 0.99

```
