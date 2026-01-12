[![author](https://img.shields.io/badge/author-Luiz_Carlos-blue.svg)](https://www.linkedin.com/in/luiz-carlos-vieira-4582797b/) [![](https://img.shields.io/badge/python-3.10+-yellow.svg)](https://www.python.org/downloads/release/python) [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/ziul-bio/umGPT/issues) [![Other Projects](https://img.shields.io/badge/Others-Projects-red.svg?style=flat)](https://github.com/ziul-bio?tab=repositories)


<p align="center">
  <img src="images/banner.png" >
</p>

# Intrinsic dataset features drive mutational effect prediction by protein language models


## Abstract:
Protein language models (pLMs) are widely used for predicting protein fitness landscapes, but their low performance on certain datasets remains poorly understood. We evaluated supervised transfer learning on 41 viral and 33 cellular deep mutational scanning datasets using embeddings from multiple pLMs. Viral datasets consistently exhibited lower predictive performance compared to cellular datasets, independent of model architecture, size, or transfer learning strategy. Surprisingly, a simple baseline that predicts site mean fitness outperformed supervised models on most datasets, highlighting the dominant role of site effects. Analysis of site variability using two metrics, Relative Variability of Site Means (RVSM) and Fraction of Highly Variable Sites (FHVS), revealed that limited within-site fitness variation in proteins constrains model performance. Moreover, splitting data by site, rather than pooling, revealed that supervised models often rely on site effects rather than capturing broader mutational patterns. Finally, RVSM and FHVS strongly predicted the performance of models across datasets in ProteinGym, indicating that fitness variability alone largely determines predictability. These findings highlight limitations of current pLMs on certain proteins and suggest that dataset composition, rather than model architecture or training, is the primary driver of predictive success.


# Repreducing results

## Set project path and create a virtual environment
```bash
bash setup.sh
```

## Extract Embeddings
```bash
bash scripts/run_extract_esm2.sh
bash scripts/run_extract_esmc.sh
```

## Run regression
```bash
bash scripts/run_lassoCV.sh
```

## Run finetune
```bash
bash scripts/run_ESM2ne_viral.sh
bash scripts/run_ESM2ne_cellular.sh

```

## Plotting figures in the paper
```R
notebooks/plots.Rmd
```

---