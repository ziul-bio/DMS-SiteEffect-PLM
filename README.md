[![author](https://img.shields.io/badge/author-Luiz_Carlos-blue.svg)](https://www.linkedin.com/in/luiz-carlos-vieira-4582797b/) [![](https://img.shields.io/badge/python-3.10+-yellow.svg)](https://www.python.org/downloads/release/python) [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/ziul-bio/umGPT/issues) [![Other Projects](https://img.shields.io/badge/Others-Projects-red.svg?style=flat)](https://github.com/ziul-bio?tab=repositories)


<p align="center">
  <img src="images/banner.png" >
</p>

# ViCAM
<sub> *A viral pLM*  fine-tuned to understand viral protein language </sub>




## About ViCAM:

* Viral fine-tuning of CAMbriam

* 

* Feel free to use all and share some contribuition to improve them too.


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



---


## To do list:
- [ ] Download database: RVDB
- [ ] Download datasets: 51 viral DMS
- [ ] Implement the MLM head on the ESMC
- [ ] Train the ViCAM
- [ ] Transfer leaninrg with the original ESMC 300M on the 51 DMS to benchemark our model. 
- [ ] 
- [ ] 
- [ ] 
- [ ] 
- [ ] 