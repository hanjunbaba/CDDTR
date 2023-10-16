# CDDTR
## Cross-domain auto encoders for predicting cell type specific drug-induced transcriptional responses
*欢迎得到您的访问！*

*论文还未接收，接收后会将论文地址粘贴在此。*

*Welcome to get your visit!*

*The paper has not been received, and the address of the paper will be pasted here after acceptance.*


## 1.介绍/Introduction
&emsp;&emsp;基因表达谱在医学和生物学等各个领域发挥着重要作用。癌症细胞系中药物诱导的基因表达谱有助于研究药物的重新定位、药物作用机制、细胞活性预测、细胞系中的药物反应预测和药物治疗的脱靶效应。此外，随着单细胞组学技术的发展，癌症细胞系或组织中药物扰动的基因表达谱也可以在细胞水平上进行研究。<br>
&emsp;&emsp;目前，一些公共可用的数据库已经积累了药物扰动下人类癌症细胞系的基因表达谱。然而，对于药物和细胞系的巨大组合，仍然有一些药物对某些细胞系没有转录干扰谱。朱等人利用深度学习，利用药物化学结构的简化分子输入线输入系统（SMILES）作为化学编码输入，拟合LINCS中的转录扰动谱，并提出了一种称为DLEPS的药效预测系统。该系统通过在肥胖、高尿酸血症和非酒精性脂肪性肝炎相关小鼠疾病模型中测试顶级候选药物进行了实验验证。然而，这种用于预测转录扰动谱的系统忽略了癌症细胞系的细胞类型特异性。<br>
&emsp;&emsp;一些研究工作利用已知的药物在某些细胞系上的转录扰动谱来预测药物在其他细胞系的扰动谱。Hodos等人基于药物、细胞系和基因将扰动转录组学数据排列成三维张量，然后使用局部算法（药物邻居谱预测，DNPP）和全局算法（快速低秩张量完成，FaLRTC）来预测未测量的细胞类型特异性基因表达谱。Umarov等人提出了一种名为DeepCellState的方法，该方法受到deepfake技术的启发，构建了一个基于自动编码器的深度学习模型。DeepCellState为每种细胞类型使用通用编码器和单独的解码器，根据另一种细胞类型的响应预测一种细胞的响应。然而，这些方法忽略了不同细胞类型的扰动轮廓之间的映射建模，因此可以进一步提高预测性能。<br>
&emsp;&emsp;我们遵循DeepCellState的思想，利用一个细胞系的扰动轮廓来预测另一个细胞株的扰动轮廓，并开发了一种基于跨域自动编码器的方法。跨域自动编码器的主要目标是学习两个或多个不同域之间的数据映射关系，从而实现从不同域转换和传输数据的能力。目前，跨领域模型已经有许多应用，如图像风格转换、跨语言文本翻译、语音转换和跨模态数据生成。在这篇文章中，我们通过两种细胞系各自的编码器获得了相似的特征，然后通过特定的解码器将获得的潜在特征解码为细胞系特异性基因表达谱，从而实现了使用另一种细胞系的谱预测一个细胞系的基因表达谱的能力。<br><br>
&emsp;&emsp;Gene expression profiles play an important role in various fields such as medicine and biology. Drug-induced gene expression profiles in cancer cell lines is useful for studying drug repositioning, drug action mechanisms, cell activity prediction, drug response prediction in cell lines and off-target effects of drug therapy. In addition, with the development of single-cell omics technology, gene expression profiles of drug perturbations in cancer cell lines or tissues can also be studied at the cellular level.<br>
&emsp;&emsp;Currently, some public available databases have accumulated gene expression profiles of human cancer cell lines under drug perturbation. However, for the vast combination of drugs and cell lines, there are still some drugs that do not have transcriptional perturbation profiles for certain cell lines. Zhu et al. utilized deep learning to fit transcriptional perturbation profiles in LINCS using the simplified molecular-input line-entry system (SMILES) of drug chemical structure as chemical coding input, and proposed a drug efficacy prediction system called DLEPS. The system was experimentally validated by testing the top drug candidates in obesity, hyperuricemia and nonalcoholic steatohepatitis-related mouse disease models. However, this system for the prediction of transcriptional perturbation profiles ignores the cell type specificity of cancer cell lines.<br>
&emsp;&emsp;Some research works utilize known transcriptional perturbation profiles of drugs on certain cell lines to predict the perturbation profiles of drugs on other cell lines. Hodos et al. arranged perturbation transcriptomics data into a three-dimensional tensor based on drugs, cell lines and genes, and then used a local algorithm (drug neighbor profile prediction, DNPP) and a global algorithm (Fast Low-Rank Tensor Completion, FaLRTC) to predict unmeasured cell type specific gene expression profiles. Umarov et al. proposed a method named DeepCellState, which was inspired by the deepfake technique to construct a deep learning model based on the autoencoder. Using a common encoder and separate decoders for each cell type, DeepCellState predicts the response in one cell type based on the response in another. However, these methods neglect the modeling of mappings between perturbation profiles of different cell types, so the predictive performance could be further improved.<br>
&emsp;&emsp;We follow the idea of DeepCellState, which utilizes the perturbation profile of one cell line to predict the perturbation profile of another cell line, and develop a method based on cross-domain auto encoders. The main goal of cross-domain auto encoders is to learn the data mapping relationships between two or more different domains, thereby achieving the ability to transform and transfer data from different domains. Currently, there are already many applications of cross-domain models, such as image style conversion, cross-language text translation, voice and speech conversion, and cross-modal data generation. In this article, we obtain similar features for two cell lines through their respective encoders, and then decode the obtained latent features into cell line-specific gene expression profiles through a specific decoder, thereby achieving the ability to predict gene expression profiles of one cell line using profiles of another. <br>

## 2.环境准备/Environmental preparation
请在您的环境下安装如下：（注意，在安装tensorflow==2.4.0的时候会自动帮你安装好numpy，您可以卸载numpy来安装我们使用的版本，或者您可以更改当前numpy下对应的pandas版本）

    tensorflow-gpu==2.4.0
	tensorflow==2.4.0
	numpy==1.19.5
	cmapPy==4.0.1
	pandas==1.3.5
	scikit-learn==1.0.2
	scipy==1.10.1
	seaborn==0.12.1
或者可以直接使用pip来配置我们的环境变量（requirements.txt在文件中已经给出）
```python 
>> pip install -r requirements.txt
```
## 3.数据集/Dataset
在这项研究中，我们使用了来自II期LINCS-L1000数据的化学诱导的转录基因谱变化（GSE70138）。在我们的分析方案中，我们采用了978个Landmark基因的表达特征。<br><br>
In this study, we used chemically induced transcriptional gene profile changes (GSE70138) from Phase II LINCS-L1000 data. In our analysis plan, we used the expression characteristics of 978 Landmark genes.
### 3.1数据集准备/Dataset preparation
我们需要下载对应的数据集，登录： https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138 和 https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742 下载以下表格中数据集：<br>
We need to log in to: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138 and https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742 Download the dataset in the following table:
| 文件/Files |
| --- |
| GSE92742_Broad_LINCS_sig_info.txt |
| GSE92742_Broad_LINCS_gene_info.txt |
| GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx |
| GSE70138_Broad_LINCS_sig_info.txt |
| GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328.gctx |
### 3.2制作数据文件/Create a data file
将下载好的数据文件放到data文件夹下，运行data_parser.py，可在pycharm中直接运行，也可以在控制台输入（请注意，'Path to store CDDTR'是你存放CDDTR的路径）：
```python  
>> cd 'Path to store CDDTR'/CDDTR/data
>> python data_parser.py
```
## 4.训练（测试）模型/Train（test）model
