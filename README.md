## TRNAS
TRNAS: A Training-Free Robust Neural Architecture Search

## Code used for "TRNAS: A Training-Free Robust Neural Architecture Search". This paper was accepted in the ICCV conference 2025. 

## Abstract
Deep Neural Networks (DNNs) have been successfully applied in various computer tasks. However, they remain vulnerable to adversarial attacks, which could lead to severe security risks. 
In recent years, robust neural architecture search (NAS) has gradually become an emerging direction for designing adversarially robust architectures. 
However, existing robust NAS methods rely on repeatedly training numerous DNNs to evaluate robustness, which makes the search process extremely expensive. 
In this paper, we propose a training-free robust NAS method (TRNAS) that significantly reduces search costs. 
First, we design a zero-cost proxy model (R-Score) that formalizes adversarial robustness evaluation by exploring the theory of DNN's linear activation capability and feature consistency. 
This proxy only requires initialized weights for evaluation, thereby avoiding expensive adversarial training costs.  
Secondly, we introduce a multi-objective selection (MOS) strategy to save candidate architectures with robustness and compactness. 
Experimental results show that TRNAS only requires 0.02 GPU days to find a promising robust architecture in a vast search space including approximately 10$^{20}$ networks. 
TRNAS surpasses other state-of-the-art robust NAS methods under both white-box and black-box attacks.  
Finally, we summarize a few meaningful conclusions for designing the robust architecture and promoting the development of robust NAS field. 

## 1. Environment requirements:
The environment of R-Score is esay. After installing torch2.x, you can install the missing packages one by one using pip. Final, we will provide English tutorials and chiese tutorials.

### 1.1 Basic Requirements
- python=3.1x
- pytorch=2.x
- numpy, pathlib, scipy

## Architecture Evaluation 
### 2.1 Architecture Evalute by R-Score
<pre><code>cd proxy_model_eval
python 0.rscore_last.py 
</code></pre> 
### 2.2 Fully Train the Searched robust DNNs
<pre><code>python adv_train.py --arch xxx
</code></pre>

## If you have any questions, please email me. I will respond as soon as I have time. Thank you for your attention.

## unfinished and to be continued

## Acknowledgement
Some of the codes are built by:

1. [25 ICLR: ZCPRob](https://github.com/fyqsama/Robust_ZCP)

2. [24 NIPS: CRoZe](https://github.com/HyeonjeongHa/CRoZe)

3. [24 ICLR: SWAP](https://github.com/pym1024/SWAP)

4. [21 ICCV: AdvRush](https://github.com)

Thanks them for their great works!
