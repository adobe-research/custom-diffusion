# Custom-Diffusion


### [website](https://www.cs.cmu.edu/~custom-diffusion/)  |  [code](https://github.com/adobe-research/custom-diffusion)  | [paper](https://arxiv.org/abs/2112.09130)


<br>

<div class="gif">
<p align="center">
<img src='assets/results.gif' align="center" width=800>
</p>
</div>

While generative models produce high-quality images of concepts learned from a large-scale database, a user often wishes to synthesize instantiations of their own concepts (for example, their family, pets, or items). Can we teach a model to quickly acquire a new concept, given a few examples? Furthermore, can we compose multiple new concepts together? We propose Custom Diffusion, an efficient method for augmenting existing text-to-image models. We find that only optimizing a few parameters in the text-to-image conditioning mechanism is sufficiently powerful to represent new concepts, while enabling fast tuning (~6 min). 

Additionally, we show that multiple new concepts can be combined into a single model via closed-form constrained optimization. Our fine-tuned model generates variations of multiple, new concepts and seamlessly composes them with existing concepts in novel settings. Our method outperforms several baselines and concurrent works, both in qualitative and quantitative evaluations, while being memory and computationally efficient.


## Method


<div>
<p align="center">
<img src='assets/methodology.png' align="center" width=800>
</p>
</div>


Given the few user-provided images of a concept, our method augments a pre-trained text-to-image diffusion model, enabling new generations of the concept in unseen contexts. 
We fine-tune a small subset of model weights, namely the key and value mapping from text to latent features in the cross-attention layers of the diffusion model. 
Our method also uses a small set of regularization images (200) to prevent overfitting. For personal categories we add a new modifier token V* in front of the category name e.g., V* dog.
For multiple-concepts we jointly train on the dataset for the two concepts. Our method also enables post-hoc merging of two fine-tuned models using optimization. 
For more details please refer to our paper.  


## Single-Concept Results


<div>
<p align="center">
<img src='assets/moongate.png' align="center" width=800>
</p>
<p align="center">
<img src='assets/tortoise_plushy.png' align="center" width=800>
</p>
</div>



## Multi-Concept Results


<div>
<p align="center">
<img src='assets/woodenpot_cat.png' align="center" width=800>
</p>
<p align="center">
<img src='assets/table_chair.png' align="center" width=800>
</p>
</div>


## References

```
@article{kumari2022customdiffusion,
  title={Multi-concept Customization of Text-to-Image Diffusion},
  author={Kumari, Nupur and Zhang, Bingliang and Zhang, Richard and Shechtman, Eli and Zhu, Jun-Yan},
  journal = {arXiv},
  year      = {2022}
}
```

## Acknowledgments
We are grateful to Nick Kolkin, David Bau, Sheng-Yu Wang, Gaurav Parmar, John Nack, and Sylvain Paris for their helpful comments and discussion, and to Allie Chang, Chen Wu, Sumith Kulal, Minguk Kang, and Taesung Park for proofreading the draft. We also thank Mia Tang and Aaron Hertzmann for sharing their artwork. This work was partly done by Nupur Kumari during the Adobe internship. The work is partly supported by Adobe Inc. 
