# CustomConcept101

We release a dataset of 101 concepts with 3-15 images for each concept for evaluating model customization methods. For a more detailed view of target images please refer to our [webpage](https://www.cs.cmu.edu/~custom-diffusion/dataset.html). 

<br>
<div>
<p align="center">
<img src='../assets/sample_images.png' align="center" width=800>
</p>
</div>



## Download dataset

```
pip install gdown
gdown 1jj8JMtIS5-8vRtNtZ2x8isieWH9yetuK
unzip benchmark_dataset.zip
```

## Evaluation

We provide a set of text prompts for each concept in the [prompts](prompts/) folder. The prompt file corresponding to each concept is mentioned in [dataset.json](dataset.json) and [dataset_multiconcept.json](dataset_multiconcept.json). The CLIP feature based image and text similarity can be calculated as:

```
python evaluate.py --sample_root {folder} --target_path {target-folder} --numgen {numgen}
```

* `sample_root`: the root location to generated images. The folder should contain subfolder `samples` with  generated images. It should also contain a `prompts.json` file with `{'imagename.stem': 'text prompt'}` for each image in the samples subfolder.  
* `target_path`: file to target real images.
* `numgen`: number of images in the `sample_root/samples` folder
* `outpkl`: the location to save evaluation results (default: evaluation.pkl)

## Results
We compare our method (Custom Diffusion) with [DreamBooth](https://dreambooth.github.io) and [Textual Inversion](https://textual-inversion.github.io) on this dataset. We trained DreamBooth and Textual Inversion according to the suggested hyperparameters in the respective papers. Both Ours and DreamBooth are trained with generated images as regularization.

**Single concept**

<table>
  <tr>
    <td></td>
    <td colspan=3 align=center > 200 DDPM </td>
    <td colspan=3 align=center> 50 DDPM </td>
  </tr>
  <tr>
    <td></td>
    <td>Textual-alignment (CLIP)</td>
    <td>Image-alignment (CLIP)</td>
    <td>Image-alignment (DINO)</td>
    <td>Textual-alignment (CLIP)</td>
    <td>Image-alignment (CLIP)</td>
    <td>Image-alignment (DINO)</td>
  </tr>
  <tr>
    <td>Textual Inversion</td>
    <td> 0.6126 </td>
    <td> 0.7524 </td>
    <td> 0.5111 </td>
    <td> 0.6117 </td>
    <td> 0.7530 </td>
    <td> 0.5128 </td>
  </tr>
  <tr>
    <td>DreamBooth</td>
    <td> 0.7522  </td>
    <td> 0.7520 </td>
    <td> 0.5533 </td>
    <td> 0.7514  </td>
    <td> 0.7521 </td>
    <td> 0.5541  </td>
  </tr>
  <tr>
    <td> Custom Diffusion (Ours)</td>
    <td> 0.7602 </td>
    <td> 0.7440 </td>
    <td> 0.5311 </td>
    <td> 0.7583 </td>
    <td> 0.7456 </td>
    <td> 0.5335 </td>
  </tr>
</table>

**Multiple concept**

<table>
  <tr>
    <td></td>
    <td colspan=3 align=center > 200 DDPM </td>
    <td colspan=3 align=center> 50 DDPM </td>
  </tr>
  <tr>
    <td></td>
    <td>Textual-alignment (CLIP)</td>
    <td>Image-alignment (CLIP)</td>
    <td>Image-alignment (DINO)</td>
    <td>Textual-alignment (CLIP)</td>
    <td>Image-alignment (CLIP)</td>
    <td>Image-alignment (DINO)</td>
  </tr>
  <tr>
    <td>DreamBooth</td>
    <td> 0.7383 </td>
    <td> 0.6625 </td>
    <td> 0.3816 </td>
    <td> 0.7366 </td>
    <td> 0.6636 </td>
    <td> 0.3849 </td>
  </tr>
  <tr>
    <td>Custom Diffusion (Opt)</td>
    <td> 0.7627 </td>
    <td> 0.6577 </td>
    <td> 0.3650 </td>
    <td> 0.7599 </td>
    <td> 0.6595 </td>
    <td> 0.3684 </td>   
  </tr>
  <tr>
    <td> Custom Diffusion (Joint)</td>
    <td> 0.7567 </td>
    <td> 0.6680 </td>
    <td> 0.3760 </td>
    <td> 0.7534 </td>
    <td> 0.6704 </td>
    <td> 0.3799 </td>
  </tr>
</table>

## Evaluation prompts 
We used ChatGPT to generate 40 image captions for each concept with the instructions to either (1) change the background of the scene while keeping the main subject, (2) insert a new object/living thing in the scene along with the main subject, (3) style variation of the main subject, and (4) change the property or material of the main subject. The generated text prompts are manually filtered or modified to get the final 20 prompts for each concept. A similar strategy is applied for multiple concepts. Some of the prompts are also inspired by other concurrent works e.g. [Perfusion](https://research.nvidia.com/labs/par/Perfusion/), [DreamBooth](https://dreambooth.github.io), [SuTI](https://open-vision-language.github.io/suti/), [BLIP-Diffusion](https://dxli94.github.io/BLIP-Diffusion-website/) etc.
  


## License
Images taken from UnSplash are under [Unsplash License](https://unsplash.com/license). Images captured by ourselves are released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en) license. Flower category images are downloaded from Wikimedia/Flickr/Pixabay and the link to orginial images can also be found [here](https://www.cs.cmu.edu/~custom-diffusion/assets/urls.txt) for attribution.   


## Acknowledgments
We are grateful to Sheng-Yu Wang, Songwei Ge, Daohan Lu, Ruihan Gao, Roni Shechtman, Avani Sethi, Yijia Wang, Shagun Uppal, and Zhizhuo Zhou for helping with the dataset collection, and Nick Kolkin for the feedback.
