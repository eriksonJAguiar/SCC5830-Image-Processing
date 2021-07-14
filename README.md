# **SCC5830 Image Processing -- Final Project**

- **Author:** Erikson Julio de Aguiar (graduate student)
- **proposal objective:** Analyze the impacts of restoration and enhancement methods to improve images over the rescaling attack


## Abstract

Adversarial attacks insert perturbations on medical images may cause misclassifications in machine learning models. As a result, diseases are incorrectly classified forward to poor diagnosis and not assist physicians. Furthermore, malicious agents might generate perturbations and inserted them in images, producing noise to build adversarial features. One of these is the image scaling attacks aim to produce another image when resize the source image. Our proposal aims to develop a method to detect attacked images and restore them. We will use the difference between histograms to identify the malicious image and restoration methods to recovery affected pixels of the source image. In this project, we hope to improve image quality to protect against image scaling attacks. To evaluate, we will use as a source image a dataset from Kaggle with images of chest X-ray and target image. We will apply non-medical images to fool tools when there is resized.

- Keywords:
  - Main task: Image restoration and enhancement
  - Application: Medical images, adversarial attacks

**Rescaling attack overview:**

<img src="https://raw.githubusercontent.com/eriksonJAguiar/scc5830_final_project/main/images/image-scaling.png" alt="drawing" width="600"/>

**Related papers:**

- Xiao, Q., Chen, Y., Shen, C., Jiaotong, X., Chen, Y., Cheng, P., Li, K."Seeing is not believing: Camouflage attacks on image scaling algorithms." 28th {USENIX} Security Symposium ({USENIX} Security 19). 2019.![https://www.usenix.org/conference/usenixsecurity19/presentation/xiao](https://www.usenix.org/conference/usenixsecurity19/presentation/xiao). 
- Quiring, Erwin, et al. "Adversarial preprocessing: Understanding and preventing image-scaling attacks in machine learning." 29th {USENIX} Security Symposium ({USENIX} Security 20). 2020. ![https://www.usenix.org/conference/usenixsecurity20/presentation/quiring](https://www.usenix.org/conference/usenixsecurity20/presentation/quiring)

## Dataset

- Dataset contains 5,863 X-Ray images (JPEG) with 2 classes (Pneumonia/Normal). Dabaset is available on [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

**Input images example:**
<div style="display: inline-block">
    <img src="https://github.com/eriksonJAguiar/scc5830_final_project/blob/main/images/chest/chest1.jpeg" width="200"/>
    <img src="https://raw.githubusercontent.com/eriksonJAguiar/scc5830_final_project/main/images/cat.jpg" alt="drawing" width="100"/>
  </div>
  
  P.s. Input image example can find on ![chest_images](https://github.com/eriksonJAguiar/scc5830_final_project/tree/main/images/chest_images)
  
 **Result image after attack**
 
 <img src="https://raw.githubusercontent.com/eriksonJAguiar/scc5830_final_project/main/images/attacks/img1_attack_CV_LANCZOS4.jpeg" width="200"/>
 
 P.s. Attack images examples can find on ![attacks_experiments](https://github.com/eriksonJAguiar/scc5830_final_project/tree/main/images/attacks_experiments) (Nearest-neighbor interpolation), also others crafted images are available on ![attacks](https://github.com/eriksonJAguiar/scc5830_final_project/tree/main/images/attacks)  (different interpolations methods) 
 
  **Restored image after attack**
 
 - Median filter:
 <img src="https://raw.githubusercontent.com/eriksonJAguiar/scc5830_final_project/main/images/restored/img8_restored_median.jpeg" width="200"/>
 
 - Remove peaks:
 <img src="https://raw.githubusercontent.com/eriksonJAguiar/scc5830_final_project/main/images/restored/img8_restored_peak.jpeg" width="200"/>
 
 - Minimum filter:
 <img src="https://raw.githubusercontent.com/eriksonJAguiar/scc5830_final_project/main/images/restored/img8_restored_min.jpeg" width="200"/>
  
## Experiments

- We suggest using our Jupyter Notebook available on ![Jupyter Notebook](https://github.com/eriksonJAguiar/scc5830_final_project/blob/main/image_scaling.ipynb) to reproduce experiments

## How to run?

### build environment
```pyhthon
    python3 -m venv attack-env
    pip3 install -r requiments.txt
```

### Modules available

**attack_module -> ImageScaling class: related to attack method:**

- Bileaner interpolation [b_resize]
- Neighbor-nearest interpolation [nn_resize]
-  Build attack [Build_attack](our implementation)
-  Build Attack with library built for ![Quiring (2020)](https://github.com/EQuiw/2019-scalingattack)
-  Build repository [build_repository] -> Create repository to load attack image
-  run experiments[run_experiments] -> run a batch of experiments
-  run expertiment one image [run_one_image] -> run experiments for one image

**defense_module -> Defense class: related to defesive method:***

- Median filter to restore image [restoration_image_median] 
- Remove peaks and  to restore image [restoration_image_median] 
- Minimum filter to detect attac image [restoration_image_median] 
- Unshap mask to test high-pass filter [unsharp_mask] 

## References

- Xiao, Q., Chen, Y., Shen, C., Jiaotong, X., Chen, Y., Cheng, P., Li, K."Seeing is not believing: Camouflage attacks on image scaling algorithms." 28th {USENIX} Security Symposium ({USENIX} Security 19). 2019.
- Quiring, Erwin, et al. "Adversarial preprocessing: Understanding and preventing image-scaling attacks in machine learning." 29th {USENIX} Security Symposium ({USENIX} Security 20). 2020.
- Quiring, E., Klein, D., Arp, D., Johns, M., Rieck, Konrad. "Backdooring and poisoning neural networks with image-scaling attacks." 2020 IEEE Security and Privacy Workshops (SPW). IEEE, 2020.
- Quiring, E., Klein, D., Arp, D., Johns, M., Rieck, Konrad. "Image-Scaling Attacks in Machine Learning", 2020. 
- Qayyum, A., Qadir, J., Bilal, M., & Al-Fuqaha, A. (2021). Secure and Robust Machine Learning for Healthcare: A Survey. IEEE Reviews in Biomedical Engineering, 14, 156–180. https://doi.org/10.1109/RBME.2020.3013489
- Finlayson, S. G., Bowers, J. D., Ito, J., Zittrain, J. L., Beam, A. L., & Kohane, I. S. (2019). Adversarial attacks on medical machine learning. Science. https://doi.org/10.1126/science.aaw4399.
