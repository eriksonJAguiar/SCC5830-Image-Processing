# SCC5830 Image Processing -- Final Project

- **Author:** Erikson Julio de Aguiar (graduate student)
- **proposal objective:** Image scaling attacks detection on medical images

## Abstract

Adversarial attacks insert perturbations on medical images may cause misclassifications in machine learning models. As a result, diseases are incorrectly classified forward to poor diagnosis and not assist physicians. Furthermore, malicious agents might generate perturbations and inserted them in images, producing noise to build adversarial features. One of these is the image scaling attacks aim to produce another image when resize the source image. Our proposal aims to develop a method to detect attacked images and restore them. We will use the difference between histograms to identify the malicious image and restoration methods to recovery affected pixels of the source image. In this project, we hope to improve image quality to protect against image scaling attacks. To evaluate, we will use as a source image a dataset from Kaggle with images of chest X-ray and target image. We will apply non-medical images to fool tools when there is resized.

- Keywords:
  - Main task: Image restoration
  - Application: Medical images with perturbations

## Images examples

- Dataset contains 5,863 X-Ray images (JPEG) with 2 classes (Pneumonia/Normal). Dabaset is available on [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

**Images example:**

![](https://raw.githubusercontent.com/eriksonJAguiar/scc5830_final_project/main/chest.png)

## Next steps
- Take a look on ![Project 1](https://github.com/eriksonJAguiar/scc5830_final_project/projects/1)

## References

- Xiao, Q., Chen, Y., Shen, C., Jiaotong, X., Chen, Y., Cheng, P., Li, K."Seeing is not believing: Camouflage attacks on image scaling algorithms." 28th {USENIX} Security Symposium ({USENIX} Security 19). 2019.
- Quiring, Erwin, et al. "Adversarial preprocessing: Understanding and preventing image-scaling attacks in machine learning." 29th {USENIX} Security Symposium ({USENIX} Security 20). 2020.
- Quiring, E., Klein, D., Arp, D., Johns, M., Rieck, Konrad. "Backdooring and poisoning neural networks with image-scaling attacks." 2020 IEEE Security and Privacy Workshops (SPW). IEEE, 2020.
- Quiring, E., Klein, D., Arp, D., Johns, M., Rieck, Konrad. "Image-Scaling Attacks in Machine Learning", 2020. 
- Qayyum, A., Qadir, J., Bilal, M., & Al-Fuqaha, A. (2021). Secure and Robust Machine Learning for Healthcare: A Survey. IEEE Reviews in Biomedical Engineering, 14, 156–180. https://doi.org/10.1109/RBME.2020.3013489
- Finlayson, S. G., Bowers, J. D., Ito, J., Zittrain, J. L., Beam, A. L., & Kohane, I. S. (2019). Adversarial attacks on medical machine learning. Science. https://doi.org/10.1126/science.aaw4399.
