# DeepEM-for-Weakly-Supervised-Detection
MICCAI18 early accepted paper DeepEM: Deep 3D ConvNets with EM for Weakly Supervised Pulmonary Nodule Detection

Please add paper into reference if the repository is helpful to you. "DeepEM: Deep 3D ConvNets With EM For Weakly Supervised Pulmonary Nodule Detection." MICCAI, 2018"

Please read our DeepLung paper "Zhu, Wentao, Chaochun Liu, Wei Fan, and Xiaohui Xie. "DeepLung: Deep 3D Dual Path Nets for Automated Pulmonary Nodule Detection and Classification." IEEE WACV, 2018." to get an idea for the baseline method. 

To run the code, first get familiar with it (environments, required packages) from DeepLung code https://github.com/wentaozhu/DeepLung.git

runweaktrainingressamp.sh is for DeepEM with sampling.

runweaktrainingresv2.sh is for DeepEM with MAP.

You can download NLST dataset from https://biometry.nci.nih.gov/cdas/nlst/ and use the scripts in ./NLST/ to clean the dataset.

The used TianChi hold out test set is in https://drive.google.com/drive/folders/1YEnxCtDplTfTp7OVKUGrXy1N8s2YZ7gP?usp=sharing.

For any problems, feel free to contact Wentao Zhu (wentaozhu1991@gmail.com)
