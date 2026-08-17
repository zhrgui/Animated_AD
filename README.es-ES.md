

# Comprensión centrada en personajes de películas animadas

Zhongrui Gui<sup>1</sup>, Junyu Xie<sup>1</sup>, Tengda Han<sup>1</sup>, Weidi Xie <sup>2</sup>, Andrew Zisserman<sup>1</sup>

<sup>1</sup> Visual Geometry Group (VGG), University of Oxford <br>
<sup>2</sup> School of Artificial Intelligence (SAI), Shanghai Jiao Tong University

<a src="https://img.shields.io/badge/cs.CV-2504.01020-b31b1b?logo=arxiv&logoColor=red" href="https://arxiv.org/pdf/2509.12204">  
<img src="https://img.shields.io/badge/cs.CV-2504.01020-b31b1b?logo=arxiv&logoColor=red"></a>
<a href="https://www.robots.ox.ac.uk/~vgg/research/animated_ad/" alt="Project page"> 
<img alt="Project page" src="https://img.shields.io/badge/project_page-Animated__AD-blue"></a>
<br>
<br>
<p align="center">
  <img src="teasor.png"  width="750"/>
</p>


## Método y Evaluación
En este trabajo, proponemos construir automáticamente un banco de personajes audiovisuales para habilitar el reconocimiento audiovisual de personajes animados. Además, aprovechamos los resultados para tareas posteriores, incluyendo la Generación de Descripción de Audio (AD) y la Subtitulación Consciente de Personajes. Nuestro trabajo consta de varios componentes principales, que listamos a continuación.

### Pipeline
* Consulte [aquí](https://github.com/g2zr004/Animated_AD/tree/main/build_character_bank) para construir el **Banco de Personajes Audiovisuales**. 
* Consulte [aquí](https://github.com/g2zr004/Animated_AD/tree/main/character_recognition) para el **Reconocimiento Audiovisual de Personajes Animados**.
* Consulte [aquí](https://github.com/g2zr004/Animated_AD/tree/main/app) para la **Aplicación en Tareas Posteriores**.

### Evaluación
* Los videos se pueden descargar [aquí](https://www.dropbox.com/scl/fo/ek8b9hzbtos3gxwsbrqdb/ACQURGZ_Gmrb35UWDhUFZas?rlkey=r1er2iswst6kymueb5z2wi9y5&st=xeh1xcqc&dl=0).
* Todas las anotaciones y la información correspondiente se pueden encontrar [aquí](https://drive.google.com/drive/folders/1Jb3N1fMAAA8cRxrAFUoAWqFWdwtTuUcE?usp=sharing).
* Los scripts de evaluación, que incluyen mIoU de Caja de Personaje, AP de Nombre de Personaje y AP de Reconocimiento de Audio, se pueden encontrar [aquí](https://github.com/g2zr004/Animated_AD/tree/main/character_recognition/eval). Para CRITIC y CIDEr, consulte el repositorio original [AutoAD](https://github.com/TengdaHan/AutoAD/tree/main/autoad_iii/metrics).

### Resultados Predichos
* Los resultados del reconocimiento visual de personajes se pueden descargar [aquí](https://drive.google.com/drive/folders/1YwACigVKwHRNtyXcBV_7edXqQQWP9EBb?usp=sharing).
* Los resultados del reconocimiento de personajes por audio se pueden descargar [aquí](https://drive.google.com/drive/folders/1MAfKV5z60ZOmdlN3l37MFos1Qpw3RNJU?usp=sharing).
* Las predicciones de AD (por Qwen2-VL con LLaMA3 o VideoLLaMA2 con LLaMA3) se pueden descargar [aquí](https://drive.google.com/drive/folders/1Fo-x-5KcQroCiEwp-K1LybdREqlV_Oms?usp=sharing).
* Los resultados de la subtitulación consciente de personajes se pueden descargar [aquí](https://drive.google.com/drive/folders/1_WVI8LUMcCBLOxXhtHkonbbH12E3DzTg?usp=sharing).

## Instalación
El entorno base se basa principalmente en [DINOv2](https://github.com/facebookresearch/dinov2) y [SAM2](https://github.com/facebookresearch/sam2). Para configurar las dependencias requeridas, siga las instrucciones a continuación:

```shell
conda env create -f conda.yaml
conda activate animated_ad

cd ..
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
```

Este entorno está configurado para la construcción automática del banco de personajes y el reconocimiento visual de personajes.


## Cita
¡Si encuentra este repositorio útil, le pedimos que considere citar nuestro trabajo! &#128522;
```
@article{gui2025character,
          title={Character-Centric Understanding of Animated Movies},
          author={Gui, Zhongrui and Xie, Junyu and Han, Tengda and Xie, Weidi and Zisserman, Andrew},
          journal={arXiv preprint arXiv:2509.12204},
          year={2025}
        }
```

## Referencias
AutoAD-Zero: [https://github.com/Jyxarthur/AutoAD-Zero](https://github.com/Jyxarthur/AutoAD-Zero) <br>
Qwen2-VL: [https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) <br>
LLaMA3: [https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) <br>
