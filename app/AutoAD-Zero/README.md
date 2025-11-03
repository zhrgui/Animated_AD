# AutoAD-Zero: A Training-Free Framework for Zero-Shot Audio Description

Junyu Xie<sup>1</sup>, Tengda Han<sup>1</sup>, Max Bain<sup>1</sup>, Arsha Nagrani<sup>1</sup>, Gül Varol<sup>1</sup> <sup>2</sup>, Weidi Xie<sup>1</sup> <sup>3</sup>, Andrew Zisserman<sup>1</sup>

<sup>1</sup> Visual Geometry Group, Department of Engineering Science, University of Oxford <br>
<sup>2</sup> LIGM, École des Ponts, Univ Gustave Eiffel, CNRS <br>
<sup>3</sup> CMIC, Shanghai Jiao Tong University

<a src="https://img.shields.io/badge/cs.CV-2407.15850-b31b1b?logo=arxiv&logoColor=red" href="https://arxiv.org/abs/2407.15850">  
<img src="https://img.shields.io/badge/cs.CV-2407.15850-b31b1b?logo=arxiv&logoColor=red"></a>
<a href="https://www.robots.ox.ac.uk/~vgg/research/autoad-zero/" alt="Project page"> 
<img alt="Project page" src="https://img.shields.io/badge/project_page-autoad--zero-blue"></a>
<a href="https://www.robots.ox.ac.uk/~vgg/research/autoad-zero/#tvad" alt="Dataset"> 
<img alt="Dataset" src="https://img.shields.io/badge/dataset-TV--AD-purple"></a>
<br>


## Requirements
* **Basic Dependencies:** 
```pytorch==2.0.0```,
```Pillow```,
```pandas```,
```decord```,
```opencv```,
```moviepy==1.0.3```
```flash-attn==2.5.6```
```transformers==4.46.0```
```accelerate==0.26.1```

* **[VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2)**:
After installation, modify the `sys.path.append("/path/to/VideoLLaMA2")` in `stage1/main.py` and `stage1/utils.py`. Please download the VideoLLaMA2-7B checkpoint [here](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-7B).

* Set up cache model path (for LLaMA3, etc.) by modifying `os.environ['TRANSFORMERS_CACHE'] = "/path/to/cache/"` in `stage1/main.py` and `stage2/main.py`

## Character Recognition
To extract the boxes for visual prompt from visual character recognition results for Audio Description generation, run:
```shell
python char_recog/main.py --anno_path {anno_path} --track_preds {track_preds} --video_dir {video_dir} --output_dir {output_dir} --movie_title_to_imdbid_file {movie_title_to_imdbid_file} --score_thresh {score_thresh}
```

Otherwise, directly use ```resources/cmdad_anno_with_face_0.45.csv``` for inference.

## Inference
#### Stage I: VLM-Based Dense Video Description
```shell
python stage1/main.py --dataset {dataset} --video_dir {video_dir} --anno_path {anno_path} --model_path {videollama2_ckpt_path} --output_dir {output_dir}
```
`--dataset`: choices are `cmdad`, `madeval`, and `tvad`. <br>
`--video_dir`: directory of video datasets, example file structures can be found in `resources/example_file_structures` (files are empty, for references only). <br>
`--anno_path`: path to AD annotations *(with predicted face IDs and bboxes)*, available in `resources/annotations`. <br>
`--charbank_path`: path to external character banks, available in `resources/charbanks`. <br>
`--model_path`: path to videollama2 checkpoint. <br>
`--output_dir`: directory to save output csv. <br>

#### Stage II: LLM-Based AD Summary
```shell
python stage2/main.py --dataset {dataset} --pred_path {stage1_result_path} 
```
`--dataset`: choices are `cmdad`, `madeval`, and `tvad`. <br>
`--pred_path`: path to the stage1 saved csv file.


## Inference with GPT-4o via OpenAI API
Note: Before starting, insert OpenAI API keys into the corresponding `main.py` file. <br>
Note: This is not officially tested and reported in the original paper. You may want to adjust the text prompts to get improved / more robust outputs.

#### Stage I: VLM-Based Dense Video Description
```shell
python stage1_gpt/main.py --dataset {dataset} --video_dir {video_dir} --anno_path {anno_path} --charbank_path {charbank_path} --output_dir {output_dir}
```

#### Stage II: LLM-Based AD Summary
```shell
python stage2_gpt/main.py --dataset {dataset} --pred_path {stage1_result_path} 
```

## Citation
If you find this repository helpful, please consider citing our work:
```
@InProceedings{xie2024autoad0,
	title={AutoAD-Zero: A Training-Free Framework for Zero-Shot Audio Description},
	author={Junyu Xie and Tengda Han and Max Bain and Arsha Nagrani and G\"ul Varol and Weidi Xie and Andrew Zisserman},
	booktitle={ACCV},
	year={2024}
}
```

## References
VideoLLaMA2: [https://github.com/DAMO-NLP-SG/VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2) <br>
LLaMA3: [https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
