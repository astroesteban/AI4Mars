---
language: en
license: mit
library_name: fastai
base_model: U-Net ResNet-18 Backbone
datasets:
- ai4mars_v0.1
metrics:
- mIoU
---

# Model Card for AI4Mars Model

Semantic segmentation of the Martian terrain based on the following classes

| RGB         | Key             |
|-------------|-----------------|
| 0,0,0       | soil            |
| 1,1,1       | bedrock         |
| 2,2,2       | sand            |
| 3,3,3       | big rock        |
| 255,255,255 -> 4, 4, 4 | NULL (no label) |

The model was trained on the AI4Mars v0.1 dataset.


## Model Details

### Model Description

U-Net model with a pretrained ResNet-18 backbone.

semantic segmentation of the ai4mars terrain dataset

- **Developed by:** Esteban Duran
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** [More Information Needed]
- **Model type:** [More Information Needed]
- **Language(s) (NLP):** en
- **License:** mit
- **Finetuned from model [optional]:** U-Net ResNet-18 Backbone

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/astroesteban/AI4Mars
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]

## Uses

This model serves as a reference on how to train a simple segmentation learner
on the AI4Mars dataset.

### Direct Use

The model was exported by the Fast.AI library. It can be loaded for inference
as follows:

```py
from fastai.imports import *
from fastai.vision.all import *

learner = load_learner("/workspace/models/ai4mars_unet_resnet18.pkl")

pred = learner.predict(img)
```

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

[More Information Needed]

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

[More Information Needed]

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Use the code below to get started with the model.

```py
from fastai.imports import *
from fastai.vision.all import *

learner = load_learner("/workspace/models/ai4mars_unet_resnet18.pkl")

pred = learner.predict(img)
```

## Training Details

### Training Data

[NASA AI4Mars Dataset](https://data.nasa.gov/Space-Science/AI4MARS-A-Dataset-for-Terrain-Aware-Autonomous-Dri/cykx-2qix/about_data)

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

The input images were resized from 1024x1024 to 256x256 due to the limitations
of the GPU.

The masks had a `null` value of 255 that was replaced with 4. See the Jupyter
Notebook for details.

#### Training Hyperparameters

- **Training regime:** fp32

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

[More Information Needed]

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

mIOU because that is what was used in the original AI4Mars dataset paper.

[AI4Mars Paper](https://data.nasa.gov/api/views/cykx-2qix/files/247093ee-6bcd-45df-8b61-04f13b0346ac?download=true&filename=AI4Mars-AI4Space-final.pdf)

### Results

[More Information Needed]

#### Summary



## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** NVIDIA RTX A2000 8GB Laptop GPU
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications [optional]

### Model Architecture and Objective

U-Net with ResNet-18 backbone

### Compute Infrastructure

Laptop

#### Hardware

__CUDA VERSION: 8902
__Number CUDA Devices: 1
__CUDA Device Name: NVIDIA RTX A2000 8GB Laptop GPU
__CUDA Device Total Memory [GB]: 8.58947584

#### Software

- FastAI v2.7.14
- PyTorch v2.2.2

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]