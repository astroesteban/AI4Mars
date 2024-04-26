# AI4Mars

A Machine learning Model for Martian Terrain Image Segmentation

## Description

The AI4Mars project provides an environment to train and evaluate a deep
learning model on the NASA [AI4Mars](https://data.nasa.gov/Space-Science/AI4MARS-A-Dataset-for-Terrain-Aware-Autonomous-Dri/cykx-2qix/about_data)
dataset. 

The dataset is described as:

> This dataset was built for training and validating terrain classification models for Mars, which may be useful in future autonomous rover efforts. It consists of ~326K semantic segmentation full image labels on 35K images from Curiosity, Opportunity, and Spirit rovers, collected through crowdsourcing. Each image was labeled by 10 people to ensure greater quality and agreement of the crowdsourced labels. It also includes ~1.5K validation labels annotated by the rover planners and scientists from NASA’s MSL (Mars Science Laboratory) mission, which operates the Curiosity rover, and MER (Mars Exploration Rovers) mission, which operated the Spirit and Opportunity rovers.

The focus of this project is to train a semantic segmentation model for a
hypothetical deployment on a Mars rover. Therefore, the objective of this
project is **not** to create a state-of-the-art semantic segmentation model,
but rather to take tried and true model architectures and techniques to create
a safe and reliable model for hypothetical deployment on spacecraft flight
software.

## Getting Started

### Dependencies

* Visual Studio Code
* Docker
