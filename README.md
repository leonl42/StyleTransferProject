In this project we would like to represent this dragon:

![Alt text](./dragon.jpg?raw=true "Title")

In the style of this wave: 

![Alt text](./wave.jpg?raw=true "Title")

To do so I used 4 approaches.

1. Directly optimizing the image (1_style_gatyes.ipynb)
2. Basic feedforward model (Johnson et al. (2016)) (2_style_feedforward.ipynb)
3. AdaIN (Huang & Belongie (2017)) (3_style_adain.ipynb)
4. Style-Attentional Networks (Park & Lee (2019)) (4_style_SANet.iypnb)

Each approache is in a separate notebook. 

**If you wish to just do Inference:** 
1. Create an environment from the environment.yml
2. Only run the cells that have a comment with "INFERENCE" in the top (might have to correct paths in the project)

**If you wish to also retrain the models:** 
1. Create an environment from the environment.yml
2. Download the coco dataset by opening this link: http://images.cocodataset.org/zips/train2017.zip
3. Just run one cell after the other (might have to correct paths in the project)


**Some important things about the notebooks:**
1. Having a GPU is not necessariy for inference. Furthermore, this project was created with the intend to run it on a local machine. If you wish to run it on colab,
you might have to change all the paths and the imports aswell. (The .ipynb notebooks import from the .py files in utils/)
3. Hyperparameters are in enum style (All caps). These can be changed. Almost all hyperparameters are in the notebook. There are only one or two in the .py files in the utils folder.
But do not wory about them. 
4. I use a path naming style where folders end with "/". So specifying your image path
could look like this "C:/Users/Documents/images/".
5. All notebooks can be run independently from each other. But it might make sense to start with 1_style_gatyes.ipynb since it contains the most comments.
Furthermore, training loops are similar to an extend in all notebooks.
6. Note that models have only been trained for a short amount of time on a small datasets with non-fitted hyperparameters. Results are expected to not look great.
Furthermore, note that AdaIN and SANet can be trained with multiple style images, which is not done here. We use the same style image.
