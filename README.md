# Game-Theoretic-Mixed-Experts

Code repository for "Game Theoretic Mixed Experts for Combinational Adversarial Machine Learning" (https://arxiv.org/abs/2211.14669)

Conda environment file can be found in GaME.yml. Creating the environment directly from the file doesn't always work, but it can
be a useful way to see all the library versions used. Notable version restrictions:

Python 3.6.13
Scikit-image 0.14.2
torch 1.7.1+cuda

To Generate all samples to create the Game table:

1. Insert all model files into SavedModels/{model type}
	a. The code currently supports the following model types: Vanilla. BaRT, Trash is Treasure, FAT, and SNN
	b. All models used in the paper can be found at https://drive.google.com/drive/folders/1syJB5Ge4Awc6yb6HSQH6IcVN_UQBhCjx?usp=sharing
	c. The code may have to be adjusted to properly load and handle new model types, see Automate.py/LoadModels
2. run Automate.py for CIFAR-10 and Automate_TIN.py for Tiny ImageNet
	a. Each file has 3 run parameters: which gpu to use, what the save file's number should be (use this to run parallel
	experiments), and what type of experiments to run ("ensemble" to attack each pair of defenses with AE-SAGA, "Transfer"
	to run transfer attacks (CIFAR-10 only), "sequential" to run APGD/MIME on single model defenses)
3. After Automate.py or Automate_TIN.py are complete there will be multiple save files on 