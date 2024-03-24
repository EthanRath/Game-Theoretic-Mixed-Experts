# Game-Theoretic-Mixed-Experts

This is a work in progress, some features still need to be refined for ease of use.

Code repository for "Game Theoretic Mixed Experts for Combinational Adversarial Machine Learning"

Conda environment file can be found in GaME.yml. Creating the environment directly from the file doesn't always work, but it can
be a useful way to see all the library versions used. Notable version restrictions:

Python 3.6.13
Scikit-image 0.14.2
torch 1.7.1+cuda

To Generate all samples to create the Game table:

1. Insert all model files into SavedModels/{model type}
	
	a. The code currently supports the following model types: Vanilla. BaRT, Trash is Treasure, FAT, and SNN
	
	b. All models used in the paper will be uploaded to google drive upon publication
	
	c. The code may have to be adjusted to properly load and handle new model types, see Automate.py/LoadModels
2. run Automate.py for CIFAR-10 and Automate_TIN.py for Tiny ImageNet
	
	a. Each file has 3 run parameters: which gpu to use, what the save file's number should be (use this to run parallel
	experiments), and what type of experiments to run ("ensemble" to attack each pair of defenses with AE-SAGA, "Transfer"
	to run transfer attacks (CIFAR-10 only), "sequential" to run APGD/MIME on single model defenses)

To generate and solve the Game table:

1. Run GaME.py

	a. The file has 4 parameters: the dataset being used ("tiny" or "cifar10"), the maximum ensemble size n, what voting type
	("fh" or "fs"), and whether to use a uniform ensemble distribution ("true" or "false").

New Attacks:

1. MIME "MIM_EOT_Wrapper" found in "Attacks/AttackWrappersProtoSAGA.py"

2. AE-SAGA "SelfAttentionGradientAttack_EOT" found in "Attacks/AttackWrappersProtoSAGA.py"
