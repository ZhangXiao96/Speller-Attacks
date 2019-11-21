# Speller Attacks

This project attacked widely-used EEG spellers, e.g., P300 spellers and SSVEP spellers, with adversarial perturbation templates. We constructed these templates to perform target attacks on EEG spellers, demonstrating that the spellers can be manipulated to output anything the attacker wants when added the templates.

## 1 Requirements

Tensorflow = 1.13.1 [https://tensorflow.google.cn/](https://tensorflow.google.cn/ "https://tensorflow.google.cn/")

pyriemann = 0.2.5 [https://pyriemann.readthedocs.io/en/latest/](https://pyriemann.readthedocs.io/en/latest/ "https://pyriemann.readthedocs.io/en/latest/")

mne = 0.17.1 [http://www.martinos.org/mne/stable/index.html](http://www.martinos.org/mne/stable/index.html "http://www.martinos.org/mne/stable/index.html")


## 2 P300 Spellers

The traditional EEG extraction methods and classifiers are re-implemented in Tensorflow, so that you can construct the adversarial examples for other EEG pipelines, instead of the demo we used in our paper.

### 2.1 Blocks

Sorts of classic methods used in EEG have been re-implemented in Tensorflow in our library as you could seen in *lib/Blocks.py*. 

The blocks including:

**Processing Blocks**: xDAWN, CSP, ICA, PCA, covariance feature, tangent space feature ('riemann', 'logdet', 'logeuclid', 'euclid')

**Classifiers**: logistic regression, SVM, LDA, MDM ('riemann', 'logdet', 'logeuclid', 'euclid')

### 2.2 Pipeline

Here is a small example for how to build the precessing pipeline. Let's assume you want to build a pipeline including (xDAWN, Covariance, tangent space feature ('riemann'), logistic regression) for P300 classification. Then, you could simply use

    from lib import Blocks
    from lib import Pipeline
    
    processers = [
	    Blocks.Xdawn(n_filters=8, with_xdawn_templates=True, apply_filters=True),
	    Blocks.CovarianceFeature(with_mean_templates=False),
	    Blocks.TangentSpaceFeature(mean_metric='riemann', name='TangentSpace({})'.format('riemann'))
    ]
    
    classifier = Blocks.LogisticRegression()
    
    model = Pipeline(processers, classifier)

to build it. And if you want to **see the information of the pipeline**:

	model.pipeline_information()

Then if you want to **train the pipeline**:
	
	model.fit(epochs, y)

if you want to **get the prediction of the pipeline**:

	model.predict(epochs)

when you want to **get the model re-implemented in Tensorflow**:

	keras_model = model.get_keras_model(input_shape=(n_channel, length))

when you want to **save the model**:
	
	model.save(save_path)

and of course **load the model** (Please be aware that now only the tensorflow version could be used after being loaded):

	model.load(load_path)

### 2.3 Generate Adversarial Perturbation Templates

We give out all the code to reproduce the results in our paper. Please check our paper and files for more information.

**Related files**:

- train.py: training the victime model.
- eval.py: eval the results on clean signals.
- generate_adversarial_templates.py: generate adversarial perturbation templates.
- target_attack_with_templates.py: eval the results on perturbed signals.

### 2.4 Visualization

![](https://github.com/ZhangXiao96/Speller-Attacks/blob/master/pictures/P300_Attacker_Scores.jpg)

![](https://github.com/ZhangXiao96/Speller-Attacks/blob/master/pictures/P300_Compare.jpg)

![](https://github.com/ZhangXiao96/Speller-Attacks/blob/master/pictures/P300_Analysis.jpg)


## 3 SSVEP Spellers

Here we attacked SSVEP Spellers which used CCA as its base model.


### 3.1 Generate Adversarial Perturbation Templates

We give out all the code to reproduce the results in our paper. Please check our paper and files for more information.

**Related files**:

- EvalCCA.py: eval the results on clean signals.
- generate_universal_noise.py: generate adversarial perturbation templates.
- EvalCCAwithAttacks.py: eval the results on perturbed signals.

### 3.2 Visualization

![](https://github.com/ZhangXiao96/Speller-Attacks/blob/master/pictures/SSVEP_Attacker_Scores.jpg)

![](https://github.com/ZhangXiao96/Speller-Attacks/blob/master/pictures/P300_Compare.jpg)

![](https://github.com/ZhangXiao96/Speller-Attacks/blob/master/pictures/SSVEP_Analysis1.jpg)

![](https://github.com/ZhangXiao96/Speller-Attacks/blob/master/pictures/SSVEP_Analysis2.jpg)




