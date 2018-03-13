## Beat by Beat: Classifying Cardiac Arrhythmias with Recurrent Neural Networks

Contact: Patrick Schwab, ETH Zurich <patrick.schwab@hest.ethz.ch>

Authors: See AUTHORS.txt

License: GPLv3; See LICENSE.txt

Description: Predicts the rhythm of given ECG signals using ensembles of recurrent neural networks. We delineated our approach in [this manuscript](https://arxiv.org/abs/1710.06319). This solution is an entry to the [PhysioNet / CinC challenge 2017](https://physionet.org/challenge/2017/).


### Citation

If you reference our methodology, code or results in your work, please consider citing:

    @inproceedings{schwab2017beat,
      title={Beat by Beat: Classifying Cardiac Arrhythmias with Recurrent Neural Networks},
      author={Schwab, Patrick and Scebba, Gaetano and Zhang, Jia and Delai, Marco and Karlen, Walter},
      booktitle={arXiv preprint arXiv:1710.06319},
      year={2017}
    }


### Installation

Requires:
- [pip](https://pip.pypa.io/en/stable/)
- Keras >= 1.2.2
- Theano >= 0.8.2
- matplotlib >= 1.3.1
- pandas >= 0.18.0
- h5py >= 2.6.0
- scikit-learn == 0.17.1
- pywavelets == 0.2.2
- imbalanced_learn == 0.2.1
- pyhsmm == 0.1.7

To train models you need to download the [PhysioNet 2017 challenge data](physionet.org/challenge/2017/).


### ATTENTION - PATCHES REQUIRED:

To save bidirectional RNNs you need to additionally patch the version of Keras installed by pip using [this patch](https://github.com/fchollet/keras/commit/b5dc734f4e08b997ae50c4e29a5c4b589595b188).

To train HSMMs you need to patch the PyHSMM library in `pyhsmm_states.py` line 1071 to:
```obs, offset = obs[:,state], offset```
