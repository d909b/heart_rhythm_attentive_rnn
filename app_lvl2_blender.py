"""
app_lvl2_blender.py - Blender model to combine predictions from a model ensemble.

Copyright (C) 2017  Patrick Schwab, ETH Zurich

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import gc
import sys
import numpy as np
from os.path import join, isfile, dirname, basename
from af_classifier.feature_extractor import *
from af_classifier.hmm_model import load_hmm_models, HMMModel, get_normalising_factors
from af_classifier.model_factory import ModelFactory
from af_classifier.data_source import PhysioNet2017DataSource
from af_classifier.apps.util import initialise_app, read_ensemble_json, equal_class_weights
from af_classifier.pipeline import split_and_normalize_data_set, configure_and_extract_from_data_set, \
    load_sequence_data, create_synthetic_balanced_data_set, equalize_data_set

if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


def collect_level1_features(args, data_set):
    ensemble = read_ensemble_json(args["ensemble"])
    ensemble_dir = dirname(args["ensemble"])

    args_copy = dict(args)

    # Extract features from the whole data set and split later.
    args_copy["test_set_fraction"] = 1.

    # Pre-normalize data set outside the loop.
    _, seq_data = split_and_normalize_data_set(args_copy, data_set)

    outputs, architectures = [], {}
    for model in ensemble:
        old_weights = None
        args_copy = dict(args)

        # Use the original input features.
        args_copy["features"] = model["input"]
        extracted_data, _, _ = configure_and_extract_from_data_set(args_copy, seq_data)

        if model["name"] == "$HMM_MODELS":
            models = load_hmm_models(args["hmm_models"])

            if not isfile(args["likelihood_normalization_params"]):
                print("ERROR: Likelihood normalisation parameter file was not found at",
                      args["likelihood_normalization_params"], file=sys.stderr)

                # This error is unrecoverable.
                sys.exit(0)
            else:
                print("INFO: Loading normalising factors from", args["likelihood_normalization_params"],
                      file=sys.stderr)
                factors = np.load(args["likelihood_normalization_params"])["arr_0"]

            loaded_model = HMMModel(models, factors)
        else:
            if "architecture" in model:
                architecture = model["architecture"]
                if architecture in architectures:
                    # If this architecture was already loaded: Reuse it from cache and switch out weights.
                    # Previous weights need to be temporarily stored so that we can revert the model
                    # to its original state after reuse.
                    model_name = architectures[architecture]
                    loaded_model = ModelFactory.get_model(join(ensemble_dir, model_name))
                    if args["precompile"] or \
                       (args["do_train"] == "False" and not args["single_file"]):
                        old_weights = loaded_model.get_weights()
                    print("INFO: Loading weights from:", model["name"], file=sys.stderr)
                    loaded_model.load_weights(join(ensemble_dir, model["name"]))
                else:
                    # This architecture has not been cached yet but should be made available for reuse later on.
                    architectures[architecture] = model["name"]
                    loaded_model = ModelFactory.get_model(join(ensemble_dir, model["name"]))
            else:
                loaded_model = ModelFactory.get_model(join(ensemble_dir, model["name"]))

        prediction = loaded_model.predict(extracted_data.x)
        outputs.append(prediction)

        if old_weights is not None:
            # Restore weights after collecting predictions.
            loaded_model.set_weights(old_weights)

    adjusted_sampling_frequency = args["sampling_frequency"] / float(args["downsample"])

    # Add wavelet entropy.
    extracted_data = seq_data.extract([WaveletEntropyFeatureExtractor(4)])
    outputs.append(extracted_data.x)

    # Add AAD wavelet entropy over beats.
    extracted_data = seq_data.extract([QRSFeatureExtractor(adjusted_sampling_frequency)])

    if extracted_data.x[0].ndim == 1:
        extracted_data.x = [extracted_data.x]

    extracted_data = extracted_data.extract([WaveletEntropyFeatureExtractor(4)])
    extracted_data = extracted_data.extract([AbsoluteAverageDeviationFeatureExtractor()])
    outputs.append(extracted_data.x)

    # Add AAD delta RR.
    extracted_data = seq_data.extract([DeltaRRFeatureExtractor(300, normalized_distances=True)])
    extracted_data = extracted_data.extract([AbsoluteAverageDeviationFeatureExtractor()])
    outputs.append(extracted_data.x)

    # Add WLE
    extracted_data = seq_data.extract([RelativeWaveletEnergiesFeatureExtractor(4, with_total=False)])
    outputs.append(extracted_data.x)

    seq_data.x = np.concatenate(outputs, axis=-1)

    return seq_data


def main():
    args = initialise_app()

    is_training = args["do_train"] != "False"
    is_precompile = args["precompile"]

    if not is_training and args["single_file"]:
        if args["early_gc_disable"]:
            gc.disable()

        # We need to predict at least once to pre-compile models.
        raw_sequence_data = PhysioNet2017DataSource()
        raw_sequence_data.load_single(args["single_file"])

        args_copy = dict(args)

        # No need to split into test / train set. We are only evaluating once.
        args_copy["test_set_fraction"] = 1.
        # No need to save meta params for single predictions.
        args_copy["meta_params_file"] = None

        seq_train = collect_level1_features(args_copy, raw_sequence_data.copy())

        # Don't waste cycles on GC - app is terminated after one prediction anyway.
        gc.disable()
    else:
        if args["cache_level1"]:
            raw_sequence_data = load_sequence_data(args)
            seq_train = collect_level1_features(args, raw_sequence_data)
            pickle.dump(seq_train,
                        open(args["cache_file"], "w"),
                        pickle.HIGHEST_PROTOCOL)
            print("INFO: Saved level 1 cache.", file=sys.stderr)
        else:
            print("INFO: Loading level 1 cache.", file=sys.stderr)
            seq_train = pickle.load(open(args["cache_file"], "r"))

        if args["load_meta_params"]:
            print("INFO: Loading meta params from file: ", args["load_meta_params"], file=sys.stderr)
            split_indices, _ = pickle.load(open(args["load_meta_params"], "r"))
        else:
            print("INFO: Not using meta params.", file=sys.stderr)
            split_indices = None

        print('INFO: Train set balance is: ', seq_train.get_class_balance(), file=sys.stderr)

    # Remove superfluous secondary output nodes.
    seq_train.x = np.concatenate((seq_train.x[:, :30:2],
                                  seq_train.x[:, 30:]),
                                 axis=-1)

    if not args["load_existing"]:
        from af_classifier.model_builder import ModelBuilder

        num_outputs = 4
        num_inputs = seq_train.x[0].shape[-1]

        nn_model = ModelBuilder.build_nn_model(num_inputs,
                                               num_outputs,
                                               p_dropout=float(args["dropout"]),
                                               num_units=int(args["num_units"]),
                                               learning_rate=float(args["learning_rate"]),
                                               num_layers=int(args["num_recurrent_layers"]),
                                               noise=float(args["noise"]))
    else:
        print("INFO: Loading existing model: ", args["load_existing"], file=sys.stderr)
        nn_model = ModelFactory.get_model(args["load_existing"])

    if is_training:
        from af_classifier.model_trainer import ModelTrainer

        if args["do_hyperopt"]:
            # Perform a hyper parameter search optimising loss.
            from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

            # Define hyper parameter space over which to do optimisation.
            space = {
                'num_layers': hp.uniform('num_layers', 1, 5),
                'num_units': hp.uniform('num_units', 64, 256),
                'dropout': hp.uniform('dropout', 0.4, .85),
                'num_epochs': hp.uniform('num_epochs', 120, 1500)
            }

            def train_hyper_opts(params):
                from sklearn.cross_validation import StratifiedKFold

                num_folds = int(1./float(args["test_set_fraction"]))

                seq_train.to_indexed()

                # Cross validation on validation set to select blender model hyperparameters.
                skf = StratifiedKFold(seq_train.y, n_folds=num_folds)

                scores = np.zeros(num_folds)
                for i, indices in enumerate(skf):
                    train_idx, test_idx = indices

                    # Prepare the training and test set for this fold.
                    train_set = seq_train.__class__(seq_train.x[train_idx],
                                                    seq_train.y[train_idx])
                    test_set = seq_train.__class__(seq_train.x[test_idx],
                                                   seq_train.y[test_idx])

                    train_set.to_categorical(num_outputs)
                    test_set.to_categorical(num_outputs)

                    num_inputs = train_set.x[0].shape[-1]

                    class_weight = None
                    if args["class_weighted"]:
                        class_weight = equal_class_weights(train_set)
                        print("INFO: Class weights are", class_weight, file=sys.stderr)

                    opt_model = ModelBuilder.build_nn_model(num_inputs,
                                                            num_outputs,
                                                            p_dropout=float(params["dropout"]),
                                                            num_units=int(np.round(params["num_units"])),
                                                            learning_rate=float(args["learning_rate"]),
                                                            num_layers=int(np.round(params["num_layers"])))

                    score, acc = ModelTrainer.train_model(opt_model,
                                                          train_set,
                                                          test_set,
                                                          int(params["num_epochs"]),
                                                          int(args["batch_size"]),
                                                          do_early_stopping=False,
                                                          with_checkpoints=False,
                                                          do_eval=True,
                                                          save_best_only=True,
                                                          checkpoint_path=join(args["output_directory"],
                                                                               args["model_name"]),
                                                          class_weight=class_weight,
                                                          report_min=False)

                    scores[i] = score

                seq_train.to_categorical(num_outputs)

                print("INFO: Tested with params:", params, file=sys.stderr)
                print("INFO: Scores were:", scores, file=sys.stderr)

                return {'loss': np.mean(scores), 'status': STATUS_OK}

            trials = Trials()
            best = fmin(train_hyper_opts, space, algo=tpe.suggest, max_evals=int(args["num_epochs"]), trials=trials)
            print("INFO: Best config was:", best, file=sys.stderr)
        else:
            seq_test = seq_train.copy()

            class_weight = None
            if args["class_weighted"]:
                class_weight = equal_class_weights(seq_train)
                print("INFO: Class weights are", class_weight, file=sys.stderr)

            ModelTrainer.train_model(nn_model,
                                     seq_train,
                                     seq_test,
                                     int(args["num_epochs"]),
                                     int(args["batch_size"]),
                                     do_early_stopping=False,
                                     with_checkpoints=True,
                                     do_eval=False,
                                     save_best_only=False,
                                     checkpoint_path=join(args["output_directory"], args["model_name"]),
                                     class_weight=class_weight)


    if args["single_file"]:
        # Run blender once to fully initialise it.
        # Without this the blender model would not be usable after unpickling.
        y = nn_model.predict(seq_train.x)

        # Select the maximum activation as our predicted class index.
        y_idx = np.argmax(y, axis=1)[0]

        print("INFO: Predictions are:", y, file=sys.stderr)
        print("F:", seq_train.x[0], file=sys.stderr)

        print_answer_line(basename(args["single_file"]), y_idx)

        if is_precompile:
            ModelFactory.save_all_models_precompiled()
    else:
        from af_classifier.model_trainer import ModelTrainer

        ModelTrainer.create_confusion_matrix(nn_model, seq_train)

        if not is_training:
            seq_validation = load_sequence_data({"dataset": args["level2_dataset"]})

            args_copy = dict(args)

            # No need to split into test / train set. We are only evaluating once.
            args_copy["test_set_fraction"] = 1.
            # No need to save meta params for single predictions.
            args_copy["meta_params_file"] = None

            seq_validation = collect_level1_features(args_copy, seq_validation)

            # Remove superfluous secondary output nodes.
            seq_validation.x = np.concatenate((seq_validation.x[:, :30:2],
                                               seq_validation.x[:, 30:]),
                                              axis=-1)

            y_pred = nn_model.predict(seq_validation.x)

            file_list, _ = PhysioNet2017DataSource.read_reference(args["level2_dataset"])

            # Prepare the answers.txt output for the entry.zip distribution.
            y_pred_idx = np.argmax(y_pred, axis=-1)
            for i, y_idx in enumerate(y_pred_idx):
                print_answer_line(file_list[i], y_idx)

            # ModelFactory.save_all_models_precompiled()


def print_answer_line(entry_name, y_idx):
    print(entry_name + "," + PhysioNet2017DataSource.CODE_MAP[y_idx] + "\n", end='')

if __name__ == '__main__':
    main()
