#! /bin/bash
#
# next.sh - Predicts what type of rhythm a given single ECG file (passed as the first parameter) contains.
#
# Copyright (C) 2017  Patrick Schwab, ETH Zurich
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

set -e
set -o pipefail

export PYTHONPATH=
export KERAS_BACKEND=theano
export THEANO_FLAGS="mode=FAST_COMPILE,optimizer=fast_compile,optimizer_excluding=inplace,reoptimize_unpickled_function=False,allow_gc=False,optimizer_including=local_remove_all_assert"

RECORD=$1

python app_lvl2_blender.py \
--dataset="$(pwd)" \
--single_file="${RECORD}" \
--do_train=False \
--multi_class \
--early_gc_disable \
--load_meta_params="./models/meta-params.pickle" \
--ensemble="./models/ensemble_precompiled.json" \
--encoder_model="./models/ae_encoder.h5.pickle" \
--morph_encoder_model="./models/stae_3l_p0_gleveled_dim24-ae-75-0.0528_encoder.h5.pickle,./models/stae_3l_p0_gleveled_dim24-ae-301-0.0434_1_encoder.h5.pickle,./models/stae_3l_p0_gleveled_dim24-ae-300-0.0811_2_encoder.h5.pickle" \
--load_existing="./models/bl_131_full_val-blender-60-0.4171.h5.pickle" >> answers.txt