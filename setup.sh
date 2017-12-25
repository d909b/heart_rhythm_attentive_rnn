#! /bin/bash
#
# setup.sh - Pre-compiles prediction models in order to speed up the consecutive invocations that follow.
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

# Necessary exports.
export KERAS_BACKEND=theano
export THEANO_FLAGS="mode=FAST_COMPILE,optimizer=fast_compile,optimizer_excluding=inplace:inplace_opt,reoptimize_unpickled_function=False,optimizer_including=local_remove_all_assert"


# Pre-compile ML model to speed up evaluation later.
python app_lvl2_blender.py \
--dataset="$(pwd)" \
--single_file="sample" \
--load_meta_params="./models/meta-params.pickle" \
--load_existing="./models/bl_131_full_val-blender-60-0.4171.h5" \
--do_train=False \
--precompile \
--multi_class
