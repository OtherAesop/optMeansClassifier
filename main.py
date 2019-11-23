# optMeansClassifier, a k-means classifier for the OptDigits data set
# Copyright (C) 2019  Ian Gore
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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import utility as helper
from kmeans import kmeans
from time import time


def test_network(training_file, testing_file, k=10, verbose=False, save=False):  # function for neatness
    print(f"Beginning training and testing of {training_file} and {testing_file}...")
    start = time()
    mse, mss, ent, seed, c_matrix = kmeans(training_file, testing_file, k, verbose)
    result = (mse, mss, ent, seed)
    end = time()
    test_time = end - start
    print(f'...ending training and testing of {training_file} and {testing_file}, process completed'
          f' in {helper.translate_seconds(test_time)} (HH:MM:SS).\n')
    if save:
        helper.save(helper.translate_seconds(test_time), k, mse, mss, ent, seed, c_matrix)
    return test_time, result, c_matrix


total_time, results, matrix = test_network("datasets/optdigits.train", "datasets/optdigits.test", 10, True, True)

total_time_, results_, matrix_ = test_network("datasets/optdigits.train", "datasets/optdigits.test", 30, True, True)


print(f'All tests completed in {helper.translate_seconds(total_time + total_time_)}.\n')
