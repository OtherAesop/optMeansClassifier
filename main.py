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


def test_network(training_file, testing_file, k=10, iterations=5, verbose=False):  # function for neatness
    print(f"Beginning training and testing of {training_file} and {testing_file}...")
    start = time()
    results = list()
    for _ in range(iterations):
        seed, result1, result2 = kmeans(training_file, testing_file, k, 1, verbose)
        results.append((seed, result1, result2))
    end = time()
    test_time = end - start
    print(f'...ending training and testing of {training_file} and {testing_file}, process completed'
          f' in {helper.translate_seconds(test_time)} (HH:MM:SS).\n')
    return test_time, results


total_time, results = test_network("datasets/optdigits.train", "datasets/optdigits.test", 10, 1, True)


print(f'All tests completed in {helper.translate_seconds(total_time)}.\n')
