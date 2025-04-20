import unittest
import time
import tracemalloc
from MatrixChainMultiplication import matrix_multiplication


class TestMatrixChainMultiplication(unittest.TestCase):
    def measure_performance(self, matrices):
        tracemalloc.start()
        start_time = time.time()

        result = matrix_multiplication(matrices)

        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Result: {result}")
        print(f"Time Taken: {end_time - start_time:.6f} seconds")
        print(f"Memory Usage: Current = {current / 1024:.2f} KB, Peak = {peak / 1024:.2f} KB")

        return result

    def test_single_matrix(self):
        matrices = [[2, 3]] # 1x2
        result = matrix_multiplication(matrices)
        self.assertEqual(result, 0)  # single matrix, no multiplication

    def test_two_matrices(self):
        matrices = [
            [
                [1, 2, 3],
                [4, 5, 6]],  # 2x3
            [
                [7, 8], 
                [9, 10],
                [11, 12]]  # 3x2
        ]
        result = matrix_multiplication(matrices)
        self.assertEqual(result, 12) # (A*B)=2*3*2=12

    def test_three_matrices(self):
        matrices = [
            [
                [1, 2],
                [3, 4]],  # 2x2
            [
                [5, 6, 7],
                [8, 9, 10]],  # 2x3
            [
                [11],
                [12], 
                [13]]  # 3x1
        ]
        result = matrix_multiplication(matrices)
        self.assertEqual(result, 10) # (A * B) * C = (2 * 2 * 3) + (2 * 3 * 1) = 18 | A * (B * C) = (2 * 3 * 1) + (2 * 2 * 1) = 10

    def test_three_matrices_two(self):
        matrices = [
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25, 26, 27, 28, 39, 30],
                [31, 32, 33, 34, 35, 36, 37, 38, 49, 40]], # 4x10
            [
                [41, 42, 43, 44],
                [45, 46, 47, 48],
                [49, 50, 51, 52],
                [53, 54, 55, 56],
                [57, 58, 59, 60],
                [61, 62, 63, 64],
                [65, 66, 67, 68],
                [69, 70, 71, 72],
                [73, 74, 75, 76],
                [77, 78, 79, 80]], # 10x4
            [
                [81],
                [82],
                [83],
                [84]], # 4x1
        ]
        result = matrix_multiplication(matrices)
        self.assertEqual(result, 80)

    def test_invalid_dimensions(self):
        matrices = [
            [
                [1, 2],
                [3, 4]],  # 2x2
            [
                [5, 6],
                [7, 8],
                [9, 10]]  # 3x2
        ]
        with self.assertRaises(ValueError):  # invalid dimensions; this should throw an error
            matrix_multiplication(matrices)

    def test_low_matrices(self):
        # 10 matrices
        matrices = [
            [
                [1, 1, 1],
                [1, 1, 1]],  # 2x3
            [
                [1, 1], 
                [1, 1],
                [1, 1]],  # 3x2
            [
                [1, 1, 1],
                [1, 1, 1]],  # 2x3
            [
                [1, 1], 
                [1, 1],
                [1, 1]],  # 3x2
            [
                [1, 1, 1],
                [1, 1, 1]],  # 2x3
            [
                [1, 1], 
                [1, 1],
                [1, 1]],  # 3x2
            [
                [1, 1, 1],
                [1, 1, 1]],  # 2x3
            [
                [1, 1], 
                [1, 1],
                [1, 1]],  # 3x2
            [
                [1, 1, 1],
                [1, 1, 1]],  # 2x3
            [
                [1, 1], 
                [1, 1],
                [1, 1]],  # 3x2
        ]
        print("Low")
        self.measure_performance(matrices)

    def test_med_matrices(self):
        # 30 matrices
        matrices = [
            [
                [1, 1, 1],
                [1, 1, 1]],  # 2x3
            [
                [1, 1], 
                [1, 1],
                [1, 1]],  # 3x2
            [
                [1, 1, 1],
                [1, 1, 1]],  # 2x3
            [
                [1, 1], 
                [1, 1],
                [1, 1]],  # 3x2
            [
                [1, 1, 1],
                [1, 1, 1]],  # 2x3
            [
                [1, 1], 
                [1, 1],
                [1, 1]],  # 3x2
            [
                [1, 1, 1],
                [1, 1, 1]],  # 2x3
            [
                [1, 1], 
                [1, 1],
                [1, 1]],  # 3x2
            [
                [1, 1, 1],
                [1, 1, 1]],  # 2x3
            [
                [1, 1], 
                [1, 1],
                [1, 1]],  # 3x2
            
        ] * 3
        print("Med")
        self.measure_performance(matrices)

    def test_high_matrices(self):
        # 50 matrices
        matrices = [
            [
                [1, 1, 1],
                [1, 1, 1]],  # 2x3
            [
                [1, 1], 
                [1, 1],
                [1, 1]],  # 3x2
            [
                [1, 1, 1],
                [1, 1, 1]],  # 2x3
            [
                [1, 1], 
                [1, 1],
                [1, 1]],  # 3x2
            [
                [1, 1, 1],
                [1, 1, 1]],  # 2x3
            [
                [1, 1], 
                [1, 1],
                [1, 1]],  # 3x2
            [
                [1, 1, 1],
                [1, 1, 1]],  # 2x3
            [
                [1, 1], 
                [1, 1],
                [1, 1]],  # 3x2
            [
                [1, 1, 1],
                [1, 1, 1]],  # 2x3
            [
                [1, 1], 
                [1, 1],
                [1, 1]],  # 3x2
            
        ] * 5
        print("High")
        self.measure_performance(matrices)

if __name__ == "__main__":
    unittest.main()
