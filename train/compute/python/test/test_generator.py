import copy
import unittest

from param_bench.train.compute.python.lib.generator import (
    full_range,
    IterableList,
    ListProduct,
    TableProduct,
)


class TestGenerator(unittest.TestCase):
    def test_full_range(self):
        def gen(start, end, step):
            result = []
            x = full_range(start, end, step)
            for i in x:
                result.append(i)
            return result

        result = gen(-3, 2, 1)
        expected = [-3, -2, -1, 0, 1, 2]
        self.assertEqual(result, expected)

        expected = [5, 7, 9, 11]
        result = gen(5, 11, 2)
        self.assertEqual(result, expected)

        result = gen(3, 11, 3)
        expected = [3, 6, 9]
        self.assertEqual(result, expected)

    def test_iterable_List(self):
        simple_list = IterableList([2, 4, 6, 8])
        result = []
        for item in simple_list:
            result.append(item)
        expected = [2, 4, 6, 8]

        self.assertEqual(result, expected)

        # Case with IterableList with nested ListProduct
        simple_list = IterableList([1, 2, ListProduct([1, full_range(3, 5)])])
        result = []
        for item in simple_list:
            if isinstance(item, ListProduct):
                this_result = []
                for sub_item in item:
                    this_result.append(copy.deepcopy(sub_item))
                result.append(this_result)
            else:
                result.append(item)
        expected = [1, 2, [[1, 3], [1, 4], [1, 5]]]

        self.assertEqual(result, expected)

        # Empty List Case
        simple_list = IterableList([])
        result = []
        for item in simple_list:
            result.append(item)
        expected = []

        self.assertEqual(result, expected)

    def test_list_product(self):
        iter_list = [1, full_range(3, 5), 2, full_range(7, 13, 3)]
        result = []
        for gen_list in ListProduct(iter_list):
            result.append(copy.deepcopy(gen_list))
        expected = [
            [1, 3, 2, 7],
            [1, 3, 2, 10],
            [1, 3, 2, 13],
            [1, 4, 2, 7],
            [1, 4, 2, 10],
            [1, 4, 2, 13],
            [1, 5, 2, 7],
            [1, 5, 2, 10],
            [1, 5, 2, 13],
        ]
        self.assertEqual(result, expected)

        iter_list = [
            1,
            ListProduct([2, full_range(3, 5)]),
            6.7,
            TableProduct(
                {
                    "A": ListProduct([6, full_range(2, 7, 2)]),
                    "B": IterableList(["str 1", "str 2"]),
                }
            ),
            "str 3",
        ]
        result = []
        for gen_list in ListProduct(iter_list):
            result.append(copy.deepcopy(gen_list))
        expected = [
            [1, [2, 3], 6.7, {"A": [6, 2], "B": "str 1"}, "str 3"],
            [1, [2, 3], 6.7, {"A": [6, 2], "B": "str 2"}, "str 3"],
            [1, [2, 3], 6.7, {"A": [6, 4], "B": "str 1"}, "str 3"],
            [1, [2, 3], 6.7, {"A": [6, 4], "B": "str 2"}, "str 3"],
            [1, [2, 3], 6.7, {"A": [6, 6], "B": "str 1"}, "str 3"],
            [1, [2, 3], 6.7, {"A": [6, 6], "B": "str 2"}, "str 3"],
            [1, [2, 4], 6.7, {"A": [6, 2], "B": "str 1"}, "str 3"],
            [1, [2, 4], 6.7, {"A": [6, 2], "B": "str 2"}, "str 3"],
            [1, [2, 4], 6.7, {"A": [6, 4], "B": "str 1"}, "str 3"],
            [1, [2, 4], 6.7, {"A": [6, 4], "B": "str 2"}, "str 3"],
            [1, [2, 4], 6.7, {"A": [6, 6], "B": "str 1"}, "str 3"],
            [1, [2, 4], 6.7, {"A": [6, 6], "B": "str 2"}, "str 3"],
            [1, [2, 5], 6.7, {"A": [6, 2], "B": "str 1"}, "str 3"],
            [1, [2, 5], 6.7, {"A": [6, 2], "B": "str 2"}, "str 3"],
            [1, [2, 5], 6.7, {"A": [6, 4], "B": "str 1"}, "str 3"],
            [1, [2, 5], 6.7, {"A": [6, 4], "B": "str 2"}, "str 3"],
            [1, [2, 5], 6.7, {"A": [6, 6], "B": "str 1"}, "str 3"],
            [1, [2, 5], 6.7, {"A": [6, 6], "B": "str 2"}, "str 3"],
        ]
        self.assertEqual(result, expected)

        iter_list = []
        result = []
        for gen_list in ListProduct(iter_list):
            result.append(gen_list)
        expected = [[]]
        self.assertEqual(result, expected)

    def test_table_product(self):
        iter_dict = {"A": 1, "B": full_range(3, 5), "C": 2, "D": full_range(7, 13, 3)}
        result = []
        for gen_dict in TableProduct(iter_dict):
            result.append(copy.deepcopy(gen_dict))
        expected = [
            {"A": 1, "B": 3, "C": 2, "D": 7},
            {"A": 1, "B": 3, "C": 2, "D": 10},
            {"A": 1, "B": 3, "C": 2, "D": 13},
            {"A": 1, "B": 4, "C": 2, "D": 7},
            {"A": 1, "B": 4, "C": 2, "D": 10},
            {"A": 1, "B": 4, "C": 2, "D": 13},
            {"A": 1, "B": 5, "C": 2, "D": 7},
            {"A": 1, "B": 5, "C": 2, "D": 10},
            {"A": 1, "B": 5, "C": 2, "D": 13},
        ]
        self.assertEqual(result, expected)

        iter_dict = {
            "A": 1,
            "B": ListProduct([3, full_range(4, 5)]),
            "C": 2,
            "D": TableProduct({"E": full_range(7, 9), "F": 10}),
        }
        result = []
        for gen_dict in TableProduct(iter_dict):
            result.append(copy.deepcopy(gen_dict))
        expected = [
            {"A": 1, "B": [3, 4], "C": 2, "D": {"E": 7, "F": 10}},
            {"A": 1, "B": [3, 4], "C": 2, "D": {"E": 8, "F": 10}},
            {"A": 1, "B": [3, 4], "C": 2, "D": {"E": 9, "F": 10}},
            {"A": 1, "B": [3, 5], "C": 2, "D": {"E": 7, "F": 10}},
            {"A": 1, "B": [3, 5], "C": 2, "D": {"E": 8, "F": 10}},
            {"A": 1, "B": [3, 5], "C": 2, "D": {"E": 9, "F": 10}},
        ]
        self.assertEqual(result, expected)

        iter_dict = {}
        result = []
        for gen_dict in TableProduct(iter_dict):
            result.append(gen_dict)
        expected = [{}]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
