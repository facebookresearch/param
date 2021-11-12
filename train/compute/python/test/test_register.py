import unittest

from ..lib.data import register_data_generator, DataGenerator, data_generator_map
from ..lib.iterator import ConfigIterator, register_config_iterator, config_iterator_map
from ..lib.operator import (
    register_operator,
    register_operators,
    OperatorInterface,
    op_map,
)


class TestRegister(unittest.TestCase):
    def test_register_config_iterator(self):
        class TestConfigIterator(ConfigIterator):
            pass

        name = "__TestConfigIterator__"
        register_config_iterator(name, TestConfigIterator)
        self.assertTrue(name in config_iterator_map)
        self.assertRaises(
            ValueError, register_config_iterator, name, TestConfigIterator
        )

    def test_register_data_generator(self):
        class TestDataGenerator(DataGenerator):
            pass

        name = "__TestDataGenerator__"
        register_data_generator(name, TestDataGenerator)
        self.assertTrue(name in data_generator_map)
        self.assertRaises(ValueError, register_data_generator, name, TestDataGenerator)

    def test_register_operator(self):
        class TestOperator(OperatorInterface):
            pass

        name = "__TestOperator__"
        register_operator(name, TestOperator)
        self.assertTrue(name in op_map)
        self.assertRaises(ValueError, register_operator, name, TestOperator)

        name_1 = "__TestOperator_1__"
        name_2 = "__TestOperator_2__"
        register_operators({name_1: TestOperator, name_2: TestOperator})
        self.assertTrue(name_1 in op_map)
        self.assertTrue(name_2 in op_map)
        self.assertRaises(ValueError, register_operators, {name_1: TestOperator})
        self.assertRaises(ValueError, register_operators, {name_2: TestOperator})
        self.assertRaises(
            ValueError, register_operators, {name_1: TestOperator, name_2: TestOperator}
        )


if __name__ == "__main__":
    unittest.main()
