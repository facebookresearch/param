from typing import Any, Dict, List, Set


def full_range(a: int, b: int, s: int = 1):
    """
    Returns inclusive range: a <= x <= b, by step of s
    """
    return range(a, b + 1, s)


# Repeatable iterator for lists.
class IterableList:
    def __init__(self, items: List[Any]):
        self.items = items

    def __iter__(self):
        return self.Iterator(self.items)

    class Iterator:
        def __init__(self, items: List[Any]):
            self.iter = iter(items)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.iter)


class ListProduct:
    """
    ListProduct takes a list of repeatable iterables (like range()), and
    generates the Cartesian product of the iterables.

    Important:
    The list returned will be mutated in place for each iteration. If the
    user code wants to keep a copy or modify the generated list, it should
    make a copy of the returned result using `copy.deepcopy(generated_list)`.
    Example:
    ```
    result = []
    for gen_list in ListProduct(iter_list):
        result.append(copy.deepcopy(gen_list))
    ```

    This interface wraps the Iterator so that a new iterator is created once
    the generator is exhausted. This allows repeatable iterations, i.e.

    iter_list_1 = [range(2, 6, 2), range(1, 3, 1), range(2, 4, 1)]
    iter_list_2 = [range(2, 6, 2), range(1, 3, 1)]
    prod = ListProduct([ListProduct(iter_list_1), ListProduct(iter_list_2)])
    for i in prod:
        print("i",i)

    for i in prod:
        print("i",i)
    """

    def __init__(self, iter_list: List[Any]):
        self.iter_list: List[Any] = iter_list

    def __iter__(self):
        return self.Iterator(self.iter_list, [None] * len(self.iter_list), 0)

    class Iterator:
        def __init__(self, iter_list: List[Any], val_list: List[Any], idx: int):
            self.generator = self._generate_next(iter_list, val_list, idx)

        def __iter__(self):
            return self

        def _generate_next(self, iter_list: List[Any], val_list: List[Any], idx: int):
            if iter_list:
                # If current item is iterable, loop through and recursive to next
                # item in the list
                if type(iter_list[0]) in iterable_types:
                    for i in iter_list[0]:
                        val_list[idx] = i
                        if len(iter_list) == 1:
                            yield val_list
                        else:
                            yield from self._generate_next(
                                iter_list[1:], val_list, idx + 1
                            )
                # If current item is not iterable, just assign and recursive to next
                # item in the list
                else:
                    val_list[idx] = iter_list[0]
                    if len(iter_list) == 1:
                        yield val_list
                    else:
                        yield from self._generate_next(iter_list[1:], val_list, idx + 1)
            else:
                yield iter_list

        def __next__(self):
            return next(self.generator)


class TableProduct:
    def __init__(self, table: Dict[Any, Any]):
        self.table: Dict[Any, Any] = table
        self.result: Dict[Any, Any] = {}

    def __iter__(self):
        iterable_keys = []
        self.result = dict.fromkeys(self.table)
        # check which key/val has iterables, copy non iterable values to
        # result table
        for key, val in self.table.items():
            # Only works with new classes with __iter__ interface.
            # If needed use iter() to check, but more expensive.
            if type(val) in iterable_types:
                iterable_keys.append(key)
            else:
                self.result[key] = val
        return self.Iterator(self.table, iterable_keys, self.result, 0)

    class Iterator:
        def __init__(
            self,
            table: Dict[Any, Any],
            iterable_keys: List[Any],
            result: Dict[Any, Any],
            idx: int,
        ):
            self.generator = self._generate_next(table, iterable_keys, result, idx)

        def __iter__(self):
            return self

        def _generate_next(
            self,
            table: Dict[Any, Any],
            iterable_keys: List[Any],
            result: Dict[Any, Any],
            idx: int,
        ):
            if table:
                if not iterable_keys:
                    yield table
                else:
                    for val in table[iterable_keys[0]]:
                        result[iterable_keys[0]] = val
                        if len(iterable_keys) == 1:
                            yield result
                        else:
                            yield from self._generate_next(
                                table, iterable_keys[1:], result, idx + 1
                            )
            else:
                yield table

        def __next__(self):
            return next(self.generator)


iterable_types: Set[Any] = {range, IterableList, ListProduct, TableProduct}
