"""有界预取队列：顺序、异常传播与停止。"""

import threading
import unittest

from pipeline_utils import ROW_ID_FIELD, prefetch_iter, prune_tree, tag_rows


class TagRowsTests(unittest.TestCase):
    def test_ids_are_unique_and_source_is_not_mutated(self):
        source = [{"sentence": "甲"}, {"sentence": "乙"}]
        rows = list(tag_rows(source))
        self.assertEqual([row[ROW_ID_FIELD] for row in rows], [0, 1])
        self.assertNotIn(ROW_ID_FIELD, source[0])
        self.assertNotIn(ROW_ID_FIELD, source[1])


class PruneTreeTests(unittest.TestCase):
    def test_false_parent_keeps_fail_open_descendant(self):
        tree = [
            {
                "index": 1,
                "word": "前",
                "children": [{"index": 2, "word": "上", "children": []}],
            }
        ]
        kept = prune_tree(tree, {1}, {2})
        self.assertEqual([node["index"] for node in kept], [2])

    def test_false_leaf_is_removed_and_valid_sibling_kept(self):
        tree = [
            {"index": 1, "word": "前", "children": []},
            {"index": 2, "word": "上", "children": []},
        ]
        kept = prune_tree(tree, {1}, {2})
        self.assertEqual([node["index"] for node in kept], [2])


class PrefetchTests(unittest.TestCase):
    def test_order_and_all_items(self):
        self.assertEqual(list(prefetch_iter(range(50), maxsize=3)), list(range(50)))

    def test_producer_exception_is_raised_in_consumer(self):
        def source():
            yield 1
            raise RuntimeError("抽取失败")

        with self.assertRaisesRegex(RuntimeError, "抽取失败"):
            list(prefetch_iter(source(), maxsize=2))

    def test_consumer_stop_does_not_leave_thread_blocked(self):
        release = threading.Event()

        def source():
            try:
                index = 0
                while True:
                    yield index
                    index += 1
            finally:
                release.set()

        iterator = prefetch_iter(source(), maxsize=1)
        self.assertEqual(next(iterator), 0)
        iterator.close()
        self.assertTrue(release.wait(timeout=2), "生产者线程没有收到停止信号")


if __name__ == "__main__":
    unittest.main()
