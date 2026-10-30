"""切句口径：句末标点 + 各种空格/不可见分隔符。"""

import unittest

import config
import sentence_split


class SentenceSplitTests(unittest.TestCase):
    def test_whitespace_variants_split_sentences(self):
        text = "第一段内容足够长\u3000第二段内容也够长\u00a0第三段内容还是长"
        sentences = sentence_split.split_sentences(text, lower_bound=1)
        self.assertEqual(sentences, ["第一段内容足够长", "第二段内容也够长", "第三段内容还是长"])

    def test_zero_width_and_bom_split(self):
        text = "前半句足够长\u200b后半句也够长\ufeff最后一段仍够长"
        sentences = sentence_split.split_sentences(text, lower_bound=1)
        self.assertEqual(sentences, ["前半句足够长", "后半句也够长", "最后一段仍够长"])

    def test_sentence_punctuation_still_splits(self):
        sentences = sentence_split.split_sentences("第一句足够长。第二句也足够长！", lower_bound=1)
        self.assertEqual(sentences, ["第一句足够长。", "第二句也足够长！"])

    def test_length_and_vocab_filters_apply_after_strip(self):
        # 首尾空白不应该把 7 个字的句子"撑"到 8 字下界
        self.assertEqual(sentence_split.split_sentences("  九个字的一句话  ", lower_bound=8, upper_bound=100), [])
        self.assertEqual(
            sentence_split.split_sentences("  九个字的一句话  ", lower_bound=7, upper_bound=100),
            ["九个字的一句话"],
        )
        self.assertEqual(sentence_split.split_sentences("包含方位词上的句子", vocabs=["上"], lower_bound=1), ["包含方位词上的句子"])
        self.assertEqual(sentence_split.split_sentences("不含目标词的句子", vocabs=["上"], lower_bound=1), [])

    def test_default_bounds_come_from_config(self):
        article = "短。" + "长" * config.SENTENCE_LEN_UPPER_BOUND + "。"
        sentences = sentence_split.split_sentences(article)
        self.assertEqual(sentences, [])


if __name__ == "__main__":
    unittest.main()
