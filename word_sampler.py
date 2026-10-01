# encoding: utf8

"""按「被提取出来的词」分层抽样语料：每个方位词/趋向动词各抽一部分，落一份小语料回 results/。

要解决的问题：一份语料（一行一个句子 + 它命中了哪些筛选规则 + 命中词在依存树里的位置）
动辄上百万行，没法整体送人工或模型逐条看。这里按「词」分层各抽 --ratio 条，得到一份能通读的样本。

关键：目标词要**按词表在 match_tree 里递归找**，不能取 match_tree[0]。原因是趋向动词语料里
有一半以上的规则，其词表命中的是「搭配动词」而不是趋向动词：

    filter_methods = [root, direction_verb_as_center]          → 根节点就是趋向动词（上/进/出来）
    filter_methods = [root, manner_verb_one_complement]        → 根节点是方式动词（凑/走），趋向动词在子节点
    filter_methods = [root, accompany_verb_two_complement]     → 根节点是伴随动词（带/送），趋向动词是两个子节点
    filter_methods = [root, locality_phrase]                   → 根节点就是方位词

实测取 match_tree[0] 会把 走/跑/带/送/凑 这类搭配动词也当成「被提取出来的词」：趋向动词语料的根节点
一共 85 种，其中 67 种不是趋向动词，涉及 59240 条记录，与「方位词/趋向动词」的口径不符。

还有两条踩过的坑，取词与抽样时都避开了：

1. match_tree 的同一个 token 会重复出现。sentenceFilters.OrAllJudgeMethod.explain 先按 index 合并
   各规则的节点，再把子节点 extend 进去，所以同一子节点会重复多次（实测最多 4 次）、顺序还不稳定。
   → 按 index 去重，不能按位置取「最后一个子节点」。
2. 根节点未必是搭配动词。direction_verb_as_center 与 *_one_complement 同时命中的记录里，
   配置顺序让根节点是趋向动词、搭配动词退到 node[1]。→ 任何位置规则都会取错，只按词表判断才对。

一条记录里出现多个目标词时**重复计入每个词**（落成多行），所以输出行数多于语料的命中行数。

内存：本模块自己的数据结构只占几十 MB（两遍流式扫，不把语料读进内存）。但进程整体常驻约 0.8GB ——
`from sentenceFilters import ...` 会连带 `sentenceTable` 里的 `from ltp import LTP` / `import torch`，
在跑之前就吃掉了（实测裸解释器 11MB、import torch 后 639MB、import ltp 后 774MB）。这条依赖是为了
复用 FilterGenerator 展开 `_vocab_file`（词表只有一处真源），不好省；只是别把这里的「O(1) 内存」
当成整个进程的内存。启动也多花约 3.5 秒。

用法：
    .venv/bin/python word_sampler.py                                  # 两份语料各抽 10%
    .venv/bin/python word_sampler.py --ratio 0.05 --seed 0            # 换比例，固定种子
    .venv/bin/python word_sampler.py --corpus results/direction_corpus.jsonl
    .venv/bin/python word_sampler.py --dry-run                        # 只统计逐词计数，不落盘
    .venv/bin/python word_sampler.py --min-per-word 1                 # 长尾词至少抽一条
    .venv/bin/python word_sampler.py --corpus results/direction_corpus.jsonl \
        --vocab /path/to/words.json5                                  # 换一份目标词表（一次只跑一份语料）
"""

import argparse
import json
import random
from pathlib import Path

from sentenceFilters import FilterGenerator, load_vocab_file

_PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_OUT_DIR = _PROJECT_ROOT / "results"
"""输出目录，与语料同目录"""

DEFAULT_RATIO = 0.01
"""每个词默认抽取的比例，向下取整"""

DEFAULT_SEED = 0
"""默认随机种子，固定住以便复现"""

CORPUS_SPECS: dict[str, tuple[str, tuple[str, ...] | None]] = {
    # 语料文件名 -> (规则配置, 只取哪些键的词表；None 表示取各顶层规则自己的词表)
    #
    # 方位词语料的三条规则各自用 _vocab_file 指向 localization_vocab.json5，取顶层规则自己的词表即可。
    # 趋向动词语料则要挑出 direction_verb_ 前缀的那些键：accompany_/cause_/manner_ 三类顶层规则的
    # 词表是搭配动词（搬/带/插/凑…），混进来就会把搭配动词当成目标词。
    "direction_corpus.jsonl": ("config_files/direction_expressions.json5", ("direction_verb_",)),
    "location_corpus.jsonl": ("config_files/localization_expressions.json5", None),
}
"""已知语料与其目标词表的来源。命中不了时要在命令行显式给 --vocab，不做猜测。"""


# --------------------------------------------------------------------------
# 一、目标词表：从筛选配置推导
# --------------------------------------------------------------------------


def _selects(key: str, depth: int, prefixes: tuple[str, ...] | None) -> bool:
    """这个键下的词表算不算目标词表

    prefixes 为 None 时只认顶层规则（depth == 1）自己的词表；否则认任意层级上键名带该前缀的节点。
    后者是为了取 direction_verb_as_first/second/single_complement 这几个挂在 _link 底下的子条件词表。
    """
    return depth == 1 if prefixes is None else key.startswith(prefixes)


def _walk_vocab(node: object, key: str, depth: int, prefixes: tuple[str, ...] | None, found: dict[str, set[str]]) -> None:
    """递归收集配置里符合 _selects 的节点的 _vocab

    注意要一路走进 _link：子条件的词表就挂在它下面，不能因为它是下划线开头的保留字就跳过。
    _vocab/_pos/_syntax/_semantic 是叶子（字符串或列表），不会匹配到 _selects 的键名判断。
    """
    if not isinstance(node, dict):
        return
    words = node.get("_vocab")
    if isinstance(words, list) and _selects(key, depth, prefixes):
        found.setdefault(key, set()).update(w for w in words if isinstance(w, str))
    for child_key, child in node.items():
        _walk_vocab(child, child_key, depth + 1, prefixes, found)


def derive_vocab(expressions: str | Path, prefixes: tuple[str, ...] | None) -> tuple[set[str], list[str]]:
    """从筛选配置推导目标词表

    走 FilterGenerator.load_config() 而不是自己再读一遍配置：它会把 _vocab_file 就地展开成 _vocab，
    这样词表只有一处真源（见 sentenceFilters.FilterGenerator._resolve_vocab_files）。

    刻意不用 FilterGenerator.collect_screen_vocab()：那是**初筛**用的超集，会做子串最小化，
    把 出来/上来/过去 这些按子串吞成 出/上/过，剩下 8 个单音节词，实测会漏掉约 1.3 万条双音节记录。

    Returns:
        tuple[set[str], list[str]]: (目标词表, 词表来自哪些键)。词表为空时抛错列出实际有哪些键，
            因为多半是配置改名了，报错要足以让人当场改对 CORPUS_SPECS。
    """
    config_data = FilterGenerator(str(expressions)).load_config()
    found: dict[str, set[str]] = {}
    _walk_vocab(config_data, "root", 0, prefixes, found)

    vocab = set().union(*found.values()) if found else set()
    if not vocab:
        rule_keys = [k for k in config_data if not k.startswith("_")]
        raise ValueError(
            f"从 {expressions} 里推不出目标词表（取词表的前缀：{prefixes}）；"
            f"该配置的顶层规则有：{'、'.join(rule_keys) or '（无）'}"
        )
    return vocab, sorted(found)


# --------------------------------------------------------------------------
# 二、取词与统计
# --------------------------------------------------------------------------


def target_nodes(record: dict, vocab: set[str]) -> list[dict]:
    """一条记录里被提取出来的目标词节点，按 index 升序返回

    递归 walk match_tree 的每个节点与其 children，收 word 落在词表里的节点。子节点也要看：
    *_complement 规则的搭配动词在根上、趋向动词在子节点里（见模块 docstring）。

    返回的是节点（[{"index": i, "word": w}, ...]）而不是词，因为同一个词可能出现在多个
    index 上，调用方（spatial_tools.pick_focus_word）要拿这个 index 去定位句子里的词；
    只关心有哪些词时用 target_words。

    两条约束：

    1. **父节点已命中词表时，不再认这个子节点。** 分界点不是「子节点在不在词表里」，而是
       「这个子节点是规则要提取的目标词，还是规则用来满足结构约束的上下文词」：
       - 父节点是搭配动词（不在词表里）→ 子节点是 *_complement 规则的目标词，要收（凑 → 出来）。
       - 父节点自己就在词表里 → 子节点只是被命中的那个词的结构上下文，不收：如 direction_verb_as_center
         的 `_link.has_locative_complement`（._semantic == "LOC"）会把另一个趋向动词当成处所补语收进来
         （过 → 下去、上 → 下、出 → 下），locality_phrase 的 `_link.has_nominal_modifier` 会把
         方位词模样的定语收进来（旁边 → 边、中 → 里、下巴 → 下）。
       这些记录都不是为子节点那个词抽出来的，收进来会让该词的样本混进「从未按它筛出」的句子。
       实测边界影响很小：趋同动词语料丢 126 处（118 条记录，0.08%），方位词语料丢 22 处（20 条记录，
       0.001%）；其中少数（上进下出、不上不下）确实是句子里另一个独立的趋向动词，但它本身没被规则的
       词表命中，按「被提取出来的词」的口径不入选——这是刻意的保守选择。
       已验证：两份语料都没有记录因此变成「一个目标词都取不到」（0 条）。
       注意不能反过来只看顶层节点：那会把 *_complement 规则的目标词全丢掉。
    2. **按 index 去重**：OrAllJudgeMethod 会把同一个 token 的子节点重复拼进来（实测最多 4 次）、
       顺序还不稳，同一个 index 收敛成一项，顺带把「同一 token 被多条规则命中」也算一次。

    这里不 import spatial_tools.pick_word_info：那个函数取的是 match_tree[0]，语义是「这条记录被
    哪条筛选规则命中、待改写的词是哪个」，服务于改写环节；本模块要的是「哪个词是方位词/趋向动词」，
    两者在 *_complement 规则上不是同一个词（详见模块 docstring）。
    """
    by_index: dict[int, str] = {}
    stack: list[tuple[object, object | None]] = [(node, None) for node in record.get("match_tree") or []]
    while stack:
        node, parent = stack.pop()
        if not isinstance(node, dict):
            continue
        index = node.get("index")
        word = node.get("word")
        parent_word = parent.get("word") if isinstance(parent, dict) else None
        if isinstance(index, int) and word in vocab and parent_word not in vocab:
            by_index.setdefault(index, word)
        stack.extend((child, node) for child in node.get("children") or [])
    return [{"index": index, "word": word} for index, word in sorted(by_index.items())]


def target_words(record: dict, vocab: set[str]) -> list[str]:
    """一条记录里被提取出来的所有目标词，去重后按字典序返回

    取词口径全在 target_nodes 里（含上面那两条约束），这里只是把节点收成词。

    **按词还要再去重一次**（target_nodes 只按 index 去重，同一个词可以在两个 index 上各留一项）：
    不去重的话 scan_by_word 会把同一行 append 两次，同一个词重复计数、还可能被重复抽样。
    """
    return sorted({node["word"] for node in target_nodes(record, vocab)})


def pre_head_indices(record: dict) -> set[int]:
    """match_tree 里「排在父节点之前」的子节点下标

    与 target_nodes 是两个不同的问题：target_nodes 问「哪些节点是词表里的目标词」，
    这里问「这个节点在树里排在它的父节点前还是后」。服务于 spatial_tools.pick_focus_word
    的候选排序（同一句话里有多个目标词时该取哪个）。

    *_complement 规则把搭配动词当根、把它满足的各个 link 条件当子节点，而其中的方向补语在
    汉语里一律跟在动词之后；被 has_locative_complement 收进来的另一个词却可以在动词之前。
    实测「…再往上撒上适量的脆麦片…」（cause_verb_one_complement）的树是

        {"index": 16, "word": "撒", "children": [{"index": 15, "word": "上"}, {"index": 17, "word": "上"}]}

    排在动词前的 上@15 是介词「往」的宾语，真正要抽的 direction_verb_as_single_complement
    是排在动词后的 上@17：只按 index 最小取词会取到前者。

    同一个下标两种位置都出现过时**不算**（OrAllJudgeMethod 会把同一个 token 的子节点重复拼进来，
    位置可能不一致——见 target_nodes 的第 2 条约束），取保守的那一侧，结论与遍历顺序无关。
    """
    before: set[int] = set()
    after: set[int] = set()
    stack: list[tuple[object, object | None]] = [(node, None) for node in record.get("match_tree") or []]
    while stack:
        node, parent = stack.pop()
        if not isinstance(node, dict):
            continue
        index = node.get("index")
        parent_index = parent.get("index") if isinstance(parent, dict) else None
        if isinstance(index, int) and isinstance(parent_index, int):
            (before if index < parent_index else after).add(index)
        stack.extend((child, node) for child in node.get("children") or [])
    return before - after


def scan_by_word(corpus: Path, vocab: set[str]) -> tuple[dict[str, list[int]], dict[str, int]]:
    """第一遍：流式扫语料，把行号按词归档

    只留 dict[词, [行号]]（方位词语料展开后约 176 万个 int，几十 MB），不把语料读进内存——
    单条记录带 60 来个 token 的 sentence 列表，全读进来要上 GB。

    Returns:
        tuple[dict[str, list[int]], dict[str, int]]: (词 -> 行号列表，各计数)
    """
    lines_by_word: dict[str, list[int]] = {}
    counts = {"total": 0, "blank": 0, "no_word": 0, "multi_word": 0}
    with open(corpus, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                counts["blank"] += 1
                continue
            counts["total"] += 1
            words = target_words(json.loads(line), vocab)
            if not words:
                counts["no_word"] += 1
                continue
            if len(words) > 1:
                counts["multi_word"] += 1
            for word in words:
                lines_by_word.setdefault(word, []).append(line_no)
    return lines_by_word, counts


def sample_lines(
    lines_by_word: dict[str, list[int]],
    ratio: float,
    seed: int,
    min_per_word: int,
) -> tuple[dict[int, list[str]], dict[str, int]]:
    """按词抽样，返回 (行号 -> 该行被抽中的词列表, 词 -> 抽中条数)

    每个词用一个从 --seed 派生的独立随机种子（Random("<seed>:<词>")）：
    共用一个 RNG 的话，某个词抽到哪些行取决于词的遍历顺序和别的词的规模，以后往词表里加一个词
    就会把**所有**词已有的样本重新洗一遍，历史标注就对不上了。按词派生后加词只影响新词本身。

    样本量取 max(--min-per-word, int(条数 * ratio))，int 向下取整即「不超过设定比例」；
    --min-per-word 是给长尾词兜底用的。-词抽 0 条不静默吞掉，会写进 summary 的 words_with_zero_samples。
    """
    chosen: dict[int, list[str]] = {}
    sampled_by_word: dict[str, int] = {}
    for word in sorted(lines_by_word):
        line_nos = lines_by_word[word]
        k = min(len(line_nos), max(min_per_word, int(len(line_nos) * ratio)))
        sampled_by_word[word] = k
        for line_no in sorted(random.Random(f"{seed}:{word}").sample(line_nos, k)):
            chosen.setdefault(line_no, []).append(word)
    return chosen, sampled_by_word


# --------------------------------------------------------------------------
# 三、落盘
# --------------------------------------------------------------------------


def write_sample(corpus: Path, chosen: dict[int, list[str]], out_path: Path) -> int:
    """第二遍：流式扫语料，把抽中的行原样写出并补上 sampled_word / line_no

    按语料原顺序输出（同一行命中多个词就按字典序连着写多条），只重放一次文件、O(1) 内存。
    不按词分组的写法虽然更好读，但要把十几万条记录全缓存下来，代价不值。

    这里没有调 spatial_tools.write_jsonl：它一次收一份完整的记录列表，要先把十几万条读进内存，
    正是本模块要避开的；spatial_tools.append_jsonl 每条都开关一次文件，十几万次也受不了。
    落盘写法（json.dumps(..., ensure_ascii=False) + "\\n"、encoding="utf-8"）与那两个函数一致。
    """
    written = 0
    with open(corpus, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for line_no, line in enumerate(f, start=1):
            words = chosen.get(line_no)
            if not words:
                continue
            record = json.loads(line)
            for word in sorted(words):
                out.write(json.dumps(record | {"sampled_word": word, "line_no": line_no}, ensure_ascii=False) + "\n")
                written += 1
    return written


def out_paths(corpus: Path, out_dir: Path) -> tuple[Path, Path]:
    """输出文件与汇总文件路径：sampled_<语料名>.jsonl 与同名 .summary.json"""
    out_file = out_dir / f"sampled_{corpus.name}"
    return out_file, out_dir / f"sampled_{corpus.stem}.summary.json"


def write_summary(path: Path, summary: dict) -> None:
    """汇总落盘，与 spatial_runner.ResultWriter.finish 同一写法"""
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# 四、命令行入口
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按「被提取出来的词」（方位词/趋向动词）分层抽样语料",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法：")[-1],
    )
    parser.add_argument(
        "--corpus",
        action="append",
        default=None,
        help=f"输入语料 jsonl 路径，可重复；默认 {' 与 '.join(str(DEFAULT_OUT_DIR / n) for n in CORPUS_SPECS)}",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_RATIO,
        help=f"每个词抽取的比例，取值 (0, 1]，默认 {DEFAULT_RATIO}（下取整）",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"随机种子，默认 {DEFAULT_SEED}")
    parser.add_argument("--min-per-word", type=int, default=0, help="每个词至少抽几条，给长尾词兜底，默认 0（不得为负）")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录，默认 results/")
    parser.add_argument(
        "--vocab",
        default=None,
        help="覆盖由语料推导出的目标词表（json5 词条数组路径）。只对本次的**一份**语料生效，"
        "因此与多份 --corpus 同时给出时会直接报错，避免把同一份词表套到别的语料上",
    )
    parser.add_argument("--dry-run", action="store_true", help="只统计并打印逐词计数，不落盘")
    args = parser.parse_args(argv)
    if not 0 < args.ratio <= 1:
        parser.error(f"--ratio 必须在 (0, 1] 之间，收到 {args.ratio}")
    if args.min_per_word < 0:
        parser.error(f"--min-per-word 不能为负，收到 {args.min_per_word}")
    if args.vocab and len(_corpora(args)) > 1:
        parser.error("--vocab 一次只能配一份语料（否则会把同一份词表套到别的语料上），请配合 --corpus 只跑一份")
    return args


def _corpora(args: argparse.Namespace) -> list[Path]:
    """本次要处理的语料：--corpus 给了就按给的来，否则默认两份都跑"""
    return [Path(p) for p in (args.corpus or [DEFAULT_OUT_DIR / name for name in CORPUS_SPECS])]


def vocab_for(corpus: Path, override: str | None) -> tuple[set[str], str]:
    """这份语料的目标词表与它的来源说明

    --vocab 的优先级最高：给了就完全按它来，连语料是否在 CORPUS_SPECS 里都不必答对。
    否则按文件名查表，查不到就报错——宁可当场报，也不要静默抽出一份错口径的样本。
    """
    if override:
        return load_vocab_file(override), str(override)
    spec = CORPUS_SPECS.get(corpus.name)
    if spec is None:
        raise SystemExit(
            f"不知道 {corpus.name} 该用哪份筛选配置取词表。"
            f"已登记：{'、'.join(CORPUS_SPECS)}；"
            "其他语料请用 --vocab 直接指定目标词表。"
        )
    expressions, prefixes = spec
    vocab, sources = derive_vocab(_PROJECT_ROOT / expressions, prefixes)
    return vocab, f"{expressions} 的 {'/'.join(sources)}"


def run_corpus(corpus: Path, args: argparse.Namespace) -> dict:
    """处理一份语料：扫一遍统计 + 抽样 + （非 dry-run 时）再扫一遍落盘"""
    vocab, vocab_source = vocab_for(corpus, args.vocab)
    lines_by_word, counts = scan_by_word(corpus, vocab)
    chosen, sampled_by_word = sample_lines(lines_by_word, args.ratio, args.seed, args.min_per_word)

    expanded = sum(len(nos) for nos in lines_by_word.values())
    sampled = sum(sampled_by_word.values())
    print(f"\n输入语料：{corpus}（{counts['total']} 条）")
    print(f"目标词表：{vocab_source}（{len(vocab)} 个词）")
    print(
        f"命中情况：{expanded} 条（展开后，含 {counts['multi_word']} 条含多个目标词）"
        f"，取不到目标词的 {counts['no_word']} 条"
    )
    print(f"抽样（ratio={args.ratio} seed={args.seed}）：{sampled} 条")
    for word in sorted(lines_by_word, key=lambda w: (-len(lines_by_word[w]), w)):
        print(f"  {word:<8}{len(lines_by_word[word]):>9} 条 → 抽 {sampled_by_word[word]:>7} 条")
    if counts["no_word"]:
        # 词表推错（漏了词、或规则改名导致取词表的前缀不再命中）时，表现是「这些记录一条也配不上词」，
        # 而不是报错。这里必须喊出来：静默少抽一批记录，看起来和「本来就只有这么多」一模一样。
        print(
            f"警告：{counts['no_word']} 条记录在 match_tree 里找不到目标词（占 {counts['no_word'] / counts['total']:.2%}）。"
            "多半是目标词表推得不全——请核对 CORPUS_SPECS 的取词表前缀，或用 --vocab 显式给词表。"
        )

    summary = {
        "corpus": str(corpus),
        "vocab_source": vocab_source,
        "vocab_size": len(vocab),
        "seed": args.seed,
        "ratio": args.ratio,
        "min_per_word": args.min_per_word,
        "total_records": counts["total"],
        "blank_lines": counts["blank"],
        "records_without_target_word": counts["no_word"],
        "records_with_multiple_words": counts["multi_word"],
        "distinct_words": len(lines_by_word),
        "expanded_records": expanded,
        "sampled_records": sampled,
        "words": {w: {"total": len(nos), "sampled": sampled_by_word[w]} for w, nos in sorted(lines_by_word.items())},
        "words_with_zero_samples": sorted(w for w, k in sampled_by_word.items() if k == 0),
        # 词表里有、但一条记录都没配上的词。这类词在 words 里根本不出现，只看 dict
        # 会以为是「这个词本来就没什么材料」；单独列出来才能区分「词表收错了」和「长尾」。
        "vocab_words_without_any_record": sorted(vocab - set(lines_by_word)),
    }

    # 两条「哪些词没抽到/没配上」的提示放在提前返回之前：--dry-run 的用途正是体检词表与抽样口径，
    # 这时候看不到提示就白跑了。
    if summary["words_with_zero_samples"]:
        print(f"注意：以下词一条都没抽到（可调大 --ratio 或用 --min-per-word）：{'、'.join(summary['words_with_zero_samples'])}")
    if summary["vocab_words_without_any_record"]:
        print(
            f"注意：目标词表里有 {len(summary['vocab_words_without_any_record'])} 个词在本语料里一条记录都没配上："
            f"{'、'.join(summary['vocab_words_without_any_record'])}（可能是词表收宽了，也可能这批语料确实没有）"
        )

    if args.dry_run:
        print("（--dry-run：不落盘）")
        return summary

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file, summary_file = out_paths(corpus, out_dir)
    if out_file.exists():
        # 文件名是固定的（种子固定才谈得上复现），所以重跑必然覆盖；说一句免得以为在看上一版结果
        print(f"覆盖已有文件：{out_file}")
    written = write_sample(corpus, chosen, out_file)
    summary["out_file"] = str(out_file)
    summary["summary_file"] = str(summary_file)
    write_summary(summary_file, summary)
    print(f"输出文件：{out_file}（{written} 条）")
    print(f"汇总文件：{summary_file}")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    corpora = _corpora(args)
    for corpus in corpora:
        if not corpus.is_file():
            raise SystemExit(f"找不到语料文件 {corpus}")
    for corpus in corpora:
        run_corpus(corpus, args)


if __name__ == "__main__":
    main()
