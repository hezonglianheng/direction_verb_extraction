# encoding: utf8

"""空间语义实验的跑批入口：读语料 → 建图 → 并发跑 → 落盘 → 汇总。

三层分工：
    spatial_agents.py   六个 agent：输入什么、产出什么（不 import langgraph）
    spatial_graphs.py   状态、节点适配器、各图 builder、图注册表
    spatial_runner.py   本文件：驱动、并发、落盘、命令行

pair 图（默认，--graph pair）的数据流：

    START ──► alter ──改写未命中──► save ──► END
                │改写成功
                ▼
             reality ──不是现实空间用法──► save ──► END
                │是（或没判出来）
                ▼
             semantic ──任一句不通顺──► save ──► END
                │两句都通顺
                ▼
             feature ──► schema ──► judge ──► save ──► END

三张图**都在各自的入口节点（alter / read）之后、LLM agent 之前**插了一个 reality 闸门：
只看关注词在这一句里是不是用来说现实世界中的空间场景，判 false 就经 save 落盘后结束
（判不好的记录也留着，好回头查是被哪条判据挡下的）；判不出来时一律放行，理由见
spatial_graphs.reality_gate —— 闸门的错法只能是漏掉几句引申用法，不能是把整批记录挡在门外。

semantic / feature / schema 三个节点都是**原句与改写句各跑一遍**（两侧各一份结果，落盘时记在 a/b
两个键下），judge 是两侧合并的地方：它一份记录只跑一次，同时读两侧的 schema 与关注词。
两侧**各看各的关注词**——原句那侧看语料记录里那个词（focus_word），改写句那侧看规则表换上去的
那个词（focus_word_b，由 alter 节点算出）。所以 b 侧的 schema 收窄到的是**替换词**所属的词组，
判定表里 上12 × 里1 这样按两组名字写的格子才对得上；两侧共用一个词的话 b 侧会收窄回原词那一组，
判定就退化成同组自比了（见 spatial_tools.replacement_focus_word）。两处的下标都落在原句的
分词结果上（b 侧是照搬 a 侧的），消息里对 b 侧的措辞已交代不要拿它去数改写句的分词行
（改写句要重新分词，替换点可能并词，见 spatial_agents.FOCUS_INDEX_NOTE_ALTER）。

一条语料里的方位词往往能换成同义词组里的好几个词（见 spatial_tools.alter_sentence），
所以跑批时会先把它展开成多个待选句、每个待选句一个状态：**一条语料通常产出多条记录**，
记录里的 candidate 块（index/total/word/replacement/group）标明这是第几个待选句。
--limit / --offset 仍按语料条数算，不按展开后的状态数算。

每跑一次产出一个 jsonl，一条记录就是一条完整链路的结果（两个句子、各自的语义特征、
各自对应的 schema、两侧各自的关注词、判定信息，外加逐节点的上下文历史），
另在同目录写一份 .summary.json 汇总。
记录在 save 节点流式落盘，长跑中途也能 tail 到进度；并发跑时输出顺序不等于输入顺序，
按记录里的 line_no（同一句再按 candidate.index）可还原。

用法：
    .venv/bin/python spatial_runner.py --limit 5                        # 跑前 5 条（pair 图，方位词）
    .venv/bin/python spatial_runner.py --limit 5 --no-llm               # 只跑纯规则与查表，不花钱
    .venv/bin/python spatial_runner.py --workers 4 --limit 200          # 并发跑
    .venv/bin/python spatial_runner.py --graph single --limit 5         # 换一张图
    .venv/bin/python spatial_runner.py --list-graphs                    # 有哪些图可用
    .venv/bin/python spatial_runner.py --corpus-type 趋向动词 --limit 20   # 换语料类型（别名 direction 也行）
    .venv/bin/python spatial_runner.py --corpus results/direction_corpus.jsonl --corpus-type 趋向动词 --limit 20
    .venv/bin/python spatial_runner.py --corpus-type 趋向动词 --alter-rules <路径> --limit 5   # 换改写的表

四张配置表（改写规则 / 语义特征类别 / schema / 判定）**按语料类型分两组**存放在
config_files/localization/（方位词）与 config_files/direction/（趋向动词）下，文件名相同；
--corpus-type 决定读哪一组（默认方位词），启动时逐张打印实际路径与内容摘要，缺表/空表当场告警。
同两个目录下还各有一份 reality 闸门的提示语（reality.txt）：闸门的判据随词类不同（方位词看的是
「说的是一处位置还是虚化用法」，趋向动词看的是「说的是位移还是时体/结果」），所以它跟着配置表走，
不放在 system_prompt/ 下，也由 --corpus-type 一并选中。
--corpus-type 不给时**不按语料文件名推断**（理由见该选项的 help）：语料名像另一类时会额外提醒一句。
"""

import argparse
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config
import spatial_agents as sa
import spatial_graphs as sg
import spatial_tools as st

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = _PROJECT_ROOT / "results" / st.CORPUS_TYPE_CORPORA[st.DEFAULT_CORPUS_TYPE]
"""默认语料：默认语料类型（方位词）的那一份。按类型取用见 st.CORPUS_TYPE_CORPORA"""
DEFAULT_MODEL = sa.DEFAULT_MODEL
"""默认模型配置名（model_config/api_keys.json 里的键）"""


# --------------------------------------------------------------------------
# 一、结果落盘
# --------------------------------------------------------------------------


class ResultWriter:
    """结果落盘器：加锁追加 jsonl，并按各图自己的口径统计计数。

    多线程跑图时 save 节点会并发调用，故用锁串起"写文件 + 计数"；
    每条写完即关文件，长跑中途 tail 也能看到完整行。

    统计口径由图决定（spatial_graphs.GraphSpec.counter_names / tally）：
    pair 图统计两侧的 schema 来源，单句图只有一句——写死成 pair 的形状会让单句图
    静默统计出一堆 "none"。
    """

    def __init__(
        self,
        path: str | Path,
        *,
        counter_names: tuple[str, ...],
        tally,
        corpus_type: str | None = None,
        config_tables: dict[str, str] | None = None,
    ):
        self.corpus_type = corpus_type
        """本次跑批的语料类型；只写进汇总，不进记录（记录形状不随本次改动变）"""
        self.config_tables = dict(config_tables or {})
        """本次实际读的四张配置表：{逻辑表名: 路径}，同样只进汇总（--alter-rules 换过表时很关键）"""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.count = 0
        self.counter_names = tuple(counter_names)
        self.counters: dict[str, Counter] = {name: Counter() for name in self.counter_names}
        self.tally = tally
        self.started_at = time.time()

    def write(self, record: dict) -> None:
        with self.lock:
            st.append_jsonl(record, self.path)
            self.count += 1
            self.tally(record, self.counters)
            if self.count % 10 == 0:
                print(f"  已写出 {self.count} 条", flush=True)

    def finish(self) -> dict:
        # corpus_type / config_tables 是这次跑批「读的是哪一组表」的唯一凭据：user message
        # 不落盘（记录里只有 node 与耗时），事后要复现或核对口径就得靠这两项
        provenance = {"corpus_type": self.corpus_type} if self.corpus_type else {}
        if self.config_tables:
            provenance["config_tables"] = self.config_tables
        summary = {
            "output_file": str(self.path),
            **provenance,
            "count": self.count,
            "elapsed_seconds": round(time.time() - self.started_at, 1),
            **{name: dict(counter) for name, counter in self.counters.items()},
        }
        summary_path = self.path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["summary_file"] = str(summary_path)
        return summary


# --------------------------------------------------------------------------
# 二、命令行入口
# --------------------------------------------------------------------------


def _corpus_type_arg(value: str) -> str:
    """argparse 的 type：把类型规范化，认不出时报参数错（退出码 2）

    不用 type=st.normalize_corpus_type 直接挂上去：argparse 只会把异常类型名印出来，
    ValueError 的正文会丢掉，而这条消息正是要告诉人「有哪些类型可用」。
    """
    try:
        return st.normalize_corpus_type(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="空间语义实验跑批：按指定的图把语料跑一遍，结果写入 jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法：")[-1],
    )
    parser.add_argument("--graph", default="pair", choices=sorted(sg.GRAPHS), help="跑哪张图（见 --list-graphs）")
    parser.add_argument("--list-graphs", action="store_true", help="打印可用的图与各自需要的 agent，然后退出")
    parser.add_argument(
        "--corpus-type",
        default=None,
        metavar="类型",
        type=_corpus_type_arg,
        help="语料类型：方位词（别名 localization/location/loc）或 趋向动词（别名 direction/dir/direction_verb）。"
        "决定读 config_files/<类型目录>/ 下的四张表与 reality 闸门的提示语（启动时逐张打印实际路径与内容摘要）。"
        "默认方位词；**刻意不按语料文件名推断**——推断猜错时是静默读错一整组表，"
        "宁可要你显式说一句（语料名像另一类时会提醒一句，但不会替你改）。"
        "--alter-rules 显式给路径时，改写那张表以它为准（类型只管另外三张）",
    )
    parser.add_argument("--corpus", default=None, help="输入语料 jsonl 路径（默认按语料类型取 results/ 下对应那份）")
    parser.add_argument(
        "--alter-rules",
        default=None,
        help="替换规则表路径（默认读本语料类型目录下的 alter_rules.json5，见 --corpus-type）："
        "换一张表对照实验用，纯规则环节，--no-llm 下同样生效。"
        "指定的文件不存在时按空表处理（等价于不改写），启动时会告警",
    )
    parser.add_argument("--out-dir", default=None, help="输出目录（默认 results/<图的 prefix>）")
    parser.add_argument("--out-file", default=None, help="直接指定输出文件（默认按时间戳生成）")
    parser.add_argument("--limit", type=int, default=100, help="最多处理多少条记录")
    parser.add_argument("--offset", type=int, default=0, help="跳过前多少条记录")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model_config/api_keys.json 里的配置名")
    parser.add_argument("--workers", type=int, default=1, help="并发处理的记录数（LLM 调用是 IO 密集）")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="不调任何模型：只跑纯规则改写、查表与落盘，用于验证规则表与接线",
    )
    return parser.parse_args(argv)


def _print_graphs() -> None:
    print("可用的图（--graph <名字>）：")
    for spec in sg.GRAPHS.values():
        print(f"  {spec.name:<12}需要 agent：{'、'.join(spec.needs)}")
        print(f"  {' ' * 12}{spec.doc}")
        print(f"  {' ' * 12}产出：results/{spec.prefix}/")


def _warn_corpus_type_mismatch(corpus_path: Path, corpus_type: str) -> None:
    """语料文件名像**另一类**语料时提醒一句（只提醒，不改类型——类型只由 --corpus-type 决定）

    防的是 `--corpus results/direction_corpus.jsonl` 这类旧命令：不给类型就跑，四张表会静默
    按方位词那组读，结果看着像跑完了、口径却全错。这里只在文件名恰好命中另一类的默认语料名
    （含采样前缀 sampled_）时出声，不做任何推断——猜错时同样是静默读错一整组表。
    """
    name = corpus_path.name
    if name.startswith("sampled_"):
        name = name[len("sampled_") :]
    others = {corpus: kind for kind, corpus in st.CORPUS_TYPE_CORPORA.items() if kind != corpus_type}
    other_kind = others.get(name)
    if other_kind:
        print(
            f"警告：语料文件名 {corpus_path.name} 像是{other_kind}语料，但本次类型是{corpus_type}"
            f"（只由 --corpus-type 决定，不按文件名推断）；若确实是{other_kind}语料，"
            f"请加 --corpus-type {other_kind}"
        )


def _config_tables(agents) -> dict[str, str]:
    """本次实际读的配置表 {逻辑表名: 路径}（写进 .summary.json 的凭据）"""
    return {
        report["table"]: report["path"]
        for agent in agents.values()
        if (report := agent.config_report()) is not None
    }


def _print_config_tables(agents) -> None:
    """启动时逐张打印本次跑批读的配置表：路径 + 内容摘要；缺表/空表当场说清楚

    由 agent 自己报（见 spatial_agents.BaseAgent.config_report），不是这里按语料类型算路径：
    --alter-rules 把改写那张表换掉时，只有 agent 知道它实际读了哪张。
    图没建那个 agent 就不读那张表，因此不打印（不是漏印）。

    空表分两级：alter_rules 空 = **警告**（这条跑批跑不出东西，全部停在 alter_missed、
    后面五个 LLM agent 一次都不调）；另三张空 = **提示**（「空表走 LLM 兜底」是各表文件里写明的
    正常态，一律报警就是对正常跑批的误报）。

    表自己报出来的问题（agent 的 warnings）一律按**警告**打印，且与空不空无关：判定表里
    留空的格子就是这类——表填了一半，那些格没人裁定，跑批会照常降级交 LLM，但不该静悄悄过去。
    """
    reports = [report for agent in agents.values() if (report := agent.config_report()) is not None]
    if not reports:
        return
    print("配置表：")
    for report in reports:
        if report["missing"]:
            mark = "（文件不存在，已按空表处理）"
        else:
            mark = ""
        print(f"  {report['table']:<16}{report['path']}    {report['summary']}{mark}")
        for line in report.get("warnings") or []:
            print(f"      警告：{line}")
        if not report["empty"]:
            continue
        level = "警告" if report["table"] == "alter_rules" else "提示"
        # 说「没有已填内容」而不是「为空」：判定表可能只写了表头（标签列了一排、一格没填），
        # 那文件不空、summary 里也有标签数，印「为空」会与同一行自相矛盾
        print(f"      {level}：{report['table']} 没有已填内容 —— {report['degrade']}")
        if report["table"] == "alter_rules":
            print(f"            请把同义词组填进 {report['path']}，或用 --alter-rules <路径> 指定另一张表")


def _prefill(records: list[dict], agents, spec: sg.GraphSpec) -> None:
    """跑批前把整批句子一次性喂给 LTP

    一来省下逐句前向的开销（批量推理快约 4 倍，见 sentenceTable.py），
    二来让后续多线程不再有并发调同一份模型的窗口——推理是在锁外做的，
    所以哪些句子会被用到必须由图自己声明（spec.plan），并且预填充后要能自检。
    """
    sentences = [s for s in spec.plan(records, agents) if s]
    unique = list(dict.fromkeys(sentences))
    filled = st.prefill_ltp(unique)
    print(f"LTP 预填充：本图需 {len(unique)} 句，实际送入模型 {filled} 句（其余命中缓存）")

    missing = [s for s in unique if not st.cache_ready(s)]
    if missing:
        print(f"警告：预填充后仍有 {len(missing)} 句不在缓存里，多线程下可能并发推理")
    if len(unique) > config.CACHE_MAX_SIZE:
        print(
            f"警告：本图需 {len(unique)} 句，超过缓存容量 {config.CACHE_MAX_SIZE}，"
            "多线程下可能边填边淘汰；建议减小 --limit 或调大 config.CACHE_MAX_SIZE"
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_graphs:
        _print_graphs()
        return

    # args.corpus_type 已被 argparse 规范化（None 仍是 None），这里才落默认值：
    # 落到 None 上就分不清「没指定」与「显式指定方位词」，启动打印要说出类型是哪来的
    corpus_type = st.normalize_corpus_type(args.corpus_type)
    corpus_path = (
        Path(args.corpus)
        if args.corpus
        else _PROJECT_ROOT / "results" / st.CORPUS_TYPE_CORPORA[corpus_type]
    )
    print(f"语料类型：{corpus_type}（{'--corpus-type' if args.corpus_type else '默认'}）")
    print(f"配置目录：config_files/{st.CORPUS_TYPE_DIRS[corpus_type]}/")
    _warn_corpus_type_mismatch(corpus_path, corpus_type)

    spec = sg.get_graph(args.graph)
    records = st.read_corpus(corpus_path, limit=args.limit, offset=args.offset)
    if not records:
        print(f"没有从 {corpus_path} 读到记录")
        return

    # --alter-rules 换的是改写那张表，走的是 build_agents 的注入通道（见 spatial_agents.AlterAgent）。
    # 不给就由 AlterAgent 自己按语料类型取本组的那张（config_files/<类型目录>/alter_rules.json5）
    overrides = (
        {"alter": {"rules": st.load_alter_rules(args.alter_rules), "rules_path": args.alter_rules}}
        if args.alter_rules
        else {}
    )
    agents = sa.build_agents(
        spec.needs,
        model_key=args.model,
        llm=not args.no_llm,
        overrides=overrides,
        corpus_type=corpus_type,
    )
    for agent in agents.values():
        # 配置错在启动时就报，别留给多线程去抢着建（deep agent 尤其贵）
        agent.warm()

    out_dir = Path(args.out_dir) if args.out_dir else _PROJECT_ROOT / "results" / spec.prefix
    out_path = Path(args.out_file) if args.out_file else st.build_output_path(out_dir, prefix=spec.prefix)
    writer = ResultWriter(
        out_path,
        counter_names=spec.counter_names,
        tally=spec.tally,
        corpus_type=corpus_type,
        config_tables=_config_tables(agents),
    )
    graph = spec.build(agents, writer)

    print(f"输入语料：{corpus_path}（{len(records)} 条）")
    print(f"输出文件：{out_path}")
    print(f"图：{spec.name}（{spec.doc}）")
    print("agent：" + "；".join(agent.describe() for agent in agents.values()))
    # 逐张打印本次实际读的配置表（含 --alter-rules 换过的那张）：缺表/空表当场说清楚。
    # 「表为空 → 一条都不改写却不提示」曾是最容易误读成「代码没接上」的情形，这一块就是
    # 把「配置缺内容」与「代码有问题」分开的地方
    _print_config_tables(agents)
    # 关注词表推得出来吗：推不出来（筛选配置改名、取词表的前缀对不上）就该在这里报，
    # 而不是等每条记录都静默回落成 match_tree[0]
    print("关注词表：" + "；".join(f"{kind} {len(words)} 个" for kind, words in st.load_focus_vocab().items()))

    if not args.no_llm:
        _prefill(records, agents, spec)

    # 一条语料能改出好几个待选句，每个待选句一个状态、一条记录，所以状态数一般多于语料条数
    states = spec.init_states(records, agents, str(out_path))
    print(f"展开待选句：{len(records)} 条语料 → {len(states)} 个状态（每个状态落一条记录）")
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        # 结果已在 save 节点落盘，这里只需驱动跑完；map 按输入顺序产出，异常会在此处抛出
        for _ in pool.map(lambda state: graph.invoke(state, {"recursion_limit": 60}), states):
            pass

    summary = writer.finish()
    print(
        f"\n完成 {summary['count']} 条记录（{len(records)} 条语料），用时 {summary['elapsed_seconds']} 秒"
        f"（总耗时 {round(time.time() - started, 1)} 秒）"
    )
    for name in spec.counter_names:
        print(f"{name}：{summary[name]}")
    print(f"汇总文件：{summary['summary_file']}")


if __name__ == "__main__":
    main()
