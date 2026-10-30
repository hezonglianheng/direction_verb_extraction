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

这道闸门判的只是**原句、且只是本记录要替换的那个位置上的词**（a 侧的 focus_word，与改写句
换上去的词无关）。一条语料会展开出多个候选状态、每个状态都要过一遍：**同一位置**的几个候选
问的是逐字相同的一句原话，**不同位置**（如「上@5」与「上@3」）问的却是不同的问题。所以它在
跑批前判完并按整条 user message（含位置词）去重（见 _prefill_reality）：节点位置与产物形状都
不变（仍在 alter 之后，改写未命中的记录照旧没有 reality 块），省下的是同位置那几个候选的重复
调用（实测 1782 条记录只对应 200 个原句）。这份去重账写进 .summary.json 的 reality_prefill
——事后核对调用数时先看它。

semantic / feature / schema 三个节点都是**原句与改写句各跑一遍**（两侧各一份结果，落盘时记在 a/b
两个键下），judge 是两侧合并的地方：它一份记录只跑一次，同时读两侧的 schema 与关注词。
这三者的 a 侧输入只由语料与本记录那个位置决定（同一句原话、**同一位置**上的候选问的是逐字相同的
一条消息），所以结果按 user message 缓存在各自的 agent 实例里（见 spatial_agents.CACHEABLE_SOURCES）——第一次问过之后，同一条消息不再重复发请求；每次跑批各 agent 的往返数与命中数写进
.summary.json 的 agent_stats。同一句原话若有两个位置（上@5 与 上@3），那是两条不同的 a 侧消息，
各问一次。
两侧**各看各的关注词**——原句那侧看语料记录里那个词（focus_word），改写句那侧看规则表换上去的
那个词（focus_word_b，由 alter 节点算出）。所以 b 侧的 schema 收窄到的是**替换词**所属的词组，
判定表里 上12 × 里1 这样按两组名字写的格子才对得上；两侧共用一个词的话 b 侧会收窄回原词那一组，
判定就退化成同组自比了（见 spatial_tools.replacement_focus_word）。两处的下标都落在原句的
分词结果上（b 侧是照搬 a 侧的），消息里对 b 侧的措辞已交代不要拿它去数改写句的分词行
（改写句要重新分词，替换点可能并词，见 spatial_agents.FOCUS_INDEX_NOTE_ALTER）。

一条语料里的方位词往往能换成规则表里好几个替换词（见 spatial_tools.alter_sentence），
而且句内**每个**目标词位置都要参与改写（「他从山上掉到山谷里」的「上」与「里」各是一个位置，
见 spatial_tools.pick_focus_words）。所以跑批时会先把它展开成多个待选句、每个待选句一个状态：
**一条语料通常产出多条记录**，记录里的 candidate 块（index/total/focus_index/word/replacement/
group）标明这是第几个待选句、改的是句内哪个位置（一条记录只改一个位置，其余目标词原样保留）。
--limit / --offset 仍按语料条数算，不按展开后的状态数算。

每跑一次产出一个 jsonl，一条记录就是一条完整链路的结果（两个句子、各自的语义特征、
各自对应的 schema、两侧各自的关注词、判定信息，外加逐节点的上下文历史），
另在同目录写一份 .summary.json 汇总。
记录在 save 节点流式落盘，长跑中途也能 tail 到进度；并发跑时落盘顺序等于完成顺序、不等于
输入顺序，但**收尾时 ResultWriter.finish 会把文件按 (line_no, candidate.index) 重排**
（键见 spatial_graphs.record_order_key），所以跑完之后两份输出（本图与 all_in_one.py 那边）
顺序完全一致，可以逐行对账。中途 tail 到的仍是完成顺序——这是刻意保留的取舍。

那份重排后的记录**顺手还导出一份同名 .csv**（成对信息：原句/替换句/图式/同义判断，供 Excel 看，
列序见 spatial_tools.PAIR_CSV_COLUMNS），默认开启，--no-pairs-csv 可关。两份文件出自同一份排好序的
记录，行序必然一致；只有 pair 图的记录有这些成对字段，换别的图本就不导，启动时会说明一句。

用法：
    .venv/bin/python spatial_runner.py --limit 5                        # 跑前 5 条（pair 图，方位词）
    .venv/bin/python spatial_runner.py --limit 5 --no-llm               # 只跑纯规则与查表，不花钱
    .venv/bin/python spatial_runner.py --limit 5 --no-pairs-csv         # 不导成对信息 csv（默认导）
    .venv/bin/python spatial_runner.py --workers 4 --limit 200          # 并发跑
    .venv/bin/python spatial_runner.py --graph single --limit 5         # 换一张图
    .venv/bin/python spatial_runner.py --list-graphs                    # 有哪些图可用
    .venv/bin/python spatial_runner.py --corpus-type 趋向动词 --limit 20   # 换语料类型（别名 direction 也行）
    .venv/bin/python spatial_runner.py --corpus results/direction_corpus.jsonl --corpus-type 趋向动词 --limit 20
    .venv/bin/python spatial_runner.py --corpus-type 趋向动词 --alter-rules <路径> --limit 5   # 换改写的表
    .venv/bin/python spatial_runner.py --models model-a model-b --workers 4   # 多模型加权投票

多模型投票（--models）：
    每个模型独立跑完整张图，原始结果写到 ``<输出主干>.models/00_<模型>.jsonl``；
    最终投票结果写到 ``--out-file`` 指定的文件，记录形状与单模型一致，另带 voting 块。
    模型配置里的 ``voting_weight`` 是票数（缺省 1，必须 >= 1）；加权得分唯一最高的标签
    被采纳，最高分并列即平票，该条 status=discarded、admission 转人工复核。
    各模型并行执行（--model-workers，0 = 全部并行），每个模型内部仍按 --workers 并发；
    LTP 预填充是模型无关的，整批只做一次。

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
import os
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import agents as agent_factory
import config
import file_paths
import spatial_agents as sa
import spatial_graphs as sg
import spatial_tools as st
import voting

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

    收尾（finish）会把文件按 (line_no, candidate.index) 重排（见 _sort_records）：
    并发下写进来的顺序是完成顺序，重排后才与 all_in_one.py 的输出同序、可逐行对账。
    同一份排好序的记录再顺手导出一份同名 .csv（见 __init__ 的 csv_columns），行序与 jsonl 一致。

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
        reality_prefill: dict | None = None,
        csv_columns: tuple[tuple[str, str], ...] = (),
        model_name: str | None = None,
        overwrite: bool = False,
        input_records: int = 0,
        expanded_states: int = 0,
        progress_prefix: str = "",
    ):
        self.model_name = model_name
        self.progress_prefix = progress_prefix
        """进度行的前缀；多模型并行时各 writer 用它区分 [模型名]，单模型保持空串"""
        self.overwrite = bool(overwrite)
        """本次是否显式允许覆盖已有输出（--overwrite）；默认**拒绝**（见 __init__ 里的检查）"""
        self.input_records = int(input_records)
        """本次读进来的原句数（语料条数），只在汇总里用"""
        self.expanded_states = int(expanded_states)
        """本次展开出的候选状态数（= 落盘记录数的分母），只在汇总里用

        与 input_records 一起构成「原句 → 位置 → 候选」的转化链（D5）：光看 count 看不出
        一条原句展开成了几条、又卡在哪一步。"""
        """本次 --model 指定的模型配置名，追加到结果与汇总末尾；--no-llm 模式下为 None"""
        self.corpus_type = corpus_type
        """本次跑批的语料类型；只写进汇总，不进记录（记录形状不随本次改动变）"""
        self.config_tables = dict(config_tables or {})
        """本次实际读的四张配置表：{逻辑表名: 路径}，同样只进汇总（--alter-rules 换过表时很关键）"""
        self.reality_prefill = dict(reality_prefill or {})
        """闸门预填充的去重账（见 _prefill_reality）；同样只进汇总

        它解释的是「这一轮 reality 为什么比记录数少很多」——没有这项，事后看到调用数与记录数
        对不上只能靠猜。跑批是在**建好 writer 之后**才预填充的（图的接线校验要先跑，配置错不该
        等到花完预填充的钱才发现），所以这项由 main 在预填充之后补写到属性上。
        """
        self.agent_stats: dict[str, dict] = {}
        """各 agent 的调用统计：{名字: {"roundtrips", "failed", "cache_hits"[, "first_error"]}}
        （见 spatial_agents.BaseAgent.stats）

        run() 结束时才拿得到，同 reality_prefill：由 main 在收尾前补写到属性上。
        模型往返数是花费的直接口径（缓存命中省下的那部分不在这里，单列 cache_hits）；
        failed 是发出去但抛异常的调用数，**与 roundtrips 相加才是发出过的调用数**——
        整批被接口挡下时 roundtrips 是 0，光看它会误判成「一个请求都没发出去」（见
        _print_agent_stats）。失败过的 agent 另带 first_error（截断后的首个异常）。
        没跑到的 agent（--no-llm、或图没走到）报全 0，不是漏记。
        """
        self.csv_columns = tuple(csv_columns)
        """成对记录拍平 csv 的列序（见 spatial_graphs.GraphSpec.csv_columns）；空 = 本次不导

        由驱动按图与开关传入（--no-pairs-csv 关掉，非 pair 图本就传空）——writer 不认识图，
        也不该认识，收尾只照这份列序把已排序的记录导成同名 .csv（见 finish）。
        """
        self.extra_summary: dict = {}
        """调用方补写的汇总字段（多模型投票的 voting / models 块）

        由 main 在收尾前写入；finish 把它并进 .summary.json。单模型时为空，
        所以原有汇总形状与字段顺序不变。
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 拒绝覆盖已有输出（E1）：判断发生在**首次写入之前**——本构造是跑批里第一个碰输出
        # 文件的动作，等 write() 第一次追加时再查，文件里已经混进本轮的记录了。
        # 旧行为是不拒绝直接追加：显式 --out-file 指向旧结果时，旧记录、重复 id 与不同配置的
        # 结果会混在同一份文件里，收尾还会按 (line_no, candidate.index) 把整份重排，
        # 而 summary 的 count 只统计本轮写入，两份口径对不上。
        if not self.overwrite:
            with file_paths.open_exclusive(self.path):
                pass  # 只做独占创建：拿到一个空文件，之后的 append 都落在本轮自己的文件里
        else:
            self.path.write_text("", encoding="utf-8")  # 显式覆盖：先清空，免得混入历史记录
        self.lock = threading.Lock()
        self.count = 0
        self.counter_names = tuple(counter_names)
        self.counters: dict[str, Counter] = {name: Counter() for name in self.counter_names}
        self.tally = tally
        self.started_at = time.time()

    def write(self, record: dict) -> None:
        record = {**record, "model_name": self.model_name}
        with self.lock:
            st.append_jsonl(record, self.path)
            self.count += 1
            self.tally(record, self.counters)
            if self.count % 10 == 0:
                print(f"  {self.progress_prefix}已写出 {self.count} 条", flush=True)

    def _sort_records(self) -> list[dict]:
        """把已落盘的 jsonl 按 (line_no, candidate.index) 重排（键见 sg.record_order_key）

        write() 是各 worker 在 save 节点并发追加（见模块 docstring），落盘顺序 = 完成顺序，
        与输入顺序无关。收尾统一重排，于是本脚本与 all_in_one.py 的 finish() 走同一段代码、
        产出同一个顺序，两份文件可逐行对账。all_in_one 本来就按输入顺序写，这里重排是兜底。

        流式落盘照旧：跑的过程中文件仍是完成顺序，tail 得到进度；只收尾重排一次。
        先写临时文件再原子替换，重排途中被打断也不会把结果文件截断。

        返回排好序的记录（没有文件或文件为空时返回空表）：收尾导 csv 就在这之后、直接拿这份
        内存里的记录写（见 finish）——csv 的行序于是**必然**等于重排后 jsonl 的行序，不必再读一遍
        文件，也不会有「两份文件次序不一致」的窗口。
        """
        if not self.path.exists():
            return []
        records = st.read_jsonl(self.path)
        if not records:
            return []
        if len(records) != self.count:
            # 条数对不上说明文件被中途截断或人工拼接改过；仍照常排，只当场说清楚
            print(f"警告：输出文件里有 {len(records)} 条，本次记录数为 {self.count}，请核对该文件")
        records.sort(key=sg.record_order_key)  # 稳定排序；键由构造保证唯一
        # 临时文件用 mkstemp 生成唯一名字，不再用确定的 `<输出>.sorting`：
        # 那个名字理论上可能正好是输入文件（或输入文件的别名），直接写会覆盖原始语料，
        # 且不受 --overwrite 控制。mkstemp 在目标同目录创建，replace 仍是原子的。
        fd, temp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".sorting.tmp"
        )
        os.close(fd)
        tmp = Path(temp_name)
        try:
            st.write_jsonl(records, tmp)  # 复用唯一一份 jsonl 写法
            tmp.replace(self.path)  # Path.replace = os.replace，原子替换
        finally:
            tmp.unlink(missing_ok=True)
        print(f"  输出已按 (line_no, candidate.index) 排序（{len(records)} 条）", flush=True)
        return records

    def finish(self) -> dict:
        records = self._sort_records()
        csv_path = None
        if self.csv_columns and records:
            # 成对信息另导一份 csv 供 Excel 看（默认开；--no-pairs-csv 关）：拿刚排好的记录写，
            # 于是 csv 行序与 jsonl 行序同一份来源（见 _sort_records）
            # 路径走 st.pairs_csv_path 而不是 with_suffix(".csv")：后者在 --out-file result.csv
            # 这类路径上换出来还是它自己，收尾导 csv 会把刚写好的 jsonl 覆盖掉（E1）
            csv_path = st.pairs_csv_path(self.path)
            total, judged = st.write_pairs_csv(records, csv_path, self.csv_columns)
            print(
                f"  成对信息 csv：{csv_path}（{total} 行，其中 {judged} 行有同义判断）",
                flush=True,
            )
        # corpus_type / config_tables 是这次跑批「读的是哪一组表」的唯一凭据：user message
        # 不落盘（记录里只有 node 与耗时），事后要复现或核对口径就得靠这两项
        provenance = {"corpus_type": self.corpus_type} if self.corpus_type else {}
        if self.config_tables:
            provenance["config_tables"] = self.config_tables
        if self.reality_prefill:
            provenance["reality_prefill"] = self.reality_prefill
        if self.agent_stats:
            provenance["agent_stats"] = self.agent_stats
        summary = {
            "output_file": str(self.path),
            # csv 也是这次跑批的产物，与 output_file 并列记进汇总；没导（--no-pairs-csv 或非 pair 图）就不写
            **({"csv_file": str(csv_path)} if csv_path else {}),
            **provenance,
            # 数据契约（D3）：读取侧据此判断这份产物能不能直接解析，不必猜字段
            "schema_version": st.RESULT_SCHEMA_VERSION,
            # 分词口径（D4）：切分配置改了下标会位移，旧产物能不能复用就看这一项
            "segmentation": config.segmentation_signature(),
            # 「原句 → 候选 → 落盘记录」的转化链（D5）：count 只是最后一环，
            # 少了前两个数就看不出一条原句展开成了几条、中间掉了多少
            "input_records": self.input_records,
            "expanded_states": self.expanded_states,
            "count": self.count,
            "elapsed_seconds": round(time.time() - self.started_at, 1),
            **{name: dict(counter) for name, counter in self.counters.items()},
            "model_name": self.model_name,
            **self.extra_summary,
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的输出文件。默认**拒绝**：已有同名输出直接报错退出（E1）——"
        "旧行为是不拒绝直接追加，于是历史记录、重复 id 与不同配置的结果会混进同一份文件，"
        "收尾重排后与只统计本轮的 summary 对不上。覆盖前会先清空该文件，不会混入历史记录",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help=f"每个 agent **每条候选**的请求预算（默认 {config.CANDIDATE_MAX_REQUESTS}，取自"
        "config.CANDIDATE_MAX_REQUESTS；0 或负表示不限）。一条语料会展开出多个候选，同一位置的"
        "失败若由每个候选各自重试，重试次数会被候选数放大成整批的倍数；超预算即停，"
        "如实记为失败并转入人工复核（A3）",
    )
    parser.add_argument(
        "--batch-max-requests",
        type=int,
        default=None,
        help=f"整批共用的请求预算（默认 {config.BATCH_MAX_REQUESTS}，取自 config.BATCH_MAX_REQUESTS；"
        "0 或负表示不限）。所有 agent 共用一份计数，达到上限后余下的调用一律不再发出，"
        "记录照常落盘并转入人工复核（A3）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="最多处理多少条**可用判定语料**（screening= true/no_llm，过滤后计数）；"
        "可用数量不足时会报告。默认 100",
    )
    parser.add_argument("--offset", type=int, default=0, help="跳过前多少条可用判定语料（过滤后计数）")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", default=DEFAULT_MODEL, help="model_config/api_keys.json 里的配置名（单模型模式）")
    model_group.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="配置名",
        help="多模型投票模式：指定 model_config/api_keys.json 里的多个配置名。"
        "每个模型独立跑完整张图（各自的 agent、缓存与原始结果），再按 voting_weight "
        "对每条候选做加权简单多数；平票记为 discarded。只给一个名字时等价于 --model。"
        "不能与 --no-llm 同用（投票必须有模型判定）",
    )
    parser.add_argument(
        "--model-workers",
        type=int,
        default=config.VOTING_MODEL_WORKERS,
        help=f"多模型投票时同时跑几个模型（默认 {config.VOTING_MODEL_WORKERS}，0 = 全部并行）。"
        "每个模型内部仍按 --workers 并发处理记录，总线程数约为两者相乘",
    )
    parser.add_argument("--workers", type=int, default=1, help="并发处理的记录数（LLM 调用是 IO 密集）")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="不调任何模型：只跑纯规则改写、查表与落盘，用于验证规则表与接线",
    )
    parser.add_argument(
        "--feature-deep",
        action="store_true",
        help="feature_agent 用 deep agent（每类特征一个 subagent、主 agent 汇总）；"
        "默认是单次调用直接产出全部特征字段，一次省下约一个数量级的模型往返。"
        "deep 只在需要对照或单次调用抽得不好时开",
    )
    parser.add_argument(
        "--no-pairs-csv",
        action="store_true",
        help="不把成对信息（原句/替换句/图式/同义判断）另导一份 csv。"
        "默认**导出**到与结果 jsonl 同名、同目录的 .csv，行序与 jsonl 一致（收尾排序后同一份记录写就，"
        "见 ResultWriter.finish），供 Excel 查看；列序见 spatial_tools.PAIR_CSV_COLUMNS。"
        "只有 pair 图有这些成对字段，换别的图本就不导",
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
            print(f"            请把替换规则组填进 {report['path']}，或用 --alter-rules <路径> 指定另一张表")


def _print_agent_stats(stats: dict[str, dict]) -> None:
    """收尾打印各 agent 的模型往返数、失败数、超预算未发数与缓存命中数

    （写进汇总的那份，见 ResultWriter.agent_stats）

    这是花掉的钱的直接口径：往返数含工具往返与结构化输出那次；命中数是被按 message 缓存
    省下的重复问（命中不花往返）。全 0 的 agent 是没走到或根本没调模型（--no-llm）。

    失败数单独列，且**与往返数一样显眼**：往返数只数拿到回复的那些，一个请求都没成功时它是 0，
    而 0 次往返读起来正是「一个请求都没发出去」。整批被接口挡下（403 区域限制、欠费、连不上）
    时真相恰好相反——发是发了，全被拒。所以这里把失败数印在同一行，并在「一次都没成功却失败过」
    时额外把首个失败原因摊开：那种情况下收尾的这几行就是唯一的诊断入口，不该逼人去翻 jsonl。

    超预算未发数（budget_exceeded）与失败数是两件事：后者发出去了但被拒，前者**根本没发**。
    不印出来的话，超预算成片发生时调用账看起来只是「往返少」，看不出是预算掐断的。
    """
    if not stats:
        return
    parts = []
    budget = 0
    for name, item in stats.items():
        part = f"{name} {item['roundtrips']} 次往返"
        if item.get("failed"):
            part += f"／{item['failed']} 次失败"
        if item.get("budget_exceeded"):
            budget += item["budget_exceeded"]
            part += f"／{item['budget_exceeded']} 次超预算未发"
        if item.get("retries"):
            part += f"／{item['retries']} 次重试"
        parts.append(part + f"／{item['cache_hits']} 次命中")
    total = sum(item["roundtrips"] for item in stats.values())
    failed = sum(item.get("failed", 0) for item in stats.values())
    hits = sum(item["cache_hits"] for item in stats.values())
    tail = f"，{failed} 次失败" if failed else ""
    print(f"模型调用账（共 {total} 次往返{tail}，缓存省下 {hits} 次）：" + "；".join(parts))
    if budget:
        print(
            f"提示：有 {budget} 次调用因请求预算未发出，记录里记为 source=error 并转入人工复核；"
            "需要放宽时调 --max-requests（每条候选）或 --batch-max-requests（整批）"
        )
    if total or not failed:
        return
    # 一次往返都没成却冒出失败：整批调用没一个成功的，这是最容易被误读成「没发请求」的形状
    reason = next(
        (item["first_error"] for item in stats.values() if item.get("first_error")),
        "（异常没留下可读的 repr）",
    )
    print(f"警告：本次没有任何一次模型调用成功，{failed} 次调用全部失败（请求已发出，被接口拒绝或没能连上）")
    print(f"      首次失败：{reason}")
    print("      逐条失败原因见各记录的 status_reason；若全部是同一条 403/超时，先查模型可达性再重跑")


def _prefill(records: list[dict], agents, spec: sg.GraphSpec) -> None:
    """跑批前把整批句子一次性喂给 LTP

    一来省下逐句前向的开销（批量推理快约 4 倍，见 sentenceTable.py），
    二来让后续多线程尽量直接读缓存；未命中的推理由 sentenceTable._infer_lock 串行保护。
    哪些句子会被用到由图声明（spec.plan），预填充后再检查缓存命中情况。
    """
    sentences = [s for s in spec.plan(records, agents) if s]
    unique = list(dict.fromkeys(sentences))
    # 一次预填超过缓存容量会使早期句子被 LRU 淘汰；处理 state 尚未开始，
    # 按容量分片预填也会让后片把前片淘汰。
    # 继续使用整批预填，并在超过容量时明确提示，避免误报分片已解决重复推理。
    filled = st.prefill_ltp(unique)
    print(f"LTP 预填充：本图需 {len(unique)} 句，实际送入模型 {filled} 句（其余命中缓存）")

    missing = [s for s in unique if not st.cache_ready(s)]
    if missing:
        # 漏网的句子由 worker 现场补算，那一步走 sentenceTable._infer_lock 的串行边界，
        # 不会退化成并发调同一份模型（见该锁的说明）
        print(f"警告：预填充后仍有 {len(missing)} 句不在缓存里，多线程下可能补算（已串行保护）")
    if len(unique) > config.CACHE_MAX_SIZE:
        print(
            f"警告：需预填充 {len(unique)} 句，超过缓存容量 {config.CACHE_MAX_SIZE}；"
            "早期句子可能被淘汰并重新推理。可缩小 --limit，或提高 config.CACHE_MAX_SIZE"
        )


def _prefill_reality(records: list[dict], agents, spec: sg.GraphSpec, workers: int, *, verbose: bool = True) -> dict:
    """跑批前把闸门要判的原句一次判完（去重），随后节点按 message 读缓存

    与 _prefill 同一套路、同一个理由：同一件判定的重复请求应当在进线程池之前去掉。闸门判的
    只是原句（a 侧那个词），而一条语料按规则表能展开出 9~11 个候选状态、每个状态都要过一遍
    ——不去重就是几倍的调用（实测 1782 条记录只对应 200 个原句，其中 1283 条记录只对应 157 个）。
    注意「原句」这里要读成「原句 + 本位置那个词」：句内两个位置（上@5 与 上@3）同一句原话
    但是两个问题，prefill 按整条 user message 去重，不会把它们并掉。

    哪些原句会走到闸门由图自己声明（spec.gate_queries）：改写未命中的记录不经过闸门，
    也就不在这里判（规则表为空时正是整批都这样，那批一条都不该发）。

    失败不入缓存（见 BaseAgent.CACHEABLE_SOURCES），所以 failed 的那几个会在各自候选上重问——
    打印里如实交代，事后对不上账时能一眼看出差在哪。
    """
    if "reality" not in spec.needs:
        return {}
    stat = agents.need("reality").prefill(spec.gate_queries(records, agents), workers=workers)
    line = (
        f"reality 预填充：本图需判 {stat['unique']} 个原句"
        f"（去重前 {stat['queries']} 个待选状态），实际发出 {stat['asked']} 次"
    )
    if stat["failed"]:
        line += f"；其中 {stat['failed']} 次失败未入缓存，会在各自候选上重试"
    if stat["skipped"]:
        line += f"；另有 {stat['skipped']} 个状态没有可判的原句/关注词，未发出"
    if verbose:
        print(line)
    return stat


def _validate_run_outputs(
    corpus_path: str | Path,
    out_path: str | Path,
    *,
    csv_columns: tuple[tuple[str, str], ...] = (),
    overwrite: bool = False,
) -> None:
    """在**首次写入之前**把这次跑批要写的几个路径一起查一遍（E1）

    查三件事，任一件不成立就在写盘之前退出：
      1. 输入与任何一个输出（jsonl / 成对 csv / summary）不能是同一个文件——
         显式 `--out-file` 与输入相同时，旧行为会把结果追加进输入、收尾再排序重写它，
         原始语料与结果记录混在一个文件里。
      2. 几个输出之间不能互撞：`--out-file result.csv` 时 `with_suffix(".csv")` 换出来还是
         它自己，收尾导 csv 会覆盖刚写好的 jsonl（路径已由 st.pairs_csv_path 错开，这里复查）。
      3. 已有输出默认拒绝（--overwrite 才允许），且拒绝发生在建文件之前。

    扩展名也在这里规定：jsonl 结果、csv 视图、summary.json 各归各位，不靠调用方自觉。
    """
    out_path = Path(out_path)
    # 真正的检查在 file_paths.validate_run_outputs：两个跑批入口共用同一份口径
    # 识别相对路径、符号链接与硬链接别名（见 file_paths.same_file_path）
    file_paths.validate_run_outputs(
        corpus_path,
        out_path,
        st.pairs_csv_path(out_path) if csv_columns else None,
        overwrite=overwrite,
    )


def _build_overrides(args, spec: sg.GraphSpec) -> dict[str, dict]:
    """命令行里会影响 agent 构造的注入项（--alter-rules / --feature-deep）

    与具体模型无关，多模型时每个模型各建一份同口径的 agent。
    """
    overrides: dict[str, dict] = {}
    if args.alter_rules:
        overrides["alter"] = {"rules": st.load_alter_rules(args.alter_rules), "rules_path": args.alter_rules}
    if args.feature_deep and "feature" in spec.needs:
        overrides.setdefault("feature", {})["deep"] = True
    return overrides


def _build_model_agents(args, spec: sg.GraphSpec, corpus_type: str, model_key: str, overrides: dict[str, dict]):
    """按模型建一套 agent 并预热：配置错在启动时就报，不留给线程池去抢着建"""
    agents = sa.build_agents(
        spec.needs,
        model_key=model_key,
        llm=not args.no_llm,
        overrides=overrides,
        corpus_type=corpus_type,
        # 请求预算（A3）：两条上限各取 config，命令行给了就以命令行为准
        max_requests=config.CANDIDATE_MAX_REQUESTS if args.max_requests is None else args.max_requests,
        batch_max_requests=(
            config.BATCH_MAX_REQUESTS
            if args.batch_max_requests is None
            else args.batch_max_requests
        ),
    )
    for agent in agents.values():
        agent.warm()
    return agents


def _execute_run(
    args,
    spec: sg.GraphSpec,
    corpus_type: str,
    corpus_path,
    records: list[dict],
    agents,
    out_path: str | Path,
    *,
    model_key: str,
    prefill_ltp: bool = True,
    verbose: bool = True,
) -> dict:
    """用一套已建好的 agent 跑完整张图并落盘

    单模型与多模型的单模型分支共用这一段；多模型模式下 verbose=False（各模型的启动打印
    由协调层统一出，避免几个线程的日志交错），prefill_ltp=False（LTP 是模型无关的，
    整批只在协调层预填充一次）。
    """
    out_path = Path(out_path)
    csv_columns = spec.csv_columns if not args.no_pairs_csv else ()
    writer = ResultWriter(
        out_path,
        counter_names=spec.counter_names,
        tally=spec.tally,
        corpus_type=corpus_type,
        config_tables=_config_tables(agents),
        # 成对信息 csv：默认导出（--no-pairs-csv 关），列序由图给（只有 pair 图有）
        csv_columns=csv_columns,
        model_name=model_key if not args.no_llm else None,
        overwrite=args.overwrite,
        progress_prefix="" if verbose else f"[{model_key}] ",
    )
    graph = spec.build(agents, writer)
    # 实际生效的请求策略（超时/重试/退避）写进汇总：事后看到某轮特别慢，
    # 能直接判断是模型慢还是重试多，不必翻 config.py 与模型配置去猜
    first_agent = next(iter(agents.values()), None)
    policy = getattr(first_agent, "retry_policy", None)
    if policy:
        writer.extra_summary["request_policy"] = dict(policy)

    if verbose:
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
        if args.no_pairs_csv:
            print("成对信息 csv：本次不导出（--no-pairs-csv）")
        elif spec.csv_columns:
            print(f"成对信息 csv：收尾导出到 {st.pairs_csv_path(out_path)}（--no-pairs-csv 可关）")
        else:
            # 非 pair 图的记录里没有成对字段：默认开着也导不出东西，说清楚而不是静悄悄少一个产物
            print(f"提示：本图（{spec.name}）的记录不是成对形状，没有可导出的成对列，本次不导 csv")

    if not args.no_llm:
        if prefill_ltp:
            _prefill(records, agents, spec)
        # 闸门的原句在这里一次判完（去重）。图已经建好——接线错在花钱之前就报出来，于是这份
        # 去重账补写到 writer 上，与语料类型、配置表一并进汇总（见 ResultWriter.reality_prefill）
        writer.reality_prefill = _prefill_reality(records, agents, spec, args.workers, verbose=verbose)

    # 一条语料的每个目标词位置各能改出好几个待选句，每个待选句一个状态、一条记录，
    # 所以状态数一般多于语料条数
    states = spec.init_states(records, agents, str(out_path))
    if verbose:
        print(f"展开待选句：{len(records)} 条语料 → {len(states)} 个状态（每个状态落一条记录）")
    # 转化链的两个分母（D5）补写到 writer 上、进汇总（与 reality_prefill / agent_stats 同一套路）
    writer.input_records = len(records)
    writer.expanded_states = len(states)
    started = time.time()

    def run_state(state):
        # 一条状态 = 一条候选，候选预算在这个作用域内累计（见 AgentSet.budget_scope）；
        # 作用域按线程隔离，--workers N 下各条候选互不串账
        with agents.budget_scope():
            return graph.invoke(state, {"recursion_limit": 60})

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        # 结果已在 save 节点落盘，这里只需驱动跑完；map 按输入顺序产出，异常会在此处抛出
        for _ in pool.map(run_state, states):
            pass

    # 各 agent 的调用账（往返数、重试数与缓存命中数）：跑完才齐，收尾补写到 writer 上、进汇总
    writer.agent_stats = {name: agent.stats() for name, agent in agents.items()}
    summary = writer.finish()
    if verbose:
        print(
            f"\n完成 {summary['count']} 条记录（{len(records)} 条语料），用时 {summary['elapsed_seconds']} 秒"
            f"（总耗时 {round(time.time() - started, 1)} 秒）"
        )
        for name in spec.counter_names:
            print(f"{name}：{summary[name]}")
        _print_admission(summary)
        _print_agent_stats(writer.agent_stats)
        print(f"汇总文件：{summary['summary_file']}")
    return {"summary": summary, "writer": writer, "states": states}


def _run_single_model(args, spec, corpus_type, corpus_path, records, out_path: Path, model_key: str) -> dict:
    """原有单模型路径：行为、输出文件与汇总形状保持不变"""
    overrides = _build_overrides(args, spec)
    agents = _build_model_agents(args, spec, corpus_type, model_key, overrides)
    csv_columns = spec.csv_columns if not args.no_pairs_csv else ()
    _validate_run_outputs(corpus_path, out_path, csv_columns=csv_columns, overwrite=args.overwrite)
    return _execute_run(
        args,
        spec,
        corpus_type,
        corpus_path,
        records,
        agents,
        out_path,
        model_key=model_key,
        prefill_ltp=True,
        verbose=True,
    )


def _run_multi_model(
    args,
    spec,
    corpus_type,
    corpus_path,
    records: list[dict],
    out_path: Path,
    model_names: list[str],
    weights: dict[str, int],
) -> dict:
    """多模型投票：每个模型独立跑完整张图，再逐候选加权投票

    并行边界在**模型**这一层：模型之间互不共享 agent、缓存与 writer；
    每个模型内部仍按 --workers 并发跑候选。原始结果各写自己的文件，
    最终投票结果写用户 --out-file 指定的那一个，下游读取方式与单模型一致。
    """
    if args.model_workers < 0:
        raise SystemExit("错误：--model-workers 不能为负数")
    model_workers = args.model_workers or len(model_names)
    raw_dir, raw_paths = voting.model_output_paths(out_path, model_names)
    csv_columns = spec.csv_columns if not args.no_pairs_csv else ()
    voting_csv_columns = csv_columns
    if csv_columns:
        # 最终 csv 在原有的成对列后追加投票审计列；原始各模型的 csv 仍用原列序
        voting_csv_columns = csv_columns + (
            ("投票状态", "voting.status"),
            ("获胜标签", "voting.winner_label"),
            ("获胜模型", "voting.winner_model"),
            ("有效权重", "voting.valid_weight"),
            ("配置权重", "voting.configured_weight"),
            ("缺票模型", "voting.missing_models"),
        )

    # 所有输出（最终投票文件 + 各模型原始文件及其 csv/summary）放进同一张清单，
    # 一次检查输入/输出别名与已有文件：分开逐组检查看不出“最终文件”和某个原始文件
    # 其实是同一个硬链接/软链接的情况。
    outputs: list[Path] = [out_path]
    if csv_columns:
        outputs.append(st.pairs_csv_path(out_path))
    outputs.append(out_path.with_suffix(".summary.json"))
    for raw_path in raw_paths.values():
        outputs.append(raw_path)
        if csv_columns:
            outputs.append(st.pairs_csv_path(raw_path))
        outputs.append(raw_path.with_suffix(".summary.json"))
    file_paths.validate_output_paths([corpus_path], outputs)
    if not args.overwrite:
        for output in outputs:
            if output.exists():
                raise ValueError(
                    f"输出文件 {output} 已存在（默认拒绝覆盖，免得混入历史记录与重复 id）；"
                    "请换一个输出路径，或显式加 --overwrite 覆盖它"
                )

    print(f"多模型投票：{len(model_names)} 个模型（权重 " + "，".join(f"{n}={weights[n]}" for n in model_names) + "）")
    print(f"最终输出：{out_path}")
    print(f"各模型原始结果：{raw_dir}/")
    print(f"模型并发：{model_workers}；每个模型内部 --workers={max(1, args.workers)}")

    overrides = _build_overrides(args, spec)
    agents_by_model = {}
    for name in model_names:
        print(f"  准备模型 {name}（voting_weight={weights[name]}）")
        agents_by_model[name] = _build_model_agents(args, spec, corpus_type, name, overrides)

    first_agents = agents_by_model[model_names[0]]
    _print_config_tables(first_agents)
    print("关注词表：" + "；".join(f"{kind} {len(words)} 个" for kind, words in st.load_focus_vocab().items()))
    if args.no_pairs_csv:
        print("成对信息 csv：本次不导出（--no-pairs-csv）")
    elif spec.csv_columns:
        print(f"成对信息 csv：收尾导出到 {st.pairs_csv_path(out_path)}（--no-pairs-csv 可关）")
    else:
        print(f"提示：本图（{spec.name}）的记录不是成对形状，没有可导出的成对列，本次不导 csv")

    # LTP 是模型无关的：整批只预填充一次，避免几个模型线程重复做同一批句法分析
    if not args.no_llm:
        _prefill(records, first_agents, spec)

    started = time.time()
    results: dict[str, dict] = {}
    errors: list[tuple[str, BaseException]] = []
    with ThreadPoolExecutor(max_workers=max(1, model_workers)) as pool:
        futures = {
            pool.submit(
                _execute_run,
                args,
                spec,
                corpus_type,
                corpus_path,
                records,
                agents_by_model[name],
                raw_paths[name],
                model_key=name,
                prefill_ltp=False,
                verbose=False,
            ): name
            for name in model_names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
                print(f"[{name}] 完成 {results[name]['summary']['count']} 条，"
                      f"用时 {results[name]['summary']['elapsed_seconds']} 秒")
            except BaseException as exc:  # 先收集，等其余模型收尾后再抛
                errors.append((name, exc))
                print(f"[{name}] 失败：{exc!r}")
    if errors:
        failed_model, exc = errors[0]
        raise RuntimeError(f"多模型投票中模型 {failed_model} 运行失败：{exc!r}") from exc

    model_records = {}
    for name in model_names:
        path = raw_paths[name]
        if not path.exists():
            raise RuntimeError(f"模型 {name} 没有产出结果文件：{path}")
        model_records[name] = st.read_jsonl(path)

    final_records = voting.combine_model_records(
        spec.name,
        model_records,
        weights,
        model_names,
        output_file=out_path,
    )
    if not final_records:
        print("警告：所有模型都没有产出可投票的记录，未生成最终结果")
        return {}

    voting_writer = ResultWriter(
        out_path,
        counter_names=spec.counter_names,
        tally=spec.tally,
        corpus_type=corpus_type,
        config_tables=_config_tables(first_agents),
        csv_columns=voting_csv_columns,
        model_name="voting",
        overwrite=args.overwrite,
        input_records=len(records),
        expanded_states=len(results[model_names[0]]["states"]),
        progress_prefix="[voting] ",
    )
    voting_writer.extra_summary = {
        "voting": voting.summarize_voting(final_records),
        "model_names": list(model_names),
        "voting_weights": dict(weights),
        "models": [
            {
                "name": name,
                "weight": weights[name],
                "output_file": str(raw_paths[name]),
                "count": results[name]["summary"]["count"],
                "elapsed_seconds": results[name]["summary"]["elapsed_seconds"],
                "agent_stats": results[name]["writer"].agent_stats,
                "reality_prefill": results[name]["writer"].reality_prefill,
                "request_policy": results[name]["writer"].extra_summary.get("request_policy"),
            }
            for name in model_names
        ],
    }
    for record in final_records:
        voting_writer.write(record)
    summary = voting_writer.finish()

    print(
        f"\n完成 {summary['count']} 条最终记录（{len(records)} 条语料），"
        f"用时 {summary['elapsed_seconds']} 秒（总耗时 {round(time.time() - started, 1)} 秒）"
    )
    for name in model_names:
        stats = results[name]["writer"].agent_stats
        roundtrips = sum(item.get("roundtrips", 0) for item in stats.values())
        failed = sum(item.get("failed", 0) for item in stats.values())
        retries = sum(item.get("retries", 0) for item in stats.values())
        print(
            f"  模型 {name}（权重 {weights[name]}）：{roundtrips} 次往返"
            + (f"／{failed} 次失败" if failed else "")
            + (f"／{retries} 次重试" if retries else "")
        )
    vote_stats = summary["voting"]
    print(
        f"投票：采纳 {vote_stats['accepted']} 条，舍弃 {vote_stats['discarded']} 条"
        f"（平票 {vote_stats['tie']}，无有效票 {vote_stats['no_valid_votes']}）；"
        f"不完整 {vote_stats['incomplete']} 条"
    )
    if vote_stats["labels"]:
        print("  采纳标签：" + "，".join(f"{k} {v}" for k, v in vote_stats["labels"].items()))
    if vote_stats["missing_by_model"]:
        print("  缺票模型：" + "，".join(f"{k} {v} 条" for k, v in vote_stats["missing_by_model"].items()))
    for name in spec.counter_names:
        print(f"{name}：{summary[name]}")
    _print_admission(summary)
    print(f"汇总文件：{summary['summary_file']}")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_graphs:
        _print_graphs()
        return

    selected_models = list(args.models) if args.models else [args.model]
    if len(set(selected_models)) != len(selected_models):
        raise SystemExit("错误：--models 里有重复的模型配置名")
    multi_model = len(selected_models) > 1
    if args.models and not multi_model:
        print(f"提示：--models 只给了一个模型（{selected_models[0]}），按单模型模式运行")
    if multi_model and args.no_llm:
        raise SystemExit("错误：--models 多模型投票必须有模型判定，不能与 --no-llm 同用")

    # 权重在花钱之前校验完：写错一个字段就当场退出，而不是跑完几个模型才发现权重没生效。
    # 单模型不读模型配置：--no-llm 的用途之一就是在没有 api_keys.json 的机器上验纯规则，
    # 这里不能因为一个用不到的投票权重把那条路堵死。
    weights: dict[str, int] = {}
    if multi_model:
        for name in selected_models:
            cfg = agent_factory.load_model_config(name)
            weights[name] = voting.normalize_weight(
                cfg.get(voting.VOTING_WEIGHT_FIELD), model_name=name
            )

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
    records = st.read_corpus(corpus_path, limit=args.limit, offset=args.offset, judge_only=True)
    if args.limit is not None and len(records) < args.limit:
        print(
            f"警告：可用判定语料只有 {len(records)} 条（true/no_llm），少于计数限制 {args.limit} 条；"
            "已按实际数量运行"
            + (f"（已跳过 offset={args.offset} 条可用语料）" if args.offset else "")
        )
    if not records:
        print(f"没有从 {corpus_path} 读到可用于判定的记录")
        return

    if args.feature_deep and "feature" not in spec.needs:
        # 图不建 feature agent，这个开关就无从生效；静默忽略会让人以为 deep 跑过了
        print(f"提示：--feature-deep 对本图（{spec.name}）无效——它不建 feature agent")

    out_dir = Path(args.out_dir) if args.out_dir else _PROJECT_ROOT / "results" / spec.prefix
    out_path = Path(args.out_file) if args.out_file else st.build_output_path(out_dir, prefix=spec.prefix)

    if multi_model:
        _run_multi_model(args, spec, corpus_type, corpus_path, records, out_path, selected_models, weights)
    else:
        _run_single_model(args, spec, corpus_type, corpus_path, records, out_path, selected_models[0])


def _print_admission(summary: dict) -> None:
    """收尾把**处理成功率**与**数据准入率**并列打出来（A1）

    status_counts 说的是流程有没有跑完，而闸门判不出来时流程照样跑得完——那批记录会得到
    status=ok、空间性却根本没确认。两个数分开报，才不会把「跑完了」当成「能用了」。
    哪些维度卡住（reality/fluency/features/schema/judgement）也一并列出，
    review_required 那一行就是下一轮人工复核的入口。
    """
    admission = summary.get("admission_counts")
    if not admission:
        return
    total = sum(admission.get(key, 0) for key in ("admitted", "not_admitted"))
    if not total:
        return
    rate = admission.get("admitted", 0) / total
    print(f"数据准入：{admission.get('admitted', 0)}/{total} 条（{rate:.1%}）；"
          f"待人工复核 {admission.get('review_required', 0)} 条")
    for name in ("reality", "fluency", "features", "schema", "judgement"):
        counter = summary.get(f"{name}_admission")
        if counter:
            print(f"    {name}：" + "，".join(f"{key} {value}" for key, value in sorted(counter.items())))


if __name__ == "__main__":
    main()
