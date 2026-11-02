# encoding: utf8

from ltp import LTP
import torch

ltp = LTP(r"LTP_model\base")

# 将模型移动到 GPU 上
if torch.cuda.is_available():
    ltp.to("cuda")

# 分词 cws、词性 pos、命名实体标注 ner、语义角色标注 srl、依存句法分析 dep、语义依存分析树 sdp、语义依存分析图 sdpg
# result = ltp.pipeline(["他从楼下扔上来一串钥匙", "他从楼下扔上一串钥匙来", "他从楼下扔一串钥匙上来"], tasks = ["cws", "pos", "dep"])
# result = ltp.pipeline("当他遇到阻碍时,例如:被玩伴嘲笑或是无法拼好一块拼图时,你可有几个步骤来帮助他:1. ", tasks = ["cws", "pos", "dep"])
# result = ltp.pipeline(["他从楼下扔上来一串钥匙", "他从楼下扔上一串钥匙来", "他从楼下扔一串钥匙上来"], tasks = ["cws", "dep", "sdp"])
result = ltp.pipeline(["魏仲时这时从里间的重症病房出来,又走进了病房。"], tasks = ["cws", "dep", "sdp"])
print(result)
print(result.cws)
print(result.dep)
print(result.sdp)
