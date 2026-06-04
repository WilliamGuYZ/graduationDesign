# 📋 毕业论文待办事项清单

## ⏳ 等真实实验回来再做

### 服务器侧
- [ ] **B-2 tokenizer 长度分布**：`python scripts/check_cot_seq_length.py --num-few-shots 0/2`，p95 > 2048 需配合 `--max-length` 上调
- [ ] **B-3 vLLM + LoRA 最小冒烟**：`SAMPLE_LIMIT=5 NUM_SAMPLES=2` 跑 H4，确认 LoRA 装载
- [ ] **跑 H1~H10 全量**：`bash scripts/train_eval.sh`，FS_K=2

### 真实数据回来后的精修
- [ ] **JSON 覆盖**：真实 `eval_*_passk_*.json` 同名覆盖 `evaluation/results/` 下的 mock 文件
- [ ] **论文 6 个 .tex 精修**（用 `git diff` 看真实 vs mock 差距）：
  - `ch05-experiments/sections/sec02-results/subsections/01-base-lora-results.tex`
  - `ch05-experiments/sections/sec02-results/subsections/02-cot-results.tex`
  - `ch05-experiments/sections/sec03-discussion/subsections/01-lora-effectiveness.tex`
  - `ch05-experiments/sections/sec03-discussion/subsections/02-few-shot-cot-boundary.tex`
  - `ch05-experiments/sections/05-summary.tex`
  - `ch06-conclusion/sections/01-research-summary.tex`
- [ ] **H1-H10 模板同步精修**：§3.2 / §3.3 / §3.4 / §3.6 用真实数据更新
- [ ] **方向反转检查**：若真实数据中某个 DiD 符号反转，需重写 ch05 §5.3 对应段落的"放大/吸收/正交"判定

---

## 🎯 执行顺序

```
SSH 恢复 → 权重就位
       ↓
   B-2 / B-3 冒烟
       ↓
  bash scripts/train_eval.sh（H1-H10 全量）
       ↓
  真实 JSON 覆盖 mock JSON
       ↓
  git diff 精修 6 个 .tex + H1-H10 模板
       ↓
  提交
```