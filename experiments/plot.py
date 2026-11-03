"""Plots and markdown tables from results/ -> results/*.png, results/tables.md.

Colours are fixed per entity across every chart (full = blue, window-only = orange,
sinks = aqua, extra ablations yellow / magenta); lines are direct-labelled and
every chart has a table in tables.md, so identity never rests on colour alone.
"""

import json
import math
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PPL = RES / "ppl" / "qwen3-0.6b"
BUCKET = 1024

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
})

LABEL = {
    "full": "full cache", "sink0_w1024": "window only (1,024)", "sink4_w1020": "4 sinks + 1,020 window",
    "sink4_w1020_norot": "4 + 1,020, no re-rotation", "sink4_w1020_bf16k": "4 + 1,020, bf16 keys",
    "sink0_w1024_bf16k": "window only, bf16 keys",
    "sink4_w1020_norot_bf16k": "4 + 1,020, no re-rotation (bf16 keys)",
}
COLOR = {"full": BLUE, "sink0_w1024": ORANGE, "sink4_w1020": AQUA,
         "sink4_w1020_norot": MAGENTA, "sink4_w1020_bf16k": YELLOW, "sink0_w1024_bf16k": MAGENTA}
tables: list[str] = []


def load(cfg, model_dir=PPL):
    p = model_dir / f"{cfg}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    meta = json.loads((model_dir / f"{cfg}.json").read_text()) if (model_dir / f"{cfg}.json").exists() else {}
    return df, meta


def bucketed(df):
    b = (df.pos - 1) // BUCKET
    g = df.groupby(b).nll.mean()
    return (g.index + 0.5) * BUCKET, g.values


def mean_nll(df, lo, hi):
    s = df[(df.pos > lo) & (df.pos <= hi)].nll
    return s.mean() if len(s) else math.nan


def label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.plot([x], [y], "o", color=color, ms=4)


# ------------------------------------------------------------------ experiment 1: main
def plot_main():
    runs = {c: load(c) for c in ["full", "sink0_w1024", "sink4_w1020"]}
    runs = {c: r for c, r in runs.items() if r is not None}
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for cfg, (df, meta) in runs.items():
        x, y = bucketed(df)
        ax.plot(x, y, color=COLOR[cfg], label=LABEL[cfg])
        label_end(ax, x[-1], y[-1], LABEL[cfg] + (f"  (OOM at {meta['oom_at']:,})" if meta.get("oom_at") else ""),
                  COLOR[cfg])
    ax.axvline(40960, color=INK2, lw=1, ls=":")
    ax.set_ylim(2.5, 7.9)
    ax.text(40960, 7.9, " Qwen3 max_position_embeddings", va="top", fontsize=8, color=INK2)
    ax.axvline(57722, color=INK2, lw=1, ls=":")
    ax.text(57722, 7.6, " PG19 book 2 starts", va="top", fontsize=8, color=INK2)
    ax.set_ylim(bottom=2.5)
    ax.set_xlabel("position in stream (tokens)")
    ax.set_ylabel("mean NLL per token (nats), 1,024-token buckets")
    ax.set_title("Streaming PG19 through Qwen3-0.6B, one token at a time, 1,024-token cache")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 65536 * 1.22)
    fig.tight_layout()
    fig.savefig(RES / "ppl_vs_position.png", dpi=150)
    plt.close(fig)

    # table: mean NLL (and PPL) over ranges
    ranges = [(0, 1024), (1024, 2048), (2048, 8192), (8192, 32768), (32768, 40960), (40960, 65536)]
    rows = ["| config | " + " | ".join(f"{lo // 1024}K–{hi // 1024}K" for lo, hi in ranges) + " |",
            "|---|" + "---:|" * len(ranges)]
    for cfg, (df, meta) in runs.items():
        cells = []
        for lo, hi in ranges:
            v = mean_nll(df, lo, hi)
            cells.append("–" if math.isnan(v) else f"{math.exp(v):.2f}")
        rows.append(f"| {LABEL[cfg]} | " + " | ".join(cells) + " |")
    tables.append("### Experiment 1: perplexity by position range (exp of mean NLL)\n\n" + "\n".join(rows))

    if "full" in runs and "sink4_w1020" in runs:
        full, sink = runs["full"][0], runs["sink4_w1020"][0]
        end = int(full.pos.max())
        lines = ["| range | full | sinks | sinks vs full | window only | window vs full |", "|---|---:|---:|---:|---:|---:|"]
        for lo, hi in [(1024, end), (8192, min(end, 32768)), (32768, end)]:
            f, s = mean_nll(full, lo, hi), mean_nll(sink, lo, hi)
            w = mean_nll(runs["sink0_w1024"][0], lo, hi) if "sink0_w1024" in runs else math.nan
            lines.append(f"| {lo:,}–{hi:,} | {math.exp(f):.2f} | {math.exp(s):.2f} | {math.exp(s - f) - 1:+.1%} | "
                         f"{math.exp(w):.1f} | {math.exp(w - f):.1f}× |")
        tables.append("### Headline: perplexity relative to the full cache, over the range the full cache survived\n\n"
                      + "\n".join(lines))


# ------------------------------------------------------------------ experiment 1: ablations
def plot_ablations():
    lo, hi = 2048, 16384  # after the first eviction has settled
    sinks = [(s, load(f"sink{s}_w1020")) for s in [0, 1, 2, 4, 8]]
    sinks = [(s, r) for s, r in sinks if r is not None]
    windows = []
    for w in [256, 512, 1024, 2048]:
        a, b = load(f"sink4_w{w - 4}"), load(f"sink0_w{w}")
        windows.append((w, a, b, load(f"sink4_w{w - 4}_bf16k")))
    full = load("full")
    ref = mean_nll(full[0], lo, hi) if full else None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    if sinks:
        xs = [str(s) for s, _ in sinks]
        ys = [math.exp(mean_nll(r[0], lo, hi)) for _, r in sinks]
        bars = ax.bar(xs, ys, color=AQUA, width=0.6, edgecolor=SURFACE, linewidth=2)
        for bar, y in zip(bars, ys):
            ax.annotate(f"{y:.1f}", (bar.get_x() + bar.get_width() / 2, y), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
        if ref is not None:
            ax.axhline(math.exp(ref), color=BLUE, lw=1.5, ls="--")
            ax.text(len(xs) - 0.55, math.exp(ref), " full cache", color=INK2, fontsize=8, va="center")
            ax.set_xlim(-0.6, len(xs) + 0.3)
        ax.set_yscale("log")
        ax.set_yticks([20, 30, 50, 100, 200, 300, 500])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("number of sink tokens (window fixed at 1,020)")
        ax.set_ylabel(f"perplexity, tokens {lo // 1024}K–{hi // 1024}K (log scale)")
        ax.set_title("How many sinks?")
    ax = axes[1]
    for key, idx, color, name in [("sinks", 1, AQUA, "4 sinks + window"), ("window", 2, ORANGE, "window only"),
                                  ("bf16", 3, YELLOW, "4 sinks, bf16 keys")]:
        pts = [(w, math.exp(mean_nll(t[idx][0], lo, hi))) for t in windows for w in [t[0]] if t[idx] is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=color, ms=8)
            label_end(ax, pts[-1][0], pts[-1][1], name, color)
    if ref is not None:
        ax.axhline(math.exp(ref), color=BLUE, lw=1.5, ls="--")
        ax.text(256, math.exp(ref), "full cache", color=INK2, fontsize=8, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024, 2048], ["256", "512", "1,024", "2,048"])
    ax.set_yscale("log")
    ax.set_yticks([20, 30, 50, 100, 200, 300, 500])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(200, 4500)
    ax.set_xlabel("total cache size (tokens)")
    ax.set_title("Cache size, with and without sinks")
    fig.tight_layout()
    fig.savefig(RES / "ablations.png", dpi=150)
    plt.close(fig)

    f = lambda r: f"{math.exp(mean_nll(r[0], lo, hi)):.2f}" if r else "–"
    rows = [f"| sinks (window 1,020) | ppl {lo // 1024}K–{hi // 1024}K | bf16 keys |", "|---:|---:|---:|"]
    rows += [f"| {sc} | {f(load(f'sink{sc}_w1020'))} | {f(load(f'sink{sc}_w1020_bf16k'))} |" for sc in [0, 1, 2, 4, 8]]
    if ref is not None:
        rows.append(f"| full cache | {math.exp(ref):.2f} | |")
    t2 = ["| cache size | 4 sinks + window | window only | 4 sinks, bf16 keys | window only, bf16 keys |",
          "|---:|---:|---:|---:|---:|"]
    for w, a, b, c in windows:
        t2.append(f"| {w:,} | {f(a)} | {f(b)} | {f(c)} | {f(load(f'sink0_w{w}_bf16k'))} |")
    extra = ["| variant (4 + 1,020) | ppl 2K–16K |", "|---|---:|"]
    for cfg in ["sink4_w1020", "sink4_w1020_bf16k", "sink4_w1020_norot_bf16k"]:
        r = load(cfg)
        if r:
            extra.append(f"| {LABEL[cfg]} | {math.exp(mean_nll(r[0], lo, hi)):.3f} |")
    tables.append("### Ablation: sink count\n\n" + "\n".join(rows))
    tables.append("### Ablation: cache size\n\n" + "\n".join(t2))
    tables.append("### Ablation: position handling and key precision\n\n" + "\n".join(extra))


# ------------------------------------------------------------------ experiment 2: memory
def plot_memory():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    rows = ["| tokens | " + " | ".join(LABEL[c] for c in ["full", "sink0_w1024", "sink4_w1020"]) + " |",
            "|---:|---:|---:|---:|"]
    data, kvb, base = {}, None, None
    for cfg in ["full", "sink0_w1024", "sink4_w1020"]:
        p = PPL / f"{cfg}_mem.csv"
        if not p.exists() or not (PPL / f"{cfg}.json").exists():
            continue
        df = pd.read_csv(p)
        meta = json.loads((PPL / f"{cfg}.json").read_text())
        kvb, base = meta["kv_bytes_per_token"], meta["weights_mib"]
        data[cfg] = (df, meta)
        ax.plot(df.tokens, df.allocated_mib / 1024, color=COLOR[cfg])
        if cfg == "sink0_w1024":
            continue  # identical to the sink line (same cache size): one shared label below
        text = (LABEL[cfg] + (f"  (OOM at {meta['oom_at']:,})" if meta.get("oom_at") else "")
                if cfg != "sink4_w1020" else "4 sinks + 1,020 and window only (identical)")
        label_end(ax, df.tokens.iloc[-1], df.allocated_mib.iloc[-1] / 1024, text, COLOR[cfg])
    if kvb:
        xs = [0, 65536]
        ax.plot(xs, [(base + x * kvb / 2**20) / 1024 for x in xs], color=INK2, lw=1, ls="--")
        ax.text(30000, (base + 30000 * kvb / 2**20) / 1024 + 0.15, "weights + 112 KiB/token (theory) ",
                color=INK2, fontsize=8, ha="right", va="bottom")
        total = json.loads((PPL / "full.json").read_text())["total_vram_mib"] if (PPL / "full.json").exists() else 8188
        ax.axhline(total / 1024, color=INK2, lw=1, ls=":")
        ax.text(0, total / 1024, "  8 GB card", color=INK2, fontsize=8, va="bottom")
    ax.set_xlabel("tokens processed")
    ax.set_ylabel("allocated GPU memory (GiB)")
    ax.set_title("Memory: full cache grows 112 KiB/token, sinks stay flat")
    ax.set_xlim(0, 65536 * 1.25)
    fig.tight_layout()
    fig.savefig(RES / "memory_vs_tokens.png", dpi=150)
    plt.close(fig)
    if data:
        for t in [1024, 4096, 16384, 32768, 40960, 49152, 65536]:
            cells = []
            for cfg in ["full", "sink0_w1024", "sink4_w1020"]:
                if cfg in data:
                    df = data[cfg][0]
                    row = df[df.tokens == t]
                    cells.append(f"{row.allocated_mib.iloc[0]:,.0f}" if len(row) else "–")
                else:
                    cells.append("–")
            rows.append(f"| {t:,} | " + " | ".join(cells) + " |")
        tables.append("### Experiment 2: allocated memory (MiB) after N tokens\n\n" + "\n".join(rows))
        if "full" in data:
            df = data["full"][0]
            a, b = df.iloc[len(df) // 4], df.iloc[-1]
            slope = (b.allocated_mib - a.allocated_mib) * 2**20 / (b.tokens - a.tokens)
            tables.append(f"Measured full-cache growth: **{slope / 1024:.1f} KiB/token** "
                          f"(theory {kvb / 1024:.1f} KiB/token).")


# ------------------------------------------------------------------ experiment 3: latency
def plot_latency():
    p = RES / "latency_qwen3-0.6b.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    s = pd.DataFrame(d["summary"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)
    for ax, key, title in [(axes[0], "gpu_ms", "GPU time per token (kernel time)"),
                           (axes[1], "eager_ms", "Wall-clock per token, eager HF")]:
        for cfg, color in [("full", BLUE), ("sink4_w1020", AQUA)]:
            r = s[s.config == cfg]
            ax.plot(r.position, r[key], "o-", color=color, ms=8)
            lo = [x[0] for x in r[key + "_range"]]
            hi = [x[1] for x in r[key + "_range"]]
            ax.fill_between(r.position, lo, hi, color=color, alpha=0.15, lw=0)
            label_end(ax, r.position.iloc[-1], r[key].iloc[-1], LABEL[cfg], color)
        if key == "eager_ms" and "graph_ms" in s:
            r = s[s.config == "sink4_w1020"]
            ax.plot(r.position, r.graph_ms, "o--", color=AQUA, ms=8)
            label_end(ax, r.position.iloc[-1], r.graph_ms.iloc[-1], "sinks, CUDA graph", AQUA)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1024, 2048, 4096, 8192, 16384, 32768], ["1K", "2K", "4K", "8K", "16K", "32K"])
        ax.set_xlim(800, 32768 * 4)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("ms / token")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(RES / "latency_vs_length.png", dpi=150)
    plt.close(fig)

    rows = ["| config | context | GPU ms/token | range | linear | attention | cache copies | other | eager ms | graph ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in s.iterrows():
        b = r.breakdown_ms
        g = f"{r.graph_ms:.1f}" if "graph_ms" in r and not pd.isna(r.get("graph_ms")) else "–"
        rows.append(f"| {LABEL[r.config]} | {r.position:,} | {r.gpu_ms:.2f} | {r.gpu_ms_range[0]:.2f}–{r.gpu_ms_range[1]:.2f} "
                    f"| {b.get('linear', 0):.2f} | {b.get('attention', 0):.2f} | {b.get('cache_copy', 0):.2f} "
                    f"| {b.get('other', 0):.2f} | {r.eager_ms:.1f} | {g} |")
    tables.append(f"### Experiment 3: decode latency ({d['rounds']} rounds, medians)\n\n" + "\n".join(rows))


# ------------------------------------------------------------------ experiment 4: chat
def plot_chat():
    runs = {}
    for cfg in ["full", "sink0_w1024", "sink4_w1020"]:
        p = RES / "chat" / f"qwen3-0.6b_{cfg}.json"
        if p.exists():
            runs[cfg] = json.loads(p.read_text(encoding="utf-8"))["turns"]
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for cfg, turns in runs.items():
        t = [x["turn"] for x in turns]
        axes[0].plot(t, [x["alloc_mib"] for x in turns], color=COLOR[cfg])
        if cfg != "sink0_w1024":
            label_end(axes[0], t[-1], turns[-1]["alloc_mib"],
                      "sinks and window only" if cfg == "sink4_w1020" else LABEL[cfg], COLOR[cfg])
        axes[1].plot(t, [x["cache_len"] for x in turns], color=COLOR[cfg])
        if cfg != "sink0_w1024":
            label_end(axes[1], t[-1], turns[-1]["cache_len"],
                      "sinks and window only" if cfg == "sink4_w1020" else LABEL[cfg], COLOR[cfg])
    axes[0].set_ylabel("allocated GPU memory (MiB)")
    axes[1].set_ylabel("tokens in cache")
    for ax in axes:
        ax.set_xlabel("conversation turn")
        ax.set_xlim(0, 75)
    axes[0].set_title("Memory over a 50-turn chat")
    axes[1].set_title("Cache length over a 50-turn chat")
    fig.tight_layout()
    fig.savefig(RES / "chat.png", dpi=150)
    plt.close(fig)

    rows = ["| config | tokens seen | probe turn 3: answer | log p(4417) | probe turn 50: answer | log p(4417) | "
            "replies ended cleanly | mean distinct-2 |", "|---|---:|---|---:|---|---:|---:|---:|"]
    for cfg, turns in runs.items():
        probes = [x for x in turns if "logp_4417" in x]
        normal = [x for x in turns if "logp_4417" not in x]
        cells = []
        for pr in probes[:2]:
            cells += [f"`{pr['reply'].strip()[:40]}`", f"{pr['logp_4417']:.2f}"]
        ended = sum(x["ended"] for x in normal) / len(normal)
        d2 = sum(x["distinct2"] for x in normal) / len(normal)
        rows.append(f"| {LABEL[cfg]} | {turns[-1]['tokens_seen']:,} | " + " | ".join(cells) +
                    f" | {ended:.0%} | {d2:.3f} |")
    tables.append("### Experiment 4: long chat and planted-fact recall\n\n" + "\n".join(rows))


# ------------------------------------------------------------------ second model
def table_llama():
    d = RES / "ppl" / "llama3.2-1b"
    runs = {c: load(c, d) for c in ["full", "sink0_w1024", "sink4_w1020"]}
    runs = {c: r for c, r in runs.items() if r}
    if not runs:
        return
    rows = ["| config | 0–1K | 1K–2K | 2K–16K |", "|---|---:|---:|---:|"]
    for cfg, (df, _) in runs.items():
        rows.append(f"| {LABEL[cfg]} | " + " | ".join(f"{math.exp(mean_nll(df, lo, hi)):.2f}"
                                                   for lo, hi in [(0, 1024), (1024, 2048), (2048, 16384)]) + " |")
    tables.append("### Second model: Llama-3.2-1B perplexity (16K tokens)\n\n" + "\n".join(rows))


if __name__ == "__main__":
    for f in (plot_main, plot_ablations, plot_memory, plot_latency, plot_chat, table_llama):
        f()
    (RES / "tables.md").write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    print("\n\n".join(tables))
