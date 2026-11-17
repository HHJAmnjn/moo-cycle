import os, sys, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import pchip_interpolate
from moomoo import OpenQuoteContext, KLType, AuType, RET_OK
from predict_single_stock import ASSETS_DB, BASE_DIR

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False

# 19年时空周期分形标准锚点 (已对齐 2026 Q4 黄金坑/跨年峰 与 2027 全年去杠杆路径)
ANCHORS_1Y = {
    "QQQ": [
        ("2026-10-14", 0.0), ("2026-10-21", -3.5), ("2026-10-31", 2.3),
        ("2026-11-06", -3.0), ("2026-11-17", -7.2), ("2026-11-26", -10.7),
        ("2026-12-10", -2.5), ("2026-12-31", -8.0),
        ("2027-03-16", -19.0), ("2027-05-19", -6.0), ("2027-07-15", -22.0),
        ("2027-09-15", -30.0), ("2027-11-17", -43.0)
    ],
    "SOXX": [
        ("2026-10-14", 0.0), ("2026-10-21", -6.5), ("2026-10-31", 0.2),
        ("2026-11-06", -8.0), ("2026-11-17", -11.5), ("2026-11-26", -16.8),
        ("2026-12-10", -4.5), ("2026-12-31", -15.0),
        ("2027-03-16", -26.0), ("2027-05-19", -13.0), ("2027-07-15", -32.0),
        ("2027-09-15", -42.0), ("2027-11-17", -55.0)
    ],
    "SPY": [
        ("2026-10-14", 0.0), ("2026-10-21", -3.0), ("2026-10-31", 0.5),
        ("2026-11-06", -2.5), ("2026-11-17", -5.8), ("2026-11-26", -9.5),
        ("2026-12-10", -2.0), ("2026-12-31", -6.5),
        ("2027-03-16", -15.0), ("2027-05-19", -4.5), ("2027-07-15", -18.0),
        ("2027-09-15", -24.0), ("2027-11-17", -35.0)
    ],
    "GLD": [
        ("2026-10-14", 0.0), ("2026-10-31", 3.8), ("2026-11-17", 4.8), ("2026-11-26", 6.0),
        ("2026-12-10", 6.8), ("2026-12-31", 7.5), ("2027-03-16", 12.0), ("2027-05-19", 9.5),
        ("2027-07-15", 16.0), ("2027-09-15", 21.0), ("2027-11-17", 26.5)
    ]
}

def predict_1year(symbol: str, custom_price: float = None):
    symbol = symbol.upper()
    meta = ASSETS_DB.get(symbol, {"name": symbol, "base": "QQQ", "beta": 1.20, "noise": 0.15})
    base_sym = meta["base"] if meta["base"] in ANCHORS_1Y else "QQQ"
    beta = meta["beta"]
    drag = meta.get("drag", 0.0)
    name = meta["name"]

    today = datetime.date.today()
    end_dt = today + datetime.timedelta(days=365)

    df_kline = pd.DataFrame()
    realtime_price = None
    ctx = None
    try:
        ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
        ret, data, _ = ctx.request_history_kline(f"US.{symbol}", start="2026-08-01", end=today.strftime("%Y-%m-%d"), ktype=KLType.K_DAY, autype=AuType.QFQ, max_count=200)
        if ret == RET_OK and isinstance(data, pd.DataFrame) and not data.empty:
            df_kline = data
            df_kline["time_key"] = pd.to_datetime(df_kline["time_key"])
            df_kline.sort_values("time_key", inplace=True)
        ret_s, snap = ctx.get_market_snapshot([f"US.{symbol}"])
        if ret_s == RET_OK and not snap.empty:
            for col in ["overnight_price", "pre_price", "after_price", "last_price"]:
                if col in snap.columns and pd.notna(snap.iloc[0][col]) and float(snap.iloc[0][col]) > 0:
                    realtime_price = float(snap.iloc[0][col])
                    break
    except Exception:
        pass
    finally:
        if ctx: ctx.close()

    last_price = custom_price if custom_price else (realtime_price if realtime_price else (float(df_kline.iloc[-1]["close"]) if not df_kline.empty else 50.00))

    raw_anchors = ANCHORS_1Y[base_sym]
    a_dates = [datetime.date.fromisoformat(d) for d, _ in raw_anchors]
    a_vals = [v for _, v in raw_anchors]

    all_bdays = pd.bdate_range(a_dates[0], end_dt)
    a_days = [(d - a_dates[0]).days for d in a_dates]
    x_days = np.array([(d.date() - a_dates[0]).days for d in all_bdays])
    base_pct = pchip_interpolate(a_days, a_vals, x_days)

    # 修复双重 Beta 放大 Bug：SOXX 锚点本身已含 1.35x 弹性，3x 杠杆 ETF 直接按 3.0 倍日度复利与适度震荡损耗计算
    if abs(beta) >= 2.5:
        lev = 3.0 if beta > 0 else -3.0
        unlev = 1.0 + base_pct / 100.0
        d_ret = np.zeros(len(unlev))
        d_ret[1:] = unlev[1:] / unlev[:-1] - 1.0
        lev_ret = d_ret * lev - (drag * 0.35 / 100.0)
        lev_ret[0] = 0.0
        stock_pct = (np.cumprod(1.0 + lev_ret) - 1.0) * 100.0
    else:
        indiv_beta = 1.0 if symbol == base_sym else (beta / 1.35 if base_sym == "SOXX" else beta)
        stock_pct = base_pct * indiv_beta

    full_series = pd.Series(stock_pct, index=all_bdays)
    fut_series = full_series[full_series.index.date >= today]

    pct_at_now = float(fut_series.iloc[0])
    scale_factors = (1.0 + fut_series.values / 100.0) / (1.0 + pct_at_now / 100.0)
    proj_prices = last_price * scale_factors
    proj_pcts = (proj_prices / last_price - 1.0) * 100.0

    print("\n" + "="*82)
    print(f" 🧭 【未来一年 (365天) 19年时空周期全景推演 (已与网页罗盘100%对齐)】 {symbol} - {name}")
    print(f" 锚定日期: {today} | 当前基准价: ${last_price:.2f} | 推演终点: {end_dt}")
    print("="*82)
    checkpoints = [
        ("2026-11-26", "11/26 黄金坑底", -28),
        ("2026-12-10", "12/10 跨年反弹峰", 14),
        ("2027-03-16", "27/03 春季去杠杆底", -28),
        ("2027-05-19", "27/05 春季反抽次顶", 14),
        ("2027-07-15", "27/07 夏季主跌中继", -26),
        ("2027-11-15", "27/11 流动性踩踏底", 14)
    ]
    for cp, desc, _ in checkpoints:
        cp_dt = pd.to_datetime(cp)
        idx = fut_series.index.get_indexer([cp_dt], method="nearest")[0]
        dt_actual = fut_series.index[idx].strftime("%Y-%m-%d")
        print(f"  ▶ {dt_actual} ({desc}) : 预测目标 ${proj_prices[idx]:.2f} ({proj_pcts[idx]:+.2f}%)")
    print("="*82 + "\n")

    fig, ax = plt.subplots(figsize=(15, 7.2), dpi=180)
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")
    if not df_kline.empty:
        ax.plot(df_kline["time_key"].dt.date, df_kline["close"], color="#0F172A", lw=2.2, label=f"历史实盘K线 (截至 {today})")
    ax.plot(fut_series.index.date, proj_prices, color="#2563EB", lw=2.6, ls="--", label=f"未来一年19年周期分形推演轨 ({today} ➔ {end_dt})")
    ax.axvline(today, color="#DC2626", ls="-.", lw=1.8)
    ax.scatter([today], [last_price], color="#DC2626", s=65, zorder=6)
    ax.annotate(f"当前起点: ${last_price:.2f}", (today, last_price), textcoords="offset points", xytext=(-55, 18),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor="#DC2626", lw=1),
                color="#DC2626", fontsize=9, fontweight="bold")

    for cp, desc, dy in checkpoints:
        cp_dt = pd.to_datetime(cp)
        idx = fut_series.index.get_indexer([cp_dt], method="nearest")[0]
        d_plot = fut_series.index[idx].date()
        p_plot = proj_prices[idx]
        col = "#0284C7" if dy < 0 else "#D97706"
        ax.scatter([d_plot], [p_plot], color=col, s=50, zorder=5)
        ax.annotate(f"{d_plot.strftime('%y/%m/%d')}\n${p_plot:.1f}", (d_plot, p_plot),
                    textcoords="offset points", xytext=(0, dy), ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor=col, alpha=0.9),
                    fontsize=8.5, fontweight="bold", color=col)

    ax.set_title(f"【19年时空周期分形 · 未来一年走势推演 (与网页第四层罗盘同源对齐)】{symbol} ({name})", fontsize=13, fontweight="bold", loc="left", pad=12)
    ax.set_ylabel("价格 (USD $)", fontsize=11)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper right", fontsize=9.5)
    plt.tight_layout()

    out_png = os.path.join(BASE_DIR, f"1year_projection_{symbol}.png")
    plt.savefig(out_png, dpi=200)
    os.system(f"open '{out_png}'")

if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "SOXL"
    c_p = float(sys.argv[2]) if len(sys.argv) > 2 else None
    predict_1year(sym, c_p)
