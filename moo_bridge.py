import os, time, json, math, datetime, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import pandas as pd

try:
    from zoneinfo import ZoneInfo
    ET_TZ = ZoneInfo("America/New_York")
except Exception:
    ET_TZ = datetime.timezone(datetime.timedelta(hours=-5))

try:
    from moomoo import OpenQuoteContext, SubType, RET_OK, SysConfig, KLType, AuType
except ImportError:
    from futu import OpenQuoteContext, SubType, RET_OK, SysConfig, KLType, AuType

SYMBOL_MAP = {
    "SOXL": "US.SOXL",
    "SOXS": "US.SOXS",
    "TQQQ": "US.TQQQ",
    "SQQQ": "US.SQQQ",
    "QQQ":  "US.QQQ",
    "SPY":  "US.SPY",
    "SOXX": "US.SOXX"
}
ALL_CODES = list(SYMBOL_MAP.values())
IS_3X_SET = {"SOXL", "SOXS", "TQQQ", "SQQQ"}

AUM_BENCHMARK_USD = {
    "SOXL": 14.404e9,
    "SOXX": 34.507e9,
    "TQQQ": 24.0e9,
    "SQQQ": 3.5e9,
    "SOXS": 1.6e9,
    "QQQ":  310.0e9,
    "SPY":  600.0e9
}

LOW_52W_FALLBACK = {
    "SOXL": 29.82,
    "SOXX": 123.15,
    "TQQQ": 38.00,
    "QQQ":  540.00,
    "SPY":  580.00,
    "SOXS": 15.00,
    "SQQQ": 18.00
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(BASE_DIR, "live_price.json")

quote_ctx = None
ctx_lock = threading.Lock()
VOL_CACHE = {}
GEX_CACHE = {}
SEESAW_CACHE = {"ready": False}

PIN_CFG = {
    "SOXX": {"bias_th": -0.100, "drop_th": -0.038, "deep_drop": -0.048, "bounce_th": 0.65},
    "QQQ":  {"bias_th": -0.055, "drop_th": -0.023, "deep_drop": -0.030, "bounce_th": 0.38},
    "SPY":  {"bias_th": -0.040, "drop_th": -0.017, "deep_drop": -0.023, "bounce_th": 0.28},
}

def _clean_num(val, ndigits=2, default=0.0):
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return default
        return round(v, ndigits)
    except Exception:
        return default

def _extract_pin_stats(kdf):
    tmp = kdf[["time_key", "high", "low", "close"]].copy()
    tmp["d"] = tmp["time_key"].astype(str).str.slice(0, 10)
    tmp.drop_duplicates(subset=["d"], inplace=True)
    tmp.sort_values("d", inplace=True)
    hi = tmp["high"].astype(float)
    lo = tmp["low"].astype(float)
    cl = tmp["close"].astype(float)
    prev_c = cl.shift(1)
    tr = pd.concat([(hi - lo), (hi - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    ma20 = float(cl.tail(20).mean())
    atr14 = float(tr.tail(14).mean())
    low5 = float(lo.tail(5).min())
    return {
        "ma20": ma20,
        "atr14": atr14,
        "low5": low5,
        "last_close": float(cl.iloc[-1]),
        "last_low": float(lo.iloc[-1]),
    }

def evaluate_pin_targets(quotes_map=None, splv_ctx=None):
    pin_stats = SEESAW_CACHE.get("pin_stats", {})
    splv_surging = bool(splv_ctx and splv_ctx.get("is_live_valid") and safe_float(splv_ctx.get("splv_day_chg")) >= 0.60)
    out = {}
    for sym in ("SOXX", "QQQ", "SPY"):
        st = pin_stats.get(sym)
        if not st:
            continue
        cfg = PIN_CFG[sym]
        q = (quotes_map or {}).get(sym, {})
        cur_p = safe_float(q.get("price"), st["last_close"])
        base_c = safe_float(q.get("reg_close"), st["last_close"])
        if cur_p <= 0:
            cur_p = st["last_close"]
        if base_c <= 0:
            base_c = st["last_close"]

        day_hi = safe_float(q.get("high"), 0.0)
        day_lo = safe_float(q.get("low"), 0.0)
        day_vwap = safe_float(q.get("vwap"), 0.0)

        ma20 = st["ma20"]
        atr14 = st["atr14"]
        low5 = st["low5"]
        base_bias = (base_c / max(1e-6, ma20)) - 1.0
        cur_bias_pct = _clean_num(((cur_p / max(1e-6, ma20)) - 1.0) * 100.0, 1)
        in_deep_pit = base_bias <= cfg["bias_th"]

        if in_deep_pit:
            rf_label = "右脚接针"
            rf_low = min(low5, base_c * 0.992)
            rf_high = max(base_c * 1.006, min(base_c + 0.35 * atr14, base_c * (1.0 - cfg["drop_th"] * 0.75)))
            if rf_low >= rf_high:
                rf_low = rf_high * 0.992
        else:
            rf_label = "急杀接针"
            rf_high = min(base_c - 0.95 * atr14, base_c * (1.0 + cfg["drop_th"] * 0.65))
            rf_low = min(rf_high * 0.988, base_c - 1.45 * atr14)

        ex_high = min(ma20 * (1.0 + cfg["bias_th"]), base_c * (1.0 + cfg["drop_th"]))
        if ex_high >= rf_low * 0.996:
            ex_high = rf_low * (1.0 + cfg["drop_th"] * 0.45)
        ex_low = min(ex_high * 0.982, base_c * (1.0 + cfg["deep_drop"]) - 0.25 * atr14)
        if ex_low >= ex_high:
            ex_low = ex_high * 0.985

        rf_low, rf_high = round(rf_low, 2), round(rf_high, 2)
        ex_low, ex_high = round(ex_low, 2), round(ex_high, 2)
        dist_rf_pct = _clean_num(((rf_high / max(1e-6, cur_p)) - 1.0) * 100.0, 1)

        # ===== V6.5.7 盘中止跌微观结构识别（区分 急跌飞刀 vs 插针收回/站稳） =====
        has_intra = (day_lo > 0 and day_hi >= day_lo and day_lo >= base_c * 0.70)
        bounce_pct = _clean_num(((cur_p / day_lo) - 1.0) * 100.0, 2) if has_intra else 0.0
        day_rng = (day_hi - day_lo) if has_intra else 0.0
        pos_in_rng = ((cur_p - day_lo) / day_rng) if (day_rng > 1e-6) else 0.0
        above_vwap = bool(day_vwap > 0 and cur_p >= day_vwap)
        b_th = cfg["bounce_th"]

        is_pin_rebound = has_intra and (bounce_pct >= b_th and pos_in_rng >= 0.30)
        is_knife_drop = (not above_vwap) and (not has_intra or (bounce_pct < b_th * 0.75 and pos_in_rng < 0.25))

        if cur_p <= ex_high:
            if is_knife_drop:
                status = "knife_ex"
                status_txt = "🔪 血筹急跌下破·防飞刀"
            elif is_pin_rebound or above_vwap:
                status = "hold_ex"
                status_txt = "✅ 血筹插针收回·可试底仓"
            else:
                status = "watch_ex"
                status_txt = "⏳ 血筹区震荡·待企稳"
        elif cur_p <= rf_high:
            if is_knife_drop:
                status = "knife_rf"
                status_txt = "🔪 急跌下破中·观望防飞刀"
            elif is_pin_rebound or above_vwap:
                if splv_surging:
                    status = "watch_rf"
                    status_txt = "⚠️ 避险压制·轻仓观望"
                else:
                    status = "hold_rf"
                    status_txt = "✅ 插针收回/站稳·可试底仓"
            else:
                status = "watch_rf"
                status_txt = "⏳ 入区震荡·待站稳VWAP"
        elif has_intra and day_lo <= rf_high * 1.002 and bounce_pct >= b_th:
            status = "reclaimed"
            status_txt = "✅ 已插针收回·右侧企稳"
        else:
            status = "above"
            status_txt = f"距一档 {dist_rf_pct:+.1f}%"

        out[sym] = {
            "price": round(cur_p, 2),
            "base_c": round(base_c, 2),
            "day_low": round(day_lo, 2) if has_intra else 0.0,
            "day_vwap": round(day_vwap, 2) if day_vwap > 0 else 0.0,
            "bounce_pct": bounce_pct,
            "above_vwap": above_vwap,
            "bias20_pct": cur_bias_pct,
            "rf_label": rf_label,
            "rf_low": rf_low,
            "rf_high": rf_high,
            "ex_low": ex_low,
            "ex_high": ex_high,
            "status": status,
            "status_txt": status_txt,
        }
    return out

def _compute_seesaw_metrics(dates, soxx_closes, splv_closes, soxx_volumes, live_info=None):
    if len(dates) < 25:
        return {"ready": False, "level": "loading", "title": "SPLV/SOXX 跷跷板样本不足", "detail": "等待历史K线加载..."}

    s_list = list(soxx_closes)
    p_list = list(splv_closes)
    v_list = list(soxx_volumes)
    d_list = list(dates)
    is_live_valid = bool(live_info and live_info.get("is_live_valid"))

    if is_live_valid:
        today_et_s = live_info["today_et"]
        live_soxx_chg = live_info["soxx_day_chg"]
        live_splv_chg = live_info["splv_day_chg"]
        live_proj_vol = float(live_info.get("proj_soxx_vol", 0.0))
        if d_list[-1] == today_et_s and len(s_list) >= 2:
            s_list[-1] = s_list[-2] * (1.0 + live_soxx_chg / 100.0)
            p_list[-1] = p_list[-2] * (1.0 + live_splv_chg / 100.0)
            v_list[-1] = live_proj_vol if live_proj_vol > 0 else 0.0
        else:
            d_list.append(today_et_s)
            s_list.append(s_list[-1] * (1.0 + live_soxx_chg / 100.0))
            p_list.append(p_list[-1] * (1.0 + live_splv_chg / 100.0))
            v_list.append(live_proj_vol if live_proj_vol > 0 else 0.0)

    s_soxx = pd.Series(s_list, dtype=float)
    s_splv = pd.Series(p_list, dtype=float)
    s_vol  = pd.Series(v_list, dtype=float)
    ret_soxx_s = s_soxx.pct_change() * 100.0
    ret_splv_s = s_splv.pct_change() * 100.0

    ratio_s = s_splv / s_soxx.replace(0, float("nan"))
    ratio_ma5 = ratio_s.rolling(5).mean()
    ratio_ma20 = ratio_s.rolling(20).mean()
    ratio_std20 = ratio_s.rolling(20).std()
    ratio_z20_s = (ratio_s - ratio_ma20) / ratio_std20.replace(0, float("nan"))
    corr20_s = ret_soxx_s.rolling(20).corr(ret_splv_s)

    soxx_ma5 = s_soxx.rolling(5).mean()
    soxx_ma10 = s_soxx.rolling(10).mean()
    soxx_ma20 = s_soxx.rolling(20).mean()
    soxx_ma200 = s_soxx.rolling(200).mean() if len(s_soxx) >= 200 else s_soxx.expanding().mean()
    splv_ma10 = s_splv.rolling(10).mean()

    bias20_s = (s_soxx / soxx_ma20.replace(0, float("nan")) - 1.0) * 100.0
    vol_ma20_s = s_vol.rolling(20).mean()
    vol_ratio_s = s_vol / vol_ma20_s.replace(0, float("nan"))

    c_soxx = float(s_soxx.iloc[-1])
    c_splv = float(s_splv.iloc[-1])
    r_soxx = _clean_num(ret_soxx_s.iloc[-1], 2)
    r_splv = _clean_num(ret_splv_s.iloc[-1], 2)
    r_now = float(ratio_s.iloc[-1])
    r_prev = float(ratio_s.iloc[-2])
    r_m5 = float(ratio_ma5.iloc[-1])
    r_m5_prev = float(ratio_ma5.iloc[-2])
    z20 = _clean_num(ratio_z20_s.iloc[-1], 2)
    corr20 = _clean_num(corr20_s.iloc[-1], 2)
    bias20 = _clean_num(bias20_s.iloc[-1], 2)
    vol_ratio = _clean_num(vol_ratio_s.iloc[-1], 2)
    s_m5 = float(soxx_ma5.iloc[-1])
    s_m10 = float(soxx_ma10.iloc[-1])
    s_m200 = float(soxx_ma200.iloc[-1])
    p_m10 = float(splv_ma10.iloc[-1])

    is_bear_ma200 = c_soxx < s_m200
    regime_tag = "🐻熊市<MA200" if is_bear_ma200 else "🐂牛市≥MA200"

    l4_series = (ret_soxx_s <= -4.5) & (bias20_s <= -10.0) & (ratio_z20_s >= 1.8) & (vol_ratio_s >= 1.5)
    l2_series = (bias20_s <= -10.0) & (ret_soxx_s <= -3.5) & (vol_ratio_s >= 1.4)
    dk_series = (
        ((ret_soxx_s <= -2.5) & (ret_splv_s <= -0.5))
        | ((corr20_s > 0.0) & (ret_soxx_s <= -3.0) & (ret_splv_s <= -0.3))
    )

    sig_left_l4 = bool(l4_series.iloc[-1])
    sig_left_l2 = bool(l2_series.iloc[-1])
    sig_double_kill = bool(dk_series.iloc[-1])
    recent_shock_5d = bool((dk_series | l4_series | l2_series).tail(5).any())
    recent_high_z15 = bool(ratio_z20_s.tail(15).max() >= 1.2)

    sig_trend_strong = (z20 >= 1.5) and (r_now > r_m5) and (c_splv >= p_m10) and (c_soxx < s_m10)
    sig_trend_watch = (1.2 <= z20 < 1.5) and (r_now > r_m5) and (c_splv >= p_m10) and (c_soxx < s_m10)
    sig_day_div = (corr20 < -0.15) and (r_splv >= 0.4) and (r_soxx <= -1.5)

    sig_rev_dk_confirm = recent_shock_5d and (not sig_double_kill) and (not sig_left_l4) and (not sig_left_l2) and (r_soxx >= 1.5) and ((r_soxx - r_splv) >= 1.5)
    sig_rev_trend_confirm = recent_high_z15 and (r_now < r_m5) and (r_prev < r_m5_prev) and (c_splv < p_m10) and (c_soxx > s_m5)

    if sig_left_l4:
        level = "left_l4"
        title = "🩸 L4 极限左侧血筹接针 (15年T+20胜率87.5% · 分批底仓)"
        detail = f"SOXX({r_soxx:+.2f}%) | 20日乖离{bias20:+.1f}% | 放量{vol_ratio:.2f}x | Z={z20:+.2f}σ；历史T+1均浮亏仅-0.88%/T+20中位+6.98%，宜左侧试底仓、防T+3回踩"
    elif sig_left_l2:
        level = "left_l2"
        title = "🩸 L2 左侧放量深跌接针 (15年T+1/T+10胜率75.0%)"
        detail = f"SOXX({r_soxx:+.2f}%) | 20日乖离{bias20:+.1f}% | 放量{vol_ratio:.2f}x；15年五周期全正超额(T+1浮亏-0.92%/T+10中位+6.78%)，可分批左侧建底仓"
    elif sig_double_kill:
        level = "double_kill"
        title = "⚡ 流动性双杀出清相 (未达L2/L4血筹极值 · 禁追空/不盲接)"
        detail = f"SOXX({r_soxx:+.2f}%)与SPLV({r_splv:+.2f}%)同步下杀(乖离{bias20:+.1f}%/量比{vol_ratio:.2f}x)；严禁追空，等待5日内SOXX≥+1.5%企稳确认"
    elif sig_trend_strong:
        level = "trend_warn"
        title = f"🚨 强波段大跌预警 (Z≥1.5σ · {regime_tag})"
        detail = (
            f"比值Z={z20:+.2f}σ站上5日线；15年样本内外双期负收益(样本外T+5跌胜率59.5%/捕获43.2%"
            + ("，当前熊市<MA200风险加倍" if is_bear_ma200 else "")
            + ")"
        )
    elif sig_day_div:
        level = "day_div"
        title = "⚠️ 单日强跷跷板背离预警 (高回撤警报)"
        detail = f"强跷跷板期(相关{corr20:+.2f})SPLV({r_splv:+.2f}%)强于SOXX({r_soxx:+.2f}%)；15年回测T+5跌超-5%捕获率46.9%(近3倍基准)"
    elif sig_rev_dk_confirm:
        level = "rev_buy"
        title = "🟢 血筹/双杀后企稳反攻确认 (15年T+20胜率73.5%)"
        detail = f"近5日极值出清后SOXX强势反攻({r_soxx:+.2f}%)且跑赢SPLV({r_soxx - r_splv:+.2f}%)；15年回测T+20超额+9.1%/均涨+3.61%"
    elif sig_rev_trend_confirm:
        level = "rev_buy"
        title = "🟢 避险真撤退 · 波段转多确认 (15年T+10胜率69.1%)"
        detail = f"高位比值连续2日破5日线且SPLV跌破10日线、SOXX站稳5日线；15年回测T+10超额+7.9%/均涨+1.81%"
    elif sig_trend_watch:
        level = "watch_warn"
        title = f"🟠 跷跷板初级警戒 (1.2≤Z<1.5σ · {regime_tag})"
        detail = f"比值Z={z20:+.2f}σ进入警戒区但未达1.5σ强破位线，停止追高，紧盯是否突破1.5σ触发强预警"
    else:
        level = "neutral"
        if corr20 <= -0.15:
            title = f"🛡️ 跷跷板监控常态 ({regime_tag})"
        else:
            title = f"⚪ 跷跷板静默观察 ({regime_tag})"
        detail = f"Z={z20:+.2f}σ | 20D相关={corr20:+.2f} | 乖离={bias20:+.1f}% | 强预警≥1.5σ，左侧血筹线:乖离≤-10%+放量≥1.4x"

    disp_splv_p = _clean_num(live_info.get("splv_price", c_splv), 2) if live_info else _clean_num(c_splv, 2)
    disp_turn_m = _clean_num(live_info.get("splv_turnover_m", 0.0), 1) if live_info else 0.0
    mode_tag = f"正盘实时(${disp_turn_m:.1f}M)" if is_live_valid else f"锁上日正盘({d_list[-1][5:]})·防盘前假单"

    return {
        "ready": True,
        "level": level,
        "title": title,
        "detail": detail,
        "ratio_z20": z20,
        "corr20": corr20,
        "bias20": bias20,
        "vol_ratio": vol_ratio,
        "is_bear_ma200": is_bear_ma200,
        "splv_price": disp_splv_p,
        "splv_chg_pct": r_splv,
        "soxx_chg_pct": r_soxx,
        "splv_turnover_m": disp_turn_m,
        "is_live_valid": is_live_valid,
        "mode_tag": mode_tag,
        "as_of": d_list[-1]
    }

def refresh_splv_seesaw_cache():
    global SEESAW_CACHE
    end_d = datetime.date.today().strftime("%Y-%m-%d")
    start_long = (datetime.date.today() - datetime.timedelta(days=340)).strftime("%Y-%m-%d")
    start_60 = (datetime.date.today() - datetime.timedelta(days=95)).strftime("%Y-%m-%d")
    dfs = {}
    pin_stats = {}

    for sym in ("SOXX", "SPLV"):
        with ctx_lock:
            ctx = get_ctx()
            ret_k, kdf, _ = ctx.request_history_kline(
                f"US.{sym}", start=start_long, end=end_d, ktype=KLType.K_DAY, autype=AuType.QFQ, max_count=260
            )
        if ret_k != RET_OK or kdf is None or kdf.empty:
            return
        if sym == "SOXX":
            pin_stats["SOXX"] = _extract_pin_stats(kdf)
            tmp = kdf[["time_key", "close", "volume"]].copy()
            tmp["d"] = tmp["time_key"].astype(str).str.slice(0, 10)
            tmp.drop_duplicates(subset=["d"], inplace=True)
            tmp.rename(columns={"volume": "vol_soxx"}, inplace=True)
            dfs[sym] = tmp[["d", "close", "vol_soxx"]]
        else:
            tmp = kdf[["time_key", "close"]].copy()
            tmp["d"] = tmp["time_key"].astype(str).str.slice(0, 10)
            tmp.drop_duplicates(subset=["d"], inplace=True)
            dfs[sym] = tmp[["d", "close"]]
        time.sleep(0.12)

    for sym in ("QQQ", "SPY"):
        try:
            with ctx_lock:
                ctx = get_ctx()
                ret_k, kdf, _ = ctx.request_history_kline(
                    f"US.{sym}", start=start_60, end=end_d, ktype=KLType.K_DAY, autype=AuType.QFQ, max_count=65
                )
            if ret_k == RET_OK and kdf is not None and len(kdf) >= 20:
                pin_stats[sym] = _extract_pin_stats(kdf)
        except Exception:
            pass
        time.sleep(0.12)

    mg = pd.merge(dfs["SOXX"], dfs["SPLV"], on="d", how="inner", suffixes=("_soxx", "_splv")).sort_values("d")
    if len(mg) < 25:
        return
    dates = mg["d"].tolist()
    soxx_c = [float(x) for x in mg["close_soxx"].tolist()]
    splv_c = [float(x) for x in mg["close_splv"].tolist()]
    soxx_v = [float(x) for x in mg["vol_soxx"].tolist()]
    eod_state = _compute_seesaw_metrics(dates, soxx_c, splv_c, soxx_v, live_info=None)
    SEESAW_CACHE = {
        "ready": True,
        "dates": dates,
        "soxx_closes": soxx_c,
        "splv_closes": splv_c,
        "soxx_volumes": soxx_v,
        "pin_stats": pin_stats,
        "eod": eod_state,
        "updated_at": time.strftime("%H:%M:%S")
    }
    pins = evaluate_pin_targets(None)
    print(f"🛡️ [V6.5.6 血筹+三标的接针价已刷新] Z={eod_state['ratio_z20']:+.2f}σ | {eod_state['title']}")
    for s in ("SOXX", "QQQ", "SPY"):
        if s in pins:
            p = pins[s]
            print(f"   🎯 {s:<4s} | 前收:${p['base_c']:6.2f} | {p['rf_label']}: ${p['rf_low']:6.2f}~${p['rf_high']:6.2f} | 极限血筹针: ${p['ex_low']:6.2f}~${p['ex_high']:6.2f}")

def evaluate_live_seesaw(splv_row, soxx_row, et_now, weekend_closed, quotes_map=None):
    if not SEESAW_CACHE.get("ready"):
        return {"ready": False, "level": "loading", "title": "SPLV/SOXX 跷跷板雷达初始化中...", "detail": "后台正在拉取260日前复权历史序列", "ratio_z20": 0.0, "corr20": 0.0, "bias20": 0.0, "vol_ratio": 0.0, "splv_price": 0.0, "splv_chg_pct": 0.0, "mode_tag": "初始化", "pin_targets": {}}
    et_min = et_now.hour * 60 + et_now.minute
    is_regular = (not weekend_closed) and (570 <= et_min <= 960)

    splv_reg_p = safe_float(splv_row.get("last_price")) if splv_row is not None else 0.0
    splv_prev_c = safe_float(splv_row.get("prev_close_price")) if splv_row is not None else 0.0
    splv_turn = safe_float(splv_row.get("turnover")) if splv_row is not None else 0.0

    soxx_reg_p = safe_float(soxx_row.get("last_price")) if soxx_row is not None else 0.0
    soxx_prev_c = safe_float(soxx_row.get("prev_close_price")) if soxx_row is not None else 0.0
    soxx_vol = safe_float(soxx_row.get("volume")) if soxx_row is not None else 0.0

    splv_day_chg = ((splv_reg_p - splv_prev_c) / splv_prev_c * 100.0) if (splv_reg_p > 0 and splv_prev_c > 0) else 0.0
    soxx_day_chg = ((soxx_reg_p - soxx_prev_c) / soxx_prev_c * 100.0) if (soxx_reg_p > 0 and soxx_prev_c > 0) else 0.0

    is_live_valid = is_regular and (splv_turn >= 5_000_000.0) and (splv_reg_p > 0) and (soxx_reg_p > 0)

    proj_soxx_vol = 0.0
    if is_live_valid and et_min >= 600 and soxx_vol > 0:
        t_frac = min(1.0, max(0.08, (et_min - 570) / 390.0))
        vol_pace = min(1.0, 0.42 * (t_frac ** 0.5) + 0.58 * t_frac)
        proj_soxx_vol = soxx_vol / vol_pace

    live_info = {
        "is_live_valid": is_live_valid,
        "today_et": et_now.strftime("%Y-%m-%d"),
        "splv_price": splv_reg_p if splv_reg_p > 0 else SEESAW_CACHE["splv_closes"][-1],
        "splv_day_chg": splv_day_chg,
        "soxx_day_chg": soxx_day_chg,
        "splv_turnover_m": splv_turn / 1e6,
        "proj_soxx_vol": proj_soxx_vol,
    }
    res_obj = _compute_seesaw_metrics(
        SEESAW_CACHE["dates"], SEESAW_CACHE["soxx_closes"], SEESAW_CACHE["splv_closes"], SEESAW_CACHE["soxx_volumes"],
        live_info=live_info
    )
    res_obj["pin_targets"] = evaluate_pin_targets(quotes_map, splv_ctx={"is_live_valid": is_live_valid, "splv_day_chg": splv_day_chg})
    return res_obj

def safe_float(val, default=0.0):
    try:
        if val is None or pd.isna(val) or str(val).strip() in ("", "N/A", "--", "nan"):
            return default
        return float(val)
    except Exception:
        return default

def get_ctx():
    global quote_ctx
    if quote_ctx is None:
        SysConfig.enable_proto_encrypt(False)
        quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
        quote_ctx.subscribe(ALL_CODES + ["US.SPLV"], [SubType.QUOTE], subscribe_push=False)
    return quote_ctx

def is_us_weekend_closed(et_now):
    wd = et_now.weekday()
    et_min = et_now.hour * 60 + et_now.minute
    if wd == 5:
        return True
    if wd == 6 and et_min < 1200:
        return True
    if wd == 4 and et_min >= 1200:
        return True
    return False

def calc_dynamic_decay_from_amp(amp_5d_pct, is_3x):
    if not is_3x:
        return 0.0
    if amp_5d_pct <= 6.0:
        return 0.125
    extra = min(0.225, ((amp_5d_pct - 6.0) / 14.0) * 0.225)
    return round(0.125 + extra, 4)

def bs_gamma_fallback(spot, strike, t_years, sigma):
    """当周末或接口未返回实时Gamma时，用Black-Scholes模型结合持仓量本地推导Gamma"""
    try:
        if spot <= 0 or strike <= 0 or t_years <= 0 or sigma <= 0:
            return 0.0
        sqrt_t = math.sqrt(t_years)
        d1 = (math.log(spot / strike) + (0.045 + 0.5 * sigma * sigma) * t_years) / (sigma * sqrt_t)
        pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
        return pdf / (spot * sigma * sqrt_t)
    except Exception:
        return 0.0

def refresh_vol_and_gex_worker():
    first_run = True
    while True:
        try:
            with ctx_lock:
                ctx = get_ctx()
                ret_s, snap_df = ctx.get_market_snapshot(ALL_CODES)
                spot_map = {}
                if ret_s == RET_OK and not snap_df.empty:
                    for _, r in snap_df.iterrows():
                        sym = str(r["code"]).replace("US.", "")
                        spot_map[sym] = safe_float(r.get("last_price"))

            end_d = datetime.date.today().strftime("%Y-%m-%d")
            start_d = (datetime.date.today() - datetime.timedelta(days=20)).strftime("%Y-%m-%d")
            for sym, code in SYMBOL_MAP.items():
                try:
                    with ctx_lock:
                        ret_k, kdf, _ = ctx.request_history_kline(
                            code, start=start_d, end=end_d, ktype=KLType.K_DAY, autype=AuType.QFQ, max_count=10
                        )
                    if ret_k == RET_OK and kdf is not None and len(kdf) >= 3:
                        tail5 = kdf.tail(5)
                        amps = []
                        for _, kr in tail5.iterrows():
                            hi, lo, cl = safe_float(kr["high"]), safe_float(kr["low"]), safe_float(kr["close"])
                            if lo > 0 and cl > 0:
                                amps.append(((hi - lo) / cl) * 100.0)
                        avg_amp = round(sum(amps) / max(1, len(amps)), 2)
                        VOL_CACHE[sym] = {
                            "amp_5d_pct": avg_amp,
                            "dyn_beta_decay": calc_dynamic_decay_from_amp(avg_amp, sym in IS_3X_SET),
                            "low_52w": round(min(LOW_52W_FALLBACK.get(sym, 9999), safe_float(kdf["low"].min(), 9999)), 2),
                            "updated_at": time.strftime("%H:%M:%S")
                        }
                except Exception:
                    pass
                time.sleep(0.2)

            try:
                refresh_splv_seesaw_cache()
            except Exception as _se:
                print(f"⚠️ SPLV/SOXX 跷跷板刷新异常: {_se}")

            # 自动拉取近期有效未过期期权链并计算 GEX（带完整报错诊断）
            today_et_date = datetime.datetime.now(ET_TZ).date()
            today_et_str = today_et_date.strftime("%Y-%m-%d")

            for sym in ("SOXL", "SOXX", "QQQ"):
                code = SYMBOL_MAP[sym]
                spot = spot_map.get(sym, 0.0)
                if spot <= 0:
                    continue
                try:
                    with ctx_lock:
                        ret_e, edf = ctx.get_option_expiration_date(code=code)
                    if ret_e != RET_OK or edf is None or edf.empty:
                        msg = str(edf)[:48] if ret_e != RET_OK else "无可用期权到期日"
                        GEX_CACHE[sym] = {"source": "opend_error", "reason": f"到期日获取失败: {msg}", "zero_gamma": 0, "put_wall": 0, "secondary_put_wall": 0, "call_wall": 0, "expiry": ""}
                        if first_run: print(f"⚠️ [{sym} 期权GEX] {GEX_CACHE[sym]['reason']}")
                        continue

                    # 过滤掉已过期合约（例如周六跳过周五11/20，自动取下周五11/27）
                    future_edf = edf[edf["strike_time"] >= today_et_str]
                    exp_date = str(future_edf.iloc[0]["strike_time"]) if not future_edf.empty else str(edf.iloc[-1]["strike_time"])
                    try:
                        exp_d_obj = datetime.datetime.strptime(exp_date[:10], "%Y-%m-%d").date()
                        dte = max(1, (exp_d_obj - today_et_date).days)
                    except Exception:
                        dte = 5
                    t_years = dte / 365.0

                    with ctx_lock:
                        ret_c, cdf = ctx.get_option_chain(code=code, start=exp_date, end=exp_date)
                    if ret_c != RET_OK or cdf is None or cdf.empty:
                        msg = str(cdf)[:48] if ret_c != RET_OK else "期权链为空"
                        GEX_CACHE[sym] = {"source": "opend_error", "reason": f"期权链获取失败: {msg}", "zero_gamma": 0, "put_wall": 0, "secondary_put_wall": 0, "call_wall": 0, "expiry": exp_date}
                        if first_run: print(f"⚠️ [{sym} 期权GEX] {GEX_CACHE[sym]['reason']}")
                        continue

                    sub = cdf[(cdf["strike_price"] >= spot * 0.72) & (cdf["strike_price"] <= spot * 1.28)]
                    opt_codes = sub["code"].tolist()[:100]
                    if not opt_codes:
                        continue

                    with ctx_lock:
                        ret_o, odf = ctx.get_market_snapshot(opt_codes)
                    if ret_o != RET_OK or odf is None or odf.empty:
                        msg = str(odf)[:52] if ret_o != RET_OK else "期权快照为空"
                        GEX_CACHE[sym] = {"source": "opend_no_perm", "reason": f"无期权API权限/限频: {msg}", "zero_gamma": 0, "put_wall": 0, "secondary_put_wall": 0, "call_wall": 0, "expiry": exp_date}
                        if first_run: print(f"⚠️ [{sym} 期权GEX] {GEX_CACHE[sym]['reason']}")
                        continue

                    strike_call_gex, strike_put_gex = {}, {}
                    used_bs_fallback = False
                    for _, orow in odf.iterrows():
                        k = safe_float(orow.get("option_strike_price"))
                        oi = safe_float(orow.get("option_open_interest"))
                        gamma = safe_float(orow.get("option_gamma"))
                        otype = str(orow.get("option_type", "")).upper()
                        if k <= 0 or oi <= 0:
                            continue
                        if gamma <= 0:
                            iv_raw = safe_float(orow.get("option_implied_volatility"))
                            sigma = (iv_raw / 100.0) if iv_raw > 5.0 else (iv_raw if iv_raw > 0.05 else (1.15 if sym in IS_3X_SET else 0.35))
                            gamma = bs_gamma_fallback(spot, k, t_years, sigma)
                            used_bs_fallback = True
                        if gamma <= 0:
                            continue
                        gex_val = gamma * oi * 100.0 * spot * spot * 0.01
                        if "CALL" in otype:
                            strike_call_gex[k] = strike_call_gex.get(k, 0.0) + gex_val
                        else:
                            strike_put_gex[k] = strike_put_gex.get(k, 0.0) - gex_val

                    if strike_put_gex or strike_call_gex:
                        all_k = sorted(set(list(strike_call_gex.keys()) + list(strike_put_gex.keys())))
                        put_wall = min(strike_put_gex.items(), key=lambda x: x[1])[0] if strike_put_gex else 0.0
                        call_wall = max(strike_call_gex.items(), key=lambda x: x[1])[0] if strike_call_gex else 0.0
                        left_puts = {k: v for k, v in strike_put_gex.items() if k < put_wall}
                        sec_put = min(left_puts.items(), key=lambda x: x[1])[0] if left_puts else 0.0
                        zero_gamma = 0.0
                        cum, prev_k, prev_cum = 0.0, None, None
                        for k in all_k:
                            cum += strike_call_gex.get(k, 0.0) + strike_put_gex.get(k, 0.0)
                            if prev_cum is not None and prev_cum < 0 <= cum:
                                zero_gamma = round((prev_k + k) / 2.0, 2)
                                break
                            prev_k, prev_cum = k, cum
                        if zero_gamma == 0.0 and call_wall > 0 and put_wall > 0:
                            zero_gamma = round((put_wall + call_wall) / 2.0, 2)

                        GEX_CACHE[sym] = {
                            "zero_gamma": round(zero_gamma, 2),
                            "put_wall": round(put_wall, 2),
                            "secondary_put_wall": round(sec_put, 2),
                            "call_wall": round(call_wall, 2),
                            "expiry": exp_date,
                            "source": "opend_auto",
                            "reason": f"OpenD自动({'BS推算' if used_bs_fallback else '实盘Greek'} {exp_date})",
                            "updated_at": time.strftime("%H:%M:%S")
                        }
                        if first_run:
                            print(f"✅ [{sym} 期权GEX自动成功] 到期:{exp_date} | 看跌墙:${put_wall} | 零Gamma:${zero_gamma} | 次柱:${sec_put}")
                    else:
                        GEX_CACHE[sym] = {"source": "opend_empty_oi", "reason": "期权快照未返回持仓量OI(需开通美股期权LV1)", "zero_gamma": 0, "put_wall": 0, "secondary_put_wall": 0, "call_wall": 0, "expiry": exp_date}
                        if first_run: print(f"⚠️ [{sym} 期权GEX] {GEX_CACHE[sym]['reason']}")
                except Exception as ex:
                    GEX_CACHE[sym] = {"source": "opend_error", "reason": f"异常: {str(ex)[:45]}", "zero_gamma": 0, "put_wall": 0, "secondary_put_wall": 0, "call_wall": 0, "expiry": ""}
                    if first_run: print(f"⚠️ [{sym} 期权GEX异常] {ex}")
                time.sleep(1.2)
            first_run = False
        except Exception:
            pass
        time.sleep(180)

def extract_session_quote(row, sym):
    et_now = datetime.datetime.now(ET_TZ)
    et_min = et_now.hour * 60 + et_now.minute
    weekend_closed = is_us_weekend_closed(et_now)

    reg_p = safe_float(row.get("last_price"))
    prev_c = safe_float(row.get("prev_close_price"), reg_p)
    open_p = safe_float(row.get("open_price"), reg_p)

    if weekend_closed:
        session_name = "weekend_closed"
        session_cols = [("after_price", "after_high_price", "after_low_price", "after_volume", "after_turnover")]
    elif 240 <= et_min < 570:
        session_name = "pre"
        session_cols = [
            ("pre_price", "pre_high_price", "pre_low_price", "pre_volume", "pre_turnover"),
            ("overnight_price", "overnight_high_price", "overnight_low_price", "overnight_volume", "overnight_turnover")
        ]
    elif 570 <= et_min < 960:
        session_name = "regular"
        session_cols = []
    elif 960 <= et_min < 1200:
        session_name = "after"
        session_cols = [("after_price", "after_high_price", "after_low_price", "after_volume", "after_turnover")]
    else:
        session_name = "overnight"
        session_cols = [
            ("overnight_price", "overnight_high_price", "overnight_low_price", "overnight_volume", "overnight_turnover"),
            ("after_price", "after_high_price", "after_low_price", "after_volume", "after_turnover")
        ]

    active_p = reg_p
    hi = safe_float(row.get("high_price"), reg_p)
    lo = safe_float(row.get("low_price"), reg_p)
    vol = safe_float(row.get("volume"))
    turn = safe_float(row.get("turnover"))
    used_ext = False

    for p_col, h_col, l_col, v_col, t_col in session_cols:
        if p_col in row:
            ext_p = safe_float(row[p_col])
            if ext_p > 0:
                active_p = ext_p
                ext_h = safe_float(row.get(h_col))
                ext_l = safe_float(row.get(l_col))
                ext_v = safe_float(row.get(v_col))
                ext_t = safe_float(row.get(t_col))
                hi = ext_h if ext_h > 0 else active_p
                lo = ext_l if ext_l > 0 else active_p
                vol = ext_v
                turn = ext_t
                used_ext = True
                break

    reg_close_base = reg_p if (used_ext and reg_p > 0) else prev_c
    vwap = round(turn / vol, 2) if (vol > 0 and turn > 0) else round((hi + lo + active_p) / 3.0, 2)
    day_chg_pct = round(((reg_p - prev_c) / prev_c) * 100.0, 2) if prev_c > 0 else 0.0
    ext_chg_pct = round(((active_p - reg_p) / reg_p) * 100.0, 2) if (used_ext and reg_p > 0) else day_chg_pct
    chg_pct = day_chg_pct if session_name in ("regular", "weekend_closed") else ext_chg_pct

    reg_turn = safe_float(row.get("turnover"), turn)
    eff_turn = turn if (session_name == "regular" and turn > 0) else reg_turn
    live_mkt_val = safe_float(row.get("total_market_val"))
    aum_usd = live_mkt_val if live_mkt_val > 2e8 else AUM_BENCHMARK_USD.get(sym, 15.0e9)
    aum_turn_pct = round((eff_turn / aum_usd) * 100.0, 1) if aum_usd > 0 else 0.0

    vinfo = VOL_CACHE.get(sym, {})
    amp_5d = vinfo.get("amp_5d_pct", 18.5 if sym in IS_3X_SET else 4.2)
    dyn_decay = vinfo.get("dyn_beta_decay", calc_dynamic_decay_from_amp(amp_5d, sym in IS_3X_SET))
    low_52w = vinfo.get("low_52w", LOW_52W_FALLBACK.get(sym, 0.0))
    gex_info = GEX_CACHE.get(sym, {"source": "loading", "reason": "后台拉取期权链中...", "zero_gamma": 0, "put_wall": 0, "secondary_put_wall": 0, "call_wall": 0, "expiry": ""})

    return {
        "price": round(active_p, 2),
        "reg_price": round(reg_p, 2),
        "open": round(open_p, 2),
        "reg_close": round(reg_close_base, 2),
        "chg_pct": chg_pct,
        "day_chg_pct": day_chg_pct,
        "ext_chg_pct": ext_chg_pct,
        "high": round(max(hi, active_p), 2),
        "low": round(min(lo, active_p) if lo > 0 else active_p, 2),
        "vwap": vwap,
        "volume": int(vol),
        "turnover_usd": round(eff_turn, 2),
        "aum_usd": round(aum_usd, 2),
        "aum_turnover_pct": aum_turn_pct,
        "amp_5d_pct": amp_5d,
        "dyn_beta_decay": dyn_decay,
        "low_52w": low_52w,
        "gex": gex_info,
        "session": session_name,
        "is_weekend": weekend_closed
    }

def fetch_opend_payload():
    global quote_ctx
    try:
        with ctx_lock:
            ctx = get_ctx()
            ret, data = ctx.get_market_snapshot(ALL_CODES + ["US.SPLV"])
        if ret != RET_OK or data.empty:
            raise RuntimeError("OpenD snapshot empty")

        res = {}
        splv_row = None
        soxx_row = None
        for _, row in data.iterrows():
            sym = str(row["code"]).replace("US.", "")
            if sym == "SPLV":
                splv_row = row
                continue
            if sym == "SOXX":
                soxx_row = row
            res[sym] = extract_session_quote(row, sym)

        et_now = datetime.datetime.now(ET_TZ)
        et_min = et_now.hour * 60 + et_now.minute
        weekend_closed = is_us_weekend_closed(et_now)
        spy_q = res.get("SPY", {})
        spy_chg = spy_q.get("day_chg_pct", 0.0) if (weekend_closed or spy_q.get("session") in ("regular", "after")) else spy_q.get("chg_pct", 0.0)

        risk_meta = {
            "et_time": et_now.strftime("%H:%M"),
            "et_min": et_min,
            "is_weekend": weekend_closed,
            "is_margin_call_win": (not weekend_closed and 710 <= et_min <= 780),
            "is_rebalance_win": (not weekend_closed and 915 <= et_min <= 960),
            "spy_chg_pct": spy_chg,
            "spy_crash_alert": (spy_chg <= -2.0),
            "splv_seesaw": evaluate_live_seesaw(splv_row, soxx_row, et_now, weekend_closed, quotes_map=res)
        }

        payload = {
            "ts": time.strftime("%H:%M:%S"),
            "source": "moomoo_opend_v657",
            "risk_meta": risk_meta,
            "quotes": res
        }
        try:
            with open(OUT_JSON + ".tmp", "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(OUT_JSON + ".tmp", OUT_JSON)
        except Exception:
            pass
        return payload
    except Exception as e:
        with ctx_lock:
            if quote_ctx:
                try: quote_ctx.close()
                except Exception: pass
                quote_ctx = None
        raise e

class ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

class BridgeHandler(BaseHTTPRequestHandler):
    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/dashboard.html"):
            try:
                with open(os.path.join(BASE_DIR, "index.html"), "rb") as f:
                    content = f.read()
                self.send_response(200)
                self._send_cors()
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(content)
                return
            except Exception:
                pass

        if path == "/api/quotes_all":
            try:
                payload = fetch_opend_payload()
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self._send_cors()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                err = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(503)
                self._send_cors()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(err)
            return

        self.send_response(404)
        self._send_cors()
        self.end_headers()

    def log_message(self, format, *args):
        return

if __name__ == "__main__":
    bg = threading.Thread(target=refresh_vol_and_gex_worker, daemon=True)
    bg.start()
    try:
        test_p = fetch_opend_payload()
        s = test_p["quotes"].get("SOXL", {})
        print(f"✅ OpenD V6.5.2 启动 | 时段:{s.get('session')} | SOXL:${s.get('price')} | SPY:{test_p['risk_meta']['spy_chg_pct']:+.2f}%")
    except Exception as e:
        print(f"⚠️ OpenD 启动自检提示: {e}")

    server = ReusableHTTPServer(("127.0.0.1", 8888), BridgeHandler)
    print("🚀 Moomoo OpenD V6.5.2 已启动: http://127.0.0.1:8888 (约3秒后下方将打印期权GEX自动拉取诊断)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 本地桥接服务已停止。")
