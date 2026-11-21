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

# ETF 资产规模基准 (USD)，用于实时计算 成交额 / AUM 清算换手率
AUM_BENCHMARK_USD = {
    "SOXL": 14.404e9,
    "SOXX": 34.507e9,
    "TQQQ": 24.0e9,
    "SQQQ": 3.5e9,
    "SOXS": 1.6e9,
    "QQQ":  310.0e9,
    "SPY":  600.0e9
}

# 52周前低兜底值 (后台历史K线拉取成功后自动实时覆盖)
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
VOL_CACHE = {}   # sym -> {amp_5d_pct, dyn_beta_decay, low_52w, updated_at}
GEX_CACHE = {}   # sym -> {zero_gamma, put_wall, secondary_put_wall, call_wall, expiry, source, updated_at}

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
        quote_ctx.subscribe(ALL_CODES, [SubType.QUOTE], subscribe_push=False)
    return quote_ctx

def calc_dynamic_decay_from_amp(amp_5d_pct, is_3x):
    """将最近5日平均日内振幅映射为动态磨损系数 (12.5% ~ 35%)"""
    if not is_3x:
        return 0.0
    # 常规振幅 ~6% -> 12.5%; 暴涨暴跌振幅 >= 20% -> 最高 35%
    if amp_5d_pct <= 6.0:
        return 0.125
    extra = min(0.225, ((amp_5d_pct - 6.0) / 14.0) * 0.225)
    return round(0.125 + extra, 4)

def refresh_vol_and_gex_worker():
    """后台守护线程：每180秒静默更新5日波动率磨损 + 期权GEX，绝不阻塞实时行情"""
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

            # 1. 拉取近10日K线计算真实5日日均振幅 & 动态磨损
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
                time.sleep(0.25)

            # 2. 为核心标的 (SOXL, SOXX, QQQ) 自动计算近期期权 GEX (看跌墙 / 零Gamma / 次级柱)
            for sym in ("SOXL", "SOXX", "QQQ"):
                code = SYMBOL_MAP[sym]
                spot = spot_map.get(sym, 0.0)
                if spot <= 0:
                    continue
                try:
                    with ctx_lock:
                        ret_e, edf = ctx.get_option_expiration_date(code=code)
                    if ret_e != RET_OK or edf is None or edf.empty:
                        continue
                    exp_date = str(edf.iloc[0]["strike_time"])
                    with ctx_lock:
                        ret_c, cdf = ctx.get_option_chain(code=code, start=exp_date, end=exp_date)
                    if ret_c != RET_OK or cdf is None or cdf.empty:
                        continue
                    # 仅筛取现价 ±28% 范围内的行权价，确保快速查询且不超 OpenD 单次快照上限
                    sub = cdf[(cdf["strike_price"] >= spot * 0.72) & (cdf["strike_price"] <= spot * 1.28)]
                    opt_codes = sub["code"].tolist()[:100]
                    if not opt_codes:
                        continue
                    with ctx_lock:
                        ret_o, odf = ctx.get_market_snapshot(opt_codes)
                    if ret_o != RET_OK or odf is None or odf.empty:
                        continue

                    strike_call_gex = {}
                    strike_put_gex = {}
                    for _, orow in odf.iterrows():
                        k = safe_float(orow.get("option_strike_price"))
                        gamma = safe_float(orow.get("option_gamma"))
                        oi = safe_float(orow.get("option_open_interest"))
                        otype = str(orow.get("option_type", "")).upper()
                        if k <= 0 or gamma <= 0 or oi <= 0:
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
                        # 找看跌墙左侧最大的次级看跌柱
                        left_puts = {k: v for k, v in strike_put_gex.items() if k < put_wall}
                        sec_put = min(left_puts.items(), key=lambda x: x[1])[0] if left_puts else 0.0
                        # 寻找净 GEX 由负转正的零 Gamma 近似点
                        zero_gamma = 0.0
                        cum = 0.0
                        prev_k, prev_cum = None, None
                        for k in all_k:
                            net_k = strike_call_gex.get(k, 0.0) + strike_put_gex.get(k, 0.0)
                            cum += net_k
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
                            "updated_at": time.strftime("%H:%M:%S")
                        }
                except Exception:
                    pass
                time.sleep(0.5)
        except Exception:
            pass
        time.sleep(180)

def extract_session_quote(row, sym):
    et_now = datetime.datetime.now(ET_TZ)
    et_min = et_now.hour * 60 + et_now.minute

    reg_p = safe_float(row.get("last_price"))
    prev_c = safe_float(row.get("prev_close_price"), reg_p)
    open_p = safe_float(row.get("open_price"), reg_p)

    if 240 <= et_min < 570:
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
        session_cols = [
            ("after_price", "after_high_price", "after_low_price", "after_volume", "after_turnover")
        ]
    else:
        session_name = "overnight"
        session_cols = [
            ("overnight_price", "overnight_high_price", "overnight_low_price", "overnight_volume", "overnight_turnover"),
            ("after_price", "after_high_price", "after_low_price", "after_volume", "after_turnover"),
            ("pre_price", "pre_high_price", "pre_low_price", "pre_volume", "pre_turnover")
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
    chg_pct = round(((active_p - reg_close_base) / reg_close_base) * 100.0, 2) if reg_close_base > 0 else 0.0

    # 计算 AUM 换手率 (正盘使用当日成交额，盘前盘后保留正盘成交额参考)
    reg_turn = safe_float(row.get("turnover"), turn)
    eff_turn = turn if (session_name == "regular" and turn > 0) else max(turn, reg_turn)
    aum_usd = AUM_BENCHMARK_USD.get(sym, 15.0e9)
    aum_turn_pct = round((eff_turn / aum_usd) * 100.0, 1) if aum_usd > 0 else 0.0

    # 读取动态波动率与 GEX 缓存
    vinfo = VOL_CACHE.get(sym, {})
    amp_5d = vinfo.get("amp_5d_pct", 18.5 if sym in IS_3X_SET else 4.2)
    dyn_decay = vinfo.get("dyn_beta_decay", calc_dynamic_decay_from_amp(amp_5d, sym in IS_3X_SET))
    low_52w = vinfo.get("low_52w", LOW_52W_FALLBACK.get(sym, 0.0))
    gex_info = GEX_CACHE.get(sym, {"source": "manual_needed", "zero_gamma": 0, "put_wall": 0, "secondary_put_wall": 0, "call_wall": 0, "expiry": ""})

    return {
        "price": round(active_p, 2),
        "open": round(open_p, 2),
        "reg_close": round(reg_close_base, 2),
        "chg_pct": chg_pct,
        "high": round(max(hi, active_p), 2),
        "low": round(min(lo, active_p) if lo > 0 else active_p, 2),
        "vwap": vwap,
        "volume": int(vol),
        "turnover_usd": round(eff_turn, 2),
        "aum_usd": aum_usd,
        "aum_turnover_pct": aum_turn_pct,
        "amp_5d_pct": amp_5d,
        "dyn_beta_decay": dyn_decay,
        "low_52w": low_52w,
        "gex": gex_info,
        "session": session_name
    }

def fetch_opend_payload():
    global quote_ctx
    try:
        with ctx_lock:
            ctx = get_ctx()
            ret, data = ctx.get_market_snapshot(ALL_CODES)
        if ret != RET_OK or data.empty:
            raise RuntimeError("OpenD snapshot empty")

        res = {}
        for _, row in data.iterrows():
            sym = str(row["code"]).replace("US.", "")
            res[sym] = extract_session_quote(row, sym)

        et_now = datetime.datetime.now(ET_TZ)
        et_min = et_now.hour * 60 + et_now.minute
        spy_chg = res.get("SPY", {}).get("chg_pct", 0.0)

        risk_meta = {
            "et_time": et_now.strftime("%H:%M"),
            "et_min": et_min,
            "is_margin_call_win": (710 <= et_min <= 780),   # ET 11:50 ~ 13:00 午盘强平窗
            "is_rebalance_win": (915 <= et_min <= 960),     # ET 15:15 ~ 16:00 3x尾盘机械再平衡窗
            "spy_chg_pct": spy_chg,
            "spy_crash_alert": (spy_chg <= -2.0)
        }

        payload = {
            "ts": time.strftime("%H:%M:%S"),
            "source": "moomoo_opend_v65",
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
        print(f"✅ OpenD V6.5 自检成功 | SOXL: ${s.get('price')} ({s.get('chg_pct'):+.2f}%) | 换手/AUM: {s.get('aum_turnover_pct')}% | 动态磨损: {round(s.get('dyn_beta_decay',0)*100,1)}%")
    except Exception as e:
        print(f"⚠️ OpenD 启动自检提示: {e}")

    server = ReusableHTTPServer(("127.0.0.1", 8888), BridgeHandler)
    print("🚀 Moomoo OpenD V6.5 实时清算熔断桥接已启动: http://127.0.0.1:8888")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 本地桥接服务已停止。")
