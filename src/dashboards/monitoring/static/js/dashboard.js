/* Trading Bot Monitor — V2 hi-fi dashboard
   React 18 (UMD) + Babel-standalone JSX.
   Real-data adapter wired to /api/dashboard/state, /api/performance, and socket.io.

   NOTE on Babel-standalone: this is the JSX runtime in the browser. It is not
   ideal for production (extra ~3MB of JS, parses on every load) — see
   docs/monitoring.md for the deferred follow-up to ship a pre-built bundle.
   For now, the dashboard is internal-only and the simplicity-of-edit win
   outweighs the load-time cost. */

const { useState, useEffect, useRef, useMemo, useCallback, useContext, createContext } = React;

// ─────────────────────────────────────────── helpers ──────────

const fmtUSD = (v, opts = {}) => {
  const { sign = false, dp = 2 } = opts;
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  const a = Math.abs(n);
  const s = a < 1000 ? a.toFixed(dp) : a.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
  const prefix = n >= 0 ? (sign ? '+$' : '$') : '-$';
  return prefix + s;
};
const fmtPct = (v, dp = 2) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  return `${n >= 0 ? '+' : ''}${n.toFixed(dp)}%`;
};
const fmtNum = (v, dp = 2) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  return n.toFixed(dp);
};
const fmtTimeAgo = (ts, nowMs) => {
  if (!ts) return '—';
  const d = typeof ts === 'string' ? new Date(ts) : new Date(Number(ts));
  if (Number.isNaN(d.getTime())) return '—';
  const ref = Number(nowMs) || Date.now();
  const mins = Math.floor((ref - d.getTime()) / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
};
const fmtUptime = (sec) => {
  const s = Number(sec) || 0;
  const days = Math.floor(s / 86400);
  const hours = Math.floor((s % 86400) / 3600);
  if (days > 0) return `${days}d ${hours}h`;
  const mins = Math.floor((s % 3600) / 60);
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
};
const symU = (s) => String(s || '').toUpperCase();

// safe (non-stack-blowing) min/max for numeric arrays
const safeMin = (arr) => {
  let m = Infinity;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (Number.isFinite(v) && v < m) m = v;
  }
  return Number.isFinite(m) ? m : 0;
};
const safeMax = (arr) => {
  let m = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (Number.isFinite(v) && v > m) m = v;
  }
  return Number.isFinite(m) ? m : 0;
};

// Hook that re-renders every `intervalMs`. Used to keep "X ago" labels fresh
// even when the underlying state object hasn't changed.
function useTick(intervalMs = 30000) {
  const [, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick(t => (t + 1) | 0), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
}

// Returns true when the viewport is mobile-sized (< 768px). Re-evaluates on
// window resize / orientation change so the dashboard reflows live, not just
// on initial paint. Uses matchMedia where available with a resize fallback.
const MOBILE_BREAKPOINT = '(max-width: 768px)';
function useIsMobile() {
  const [isMobile, setIsMobile] = useState(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return false;
    return window.matchMedia(MOBILE_BREAKPOINT).matches;
  });
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mq = window.matchMedia(MOBILE_BREAKPOINT);
    const handler = (e) => setIsMobile(e.matches);
    // addEventListener is the modern API; addListener is the Safari fallback
    if (mq.addEventListener) mq.addEventListener('change', handler);
    else mq.addListener(handler);
    return () => {
      if (mq.removeEventListener) mq.removeEventListener('change', handler);
      else mq.removeListener(handler);
    };
  }, []);
  return isMobile;
}

// ─────────────────────────────────────────── data normalisation ──────────
//
// The backend speaks the existing /api/* shape. We translate it into the
// design's expected store shape so all V2 components work unchanged.
//
// Important: prefer null over zero for metrics where "no data yet" and
// "actually zero" have different semantics (sharpe, win rate, profit factor).
// The UI renders `—` for null instead of fabricating a value.

function buildEquityCurve(performance) {
  if (!performance || !performance.timestamps || !performance.balances) return [];
  const ts = performance.timestamps;
  const bal = performance.balances;
  const n = Math.min(ts.length, bal.length);
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = new Date(ts[i]).getTime();
    const v = Number(bal[i]);
    if (Number.isFinite(t) && Number.isFinite(v)) out.push({ ts: t, v });
  }
  return out;
}

function normalizePosition(p, idx) {
  const side = String(p.side || '').toUpperCase();
  const entry = Number(p.entry_price) || 0;
  const current = Number(p.current_price) || 0;
  const qty = Number(p.quantity) || 0;
  const dir = side === 'SHORT' ? -1 : 1;
  const pnl = Number(p.unrealized_pnl);
  const pnlPct = entry > 0 ? ((current - entry) / entry) * 100 * dir : 0;
  const trailSL = p.trailing_stop_price ?? p.stop_loss ?? null;
  const tp = p.take_profit ?? null;
  // Capture entry time as a real ms timestamp so the Logs view doesn't have to
  // recover it via `Date.now() - ageMs` (which drifts with render lag).
  const entryTimeMs = p.entry_time ? new Date(p.entry_time).getTime() : null;
  const ageMs = entryTimeMs != null ? Math.max(0, Date.now() - entryTimeMs) : 0;
  return {
    id: p.symbol ? `${p.symbol}-${idx}` : `pos-${idx}`,
    symbol: symU(p.symbol),
    side: side === 'SHORT' ? 'SHORT' : 'LONG',
    size: qty,
    entry,
    current,
    pnl: Number.isFinite(pnl) ? pnl : 0,
    pnlPct,
    trailSL: trailSL ?? null,
    breakeven: !!p.breakeven_triggered,
    mfe: Number(p.mfe) || 0,
    mae: Number(p.mae) || 0,
    target: { tp, sl: p.stop_loss ?? null, trail: trailSL, trailPct: trailSL && entry ? Math.abs(((trailSL - entry) / entry) * 100) : 0 },
    entryTimeMs,
    ageMs,
    strategy: p.strategy_name || null,
    confidence: Number.isFinite(Number(p.confidence)) ? Number(p.confidence) : null,
    signal: side === 'SHORT' ? 'SELL' : 'BUY',
    raw: p,
  };
}

function normalizeTrade(t, idx) {
  const side = String(t.side || '').toUpperCase();
  const sideShort = side === 'SHORT' || side === 'SELL' || side === 'S' ? 'S' : 'L';
  const exitTime = t.exit_time ? new Date(t.exit_time).getTime() : null;
  // Use a stable id: prefer the DB id, then a composite of symbol+exit_time
  // (which is unique within the trades table for a given bot). Index-based
  // ids would shift between fetches and cause the inspector to show the
  // wrong trade after a refetch.
  const stableKey = t.id != null
    ? `db-${t.id}`
    : (t.symbol && t.exit_time)
      ? `nat-${symU(t.symbol)}-${exitTime}`
      : `t${idx}-${exitTime ?? Date.now()}`;
  return {
    id: stableKey,
    symbol: symU(t.symbol),
    side: sideShort,
    qty: Number(t.quantity) || 0,
    entry: Number(t.entry_price) || 0,
    exit: Number(t.exit_price) || 0,
    pnl: Number(t.pnl) || 0,
    reason: t.exit_reason || '—',
    time: exitTime,
    raw: t,
  };
}

// Number-or-null helper: collapses non-finite inputs to null so the UI can
// distinguish "not yet known" from "actual zero". Note the explicit
// null/undefined guard — without it, `Number(null) === 0` and
// `Number(undefined) === NaN`, so a server-side `null` (the project's
// "not configured" sentinel) would silently coerce to 0 and the UI would
// render "0.00%" / "$0" instead of the intended "—" placeholder.
const numOrNull = (v) => {
  if (v === null || v === undefined || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

function normalizeState(payload, performance) {
  const m = (payload && payload.metrics) || {};
  const bot = (payload && payload.bot) || {};
  const positions = (payload && payload.positions) || [];
  const trades = (payload && payload.trades) || [];

  const initialBalance = numOrNull(bot.initial_balance);
  const balanceRaw = numOrNull(m.current_balance);
  const balance = balanceRaw != null ? balanceRaw : (initialBalance != null ? initialBalance : null);
  // Backend exposes total_pnl as realized only (SUM(pnl) FROM trades). We
  // surface it as `realized` directly to avoid the historical "totalPnl"
  // double-subtraction bug. `totalPnl` (realized + unrealized) is computed
  // separately so the UI can show both honestly. Preserve null semantics for
  // unrealized so the UI can render `—` when the metric is genuinely missing
  // (e.g. metric disabled in config) instead of silently treating it as 0.
  const realized = numOrNull(m.total_pnl);
  const unrealized = numOrNull(m.unrealized_pnl);
  const totalPnl = realized != null ? realized + (unrealized ?? 0) : null;

  const todayPnl = numOrNull(m.daily_pnl) ?? 0;
  const todayPnlPct = (initialBalance != null && initialBalance > 0)
    ? (todayPnl / initialBalance) * 100
    : null;

  const dynMult = numOrNull(m.dynamic_risk_factor) ?? 1.0;
  const dynReason = m.dynamic_risk_reason || 'normal';
  const maxOpen = numOrNull(bot.max_open_positions);

  return {
    bot: {
      name: bot.name || m.current_strategy || 'unknown',
      mode: (bot.mode || 'paper').toLowerCase(),
      status: bot.status || 'running',
      connected: bot.connected !== false,
      lastUpdate: bot.last_update ? new Date(bot.last_update).getTime() : Date.now(),
      symbols: bot.symbols && bot.symbols.length ? bot.symbols : ['BTCUSDT'],
      timeframe: bot.timeframe || '1h',
      uptime: fmtUptime(bot.uptime_seconds || m.system_uptime),
      maxOpenPositions: maxOpen,
    },
    balance,
    initialBalance,
    realized,        // realized P&L (closed trades), or null if missing
    totalPnl,        // realized + unrealized, or null
    todayPnl,
    todayPnlPct,
    weeklyPnl: numOrNull(m.weekly_pnl) ?? 0,
    unrealized: unrealized ?? 0,
    unrealizedRaw: unrealized,    // null-preserving copy for the inspector
    sharpe: numOrNull(m.sharpe_ratio),
    maxDD: numOrNull(m.max_drawdown) ?? 0,
    currentDD: numOrNull(m.current_drawdown) ?? 0,
    volatility: numOrNull(m.volatility) ?? 0,
    winRate: numOrNull(m.win_rate),               // null when no trades
    profitFactor: numOrNull(m.profit_factor),
    avgWinLoss: numOrNull(m.avg_win_loss_ratio),
    totalTrades: numOrNull(m.total_trades) ?? 0,
    activePositions: numOrNull(m.active_positions_count) ?? positions.length,
    maxPositions: maxOpen,                        // null if not configured
    totalPositionValue: numOrNull(m.total_position_value) ?? 0,
    marginUsage: numOrNull(m.margin_usage) ?? 0,
    availableMargin: numOrNull(m.available_margin) ?? 0,
    // riskPerTrade comes ONLY from the session config (via bot meta). The
    // legacy `metrics.risk_per_trade` field defaults to 1.0% server-side
    // when no config carries the knob, which is indistinguishable from an
    // explicit 1.0% — we'd resurrect the fabricated baseline iter-1 fixed
    // for `initial_balance` and `max_open_positions` if we used it as a
    // fallback. Render `—` instead.
    riskPerTrade: numOrNull(bot.risk_per_trade),
    fillRate: numOrNull(m.fill_rate) ?? 0,
    avgSlippage: numOrNull(m.avg_slippage) ?? 0,
    failedOrders: numOrNull(m.failed_orders) ?? 0,
    orderLatency: numOrNull(m.order_latency) ?? 0,
    executionQuality: numOrNull(m.execution_quality) ?? 0,
    apiLatency: numOrNull(m.api_latency) ?? 0,
    apiStatus: m.api_connection_status || 'Unknown',
    dataFeed: m.data_feed_status || 'Unknown',
    rsi: numOrNull(m.rsi) ?? 50,
    emaTrend: m.ema_trend || '—',
    priceChange24h: numOrNull(m.price_change_24h) ?? 0,
    dynamicRisk: { mult: dynMult, status: dynReason, reason: dynReason, active: !!m.dynamic_risk_active },
    positions: positions.map(normalizePosition),
    trades: trades.map(normalizeTrade),
    equityCurve: buildEquityCurve(performance),
    metricsRaw: m,
  };
}

// Patch top-level metrics-derived fields without touching positions/trades —
// used on socket metrics_update to avoid showing stale per-position prices
// while still feeling instant for the KPI strip.
function patchMetrics(prev, metrics) {
  if (!prev || !metrics) return prev;
  const m = metrics;
  const initialBalance = prev.initialBalance;
  const balance = numOrNull(m.current_balance) ?? prev.balance;
  const realized = numOrNull(m.total_pnl) ?? prev.realized;
  const unrealizedRaw = numOrNull(m.unrealized_pnl) ?? prev.unrealizedRaw ?? null;
  const unrealized = unrealizedRaw ?? 0;
  const totalPnl = realized != null ? realized + (unrealizedRaw ?? 0) : prev.totalPnl;
  const todayPnl = numOrNull(m.daily_pnl) ?? prev.todayPnl;
  return {
    ...prev,
    balance,
    realized,
    unrealized,
    unrealizedRaw,
    totalPnl,
    todayPnl,
    todayPnlPct: (initialBalance != null && initialBalance > 0)
      ? (todayPnl / initialBalance) * 100
      : prev.todayPnlPct,
    sharpe: numOrNull(m.sharpe_ratio) ?? prev.sharpe,
    maxDD: numOrNull(m.max_drawdown) ?? prev.maxDD,
    currentDD: numOrNull(m.current_drawdown) ?? prev.currentDD,
    volatility: numOrNull(m.volatility) ?? prev.volatility,
    winRate: numOrNull(m.win_rate) ?? prev.winRate,
    profitFactor: numOrNull(m.profit_factor) ?? prev.profitFactor,
    avgWinLoss: numOrNull(m.avg_win_loss_ratio) ?? prev.avgWinLoss,
    totalTrades: numOrNull(m.total_trades) ?? prev.totalTrades,
    activePositions: numOrNull(m.active_positions_count) ?? prev.activePositions,
    totalPositionValue: numOrNull(m.total_position_value) ?? prev.totalPositionValue,
    marginUsage: numOrNull(m.margin_usage) ?? prev.marginUsage,
    availableMargin: numOrNull(m.available_margin) ?? prev.availableMargin,
    rsi: numOrNull(m.rsi) ?? prev.rsi,
    emaTrend: m.ema_trend || prev.emaTrend,
    priceChange24h: numOrNull(m.price_change_24h) ?? prev.priceChange24h,
    dynamicRisk: {
      mult: numOrNull(m.dynamic_risk_factor) ?? prev.dynamicRisk.mult,
      status: m.dynamic_risk_reason || prev.dynamicRisk.status,
      reason: m.dynamic_risk_reason || prev.dynamicRisk.reason,
      active: !!m.dynamic_risk_active,
    },
    bot: { ...prev.bot, lastUpdate: Date.now() },
    metricsRaw: m,
  };
}

// ─────────────────────────────────────────── store ──────────

const StoreCtx = createContext(null);

function StoreProvider({ children }) {
  const [state, setState] = useState(null);
  const [error, setError] = useState(null);
  const [range, setRange] = useState('1W');
  const inFlightRef = useRef(null);

  // Map design's range tokens to backend ?days=. "1Y" supersedes "ALL" since
  // the backend caps performance queries at 365 days; the label was renamed
  // to avoid implying "all time" when we only fetch 1y.
  const rangeToDays = (r) => ({ '1D': 1, '1W': 7, '1M': 30, '3M': 90, '1Y': 365 }[r] || 7);

  const fetchState = useCallback(async () => {
    // Cancel any in-flight request — last writer wins.
    if (inFlightRef.current) {
      try { inFlightRef.current.abort(); } catch (_) { /* noop */ }
    }
    const ac = new AbortController();
    inFlightRef.current = ac;
    try {
      let payload = null;
      try {
        const r = await fetch('/api/dashboard/state?trades_limit=50', { cache: 'no-store', signal: ac.signal });
        if (r.ok) payload = await r.json();
      } catch (e) {
        if (e && e.name === 'AbortError') return;
        console.warn('bundled /api/dashboard/state fetch failed, falling back', e);
      }

      if (!payload) {
        const safeFetch = (url) => fetch(url, { cache: 'no-store', signal: ac.signal })
          .then(r => r.ok ? r.json() : null)
          .catch(e => { if (e && e.name === 'AbortError') throw e; console.warn(`fetch ${url} failed`, e); return null; });
        const [mRes, pRes, tRes, sRes] = await Promise.all([
          safeFetch('/api/metrics'),
          safeFetch('/api/positions'),
          safeFetch('/api/trades?limit=50'),
          safeFetch('/api/system/status'),
        ]);
        const metrics = mRes || {};
        const positions = Array.isArray(pRes) ? pRes : [];
        const trades = Array.isArray(tRes) ? tRes : [];
        const sys = sRes || {};
        const symbols = Array.from(new Set([
          ...positions.map(p => p.symbol).filter(Boolean),
          ...trades.map(t => t.symbol).filter(Boolean),
        ])).slice(0, 4);
        payload = {
          bot: {
            name: metrics.current_strategy || 'unknown',
            symbols: symbols.length ? symbols : ['BTCUSDT'],
            timeframe: '1h',
            mode: 'paper',
            status: 'running',
            connected: (sys.api_status === 'Connected') || (metrics.api_connection_status === 'Connected'),
            initial_balance: null,
            max_open_positions: null,
            uptime_seconds: numOrNull(metrics.system_uptime) ?? 0,
            last_update: metrics.last_data_update || new Date().toISOString(),
          },
          metrics,
          positions,
          trades,
        };
      }

      if (ac.signal.aborted) return;
      setState((prev) => {
        const next = normalizeState(payload, { timestamps: [], balances: [] });
        if (prev && prev.equityCurve && prev.equityCurve.length) {
          next.equityCurve = prev.equityCurve;
        }
        return next;
      });
      setError(null);
    } catch (e) {
      if (e && e.name === 'AbortError') return;
      setError(String(e.message || e));
    } finally {
      if (inFlightRef.current === ac) inFlightRef.current = null;
    }
  }, []);

  const fetchEquity = useCallback(async (r) => {
    try {
      const days = rangeToDays(r);
      const res = await fetch(`/api/performance?days=${days}`, { cache: 'no-store' });
      if (!res.ok) throw new Error(`performance ${res.status}`);
      const json = await res.json();
      const curve = buildEquityCurve(json);
      setState((prev) => prev ? { ...prev, equityCurve: curve } : prev);
    } catch (e) {
      console.warn('equity fetch failed', e);
    }
  }, []);

  // initial paint
  useEffect(() => {
    fetchState();
    fetchEquity(range);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // refetch equity when range changes
  useEffect(() => { fetchEquity(range); }, [range, fetchEquity]);

  // Periodic resync (positions/trades) — separate effect from socket so a
  // future change to the socket effect doesn't tear down the timer.
  useEffect(() => {
    const id = setInterval(fetchState, 30000);
    return () => clearInterval(id);
  }, [fetchState]);

  // socket.io live updates — patch metrics in place; positions/trades come
  // from the periodic fetchState so they don't go stale when normalised
  // off old `raw` snapshots.
  useEffect(() => {
    if (typeof io === 'undefined') return;
    const socket = io({ transports: ['websocket', 'polling'] });
    socket.on('connect', () => {
      setState((prev) => prev ? { ...prev, bot: { ...prev.bot, connected: true } } : prev);
    });
    socket.on('disconnect', () => {
      setState((prev) => prev ? { ...prev, bot: { ...prev.bot, connected: false } } : prev);
    });
    socket.on('metrics_update', (metrics) => {
      setState((prev) => patchMetrics(prev, metrics));
      setError(null); // a successful tick clears any stale error banner
    });
    return () => { socket.disconnect(); };
  }, []);

  const value = useMemo(() => ({ state, error, range, setRange, refresh: fetchState }), [state, error, range, fetchState]);
  return <StoreCtx.Provider value={value}>{children}</StoreCtx.Provider>;
}

const useStore = () => useContext(StoreCtx);

// ─────────────────────────────────────────── primitives ──────────

function HPill({ children, active = false, success = false, danger = false, ghost = false, onClick, title }) {
  const cls = ['tbm-pill', active && 'active', success && 'success', danger && 'danger', ghost && 'ghost'].filter(Boolean).join(' ');
  return <span className={cls} onClick={onClick} title={title}>{children}</span>;
}

function HKPI({ label, value, sub, size = '', color }) {
  return (
    <div className="tbm-kpi">
      <div className="tbm-kicker">{label}</div>
      <div className={`v ${size}`} style={{ color: color || 'var(--text)' }}>{value}</div>
      {sub && <div className="s">{sub}</div>}
    </div>
  );
}

function HRow({ k, v, sub, color }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', fontFamily: 'var(--mono)', fontSize: 12 }}>
      <span style={{ color: 'var(--text-3)' }}>{k}</span>
      <span style={{ color: color || 'var(--text)', fontWeight: 500 }}>
        {v}{sub && <span style={{ color: 'var(--text-3)', marginLeft: 6, fontWeight: 400 }}>{sub}</span>}
      </span>
    </div>
  );
}

function HBarLabeled({ k, v, max, sub, color }) {
  const pct = Math.max(0, Math.min(100, (Number(v) / Number(max || 1)) * 100));
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-3)', marginBottom: 4 }}>
        <span>{k}</span>
        <span style={{ color: 'var(--text-2)', fontFamily: 'var(--mono)' }}>{sub}</span>
      </div>
      <div className="tbm-bar"><span style={{ width: `${pct}%`, background: color || 'var(--accent)' }} /></div>
    </div>
  );
}

function HSparkline({ data, width = 80, height = 22, color }) {
  if (!data || data.length < 2) return <svg width={width} height={height} aria-hidden="true" />;
  const ys = data.map(d => typeof d === 'number' ? d : d.v);
  const minY = safeMin(ys), maxY = safeMax(ys);
  const sx = i => (i / (ys.length - 1)) * width;
  const sy = v => height - ((v - minY) / ((maxY - minY) || 1)) * height;
  let path = `M ${sx(0)} ${sy(ys[0])}`;
  for (let i = 1; i < ys.length; i++) {
    const cx = (sx(i - 1) + sx(i)) / 2;
    path += ` Q ${cx} ${sy(ys[i - 1])} ${cx} ${(sy(ys[i - 1]) + sy(ys[i])) / 2} T ${sx(i)} ${sy(ys[i])}`;
  }
  const c = color || (ys[ys.length - 1] >= ys[0] ? 'var(--accent-2)' : 'var(--danger)');
  return (
    <svg width={width} height={height} className="tbm-spark" style={{ display: 'inline-block' }} aria-hidden="true">
      <path d={path} fill="none" stroke={c} strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// Placeholder when we don't have per-symbol price/PnL history. Renders a
// flat baseline + label so it's visibly NOT a real chart — replaces the
// previous deterministic-noise sparkline that looked like real data.
function MissingSpark({ width = 140, height = 22 }) {
  return (
    <div style={{ width, height, display: 'flex', alignItems: 'center', justifyContent: 'center',
                   border: '1px dashed var(--border)', borderRadius: 4,
                   color: 'var(--text-3)', fontFamily: 'var(--mono)', fontSize: 9, letterSpacing: '0.08em' }}>
      no history
    </div>
  );
}

// ─────────────────────────────────────────── topbar ──────────

function HifiTopBar({ theme, onToggleTheme }) {
  const { state } = useStore();
  useTick(15000); // refresh "X ago" labels
  if (!state) return null;
  const s = state;
  const subTitle = `${s.bot.name} · ${s.bot.symbols.join(' · ')} · ${s.bot.timeframe}`;
  // Theme button shows the ACTION (what tapping does), not the current state.
  const nextThemeLabel = theme === 'dark' ? '☀ Light' : '☾ Dark';
  return (
    <div className="tbm-top">
      <div className="tbm-brand">
        <div className="tbm-logo" aria-hidden="true">A</div>
        <div>
          <div className="tbm-brand-title">Trading Bot Monitor</div>
          <div className="tbm-brand-sub">{subTitle}</div>
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        {s.bot.connected
          ? <HPill success><span className="dot pulse" />connected</HPill>
          : <HPill danger><span className="dot" />offline</HPill>}
        <span style={{ fontSize: 11, color: 'var(--text-2)', fontFamily: 'var(--mono)' }}>
          updated {fmtTimeAgo(s.bot.lastUpdate)}
        </span>
        <HPill>{s.bot.mode}</HPill>
        <HPill success={s.bot.status === 'running'} danger={s.bot.status !== 'running'}>
          <span className="dot" />{s.bot.status}
        </HPill>
        <button className="tbm-btn" onClick={onToggleTheme}
                aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>
          {nextThemeLabel}
        </button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────── left rail nav ──────────

const NAV = [
  { id: 'dash',   label: 'Dash',   icon: 'M2 9h6V2H2zM10 9h6V2h-6zM2 16h6v-7H2zM10 16h6v-7h-6z' },
  { id: 'pos',    label: 'Pos',    icon: 'M2 4h14M2 9h14M2 14h14' },
  { id: 'strat',  label: 'Strat',  icon: 'M2 16 7 8l3 5 5-9' },
  { id: 'trades', label: 'Trades', icon: 'M3 14l4-4 3 3 5-7M11 6h4v4' },
  { id: 'risk',   label: 'Risk',   icon: 'M9 2l7 4v4c0 4-3 7-7 8-4-1-7-4-7-8V6z' },
  { id: 'logs',   label: 'Logs',   icon: 'M3 3h12v12H3zM5 6h8M5 9h8M5 12h5' },
];

// ─────────────────────────────────────────── mobile topbar + tab bar ──────────

function HifiTopBarMobile({ theme, onToggleTheme }) {
  const { state } = useStore();
  useTick(15000);
  if (!state) return null;
  const s = state;
  return (
    <div className="tbm-top-mobile" style={{
      padding: 'calc(env(safe-area-inset-top, 0px) + 12px) 14px 10px',
      borderBottom: '1px solid var(--border-strong)',
      background: 'var(--bg-elev)',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      position: 'sticky', top: 0, zIndex: 5,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
        <div className="tbm-logo" aria-hidden="true">A</div>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 14, fontWeight: 600, letterSpacing: '-0.01em', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {s.bot.name}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-2)', fontFamily: 'var(--mono)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {s.bot.mode} · {s.bot.status} · {s.bot.symbols[0] || ''}{s.bot.symbols.length > 1 ? ` +${s.bot.symbols.length - 1}` : ''}
          </div>
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
        {s.bot.connected
          ? <HPill success><span className="dot pulse" />live</HPill>
          : <HPill danger><span className="dot" />off</HPill>}
        <button className="tbm-btn" onClick={onToggleTheme} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
                style={{ padding: '6px 8px' }}>
          {theme === 'dark' ? '☀' : '☾'}
        </button>
      </div>
    </div>
  );
}

function V2BottomTabBar({ tab, onTab }) {
  return (
    <nav role="tablist" aria-label="Dashboard sections" className="tbm-bottom-bar" style={{
      position: 'fixed', left: 0, right: 0, bottom: 0,
      paddingBottom: 'env(safe-area-inset-bottom, 0px)',
      background: 'var(--bg-elev)',
      borderTop: '1px solid var(--border-strong)',
      display: 'grid', gridTemplateColumns: `repeat(${NAV.length}, 1fr)`,
      zIndex: 10,
    }}>
      {NAV.map((n) => {
        const active = tab === n.id;
        return (
          <button key={n.id} onClick={() => onTab(n.id)}
                  role="tab" aria-selected={active} aria-label={n.label}
                  style={{
            border: 'none', background: 'transparent', cursor: 'pointer',
            color: active ? 'var(--accent)' : 'var(--text-3)',
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            gap: 3, padding: '10px 4px 12px', position: 'relative',
            transition: 'color 120ms', fontFamily: 'inherit',
          }}>
            {active && <span aria-hidden="true" style={{ position: 'absolute', top: 0, left: '30%', right: '30%', height: 2, background: 'var(--accent)', borderRadius: '0 0 2px 2px' }} />}
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d={n.icon} />
            </svg>
            <span style={{ fontSize: 9.5, fontWeight: 500, letterSpacing: '0.04em' }}>{n.label}</span>
          </button>
        );
      })}
    </nav>
  );
}

// ─────────────────────────────────────────── mobile views ──────────

function V2DashMobile({ overlays, setOverlays, selected, setSelected }) {
  const { state, range, setRange } = useStore();
  if (!state) return null;
  const s = state;

  const totalReturnPct = (s.totalPnl != null && s.initialBalance && s.initialBalance > 0)
    ? (s.totalPnl / s.initialBalance) * 100
    : null;

  const kpis = [
    ['sharpe', fmtNum(s.sharpe, 2)],
    ['win', s.winRate != null ? `${s.winRate.toFixed(0)}%` : '—'],
    ['risk', `${fmtNum(s.dynamicRisk.mult, 1)}x`],
    ['open', String(s.activePositions)],
    ['exposure', fmtUSD(s.totalPositionValue)],
    ['DD', `${s.maxDD.toFixed(1)}%`],
  ];

  return (
    <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* hero balance card */}
      <div className="tbm-card" style={{ padding: 16 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 4 }}>
          <span className="tbm-kicker">equity</span>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: s.todayPnl >= 0 ? 'var(--accent-2)' : 'var(--danger)' }}>
            {fmtUSD(s.todayPnl, { sign: true })} today
          </span>
        </div>
        <div style={{ fontSize: 30, fontWeight: 600, letterSpacing: '-0.02em' }}>{fmtUSD(s.balance)}</div>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 12, color: (s.totalPnl ?? 0) >= 0 ? 'var(--accent-2)' : 'var(--danger)' }}>
          {totalReturnPct != null ? `${fmtPct(totalReturnPct)} all time` : 'all-time —'}
        </div>
      </div>

      {/* 3-col KPI grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
        {kpis.map(([k, v]) => (
          <div key={k} className="tbm-card" style={{ padding: '8px 10px' }}>
            <div style={{ fontSize: 9, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.10em' }}>{k}</div>
            <div style={{ fontSize: 16, fontWeight: 600 }}>{v}</div>
          </div>
        ))}
      </div>

      {/* range chips + chart */}
      <div className="tbm-card" style={{ padding: 12, display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
          <span className="tbm-kicker">equity curve</span>
          <div style={{ display: 'flex', gap: 4 }}>
            {RANGES.map(r => <HPill key={r} active={r === range} onClick={() => setRange(r)}>{r}</HPill>)}
          </div>
        </div>
        <div style={{ height: 200 }}>
          <V2Chart data={s.equityCurve} overlays={overlays} trades={s.trades}
                   onSelectTrade={(id) => setSelected({ kind: 'trade', id })} />
        </div>
        <div style={{ display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
          <HPill active={overlays.trades} onClick={() => setOverlays({ ...overlays, trades: !overlays.trades })}>+ trades</HPill>
          <HPill active={overlays.drawdown} onClick={() => setOverlays({ ...overlays, drawdown: !overlays.drawdown })}>● drawdown</HPill>
        </div>
      </div>

      {/* positions stack */}
      <div className="tbm-card" style={{ padding: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
          <span className="tbm-kicker">open positions · {s.positions.length}</span>
          {s.positions.length > 0 && <span style={{ fontSize: 10, color: 'var(--text-3)' }}>tap to inspect</span>}
        </div>
        {s.positions.length === 0 ? (
          <div style={{ color: 'var(--text-3)', fontFamily: 'var(--mono)', fontSize: 12 }}>No open positions</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {s.positions.map(p => {
              const sel = selected.kind === 'position' && selected.symbol === p.symbol;
              return (
                <button key={p.id} onClick={() => setSelected({ kind: 'position', symbol: p.symbol })}
                        aria-pressed={sel}
                        aria-label={`${p.symbol} ${p.side} ${fmtUSD(p.pnl, { sign: true })} unrealized`}
                        style={{
                  textAlign: 'left', cursor: 'pointer',
                  background: sel ? 'var(--accent-soft)' : 'var(--bg-elev-2)',
                  border: sel ? '1px solid var(--accent)' : '1px solid var(--border)',
                  borderRadius: 10, padding: '10px 12px', color: 'var(--text)', fontFamily: 'inherit',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 600, fontSize: 13 }}>{p.symbol}</span>
                    <span className={`tbm-tag ${p.side === 'LONG' ? 'long' : 'short'}`}>{p.side}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginTop: 6 }}>
                    <span style={{ fontFamily: 'var(--font)', fontSize: 18, fontWeight: 600, color: p.pnl >= 0 ? 'var(--accent-2)' : 'var(--danger)', letterSpacing: '-0.02em' }}>
                      {fmtUSD(p.pnl, { sign: true })}
                    </span>
                    <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-3)' }}>
                      {fmtPct(p.pnlPct)}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* inline inspector — replaces the desktop side panel on mobile */}
      {(selected.kind === 'position' || selected.kind === 'trade') && (
        <div style={{ paddingTop: 4 }}>
          <div className="tbm-kicker" style={{ marginBottom: 8 }}>inspecting</div>
          {selected.kind === 'position' && <V2InspectPosition symbol={selected.symbol} />}
          {selected.kind === 'trade' && <V2InspectTrade id={selected.id} setSelected={setSelected} />}
        </div>
      )}
    </div>
  );
}

function V2PosViewMobile({ setSelected, setNavTab }) {
  const { state } = useStore();
  const s = state;
  return (
    <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <span className="tbm-kicker">positions</span>
        <span className="tbm-card-title" style={{ fontSize: 14 }}>Open · {s.positions.length}</span>
      </div>
      {s.positions.length === 0 ? (
        <div className="tbm-card" style={{ padding: 14, color: 'var(--text-3)', fontFamily: 'var(--mono)', fontSize: 12 }}>
          No open positions
        </div>
      ) : s.positions.map(p => (
        <button key={p.id}
                onClick={() => { setSelected({ kind: 'position', symbol: p.symbol }); setNavTab('dash'); }}
                className="tbm-card"
                aria-label={`Inspect ${p.symbol}`}
                style={{ textAlign: 'left', cursor: 'pointer', padding: 14, color: 'var(--text)', fontFamily: 'inherit' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontWeight: 600, fontSize: 14 }}>{p.symbol}</span>
              <span className={`tbm-tag ${p.side === 'LONG' ? 'long' : 'short'}`}>{p.side}</span>
            </div>
            <span style={{ fontSize: 18, fontWeight: 600, color: p.pnl >= 0 ? 'var(--accent-2)' : 'var(--danger)' }}>
              {fmtUSD(p.pnl, { sign: true })}
            </span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, marginTop: 10, fontFamily: 'var(--mono)', fontSize: 11 }}>
            <div>
              <div style={{ color: 'var(--text-3)', fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.10em' }}>entry</div>
              <div>{Number(p.entry).toLocaleString()}</div>
            </div>
            <div>
              <div style={{ color: 'var(--text-3)', fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.10em' }}>mark</div>
              <div>{Number(p.current).toLocaleString()}</div>
            </div>
            <div>
              <div style={{ color: 'var(--text-3)', fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.10em' }}>stop</div>
              <div>{p.trailSL ? Number(p.trailSL).toLocaleString() : '—'}</div>
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}

function V2TradesViewMobile() {
  const { state } = useStore();
  useTick(15000);
  const s = state;
  return (
    <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
        <span className="tbm-kicker">recent trades</span>
        <span className="tbm-card-title" style={{ fontSize: 14 }}>{s.trades.length}</span>
      </div>
      {s.trades.length === 0 ? (
        <div className="tbm-card" style={{ padding: 14, color: 'var(--text-3)', fontFamily: 'var(--mono)', fontSize: 12 }}>
          No trades yet
        </div>
      ) : s.trades.slice(0, 30).map(t => (
        <div key={t.id} className="tbm-card" style={{ padding: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <span style={{ fontWeight: 600, fontSize: 13 }}>{t.symbol}</span>
              <span className={`tbm-tag ${t.side === 'L' ? 'long' : 'short'}`}>{t.side === 'L' ? 'LONG' : 'SHORT'}</span>
              <span className="tbm-tag">{t.reason}</span>
            </div>
            <span style={{ fontSize: 16, fontWeight: 600, color: t.pnl >= 0 ? 'var(--accent-2)' : 'var(--danger)' }}>
              {fmtUSD(t.pnl, { sign: true })}
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--text-3)' }}>
            <span>{Number(t.entry).toLocaleString()} → {Number(t.exit).toLocaleString()}</span>
            <span>{fmtTimeAgo(t.time)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function V2Rail({ tab, onTab }) {
  const onKey = (e, idx) => {
    // Arrow-key nav per WAI-ARIA tablist pattern
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
      e.preventDefault();
      const next = NAV[(idx + 1) % NAV.length];
      onTab(next.id);
      // move focus to next button
      const sib = e.currentTarget.parentElement.querySelectorAll('[role="tab"]')[(idx + 1) % NAV.length];
      if (sib) sib.focus();
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
      e.preventDefault();
      const next = NAV[(idx - 1 + NAV.length) % NAV.length];
      onTab(next.id);
      const sib = e.currentTarget.parentElement.querySelectorAll('[role="tab"]')[(idx - 1 + NAV.length) % NAV.length];
      if (sib) sib.focus();
    }
  };
  return (
    <div role="tablist" aria-label="Dashboard sections" aria-orientation="vertical" style={{
      width: 64, background: 'var(--bg-elev)',
      borderRight: '1px solid var(--border-strong)',
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      padding: '18px 0', gap: 4,
    }}>
      <div className="tbm-logo" style={{ marginBottom: 16 }} aria-hidden="true">A</div>
      {NAV.map((n, idx) => {
        const active = tab === n.id;
        return (
          <button key={n.id} onClick={() => onTab(n.id)} title={n.label}
                  role="tab" aria-selected={active} aria-label={n.label}
                  tabIndex={active ? 0 : -1}
                  onKeyDown={(e) => onKey(e, idx)}
                  style={{
            width: 44, height: 48, borderRadius: 10,
            border: 'none', cursor: 'pointer',
            background: active ? 'var(--accent-soft)' : 'transparent',
            color: active ? 'var(--accent)' : 'var(--text-3)',
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            gap: 3, transition: 'background 120ms, color 120ms',
            position: 'relative',
            fontFamily: 'inherit',
            outline: 'none',
          }}
          onFocus={(e) => { e.currentTarget.style.boxShadow = '0 0 0 2px var(--border-focus)'; }}
          onBlur={(e) => { e.currentTarget.style.boxShadow = 'none'; }}
          onMouseEnter={(e) => { if (!active) e.currentTarget.style.color = 'var(--text)'; }}
          onMouseLeave={(e) => { if (!active) e.currentTarget.style.color = 'var(--text-3)'; }}>
            {active && <span style={{ position: 'absolute', left: -6, top: 10, bottom: 10, width: 2, borderRadius: 2, background: 'var(--accent)' }} />}
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d={n.icon} />
            </svg>
            <span style={{ fontSize: 9, fontWeight: 500, letterSpacing: '0.04em' }}>{n.label}</span>
          </button>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────── chart ──────────

function V2Chart({ data, overlays, onSelectTrade, trades }) {
  const ref = useRef(null);
  const [hover, setHover] = useState(null);
  const W = 820, H = 320;
  const pad = { l: 50, r: 14, t: 16, b: 26 };
  if (!data || data.length < 2) {
    return (
      <div style={{ flex: 1, minHeight: 320, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-3)', fontFamily: 'var(--mono)', fontSize: 12 }}>
        not enough equity data yet
      </div>
    );
  }

  const xs = data.map(d => d.ts);
  const ys = data.map(d => d.v);
  const minX = xs[0], maxX = xs[xs.length - 1];
  const minY = safeMin(ys), maxY = safeMax(ys);
  const yPad = (maxY - minY) * 0.12 || Math.max(1, maxY * 0.01);
  const y0 = minY - yPad, y1 = maxY + yPad;
  const w = W - pad.l - pad.r;
  const h = H - pad.t - pad.b;
  const sx = t => pad.l + ((t - minX) / ((maxX - minX) || 1)) * w;
  const sy = v => pad.t + (1 - (v - y0) / ((y1 - y0) || 1)) * h;

  const pts = data.map(d => [sx(d.ts), sy(d.v)]);
  let path = `M ${pts[0][0]} ${pts[0][1]}`;
  for (let i = 1; i < pts.length; i++) {
    const cx = (pts[i - 1][0] + pts[i][0]) / 2;
    path += ` Q ${cx} ${pts[i - 1][1]} ${cx} ${(pts[i - 1][1] + pts[i][1]) / 2} T ${pts[i][0]} ${pts[i][1]}`;
  }
  const areaPath = `${path} L ${pts[pts.length - 1][0]} ${pad.t + h} L ${pts[0][0]} ${pad.t + h} Z`;
  // Benchmark overlay was removed: the prior version computed the line as a
  // scaled+wiggled function of the bot's *own equity*, which made it
  // mathematically impossible to lose against the benchmark. A real
  // per-symbol buy-and-hold series needs to come from data_provider history;
  // tracked in docs/monitoring.md as a follow-up.

  // drawdown shading
  let runMax = ys[0];
  const peaks = ys.map(v => (runMax = Math.max(runMax, v)));
  const ddPath = peaks.map((p, i) => `${i === 0 ? 'M' : 'L'} ${pts[i][0]} ${sy(p)}`).join(' ')
    + ' ' + pts.slice().reverse().map(([x, y]) => `L ${x} ${y}`).join(' ') + ' Z';

  // trade markers — only trades with valid exit_time inside the visible range
  const markers = (trades || [])
    .filter(t => t && t.time != null && t.time >= minX && t.time <= maxX)
    .slice(0, 24)
    .map(t => {
      let bestI = 0, bestD = Infinity;
      for (let i = 0; i < data.length; i++) {
        const d = Math.abs(data[i].ts - t.time);
        if (d < bestD) { bestD = d; bestI = i; }
      }
      return { id: t.id, x: pts[bestI][0], y: pts[bestI][1], win: t.pnl >= 0, pnl: t.pnl, symbol: t.symbol, time: t.time };
    });

  const yticks = [0.25, 0.5, 0.75].map(p => y0 + p * (y1 - y0));
  const xticks = 5;

  function onMove(e) {
    if (!ref.current) return;
    const r = ref.current.getBoundingClientRect();
    if (r.width === 0) return;
    // SVG uses preserveAspectRatio="xMidYMid meet" now — viewBox aspect is
    // preserved, so we map cursor to viewBox coords with a single scale.
    const xVB = ((e.clientX - r.left) / r.width) * W;
    if (xVB < pad.l || xVB > pad.l + w) { setHover(null); return; }
    const t = minX + ((xVB - pad.l) / w) * (maxX - minX);
    let best = 0, dmin = Infinity;
    for (let i = 0; i < data.length; i++) {
      const dd = Math.abs(data[i].ts - t);
      if (dd < dmin) { dmin = dd; best = i; }
    }
    setHover({ i: best, x: pts[best][0], y: pts[best][1] });
  }

  return (
    <div style={{ position: 'relative', flex: 1, minHeight: 320 }}>
      <svg ref={ref} width="100%" height="100%" viewBox={`0 0 ${W} ${H}`}
           preserveAspectRatio="xMidYMid meet"
           onMouseMove={onMove} onMouseLeave={() => setHover(null)}
           role="img" aria-label="Equity curve over time"
           style={{ display: 'block', cursor: 'crosshair' }}>
        {yticks.map((tk, i) => (
          <line key={i} x1={pad.l} x2={W - pad.r} y1={sy(tk)} y2={sy(tk)} stroke="var(--grid)" strokeDasharray="2 4" />
        ))}
        {yticks.map((tk, i) => (
          <text key={'y' + i} x={pad.l - 8} y={sy(tk) + 3} textAnchor="end" className="tbm-ax">
            ${tk >= 1000 ? (tk / 1000).toFixed(1) + 'k' : tk.toFixed(0)}
          </text>
        ))}
        {Array.from({ length: xticks }, (_, i) => {
          const t = minX + (i / (xticks - 1)) * (maxX - minX);
          const x = sx(t);
          const d = new Date(t);
          return <text key={'x' + i} x={x} y={H - 8} textAnchor="middle" className="tbm-ax">
            {d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
          </text>;
        })}
        {overlays.drawdown && <path d={ddPath} fill="var(--danger)" opacity="0.08" />}
        <path d={areaPath} fill="var(--accent-2)" opacity="0.10" />
        <path d={path} fill="none" stroke="var(--accent-2)" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
        {overlays.trades && markers.map(m => (
          <g key={m.id}
             onClick={() => onSelectTrade && onSelectTrade(m.id)}
             onKeyDown={(e) => {
               if (e.key === 'Enter' || e.key === ' ') {
                 e.preventDefault();
                 onSelectTrade && onSelectTrade(m.id);
               }
             }}
             tabIndex={0}
             role="button"
             aria-label={`Trade ${m.symbol} ${m.win ? 'win' : 'loss'} ${m.pnl >= 0 ? '+' : ''}${m.pnl.toFixed(2)} on ${new Date(m.time).toLocaleDateString()}`}
             style={{ cursor: 'pointer', outline: 'none' }}>
            <circle cx={m.x} cy={m.y} r="5" fill={m.win ? 'var(--accent-2)' : 'var(--danger)'} stroke="var(--bg-elev)" strokeWidth="2" />
          </g>
        ))}
        {hover && (
          <>
            <line x1={hover.x} x2={hover.x} y1={pad.t} y2={pad.t + h} stroke="var(--text-3)" strokeDasharray="2 3" opacity="0.6" />
            <circle cx={hover.x} cy={hover.y} r="4" fill="var(--accent-2)" stroke="var(--bg-elev)" strokeWidth="2" />
          </>
        )}
      </svg>
      <div style={{ position: 'absolute', left: 60, top: 14, display: 'flex', gap: 14, fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--mono)' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 10, height: 2, background: 'var(--accent-2)' }} />equity
        </span>
        {overlays.trades && <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--accent-2)' }} />trade
        </span>}
        {overlays.drawdown && <span style={{ color: 'var(--danger)' }}>drawdown</span>}
      </div>
      {hover && data[hover.i] && (
        <div className="tbm-tooltip" style={{
          left: Math.min((hover.x / W) * 100, 80) + '%',
          top:  Math.max((hover.y / H) * 100 - 6, 2) + '%',
          transform: 'translate(8px, -100%)',
        }}>
          <div style={{ color: 'var(--text-3)', fontSize: 10 }}>
            {new Date(data[hover.i].ts).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
          </div>
          <div style={{ color: 'var(--accent-2)', fontWeight: 600 }}>${Number(data[hover.i].v).toFixed(2)}</div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────── dash main ──────────

const RANGES = ['1D', '1W', '1M', '3M', '1Y'];
const RANGE_LABEL = { '1D': 'last 24 hours', '1W': 'last 7 days', '1M': 'last 30 days', '3M': 'last 90 days', '1Y': 'last 12 months' };

function V2Main({ overlays, setOverlays, selected, setSelected }) {
  const { state, range, setRange } = useStore();
  if (!state) return null;
  const s = state;

  const totalReturnPct = (s.totalPnl != null && s.initialBalance && s.initialBalance > 0)
    ? (s.totalPnl / s.initialBalance) * 100
    : null;

  return (
    <div style={{ padding: 22, display: 'flex', flexDirection: 'column', gap: 14, minWidth: 0 }}>
      {/* KPI strip */}
      <div style={{ display: 'flex', gap: 22, paddingBottom: 14, borderBottom: '1px solid var(--border)', alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <HKPI label="equity" value={fmtUSD(s.balance)} sub={`${fmtUSD(s.todayPnl, { sign: true })} today`}
              color={s.todayPnl >= 0 ? 'var(--accent-2)' : 'var(--danger)'} />
        <HKPI label="all-time"
              value={totalReturnPct != null ? fmtPct(totalReturnPct) : '—'}
              sub={s.totalPnl != null ? fmtUSD(s.totalPnl, { sign: true }) : 'no baseline'}
              color={(s.totalPnl ?? 0) >= 0 ? 'var(--accent-2)' : 'var(--danger)'} />
        <HKPI label="open exposure" value={fmtUSD(s.totalPositionValue)} sub={s.activePositions ? `${s.activePositions} pos` : 'no pos'} />
        <HKPI label="sharpe" value={fmtNum(s.sharpe, 2)} sub="annualized" />
        <HKPI label="win rate"
              value={s.winRate != null ? `${s.winRate.toFixed(0)}%` : '—'}
              sub={`${s.totalTrades} trades`} />
        <HKPI label="dyn risk" value={`${fmtNum(s.dynamicRisk.mult, 1)}x`} sub={s.dynamicRisk.reason} />
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignSelf: 'center' }}>
          {RANGES.map(r => <HPill key={r} active={r === range} onClick={() => setRange(r)}>{r}</HPill>)}
        </div>
      </div>

      {/* Hero chart */}
      <div className="tbm-card" style={{ flex: 1, minHeight: 380, display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span className="tbm-kicker">01</span>
            <span className="tbm-card-title">Equity · {RANGE_LABEL[range]}</span>
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            <HPill active={overlays.trades} onClick={() => setOverlays({ ...overlays, trades: !overlays.trades })}>+ trades</HPill>
            <HPill active={overlays.drawdown} onClick={() => setOverlays({ ...overlays, drawdown: !overlays.drawdown })}>● drawdown</HPill>
          </div>
        </div>
        <V2Chart data={s.equityCurve} overlays={overlays} trades={s.trades}
                 onSelectTrade={(id) => setSelected({ kind: 'trade', id })} />
      </div>

      {/* Positions strip */}
      <div className="tbm-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span className="tbm-kicker">02</span>
            <span className="tbm-card-title">Open positions · {s.positions.length}</span>
          </div>
          <span style={{ fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.10em' }}>
            click a card → inspect →
          </span>
        </div>
        {s.positions.length === 0 ? (
          <div style={{ padding: '22px 4px', color: 'var(--text-3)', fontFamily: 'var(--mono)', fontSize: 12 }}>
            No open positions
          </div>
        ) : (
          <div style={{
            display: 'grid',
            gridTemplateColumns: `repeat(${Math.min(s.positions.length, 4)}, 1fr)`,
            gap: 12,
          }}>
            {s.positions.map(p => {
              const sel = selected.kind === 'position' && selected.symbol === p.symbol;
              return (
                <button key={p.id}
                        onClick={() => setSelected({ kind: 'position', symbol: p.symbol })}
                        aria-pressed={sel}
                        aria-label={`${p.symbol} ${p.side} ${fmtUSD(p.pnl, { sign: true })} unrealized`}
                        style={{
                  textAlign: 'left', cursor: 'pointer',
                  background: sel ? 'var(--accent-soft)' : 'var(--bg-elev-2)',
                  border: sel ? '1px solid var(--accent)' : '1px solid var(--border)',
                  boxShadow: sel ? '0 0 0 3px color-mix(in srgb, var(--accent) 18%, transparent)' : 'none',
                  borderRadius: 10, padding: '12px 14px',
                  transition: 'all 120ms', color: 'var(--text)', fontFamily: 'inherit',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 600, fontSize: 13 }}>{p.symbol}</span>
                    <span className={`tbm-tag ${p.side === 'LONG' ? 'long' : 'short'}`}>{p.side}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginTop: 8 }}>
                    <span style={{ fontFamily: 'var(--font)', fontSize: 22, fontWeight: 600, color: p.pnl >= 0 ? 'var(--accent-2)' : 'var(--danger)', letterSpacing: '-0.02em' }}>
                      {fmtUSD(p.pnl, { sign: true })}
                    </span>
                    <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-3)' }}>
                      {fmtPct(p.pnlPct)}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────── inspector ──────────

function V2Inspector({ selected, setSelected }) {
  const { state } = useStore();
  if (!state) return null;
  return (
    <div style={{ borderLeft: '1px solid var(--border-strong)', background: 'var(--bg-elev)', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <div style={{ padding: '14px 18px 12px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span className="tbm-kicker">inspecting</span>
        <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--mono)', textTransform: 'uppercase', letterSpacing: '0.10em' }}>
          {selected.kind === 'position' && `position · ${selected.symbol}`}
          {selected.kind === 'trade' && `trade · ${selected.id}`}
          {selected.kind === 'strategy' && 'strategy'}
          {selected.kind === 'none' && '—'}
        </span>
      </div>
      <div style={{ flex: 1, overflow: 'auto', padding: 18 }}>
        {selected.kind === 'position' && <V2InspectPosition symbol={selected.symbol} />}
        {selected.kind === 'trade' && <V2InspectTrade id={selected.id} setSelected={setSelected} />}
        {selected.kind === 'strategy' && <V2InspectStrategy />}
        {selected.kind === 'none' && (
          <div style={{ color: 'var(--text-3)', fontFamily: 'var(--mono)', fontSize: 12 }}>
            Click a position card or a trade marker on the chart to inspect.
          </div>
        )}
      </div>
    </div>
  );
}

// Hook used by Shell: resets `selected` when a *position* target disappears
// so the inspector header never points at a missing position. Lifting this
// to an effect (instead of shadowing inside `V2Inspector`) keeps the
// parent's `selected` truthful across renders — important when the same
// symbol's position closes and reopens with a different side.
//
// Note: trades are intentionally NOT reset here. The trades list is a sliding
// window (last N fetched), so a previously-selected trade can roll out of
// view on every refetch. `V2InspectTrade` renders its own explicit "no
// longer in view" placeholder + back button when it can't find the id —
// that's the right UX, and resetting here would short-circuit it.
function useStaleSelectionReset(state, selected, setSelected, setAutoSelected) {
  useEffect(() => {
    if (!state) return;
    if (selected.kind === 'position'
        && !state.positions.find(p => p.symbol === selected.symbol)) {
      setSelected({ kind: 'none' });
      // Allow auto-select to re-engage on the next opened position.
      setAutoSelected(false);
    }
  }, [state, selected, setSelected, setAutoSelected]);
}

function V2InspectPosition({ symbol }) {
  const { state } = useStore();
  const s = state;
  const p = s.positions.find(x => x.symbol === symbol);
  if (!p) return <div style={{ color: 'var(--text-3)' }}>No open position</div>;
  const tp = p.target.tp;
  const sl = p.target.trail || p.target.sl;
  const portionPct = (s.balance && s.balance > 0) ? (p.size * p.current / s.balance) * 100 : null;
  const stopDistPct = p.current && sl ? ((sl - p.current) / p.current) * 100 : null;
  const tpDistPct = p.current && tp ? ((tp - p.current) / p.current) * 100 : null;
  const risk = sl ? Math.abs(p.entry - sl) * p.size : null;
  const rr = (tp && sl) ? Math.abs((tp - p.entry) / (p.entry - sl || 1)) : null;
  const rrFill = rr ? Math.min(100, (rr / 4) * 100) : 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
          <span style={{ fontSize: 22, fontWeight: 600, letterSpacing: '-0.01em' }}>{p.symbol}</span>
          <span className={`tbm-tag ${p.side === 'LONG' ? 'long' : 'short'}`}>{p.side}</span>
        </div>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-3)' }}>
          {fmtNum(p.size, 4)} · entry {Number(p.entry).toLocaleString()}
        </div>
      </div>

      <div className="tbm-card" style={{ padding: 14 }}>
        <div className="tbm-h2">Unrealized</div>
        <div style={{ fontSize: 32, fontWeight: 600, color: p.pnl >= 0 ? 'var(--accent-2)' : 'var(--danger)', letterSpacing: '-0.02em' }}>
          {fmtUSD(p.pnl, { sign: true })}
        </div>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-3)' }}>
          mark {Number(p.current).toLocaleString()} · {fmtPct(p.pnlPct)}
        </div>
      </div>

      <div className="tbm-card" style={{ padding: 14 }}>
        <div className="tbm-h2">Stops & exposure</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <HKPI label="stop" value={sl ? Number(sl).toLocaleString() : '—'} sub={stopDistPct != null ? fmtPct(stopDistPct) : 'none'} color="var(--danger)" />
          <HKPI label="target" value={tp ? Number(tp).toLocaleString() : '—'} sub={tpDistPct != null ? fmtPct(tpDistPct) : 'none'} color="var(--accent-2)" />
          <HKPI label="size" value={fmtUSD(p.size * p.current)} sub={portionPct != null ? `${portionPct.toFixed(0)}% port` : '—'} />
          <HKPI label="risk" value={risk != null ? fmtUSD(risk) : '—'} sub="if SL hit" />
        </div>
        {rr !== null && (
          <div style={{ marginTop: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-3)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.10em' }}>
              <span>r/r ratio</span><span style={{ color: 'var(--text)' }}>1 : {rr.toFixed(1)}</span>
            </div>
            <div className="tbm-bar"><span style={{ width: `${rrFill}%`, background: 'var(--accent-2)' }} /></div>
          </div>
        )}
      </div>

      <div className="tbm-card" style={{ padding: 14 }}>
        <div className="tbm-h2">Trail / breakeven</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <HRow k="trail SL" v={p.trailSL ? Number(p.trailSL).toLocaleString() : '—'} />
          <HRow k="breakeven" v={p.breakeven ? '✓ active' : '—'} color={p.breakeven ? 'var(--accent-2)' : 'var(--text-3)'} />
          <HRow k="MFE" v={fmtUSD(p.mfe, { sign: true })} color="var(--accent-2)" />
          <HRow k="MAE" v={fmtUSD(p.mae, { sign: true })} color="var(--danger)" />
          <HRow k="age" v={`${Math.floor(p.ageMs / 3600000)}h ${Math.floor(p.ageMs % 3600000 / 60000)}m`} />
        </div>
      </div>

      {p.strategy && (
        <div className="tbm-card" style={{ padding: 14 }}>
          <div className="tbm-h2">Strategy · {p.strategy}</div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-2)', lineHeight: 1.6 }}>
            <div>signal · <span style={{ color: p.side === 'LONG' ? 'var(--accent-2)' : 'var(--danger)' }}>{p.signal}</span>{p.confidence != null && <> · {p.confidence.toFixed(2)} conf</>}</div>
          </div>
        </div>
      )}
    </div>
  );
}

function V2InspectTrade({ id, setSelected }) {
  const { state } = useStore();
  useTick(15000);
  const s = state;
  const t = s.trades.find(x => x.id === id);
  if (!t) {
    // Don't silently fall back to a different trade — that would silently
    // change the financial data the user is looking at after a refetch.
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ color: 'var(--text-3)', fontFamily: 'var(--mono)', fontSize: 12 }}>
          That trade is no longer in view (older than the most recent {s.trades.length} fetched).
        </div>
        <button
          onClick={() => setSelected(s.positions[0]
            ? { kind: 'position', symbol: s.positions[0].symbol }
            : { kind: 'none' })}
          style={{ alignSelf: 'flex-start', background: 'transparent', border: 'none',
                   color: 'var(--accent)', fontSize: 11, cursor: 'pointer', padding: 0,
                   fontFamily: 'var(--mono)' }}>
          ← back
        </button>
      </div>
    );
  }
  const movePct = t.entry ? ((t.exit - t.entry) / t.entry) * 100 * (t.side === 'L' ? 1 : -1) : 0;
  const onBack = () => {
    if (s.positions[0]) {
      setSelected({ kind: 'position', symbol: s.positions[0].symbol });
    } else {
      setSelected({ kind: 'none' });
    }
  };
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <button onClick={onBack}
              style={{ alignSelf: 'flex-start', background: 'transparent', border: 'none', color: 'var(--accent)', fontSize: 11, cursor: 'pointer', padding: 0, fontFamily: 'var(--mono)' }}>
        ← back
      </button>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 22, fontWeight: 600 }}>{t.symbol}</span>
          <span className={`tbm-tag ${t.side === 'L' ? 'long' : 'short'}`}>{t.side === 'L' ? 'LONG' : 'SHORT'}</span>
          <span className="tbm-tag">{t.reason}</span>
        </div>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-3)', marginTop: 4 }}>closed · {fmtTimeAgo(t.time)}</div>
      </div>
      <div className="tbm-card" style={{ padding: 14 }}>
        <div className="tbm-h2">Realized P&L</div>
        <div style={{ fontSize: 32, fontWeight: 600, color: t.pnl >= 0 ? 'var(--accent-2)' : 'var(--danger)', letterSpacing: '-0.02em' }}>
          {fmtUSD(t.pnl, { sign: true })}
        </div>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-3)' }}>
          {fmtPct(movePct)}
        </div>
      </div>
      <div className="tbm-card" style={{ padding: 14 }}>
        <div className="tbm-h2">Execution</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <HRow k="qty" v={fmtNum(t.qty, 4)} />
          <HRow k="entry" v={Number(t.entry).toLocaleString()} />
          <HRow k="exit" v={Number(t.exit).toLocaleString()} />
          <HRow k="reason" v={t.reason} />
        </div>
      </div>
    </div>
  );
}

function V2InspectStrategy() {
  const { state } = useStore();
  const s = state;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div>
        <div style={{ fontSize: 22, fontWeight: 600 }}>{s.bot.name}</div>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-3)' }}>
          {s.bot.symbols.join(' · ')} · {s.bot.timeframe}
        </div>
      </div>
      <div className="tbm-card" style={{ padding: 14 }}>
        <div className="tbm-h2">Recent signal</div>
        {s.positions[0] ? (
          <>
            <div style={{ fontSize: 28, fontWeight: 600, color: s.positions[0].side === 'LONG' ? 'var(--accent-2)' : 'var(--danger)' }}>
              {s.positions[0].signal}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-3)' }}>
              {s.positions[0].symbol}
              {s.positions[0].confidence != null && <> · conf {s.positions[0].confidence.toFixed(2)}</>}
            </div>
          </>
        ) : (
          <div style={{ color: 'var(--text-3)' }}>No open positions yet</div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────── other tabs ──────────

function V2PosView({ setSelected, setNavTab }) {
  const { state } = useStore();
  const s = state;
  return (
    <div style={{ padding: 22, overflow: 'auto' }}>
      <div className="tbm-card">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
          <span className="tbm-kicker">positions</span>
          <span className="tbm-card-title">Open positions · {s.positions.length}</span>
        </div>
        {s.positions.length === 0 ? (
          <div style={{ padding: '20px 0', color: 'var(--text-3)', fontFamily: 'var(--mono)', fontSize: 12 }}>No open positions</div>
        ) : (
          <table className="tbm-tbl">
            <thead>
              <tr>
                <th>Symbol</th><th>Side</th><th>Size</th><th>Entry</th><th>Current</th>
                <th>P&L</th><th>Trail SL</th><th>BE</th><th>MFE</th><th>MAE</th>
              </tr>
            </thead>
            <tbody>
              {s.positions.map(p => (
                <tr key={p.id} className="clickable"
                    onClick={() => { setSelected({ kind: 'position', symbol: p.symbol }); setNavTab('dash'); }}>
                  <td className="strong">{p.symbol}</td>
                  <td><span className={`tbm-tag ${p.side === 'LONG' ? 'long' : 'short'}`}>{p.side}</span></td>
                  <td>{fmtNum(p.size, 4)}</td>
                  <td>{Number(p.entry).toLocaleString()}</td>
                  <td>{Number(p.current).toLocaleString()}</td>
                  <td style={{ color: p.pnl >= 0 ? 'var(--accent-2)' : 'var(--danger)', fontWeight: 600 }}>
                    {fmtUSD(p.pnl, { sign: true })}{' '}
                    <span style={{ color: 'var(--text-3)', fontWeight: 400 }}>{fmtPct(p.pnlPct)}</span>
                  </td>
                  <td>{p.trailSL ? Number(p.trailSL).toLocaleString() : '—'}</td>
                  <td style={{ color: p.breakeven ? 'var(--accent-2)' : 'var(--text-3)' }}>{p.breakeven ? '✓' : '—'}</td>
                  <td style={{ color: 'var(--accent-2)' }}>{fmtUSD(p.mfe, { sign: true })}</td>
                  <td style={{ color: 'var(--danger)' }}>{fmtUSD(p.mae, { sign: true })}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function V2StratView() {
  const { state } = useStore();
  const s = state;
  // Confidence bars use real metrics where available; each row carries its
  // own raw value as the `sub` label so the bar (a normalised proxy) can't
  // be mistaken for the real number.
  const conf = s.positions[0]?.confidence ?? null;
  const rsi = numOrNull(s.rsi) ?? 50;
  const winRatePct = s.winRate != null ? s.winRate : null;
  // Sharpe bar visualises the magnitude on 0..3+. Negative Sharpe rendered red
  // so a losing streak doesn't look identical to "no data yet" (null → bar
  // hidden, sub label "—"). Using absolute value lets the bar fill toward the
  // expected scale; the colour communicates direction.
  const sharpeKnown = s.sharpe != null;
  const sharpeNorm = sharpeKnown
    ? Math.max(0, Math.min(100, (Math.abs(s.sharpe) / 3) * 100))
    : 0;
  const sharpeColor = !sharpeKnown
    ? 'var(--text-3)'
    : (s.sharpe < 0 ? 'var(--danger)' : 'var(--accent-2)');
  const sharpeSub = sharpeKnown ? fmtNum(s.sharpe, 2) : 'no history yet';
  return (
    <div className="tbm-2col-grid" style={{ padding: 22, display: 'grid', gap: 14, overflow: 'auto' }}>
      <div className="tbm-card" style={{ padding: 18 }}>
        <div className="tbm-h2">Bot</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <HRow k="strategy" v={s.bot.name} />
          <HRow k="symbols" v={s.bot.symbols.join(', ')} />
          <HRow k="timeframe" v={s.bot.timeframe} />
          <HRow k="mode" v={s.bot.mode} />
          <HRow k="status" v={s.bot.status} color={s.bot.status === 'running' ? 'var(--accent-2)' : 'var(--text-3)'} />
          <HRow k="uptime" v={s.bot.uptime} />
          <HRow k="EMA trend" v={s.emaTrend} color="var(--accent-2)" />
        </div>
      </div>
      <div className="tbm-card" style={{ padding: 18 }}>
        <div className="tbm-h2">
          Latest model signal
          {conf != null && <span style={{ color: 'var(--text-2)' }}> · conf {conf.toFixed(2)}</span>}
        </div>
        <HBarLabeled k="ML confidence (latest position)"
                     v={conf != null ? Math.round(conf * 100) : 0} max={100}
                     sub={conf != null ? conf.toFixed(2) : 'no open position'}
                     color="var(--accent-2)" />
        <div style={{ height: 6 }} />
        <HBarLabeled k={`RSI(14) · ${s.bot.symbols[0] || ''}`}
                     v={rsi} max={100} sub={rsi.toFixed(1)} color="var(--accent)" />
        <div style={{ height: 6 }} />
        <HBarLabeled k="Historical win rate"
                     v={winRatePct ?? 0} max={100}
                     sub={winRatePct != null ? `${winRatePct.toFixed(0)}%` : 'no trades yet'}
                     color="var(--accent-2)" />
        <div style={{ height: 6 }} />
        <HBarLabeled k="Sharpe (|value| normalised, 0-3+)"
                     v={sharpeNorm} max={100}
                     sub={sharpeSub}
                     color={sharpeColor} />
        <div style={{ marginTop: 12, fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--mono)', lineHeight: 1.5 }}>
          Bars are aggregate / latest-position proxies, not per-signal factors.
        </div>
      </div>
    </div>
  );
}

function V2TradesView() {
  const { state } = useStore();
  useTick(15000);
  const s = state;
  const [filter, setFilter] = useState('all');
  const filtered = s.trades.filter(t =>
    filter === 'all' ? true :
    filter === 'wins' ? t.pnl > 0 :
    filter === 'losses' ? t.pnl < 0 :
    t.symbol === filter
  );
  const symbols = Array.from(new Set(s.trades.map(t => t.symbol))).slice(0, 4);
  const wins = s.trades.filter(t => t.pnl > 0).length;
  const total = s.trades.length;
  const totalPnl = s.trades.reduce((a, t) => a + t.pnl, 0);
  const avgWin = wins ? s.trades.filter(t => t.pnl > 0).reduce((a, t) => a + t.pnl, 0) / wins : 0;
  const avgLoss = (total - wins) ? s.trades.filter(t => t.pnl < 0).reduce((a, t) => a + t.pnl, 0) / (total - wins) : 0;

  // The Trades view's stats cover the most recent N fetched trades (limit=50
  // on the bundled state endpoint). Lifetime totals live in the Dash KPI strip.
  // Labelling the KPIs "(last N)" prevents the trader from comparing them to
  // the global win-rate / total-trades values and getting confused.
  const recentLabel = `last ${total}`;
  return (
    <div style={{ padding: 22, overflow: 'auto' }}>
      <div style={{ display: 'flex', gap: 24, marginBottom: 16, flexWrap: 'wrap' }}>
        <HKPI label={`trades (${recentLabel})`} value={total} size="md" />
        <HKPI label={`wins (${recentLabel})`}
              value={total ? `${wins} (${(wins / total * 100).toFixed(0)}%)` : '0'}
              color="var(--accent-2)" size="md" />
        <HKPI label={`losses (${recentLabel})`} value={total - wins} color="var(--danger)" size="md" />
        <HKPI label={`net (${recentLabel})`}
              value={fmtUSD(totalPnl, { sign: true })}
              color={totalPnl >= 0 ? 'var(--accent-2)' : 'var(--danger)'} size="md" />
        <HKPI label={`avg win (${recentLabel})`}
              value={total ? fmtUSD(avgWin, { sign: true }) : '—'}
              color="var(--accent-2)" size="md" />
        <HKPI label={`avg loss (${recentLabel})`}
              value={total - wins ? fmtUSD(avgLoss, { sign: true }) : '—'}
              color="var(--danger)" size="md" />
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          {['all', 'wins', 'losses', ...symbols].map(f => <HPill key={f} active={f === filter} onClick={() => setFilter(f)}>{f}</HPill>)}
        </div>
      </div>
      <div className="tbm-card">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
          <span className="tbm-kicker">trades</span>
          <span className="tbm-card-title">History · {filtered.length}</span>
        </div>
        {filtered.length === 0 ? (
          <div style={{ padding: '20px 0', color: 'var(--text-3)', fontFamily: 'var(--mono)', fontSize: 12 }}>No trades match filter</div>
        ) : (
          <table className="tbm-tbl">
            <thead>
              <tr>
                <th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Exit</th>
                <th>P&L</th><th>Reason</th><th>Time</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(t => (
                <tr key={t.id}>
                  <td className="strong">{t.symbol}</td>
                  <td><span className={`tbm-tag ${t.side === 'L' ? 'long' : 'short'}`}>{t.side === 'L' ? 'LONG' : 'SHORT'}</span></td>
                  <td>{fmtNum(t.qty, 4)}</td>
                  <td>{Number(t.entry).toLocaleString()}</td>
                  <td>{Number(t.exit).toLocaleString()}</td>
                  <td style={{ color: t.pnl >= 0 ? 'var(--accent-2)' : 'var(--danger)', fontWeight: 600 }}>{fmtUSD(t.pnl, { sign: true })}</td>
                  <td><span className="tbm-tag">{t.reason}</span></td>
                  <td>{fmtTimeAgo(t.time)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function V2RiskView() {
  const { state } = useStore();
  const s = state;
  const ddPct = Math.abs(s.maxDD);
  const volNorm = Math.max(0, Math.min(100, s.volatility / 2));
  const ddBar = Math.max(0, Math.min(100, ddPct * 5));
  const exposurePct = (s.balance && s.balance > 0)
    ? (s.totalPositionValue / s.balance) * 100
    : null;
  const winRateBar = s.winRate ?? 0;
  const maxOpen = s.bot.maxOpenPositions; // null if not configured
  return (
    <div className="tbm-2col-grid" style={{ padding: 22, display: 'grid', gap: 14, overflow: 'auto' }}>
      <div className="tbm-card" style={{ padding: 18 }}>
        <div className="tbm-h2">Dynamic risk · {fmtNum(s.dynamicRisk.mult, 2)}x</div>
        <div style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 14, lineHeight: 1.5 }}>
          Position sizing scales with drawdown, volatility, and cooldowns. Current: <strong style={{ color: 'var(--accent-2)' }}>{s.dynamicRisk.reason}</strong>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <HBarLabeled k="Max drawdown (period)" v={ddBar} max={100} sub={`${ddPct.toFixed(1)}%`} color="var(--danger)" />
          <HBarLabeled k="Volatility (annualised)" v={volNorm} max={100} sub={`${s.volatility.toFixed(1)}%`} color="var(--accent)" />
          <HBarLabeled k="Win rate" v={winRateBar} max={100} sub={s.winRate != null ? `${s.winRate.toFixed(0)}%` : '—'} color="var(--accent-2)" />
          <HBarLabeled k="Dynamic factor" v={Math.max(0, Math.min(100, (s.dynamicRisk.mult / 2) * 100))} max={100} sub={`${fmtNum(s.dynamicRisk.mult, 2)}x`} color="var(--accent-2)" />
        </div>
      </div>
      <div className="tbm-card" style={{ padding: 18 }}>
        <div className="tbm-h2">Exposure caps</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <HRow k="open positions" v={maxOpen != null ? `${s.activePositions} / ${maxOpen}` : `${s.activePositions}`} sub={maxOpen == null ? 'cap not set' : null} />
          <HRow k="total exposure" v={fmtUSD(s.totalPositionValue)} sub={exposurePct != null ? `${exposurePct.toFixed(0)}% of equity` : ''} />
          <HRow k="risk / trade"
                v={s.riskPerTrade != null ? `${fmtNum(s.riskPerTrade, 2)}%` : '—'}
                sub={s.riskPerTrade != null && s.balance
                       ? `≈ ${fmtUSD(s.balance * s.riskPerTrade / 100)}`
                       : (s.riskPerTrade == null ? 'cap not configured' : '')} />
          <HRow k="margin available" v={fmtUSD(s.availableMargin)} />
        </div>
      </div>
      <div className="tbm-card" style={{ padding: 18, gridColumn: '1 / -1' }}>
        <div className="tbm-h2">Risk metrics</div>
        <div className="tbm-metrics-grid" style={{ display: 'grid', gap: 18 }}>
          <HKPI label="sharpe" value={fmtNum(s.sharpe, 2)} sub="annualized" />
          <HKPI label="max DD" value={`${s.maxDD.toFixed(1)}%`} color="var(--danger)" />
          <HKPI label="curr DD" value={`${s.currentDD.toFixed(1)}%`} color="var(--warn)" />
          <HKPI label="volatility" value={`${s.volatility.toFixed(1)}%`} sub="annualized" />
          <HKPI label="profit factor" value={s.profitFactor != null ? fmtNum(s.profitFactor, 2) : '—'} />
          <HKPI label="avg win/loss" value={s.avgWinLoss != null ? fmtNum(s.avgWinLoss, 2) : '—'} />
        </div>
      </div>
    </div>
  );
}

function V2LogsView() {
  const { state } = useStore();
  const s = state;
  // Recent activity is built ONLY from real events (closed trades and the
  // age of any current open positions). Never fabricate timestamps from
  // Date.now() — a misleading "live event log" undermines trust.
  const lines = [];
  for (const p of s.positions) {
    // Use the real entry timestamp captured at normalize-time. Recovering it
    // via `Date.now() - p.ageMs` would drift by render lag, especially on
    // positions opened seconds before the page rendered.
    if (p.entryTimeMs != null) {
      lines.push({
        t: 'OPEN',
        c: 'var(--accent)',
        m: `${p.side} ${p.symbol} ${fmtNum(p.size, 4)} @ ${Number(p.entry).toLocaleString()}`,
        ts: p.entryTimeMs,
      });
    }
  }
  for (const t of s.trades.slice(0, 30)) {
    if (t.time != null) {
      lines.push({
        t: 'EXIT',
        c: t.pnl >= 0 ? 'var(--accent-2)' : 'var(--danger)',
        m: `closed ${t.symbol} ${t.reason} · ${fmtUSD(t.pnl, { sign: true })}`,
        ts: t.time,
      });
    }
  }
  lines.sort((a, b) => b.ts - a.ts);

  return (
    <div style={{ padding: 22 }}>
      <div className="tbm-card" style={{ padding: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
          <span className="tbm-kicker">activity</span>
          <span className="tbm-card-title">Position open / close history</span>
        </div>
        {lines.length === 0 ? (
          <div style={{ color: 'var(--text-3)', fontFamily: 'var(--mono)', fontSize: 12 }}>
            No position or trade activity recorded yet.
          </div>
        ) : (
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, lineHeight: 1.7 }}>
            {lines.map((l, i) => (
              <div key={i} style={{ display: 'flex', gap: 14 }}>
                <span style={{ color: 'var(--text-3)', minWidth: 130 }}>{new Date(l.ts).toLocaleString()}</span>
                <span style={{ color: l.c, width: 56, fontWeight: 600 }}>{l.t}</span>
                <span style={{ color: 'var(--text-2)' }}>{l.m}</span>
              </div>
            ))}
          </div>
        )}
        <div style={{ marginTop: 12, fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--mono)' }}>
          Sourced from real position open times and trade exit times. Connection / system events not yet logged here.
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────── shell ──────────

function Shell() {
  const [theme, setTheme] = useState(() => localStorage.getItem('tbm-theme') || 'dark');
  const [navTab, setNavTab] = useState('dash');
  const [selected, setSelected] = useState({ kind: 'none' });
  const [autoSelected, setAutoSelected] = useState(false);
  const [overlays, setOverlays] = useState({ trades: true, drawdown: true });
  const { state, error } = useStore();

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('tbm-theme', theme);
  }, [theme]);

  // Auto-select first position ONCE when state arrives. After that, respect
  // the user's selection — including explicit nav back to {kind:'none'}.
  useEffect(() => {
    if (!autoSelected && state && state.positions.length > 0 && selected.kind === 'none') {
      setSelected({ kind: 'position', symbol: state.positions[0].symbol });
      setAutoSelected(true);
    }
  }, [state, autoSelected, selected.kind]);

  // Drop selection when its target disappears, so the inspector header never
  // points at a missing position/trade and a closed-then-reopened symbol
  // doesn't silently re-bind to a different side. Defined as a separate hook
  // (above the Shell component) for testability and to keep this body short.
  useStaleSelectionReset(state, selected, setSelected, setAutoSelected);

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark');
  const isMobile = useIsMobile();

  if (!state && !error) {
    return (
      <div className="tbm-boot">
        <div className="tbm-spinner" />loading dashboard
      </div>
    );
  }

  // Mobile layout: bottom tab bar + stacked content + sticky compact topbar.
  // Strat / Risk / Logs reuse the desktop views, with their 2-col / 6-col
  // grids dropping to 1-col via the tbm-2col-grid / tbm-metrics-grid CSS
  // breakpoints below MOBILE_BREAKPOINT.
  if (isMobile) {
    return (
      <div className="tbm tbm-mobile" data-theme={theme}>
        <HifiTopBarMobile theme={theme} onToggleTheme={toggleTheme} />
        {error && (
          <div role="alert" style={{ background: 'var(--danger-soft)', color: 'var(--danger)', padding: '8px 14px', fontFamily: 'var(--mono)', fontSize: 11, borderBottom: '1px solid var(--border)' }}>
            connection error · {error} · retrying…
          </div>
        )}
        {/* Reserve space for the fixed bottom tab bar so content can scroll under it cleanly. */}
        <div style={{
          paddingBottom: 'calc(72px + env(safe-area-inset-bottom, 0px))',
          minHeight: '100vh',
        }}>
          {!state ? (
            <div style={{ padding: 14, color: 'var(--text-3)', fontFamily: 'var(--mono)', fontSize: 12 }}>
              waiting for data…
            </div>
          ) : (
            <>
              {navTab === 'dash' && (
                <V2DashMobile overlays={overlays} setOverlays={setOverlays}
                              selected={selected} setSelected={setSelected} />
              )}
              {navTab === 'pos' && <V2PosViewMobile setSelected={setSelected} setNavTab={setNavTab} />}
              {navTab === 'strat' && <V2StratView />}
              {navTab === 'trades' && <V2TradesViewMobile />}
              {navTab === 'risk' && <V2RiskView />}
              {navTab === 'logs' && <V2LogsView />}
            </>
          )}
        </div>
        <V2BottomTabBar tab={navTab} onTab={setNavTab} />
      </div>
    );
  }

  return (
    <div className="tbm" data-theme={theme}>
      <div className="tbm-shell">
        <V2Rail tab={navTab} onTab={setNavTab} />
        <div className="tbm-content">
          <HifiTopBar theme={theme} onToggleTheme={toggleTheme} />
          {error && (
            <div role="alert" style={{ background: 'var(--danger-soft)', color: 'var(--danger)', padding: '8px 22px', fontFamily: 'var(--mono)', fontSize: 12, borderBottom: '1px solid var(--border)' }}>
              connection error · {error} · retrying…
            </div>
          )}
          {!state ? (
            <div style={{ padding: 22, color: 'var(--text-3)', fontFamily: 'var(--mono)', fontSize: 12 }}>
              waiting for data…
            </div>
          ) : (
            <>
              {navTab === 'dash' && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', flex: 1, minHeight: 0 }}>
                  <V2Main overlays={overlays} setOverlays={setOverlays} selected={selected} setSelected={setSelected} />
                  <V2Inspector selected={selected} setSelected={setSelected} />
                </div>
              )}
              {navTab === 'pos' && <V2PosView setSelected={setSelected} setNavTab={setNavTab} />}
              {navTab === 'strat' && <V2StratView />}
              {navTab === 'trades' && <V2TradesView />}
              {navTab === 'risk' && <V2RiskView />}
              {navTab === 'logs' && <V2LogsView />}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function App() {
  return (
    <StoreProvider>
      <Shell />
    </StoreProvider>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
