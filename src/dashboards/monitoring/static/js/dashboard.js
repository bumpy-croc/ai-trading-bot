/* Trading Bot Monitor — V2 hi-fi dashboard
   React 18 (UMD) + Babel-standalone JSX.
   Real-data adapter wired to /api/dashboard/state, /api/performance, and socket.io. */

const { useState, useEffect, useRef, useMemo, useCallback, useContext, createContext } = React;

// ─────────────────────────────────────────── helpers ──────────

const fmtUSD = (v, opts = {}) => {
  const { sign = false, dp = 2 } = opts;
  const n = Number(v) || 0;
  const a = Math.abs(n);
  const s = a < 1000 ? a.toFixed(dp) : a.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
  const prefix = n >= 0 ? (sign ? '+$' : '$') : '-$';
  return prefix + s;
};
const fmtPct = (v, dp = 2) => {
  const n = Number(v) || 0;
  return `${n >= 0 ? '+' : ''}${n.toFixed(dp)}%`;
};
const fmtNum = (v, dp = 2) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  return n.toFixed(dp);
};
const fmtTimeAgo = (ts) => {
  if (!ts) return '—';
  const d = typeof ts === 'string' ? new Date(ts) : new Date(Number(ts));
  if (Number.isNaN(d.getTime())) return '—';
  const mins = Math.floor((Date.now() - d.getTime()) / 60000);
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
const sym = (s) => String(s || '').toUpperCase();

// ─────────────────────────────────────────── data normalisation ──────────
//
// The backend speaks the existing /api/* shape. We translate it into the
// design's expected store shape so all V2 components work unchanged.

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
  const ageMs = p.entry_time ? Math.max(0, Date.now() - new Date(p.entry_time).getTime()) : 0;
  return {
    id: p.symbol ? `${p.symbol}-${idx}` : `pos-${idx}`,
    symbol: sym(p.symbol),
    side: side === 'SHORT' ? 'SHORT' : 'LONG',
    size: qty,
    entry,
    current,
    pnl: Number.isFinite(pnl) ? pnl : 0,
    pnlPct,
    trailSL: trailSL ?? entry,
    breakeven: !!p.breakeven_triggered,
    mfe: Number(p.mfe) || 0,
    mae: Number(p.mae) || 0,
    target: { tp, sl: p.stop_loss ?? null, trail: trailSL, trailPct: trailSL && entry ? Math.abs(((trailSL - entry) / entry) * 100) : 0 },
    ageMs,
    strategy: p.strategy_name || null,
    confidence: Number(p.confidence) || null,
    signal: side === 'SHORT' ? 'SELL' : 'BUY',
    raw: p,
  };
}

function normalizeTrade(t, idx) {
  const side = String(t.side || '').toUpperCase();
  const sideShort = side === 'SHORT' || side === 'SELL' || side === 'S' ? 'S' : 'L';
  const exitTime = t.exit_time ? new Date(t.exit_time).getTime() : Date.now();
  return {
    id: `t${idx}`,
    symbol: sym(t.symbol),
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

function normalizeState(payload, performance) {
  const m = (payload && payload.metrics) || {};
  const bot = (payload && payload.bot) || {};
  const positions = (payload && payload.positions) || [];
  const trades = (payload && payload.trades) || [];

  const initialBalance = Number(bot.initial_balance) || 1000;
  const balance = Number(m.current_balance) || initialBalance;
  const totalPnl = Number(m.total_pnl) || (balance - initialBalance);
  const unrealized = Number(m.unrealized_pnl) || 0;
  const realized = totalPnl - unrealized;
  const todayPnl = Number(m.daily_pnl) || 0;
  const todayPnlPct = initialBalance ? (todayPnl / initialBalance) * 100 : 0;

  const dynMult = Number(m.dynamic_risk_factor) || 1.0;
  const dynReason = m.dynamic_risk_reason || 'normal';

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
    },
    balance,
    initialBalance,
    totalPnl,
    todayPnl,
    todayPnlPct,
    weeklyPnl: Number(m.weekly_pnl) || 0,
    unrealized,
    realized,
    sharpe: Number(m.sharpe_ratio) || 0,
    sortino: Number(m.sortino_ratio) || 0,
    maxDD: Number(m.max_drawdown) || 0,
    currentDD: Number(m.current_drawdown) || 0,
    volatility: Number(m.volatility) || 0,
    winRate: (Number(m.win_rate) || 0) / 100,
    profitFactor: m.profit_factor != null ? Number(m.profit_factor) : null,
    avgWinLoss: m.avg_win_loss_ratio != null ? Number(m.avg_win_loss_ratio) : null,
    totalTrades: Number(m.total_trades) || 0,
    activePositions: Number(m.active_positions_count) || positions.length,
    maxPositions: 3,
    totalPositionValue: Number(m.total_position_value) || 0,
    marginUsage: Number(m.margin_usage) || 0,
    availableMargin: Number(m.available_margin) || 0,
    riskPerTrade: Number(m.risk_per_trade) || 1.0,
    fillRate: Number(m.fill_rate) || 0,
    avgSlippage: Number(m.avg_slippage) || 0,
    failedOrders: Number(m.failed_orders) || 0,
    orderLatency: Number(m.order_latency) || 0,
    executionQuality: Number(m.execution_quality) || 0,
    apiLatency: Number(m.api_latency) || 0,
    apiStatus: m.api_connection_status || 'Unknown',
    dataFeed: m.data_feed_status || 'Unknown',
    rsi: Number(m.rsi) || 50,
    emaTrend: m.ema_trend || '—',
    priceChange24h: Number(m.price_change_24h) || 0,
    dynamicRisk: { mult: dynMult, status: dynReason, reason: dynReason, active: !!m.dynamic_risk_active },
    positions: positions.map(normalizePosition),
    trades: trades.map(normalizeTrade),
    equityCurve: buildEquityCurve(performance),
    metricsRaw: m,
  };
}

// ─────────────────────────────────────────── store ──────────

const StoreCtx = createContext(null);

function StoreProvider({ children }) {
  const [state, setState] = useState(null);
  const [error, setError] = useState(null);
  const [range, setRange] = useState('1W'); // chart range
  const [equity, setEquity] = useState([]);
  const stateRef = useRef(state);
  stateRef.current = state;

  // Map design's range tokens to backend ?days=
  const rangeToDays = (r) => ({ '1D': 1, '1W': 7, '1M': 30, '3M': 90, 'ALL': 365 }[r] || 7);

  // Try the bundled endpoint first; fall back to per-resource fetches if it 404s.
  const fetchState = useCallback(async () => {
    try {
      // 1) try bundled
      let payload = null;
      try {
        const r = await fetch('/api/dashboard/state?trades_limit=50', { cache: 'no-store' });
        if (r.ok) payload = await r.json();
      } catch {}

      if (!payload) {
        // 2) fall back to existing endpoints in parallel
        const [mRes, pRes, tRes, sRes] = await Promise.all([
          fetch('/api/metrics', { cache: 'no-store' }).then(r => r.ok ? r.json() : {}).catch(() => ({})),
          fetch('/api/positions', { cache: 'no-store' }).then(r => r.ok ? r.json() : []).catch(() => []),
          fetch('/api/trades?limit=50', { cache: 'no-store' }).then(r => r.ok ? r.json() : []).catch(() => []),
          fetch('/api/system/status', { cache: 'no-store' }).then(r => r.ok ? r.json() : {}).catch(() => ({})),
        ]);
        const symbols = Array.from(new Set([
          ...(Array.isArray(pRes) ? pRes.map(p => p.symbol).filter(Boolean) : []),
          ...(Array.isArray(tRes) ? tRes.map(t => t.symbol).filter(Boolean) : []),
        ])).slice(0, 4);
        payload = {
          bot: {
            name: mRes.current_strategy || 'unknown',
            symbols: symbols.length ? symbols : ['BTCUSDT'],
            timeframe: '1h',
            mode: 'paper',
            status: 'running',
            connected: (sRes.api_status === 'Connected') || (mRes.api_connection_status === 'Connected'),
            initial_balance: 1000,
            uptime_seconds: Number(mRes.system_uptime) || 0,
            last_update: mRes.last_data_update || new Date().toISOString(),
          },
          metrics: mRes,
          positions: Array.isArray(pRes) ? pRes : [],
          trades: Array.isArray(tRes) ? tRes : [],
        };
      }

      setState((prev) => {
        const next = normalizeState(payload, { timestamps: [], balances: [] });
        if (prev && prev.equityCurve && prev.equityCurve.length) {
          next.equityCurve = prev.equityCurve;
        }
        return next;
      });
      setError(null);
    } catch (e) {
      setError(String(e.message || e));
    }
  }, []);

  const fetchEquity = useCallback(async (r) => {
    try {
      const days = rangeToDays(r);
      const res = await fetch(`/api/performance?days=${days}`, { cache: 'no-store' });
      if (!res.ok) throw new Error(`performance ${res.status}`);
      const json = await res.json();
      const curve = buildEquityCurve(json);
      setEquity(curve);
      setState((prev) => prev ? { ...prev, equityCurve: curve } : prev);
    } catch (e) {
      // soft fail — keep prior curve
      console.warn('equity fetch failed', e);
    }
  }, []);

  // initial paint
  useEffect(() => {
    fetchState();
    fetchEquity(range);
  }, [fetchState, fetchEquity]);

  // refetch equity when range changes
  useEffect(() => { fetchEquity(range); }, [range, fetchEquity]);

  // socket.io live updates — patch metrics in place to avoid full rebuild
  useEffect(() => {
    if (typeof io === 'undefined') return;
    const socket = io({ transports: ['websocket', 'polling'] });
    socket.on('connect', () => {
      setState((prev) => prev ? { ...prev, bot: { ...prev.bot, connected: true } } : prev);
      socket.emit('request_update');
    });
    socket.on('disconnect', () => {
      setState((prev) => prev ? { ...prev, bot: { ...prev.bot, connected: false } } : prev);
    });
    socket.on('metrics_update', (metrics) => {
      // The metrics_update payload is just the metrics dict — refetch full state
      // for positions/trades to stay in sync, but throttle to ~5s to avoid hammering.
      // Inline patch metrics for instant feel:
      setState((prev) => {
        if (!prev) return prev;
        const merged = normalizeState(
          { metrics, bot: undefined, positions: prev.positions.map(p => p.raw), trades: prev.trades.map(t => t.raw) },
          { timestamps: [], balances: [] }
        );
        // keep bot meta + equity curve from previous state
        merged.bot = prev.bot;
        merged.equityCurve = prev.equityCurve;
        merged.bot.lastUpdate = Date.now();
        return merged;
      });
    });
    const stateInterval = setInterval(fetchState, 30000); // resync state every 30s
    return () => { clearInterval(stateInterval); socket.disconnect(); };
  }, [fetchState]);

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
  const pct = Math.max(0, Math.min(100, (v / max) * 100));
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
  if (!data || data.length < 2) return <svg width={width} height={height} />;
  const ys = data.map(d => typeof d === 'number' ? d : d.v);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const sx = i => (i / (ys.length - 1)) * width;
  const sy = v => height - ((v - minY) / ((maxY - minY) || 1)) * height;
  let path = `M ${sx(0)} ${sy(ys[0])}`;
  for (let i = 1; i < ys.length; i++) {
    const cx = (sx(i - 1) + sx(i)) / 2;
    path += ` Q ${cx} ${sy(ys[i - 1])} ${cx} ${(sy(ys[i - 1]) + sy(ys[i])) / 2} T ${sx(i)} ${sy(ys[i])}`;
  }
  const c = color || (ys[ys.length - 1] >= ys[0] ? 'var(--accent-2)' : 'var(--danger)');
  return (
    <svg width={width} height={height} className="tbm-spark" style={{ display: 'inline-block' }}>
      <path d={path} fill="none" stroke={c} strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// deterministic per-symbol mini-spark when we don't have per-symbol price history
function genMiniSpark(seed) {
  const out = [];
  let h = 0;
  const str = String(seed || '');
  for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) >>> 0;
  for (let i = 0; i < 30; i++) {
    h = (h * 1103515245 + 12345) >>> 0;
    out.push(Math.sin(i * 0.4 + (h % 100) / 30) * 8 + (i / 5) + ((h >> (i % 8)) & 1 ? 0.6 : -0.4));
  }
  return out;
}

// ─────────────────────────────────────────── topbar ──────────

function HifiTopBar({ theme, onToggleTheme }) {
  const { state } = useStore();
  if (!state) return null;
  const s = state;
  const subTitle = `${s.bot.name} · ${s.bot.symbols.join(' · ')} · ${s.bot.timeframe}`;
  return (
    <div className="tbm-top">
      <div className="tbm-brand">
        <div className="tbm-logo">A</div>
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
        <button className="tbm-btn" onClick={onToggleTheme} title="Toggle theme">
          {theme === 'dark' ? '☾ Dark' : '☀ Light'}
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

function V2Rail({ tab, onTab }) {
  return (
    <div style={{
      width: 64, background: 'var(--bg-elev)',
      borderRight: '1px solid var(--border-strong)',
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      padding: '18px 0', gap: 4,
    }}>
      <div className="tbm-logo" style={{ marginBottom: 16 }}>A</div>
      {NAV.map(n => {
        const active = tab === n.id;
        return (
          <button key={n.id} onClick={() => onTab(n.id)} title={n.label} style={{
            width: 44, height: 48, borderRadius: 10,
            border: 'none', cursor: 'pointer',
            background: active ? 'var(--accent-soft)' : 'transparent',
            color: active ? 'var(--accent)' : 'var(--text-3)',
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            gap: 3, transition: 'background 120ms, color 120ms',
            position: 'relative',
            fontFamily: 'inherit',
          }}
          onMouseEnter={(e) => { if (!active) e.currentTarget.style.color = 'var(--text)'; }}
          onMouseLeave={(e) => { if (!active) e.currentTarget.style.color = 'var(--text-3)'; }}>
            {active && <span style={{ position: 'absolute', left: -6, top: 10, bottom: 10, width: 2, borderRadius: 2, background: 'var(--accent)' }} />}
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
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
        no equity data yet
      </div>
    );
  }

  const xs = data.map(d => d.ts);
  const ys = data.map(d => d.v);
  const minX = xs[0], maxX = xs[xs.length - 1];
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const yPad = (maxY - minY) * 0.12 || 1;
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

  // benchmark — synthesised buy-and-hold path (62% of bot's slope; just an indicative guide)
  const bench = data.map((d, i) => ({
    ts: d.ts,
    v: data[0].v * (1 + (data[data.length - 1].v / data[0].v - 1) * 0.62 * (i / (data.length - 1)) + Math.sin(i * 0.15) * 0.005),
  }));
  const bpts = bench.map(d => [sx(d.ts), sy(d.v)]);
  let bpath = `M ${bpts[0][0]} ${bpts[0][1]}`;
  for (let i = 1; i < bpts.length; i++) {
    const cx = (bpts[i - 1][0] + bpts[i][0]) / 2;
    bpath += ` Q ${cx} ${bpts[i - 1][1]} ${cx} ${(bpts[i - 1][1] + bpts[i][1]) / 2} T ${bpts[i][0]} ${bpts[i][1]}`;
  }

  // drawdown shading: fill below running peak
  let runMax = ys[0];
  const peaks = ys.map(v => (runMax = Math.max(runMax, v)));
  const ddPath = peaks.map((p, i) => `${i === 0 ? 'M' : 'L'} ${pts[i][0]} ${sy(p)}`).join(' ')
    + ' ' + pts.slice().reverse().map(([x, y]) => `L ${x} ${y}`).join(' ') + ' Z';

  // trade markers — place real closed trades on the chart by exit_time
  const markers = (trades || [])
    .filter(t => t.time >= minX && t.time <= maxX)
    .slice(0, 24)
    .map(t => {
      // find nearest equity point to interpolate y
      let bestI = 0, bestD = Infinity;
      for (let i = 0; i < data.length; i++) {
        const d = Math.abs(data[i].ts - t.time);
        if (d < bestD) { bestD = d; bestI = i; }
      }
      return { id: t.id, x: pts[bestI][0], y: pts[bestI][1], win: t.pnl >= 0 };
    });

  const yticks = [0.25, 0.5, 0.75].map(p => y0 + p * (y1 - y0));
  const xticks = 5;

  function onMove(e) {
    const r = ref.current.getBoundingClientRect();
    const sw = (W) / r.width;
    const x = (e.clientX - r.left) * sw;
    if (x < pad.l || x > pad.l + w) { setHover(null); return; }
    const t = minX + ((x - pad.l) / w) * (maxX - minX);
    let best = 0, dmin = Infinity;
    for (let i = 0; i < data.length; i++) {
      const dd = Math.abs(data[i].ts - t);
      if (dd < dmin) { dmin = dd; best = i; }
    }
    setHover({ i: best, x: pts[best][0], y: pts[best][1] });
  }

  return (
    <div style={{ position: 'relative', flex: 1, minHeight: 320 }}>
      <svg ref={ref} width="100%" height="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
           onMouseMove={onMove} onMouseLeave={() => setHover(null)}
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
        {overlays.benchmark && <path d={bpath} fill="none" stroke="var(--text-3)" strokeWidth="1.4" strokeDasharray="4 4" opacity="0.7" />}
        <path d={areaPath} fill="var(--accent-2)" opacity="0.10" />
        <path d={path} fill="none" stroke="var(--accent-2)" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
        {overlays.trades && markers.map(m => (
          <g key={m.id} onClick={() => onSelectTrade && onSelectTrade(m.id)} style={{ cursor: 'pointer' }}>
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
        {overlays.benchmark && <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 10, height: 2, background: 'var(--text-3)', borderTop: '1px dashed' }} />benchmark
        </span>}
        {overlays.trades && <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--accent-2)' }} />trade
        </span>}
        {overlays.drawdown && <span style={{ color: 'var(--danger)' }}>drawdown</span>}
      </div>
      {hover && data[hover.i] && (
        <div className="tbm-tooltip" style={{
          left: Math.min((hover.x / W) * 100, 80) + '%',
          top: Math.max((hover.y / H) * 100 - 6, 2) + '%',
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

const RANGES = ['1D', '1W', '1M', '3M', 'ALL'];
const RANGE_LABEL = { '1D': 'last 24 hours', '1W': 'last 7 days', '1M': 'last 30 days', '3M': 'last 90 days', 'ALL': 'all time' };

function V2Main({ overlays, setOverlays, selected, setSelected }) {
  const { state, range, setRange } = useStore();
  if (!state) return null;
  const s = state;

  return (
    <div style={{ padding: 22, display: 'flex', flexDirection: 'column', gap: 14, minWidth: 0 }}>
      {/* KPI strip */}
      <div style={{ display: 'flex', gap: 22, paddingBottom: 14, borderBottom: '1px solid var(--border)', alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <HKPI label="equity" value={fmtUSD(s.balance)} sub={`${fmtUSD(s.todayPnl, { sign: true })} today`} color={s.todayPnl >= 0 ? 'var(--accent-2)' : 'var(--danger)'} />
        <HKPI label="all-time" value={fmtPct((s.totalPnl / s.initialBalance) * 100)} color={s.totalPnl >= 0 ? 'var(--accent-2)' : 'var(--danger)'} />
        <HKPI label="open risk" value={fmtUSD(s.totalPositionValue)} sub={s.activePositions ? `${s.activePositions} pos` : 'no pos'} />
        <HKPI label="sharpe" value={fmtNum(s.sharpe, 2)} sub="annualized" />
        <HKPI label="win rate" value={`${(s.winRate * 100).toFixed(0)}%`} sub={`${s.totalTrades} trades`} />
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
            <HPill active={overlays.benchmark} onClick={() => setOverlays({ ...overlays, benchmark: !overlays.benchmark })}>+ benchmark</HPill>
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
                <button key={p.id} onClick={() => setSelected({ kind: 'position', symbol: p.symbol })} style={{
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
                  <div style={{ marginTop: 6 }}>
                    <HSparkline data={genMiniSpark(p.symbol)} width={140} height={22} color={p.pnl >= 0 ? 'var(--accent-2)' : 'var(--danger)'} />
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

function V2InspectPosition({ symbol }) {
  const { state } = useStore();
  const s = state;
  const p = s.positions.find(x => x.symbol === symbol) || s.positions[0];
  if (!p) return <div style={{ color: 'var(--text-3)' }}>No open position</div>;
  const tp = p.target.tp;
  const sl = p.target.trail || p.target.sl;
  const portionPct = s.balance ? (p.size * p.current / s.balance) * 100 : 0;
  const stopDistPct = p.current && sl ? ((sl - p.current) / p.current) * 100 : 0;
  const tpDistPct = p.current && tp ? ((tp - p.current) / p.current) * 100 : 0;
  const risk = sl ? Math.abs(p.entry - sl) * p.size : 0;
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
        <div style={{ marginTop: 10 }}>
          <HSparkline data={genMiniSpark(p.symbol + 'live')} width={320} height={50} color={p.pnl >= 0 ? 'var(--accent-2)' : 'var(--danger)'} />
        </div>
      </div>

      <div className="tbm-card" style={{ padding: 14 }}>
        <div className="tbm-h2">Stops & exposure</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <HKPI label="stop" value={sl ? Number(sl).toLocaleString() : '—'} sub={sl ? fmtPct(stopDistPct) : 'none'} color="var(--danger)" />
          <HKPI label="target" value={tp ? Number(tp).toLocaleString() : '—'} sub={tp ? fmtPct(tpDistPct) : 'none'} color="var(--accent-2)" />
          <HKPI label="size" value={fmtUSD(p.size * p.current)} sub={`${portionPct.toFixed(0)}% port`} />
          <HKPI label="risk" value={fmtUSD(risk)} sub="if SL hit" />
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
  const s = state;
  const t = s.trades.find(x => x.id === id) || s.trades[0];
  if (!t) return <div style={{ color: 'var(--text-3)' }}>No trade</div>;
  const movePct = t.entry ? ((t.exit - t.entry) / t.entry) * 100 * (t.side === 'L' ? 1 : -1) : 0;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <button onClick={() => setSelected(s.positions[0] ? { kind: 'position', symbol: s.positions[0].symbol } : { kind: 'none' })}
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
  // Confidence breakdown: built from real confidence + ema/rsi when available
  const conf = s.positions[0]?.confidence ?? null;
  const rsi = Number(s.rsi) || 50;
  const trendLabel = s.emaTrend || '—';
  return (
    <div style={{ padding: 22, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, overflow: 'auto' }}>
      <div className="tbm-card" style={{ padding: 18 }}>
        <div className="tbm-h2">Model</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <HRow k="strategy" v={s.bot.name} />
          <HRow k="symbols" v={s.bot.symbols.join(', ')} />
          <HRow k="timeframe" v={s.bot.timeframe} />
          <HRow k="mode" v={s.bot.mode} />
          <HRow k="status" v={s.bot.status} color={s.bot.status === 'running' ? 'var(--accent-2)' : 'var(--text-3)'} />
          <HRow k="uptime" v={s.bot.uptime} />
          <HRow k="trend" v={trendLabel} color="var(--accent-2)" />
        </div>
      </div>
      <div className="tbm-card" style={{ padding: 18 }}>
        <div className="tbm-h2">Last signal {conf != null && <span style={{ color: 'var(--text-2)' }}>· conf {conf.toFixed(2)}</span>}</div>
        <HBarLabeled k="ML confidence" v={conf != null ? Math.round(conf * 100) : 0} max={100} sub={conf != null ? conf.toFixed(2) : '—'} color="var(--accent-2)" />
        <div style={{ height: 6 }} />
        <HBarLabeled k="RSI (14)" v={rsi} max={100} sub={rsi.toFixed(1)} color="var(--accent)" />
        <div style={{ height: 6 }} />
        <HBarLabeled k="Win rate" v={s.winRate * 100} max={100} sub={`${(s.winRate * 100).toFixed(0)}%`} color="var(--accent-2)" />
        <div style={{ height: 6 }} />
        <HBarLabeled k="Sharpe (norm)" v={Math.max(0, Math.min(100, (s.sharpe / 3) * 100))} max={100} sub={fmtNum(s.sharpe, 2)} color="var(--accent-2)" />
      </div>
    </div>
  );
}

function V2TradesView() {
  const { state } = useStore();
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

  return (
    <div style={{ padding: 22, overflow: 'auto' }}>
      <div style={{ display: 'flex', gap: 24, marginBottom: 16, flexWrap: 'wrap' }}>
        <HKPI label="trades" value={total} size="md" />
        <HKPI label="wins" value={total ? `${wins} (${(wins / total * 100).toFixed(0)}%)` : '0'} color="var(--accent-2)" size="md" />
        <HKPI label="losses" value={total - wins} color="var(--danger)" size="md" />
        <HKPI label="net" value={fmtUSD(totalPnl, { sign: true })} color={totalPnl >= 0 ? 'var(--accent-2)' : 'var(--danger)'} size="md" />
        <HKPI label="avg win" value={fmtUSD(avgWin, { sign: true })} color="var(--accent-2)" size="md" />
        <HKPI label="avg loss" value={fmtUSD(avgLoss, { sign: true })} color="var(--danger)" size="md" />
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
  return (
    <div style={{ padding: 22, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, overflow: 'auto' }}>
      <div className="tbm-card" style={{ padding: 18 }}>
        <div className="tbm-h2">Dynamic risk · {fmtNum(s.dynamicRisk.mult, 2)}x</div>
        <div style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 14, lineHeight: 1.5 }}>
          Position sizing scales with drawdown, volatility, and cooldowns. Current: <strong style={{ color: 'var(--accent-2)' }}>{s.dynamicRisk.reason}</strong>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <HBarLabeled k="Drawdown" v={ddBar} max={100} sub={`${ddPct.toFixed(1)}%`} color="var(--danger)" />
          <HBarLabeled k="Volatility (annualized)" v={volNorm} max={100} sub={`${s.volatility.toFixed(1)}%`} color="var(--accent)" />
          <HBarLabeled k="Win rate" v={s.winRate * 100} max={100} sub={`${(s.winRate * 100).toFixed(0)}%`} color="var(--accent-2)" />
          <HBarLabeled k="Dynamic factor" v={Math.max(0, Math.min(100, (s.dynamicRisk.mult / 2) * 100))} max={100} sub={`${fmtNum(s.dynamicRisk.mult, 2)}x`} color="var(--accent-2)" />
        </div>
      </div>
      <div className="tbm-card" style={{ padding: 18 }}>
        <div className="tbm-h2">Exposure caps</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <HRow k="open positions" v={`${s.activePositions} / ${s.maxPositions}`} />
          <HRow k="total exposure" v={fmtUSD(s.totalPositionValue)} sub={s.balance ? `${((s.totalPositionValue / s.balance) * 100).toFixed(0)}%` : ''} />
          <HRow k="risk / trade" v={`${s.riskPerTrade}%`} sub={`≈ ${fmtUSD(s.balance * s.riskPerTrade / 100)}`} />
          <HRow k="margin used" v={fmtUSD(s.totalPositionValue)} sub={`${s.marginUsage.toFixed(0)}%`} />
          <HRow k="margin avail" v={fmtUSD(s.availableMargin)} />
        </div>
      </div>
      <div className="tbm-card" style={{ padding: 18, gridColumn: '1 / -1' }}>
        <div className="tbm-h2">Risk metrics</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 18 }}>
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
  // Compose a synthetic event log from real recent positions + trades + bot status
  const lines = [];
  if (s.bot.connected) lines.push({ t: 'INFO',  c: 'var(--text-2)', m: `data feed connected · ${s.bot.symbols.join(', ')}`, ts: Date.now() });
  if (s.bot.status === 'running') lines.push({ t: 'INFO', c: 'var(--text-2)', m: `bot running · ${s.bot.name} · ${s.bot.mode}`, ts: Date.now() - 30000 });
  for (const p of s.positions) {
    lines.push({ t: 'OPEN', c: 'var(--accent)', m: `${p.side} ${p.symbol} ${fmtNum(p.size, 4)} @ ${Number(p.entry).toLocaleString()}`, ts: Date.now() - p.ageMs });
  }
  for (const t of s.trades.slice(0, 10)) {
    lines.push({ t: t.pnl >= 0 ? 'EXIT' : 'EXIT', c: t.pnl >= 0 ? 'var(--accent-2)' : 'var(--danger)',
      m: `closed ${t.symbol} ${t.reason} · ${fmtUSD(t.pnl, { sign: true })}`, ts: t.time });
  }
  lines.sort((a, b) => b.ts - a.ts);

  return (
    <div style={{ padding: 22 }}>
      <div className="tbm-card" style={{ padding: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
          <span className="tbm-kicker">logs</span>
          <span className="tbm-card-title">Recent activity</span>
        </div>
        {lines.length === 0 ? (
          <div style={{ color: 'var(--text-3)', fontFamily: 'var(--mono)', fontSize: 12 }}>No activity yet</div>
        ) : (
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, lineHeight: 1.7 }}>
            {lines.map((l, i) => (
              <div key={i} style={{ display: 'flex', gap: 14 }}>
                <span style={{ color: 'var(--text-3)', minWidth: 90 }}>{new Date(l.ts).toLocaleTimeString()}</span>
                <span style={{ color: l.c, width: 56, fontWeight: 600 }}>{l.t}</span>
                <span style={{ color: 'var(--text-2)' }}>{l.m}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────── shell ──────────

function Shell() {
  const [theme, setTheme] = useState(() => localStorage.getItem('tbm-theme') || 'dark');
  const [navTab, setNavTab] = useState('dash');
  const [selected, setSelected] = useState({ kind: 'none' });
  const [overlays, setOverlays] = useState({ benchmark: false, trades: true, drawdown: true });
  const { state, error } = useStore();

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('tbm-theme', theme);
  }, [theme]);

  // Auto-select first position once data lands so the inspector has content
  useEffect(() => {
    if (selected.kind === 'none' && state && state.positions.length > 0) {
      setSelected({ kind: 'position', symbol: state.positions[0].symbol });
    }
  }, [state, selected]);

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark');

  if (!state && !error) {
    return (
      <div className="tbm-boot">
        <div className="tbm-spinner" />loading dashboard
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
            <div style={{ background: 'var(--danger-soft)', color: 'var(--danger)', padding: '8px 22px', fontFamily: 'var(--mono)', fontSize: 12, borderBottom: '1px solid var(--border)' }}>
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
