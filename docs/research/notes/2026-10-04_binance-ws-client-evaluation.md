# Binance WebSocket client evaluation (GH #724)

**Date**: 2026-10-04
**Author**: quant-researcher (research spike, docs only, no source changes)
**Question**: should the python-binance WebSocket layer be replaced with something more robust?
**Evidence base**: repo at `origin/develop` (python-binance pinned `==1.0.37`), GH issues
#616/#723/#724/#729/#665/#907/#908/#911/#915, `.claude/state/log.md`, the python-binance 1.0.37
wheel source read line by line, upstream repos/PyPI metadata pulled 2026-10-04, and Binance's
developer docs fetched 2026-10-04.

Convention: **[V]** = verified against a primary source (cited). **[I]** = inference. **[U]** =
could not verify.

## Verdict

1. Do not replace python-binance wholesale. REST (`Client`) and the kline stream stay.
2. Margin user-data stream: no maintained library gives us what we need today.
   - The official Binance SDK has no `listenToken` support for margin, so it cannot do cross-margin user data on the path Binance documents.
   - unicorn-binance-websocket-api (UBWA) uses a different subscribe method for margin than Binance documents [U].
   - ccxt.pro implements it correctly but changes the event shape.
3. The structural defect is real and still present in python-binance 1.0.37. After a ws-api transport reconnect the library does not re-subscribe, and it never renews the 24h `listenToken`.
4. Recommended path, in order:
   1. **Step 0 (before 2026-10-14):** confirm that Binance's 2026-09-25 margin user-data-stream migration does not break our account.
   2. **Step 1:** put an adapter boundary around the user stream. No behaviour change.
   3. **Step 2:** write a thin in-house margin user-stream client (about 300 lines on `websockets`, already installed transitively). Run it in shadow mode on prod first, then flag-promote.
   4. **If the owner declines Step 2:** stay-and-fix. That means accept periodic REST-degraded windows backstopped by the existing alerts, and make the hard-reconnect path default-on after review.

Decision owner has to make: see section 6.

## 1. What the bot needs from the WS layer, and how it is met today

Two independent `BinanceProvider` instances each own a `ThreadedWebsocketManager` (TWM) and a private
event loop: one for kline, one for the margin user stream (`binance_provider.py:2563-2595`; the
loop-per-manager rule came from PR #646 and PRs #650/#652). Only one file imports from `binance`
for WS (`binance_provider.py:65`); everything else goes through the provider API.

| Need | How it is met today | Where it fails or is weak |
|---|---|---|
| Kline stream (spot, `stream.binance.com`) | `start_kline_socket` bound to the TWM loop (`binance_provider.py:2748-2790`, `_socket_on_twm_loop` `:2720`). Callback feeds `KlineBuffer.on_kline` (lock-guarded, `kline_buffer.py:82-95`). | Library breaks its read loop after 5 reconnects, then spins `ReadLoopClosed` error callbacks with no sleep (#665; source-verified in 1.0.37, see 3.1). Recovery is our health thread: `check_kline_health` (`ws_health.py:308`), `reconnect_kline` (`binance_provider.py:3073`), REST fallback that keeps probing (#662). Self-heals in about 30 s (LESSONS §5.4). #663 was a 7 h blackout from the loop-binding bug (`:2720-2746`). |
| Margin user-data stream with listen-key lifecycle | `start_margin_socket` (`binance_provider.py:2858-2861`). We contain **no** listen-key or listen-token code (grep of `src/` and `cli/`: one comment hit). python-binance 1.0.37 creates a `listenToken` via `POST /sapi/v1/userListenToken` and subscribes `userDataStream.subscribe.listenToken` on the **shared** `client.ws_api` socket (`keepalive_websocket.py:82-95,179-196`). | No keepalive and no renewal for ws-api subscriptions (`keepalive_websocket.py:247-251`). `_listen_token_expiration` is written at `:186,:205` and read nowhere. Token max validity is 24 h [V]; the server then sends `eventStreamTerminated`. We treat that as a disconnect (`binance_provider.py:2821-2853`, #908), so we recover, but by teardown and rebuild, not renewal. |
| Reconnect with backoff | Library layer: `ReconnectingWebsocket`, `MAX_RECONNECTS=5`, jittered exponential wait (`reconnecting_websocket.py:46,300-302`). Our layer: `reconnect_user` (3 tries, 1 s/2 s, `binance_provider.py:3107-3139`), circuit after 3 unproductive reconnects (`ws_health.py:538`, `constants.py:516`), exponential degraded-probe schedule 10,20,40,80,160,280,... capped at 120 cycles (`ws_health.py:632-703`, `constants.py:529-545`), flag-gated hard reconnect that rebuilds the TWM (`binance_provider.py:3141-3196`, `ws_health.py:705`, `FEATURE_WS_USER_HARD_RECONNECT`, default OFF). | The library does not re-subscribe after a ws-api transport reconnect [V: `websocket_api.py` has no `_before_connect` override and `connect()` at `reconnecting_websocket.py:114-148` re-opens the socket only]. Our compensation is about 650 lines in the provider plus 904 lines in `ws_health.py`. |
| Ping/pong | Binance pings every 20 s and drops the connection if no pong within 60 s [V: spot WS streams and WS API docs]. python-binance passes no ping arguments (`reconnecting_websocket.py:134`), so `websockets` defaults apply: ping every 20 s, 20 s timeout [V: websockets keepalive docs; applying them here is [I]]. | None observed. Separately, `verify_user_stream_alive` sends a ws-api `ping` (`binance_provider.py:2985-3040`) when the stream is quiet. It proves the transport, not the subscription (see 3.2). |
| 24 h forced disconnect | Not handled proactively. Both stream and ws-api connections live at most 24 h [V]. The library handles the close as an error + reconnect; kline then goes error callback -> `RESYNCING` -> `reconnect_kline`. | python-binance has no 24 h rotation and no `serverShutdown` handling [V: grep of `binance/ws/*.py`; same for `src/`]. Every 24 h is a (benign) error event. |
| Event-loop/thread model | TWM thread runs a private loop; our callbacks are synchronous and short. User events go through `UserDataProcessor.enqueue` (`user_data_processor.py:38-47`, unbounded `queue.Queue`) to a worker thread. Kline events mutate `KlineBuffer` under a lock. | Loop-binding bugs: #646, #650/#652, #663. Teardown noise: #716 (KeyError), #729 (`Error stopping client` / `Task was destroyed`), -2036 on unsubscribe (#907). Suppressed by `_twm_loop_exception_handler` (`binance_provider.py:2597-2711`), which pattern-matches library traceback frames. |
| Backpressure | Library: `max_queue_size=100`, overflow raises `BinanceWebsocketQueueOverflow` and kills the read loop (`reconnecting_websocket.py:211-217`). Ours: inline sync callback, so consumption is as fast as the callback. | Not a practical risk at user-event volumes (a handful per minute). The failure mode (read loop dies) is the same class as #665. |

Incident timeline relevant to the decision:

- **2026-05-15 (#616)** margin stream in a 2-minute reconnect loop; prod ran on REST fallback.
- **2026-06-08 (#723, #729)** `FEATURE_WS_USER_HARD_RECONNECT` shipped inert and was switched on in prod that day (#729 reports 4 hard reconnects with the flag ON); the reconnect counter reached 1360 by 07-06 (flag state then not stated).
- **2026-07-06 (#907/#908, deployed via #911)** root cause was partly ours: the 120 s staleness watchdog treated silence on a heartbeat-less stream as death, so every subscription was torn down about 120 s after creation. Fix: verify with a ws-api ping and hold. A 6 h soak showed zero `REST_DEGRADED` transitions (`log.md:297`).
- **2026-08-08 22:33Z to 2026-08-13 (`log.md:969,1107`)** user stream REST-degraded for 5 days after the #908 fix, one CRITICAL alert, no recovery until a service restart. Whether `FEATURE_WS_USER_HARD_RECONNECT` was still on then: [U]; if it was, the hard reconnect did not restore the stream.

So the 2026-07 churn was mostly our own bug, and that is fixed. The 2026-08 episode shows the library-level residual is real: after a transport drop the stream may not come back without a rebuild.

Safety context [V from CLAUDE.md/LESSONS]: stop-losses are exchange-side, and REST polling backstops order/balance when the stream is down. A degraded user stream costs fill latency and noise, not unprotected positions. That sets the urgency: moderate, not urgent.

## 2. Binance API facts that constrain the choice

All fetched 2026-10-04. Binance's docs site renders client-side, so these come from the page content returned by the fetch tool, cross-checked against the python-binance and ccxt implementations where noted.

- [V] **Margin listenKey REST endpoints discontinued 2026-02-20 07:00 UTC**: `POST/PUT/DELETE /sapi/v1/userDataStream` and `/isolated`. Announced 2026-01-21. Source: <https://developers.binance.com/docs/margin_trading/change-log>.
- [V] **Replacement**: `POST /sapi/v1/userListenToken` returns `{token, expirationTime}`; subscribe with WS API method `userDataStream.subscribe.listenToken` on `wss://ws-api.binance.com:443/ws-api/v3`. Default and max validity 24 h; `isIsolated` defaults false (cross margin). To extend, mint a new token before expiry and call subscribe again with it. If not extended, `eventStreamTerminated` is pushed. Source: <https://developers.binance.com/docs/margin_trading/trade-data-stream/Listen-Token-Websocket-API> (the extend procedure came from the page text surfaced by search, then consistent with the "not extended" sentence on the fetched page).
- [V] The old stream page is marked "(deprecated) Trade Data Stream"; its text still says listen keys last 60 min and connections last 24 h. Source: <https://developers.binance.com/en/docs/products/margin-trading/trade-data-stream>. The margin best-practice page still describes the removed `POST /sapi/v1/userDataStream`, so Binance's own docs are inconsistent: <https://developers.binance.com/docs/margin_trading/best-practice>.
- [V] **New dated change, 2026-09-25, effective 2026-10-14 (10 days from this note)**: dedicated margin listenKey endpoints `POST/PUT/DELETE /sapi/v2/margin/uta/listen-key`, stream URL `wss://fstream.binance.com/private/ws/<listenKey>`, new events `marginLevelChange` and `liabilityChange`. New accounts calling the legacy `POST /sapi/v1/margin/listen-key` get error `-3086` ("Margin User Data Stream has been migrated"). Source: <https://developers.binance.com/docs/margin_trading/change-log>.
- [U] Whether our account is subject to this migration, and whether `userListenToken` keeps working for it. We call neither listen-key endpoint (python-binance uses `userListenToken`), so [I] we are probably unaffected, but nobody has checked against a live account. This is Step 0.
- [V] **Connection rules** (spot streams and WS API docs): 24 h max connection life; `serverShutdown` event before server shutdown; server ping every 20 s, disconnect if no pong within 1 min; 300 connection attempts per 5 min per IP; 1024 streams per connection. Sources: <https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md>, <https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-api.md>.
- [V] WS API user-data subscriptions are per connection (one active subscription per account per connection; 1,000 active and 65,535 lifetime subscriptions per session). `session.status` reports a `userDataStream` flag (active or not) and `session.subscriptions` lists them. Same source. These are spot docs; [U] whether `session.status` reports margin subscriptions the same way. If it does, it is a better liveness probe than `ws_ping`.
- [V] Cross-margin risk stream: `wss://margin-stream.binance.com`, listenKey, cross margin only, 24 h connection: <https://developers.binance.com/en/docs/products/margin-trading/risk-data-stream>. Not used by us.

Timeline of Binance changes to this surface: 2025-10-06 (listenToken introduced), 2025-11-10 (deprecation notice), 2026-02-20 (removal), 2026-09-25/10-14 (UTA migration). Four changes in twelve months. Any option has to absorb that cadence.

## 3. Candidates

### 3.1 Stay on python-binance (1.0.37)

- [V] Release state: 1.0.37 released 2026-06-08 is the latest on PyPI; the last commit on `master` is the same day; 533 open issues; MIT; 7.2k stars. Sources: <https://pypi.org/project/python-binance/>, <https://github.com/sammchardy/python-binance>. Cadence: monthly through early 2026, then 2026-02-16, 03-24, 06-08; nothing for four months. #908 already noted there is no newer version to upgrade to.
- [V] Changelog entries relevant to our failures (<https://github.com/sammchardy/python-binance/blob/master/docs/changelog.rst>):
  - 1.0.33: reconnect keep-alive fix.
  - 1.0.35 (2026-02-16): margin socket moved to WS API with listenToken (PR #1670, merged in response to issue #1668, opened 2026-02-03, 17 days before Binance's deadline) and "reconnects for websocket connection closed ok and ws-api" (#1655).
  - 1.0.36 (2026-03-24): "propagate connection errors to subscription queues" (#1681, from issue #1678). Before this, a dropped ws-api connection was invisible to the margin consumer. This is likely part of why the #616/#723 behaviour looked like "reconnects deliver no events".
  - 1.0.37: id adjustments only.
- [V] Still present in 1.0.37 (read from the wheel, <https://pypi.org/project/python-binance/1.0.37/#files>):
  - No re-subscribe after ws-api reconnect (`websocket_api.py`, `reconnecting_websocket.py`).
  - No listenToken renewal (`keepalive_websocket.py:186,205,247-251`).
  - No 24 h rotation, no `serverShutdown` handling.
  - Kline read loop breaks after 5 reconnects (`reconnecting_websocket.py:243-260`), and `ThreadedApiManager.start_listener` turns each `ReadLoopClosed` into an error callback with no sleep (`threaded_stream.py:72-93`). This is #665.
  - `ThreadedApiManager.stop()` closes the client before flipping socket flags (`threaded_stream.py:107-122`), then `__aexit__` unsubscribes over a fresh connection, which is the -2036 teardown noise (#907).
- Threading fit: already integrated (TWM thread plus private loop). Migration cost: zero. Licence MIT.
- Assessment: library gaps are known and bounded, and they are compensated in our code. Staying costs ongoing maintenance of the compensation layer and exposure to Binance changes with a maintainer whose last release is four months old. [I]

### 3.2 Official Binance connectors

- [V] The legacy `binance-connector` package is declared deprecated on PyPI (3.13.0, 2026-04-30, MIT): <https://pypi.org/project/binance-connector/>.
- [V] Current official SDK is the modular monorepo <https://github.com/binance/binance-connector-python> (MIT, pushed 2026-10-01, 13 open issues): one PyPI package per product (`binance-sdk-spot`, `binance-sdk-margin-trading`, ...) on a shared `binance-common`. Python >= 3.10.
- [V] Release cadence: very high and breaking. `binance-sdk-margin-trading` 6.5.0 (2026-05-22) to 8.5.0 (2026-10-01); `binance-sdk-spot` 10.0.0 (2026-07-14) to 13.0.0 (2026-10-01), three major versions in 11 weeks. Sources: <https://pypi.org/project/binance-sdk-margin-trading/>, <https://pypi.org/project/binance-sdk-spot/>. Dependencies are heavier than ours: `aiohttp>=3.14.3` (installed here: 3.12.15), `pydantic>=2.10`, `requests>=2.34.2` (we pin `requests==2.31.0`), `urllib3>=2.7`, `pycryptodome`, `websocket-client`, `websockets<16` (websockets is at 17.x upstream, 15.0.1 installed here).
- [V] Reconnect semantics (`binance-common` 4.5.0, `websocket.py`, <https://github.com/binance/binance-connector-python/tree/main/common/src/binance_common>): proactive reconnect every 23 h (`AUTO_RECONNECT_INTERVAL_SECONDS = 23*3600`), `serverShutdown` triggers a scheduled reconnect, manual pong replies (`autoping=False`), reconnect attempts 3 by default (max 10) with fixed delay, connection events `open/reconnect/close/error/ping/pong`, and `_resubscribe_user_streams` re-sends `userDataStream.subscribe` on the new connection. Better than python-binance on every robustness point except one.
- [V] That one point is decisive: **the margin SDK has no listenToken support.** `binance-sdk-margin-trading` 8.5.0, `binance-sdk-spot` 13.0.0 and `binance-common` 4.5.0 contain zero references to `listenToken`/`userListenToken` (grep of the extracted wheels). Its margin user-data path is REST `POST /sapi/v1/margin/listen-key` (the endpoint Binance calls legacy in the 2026-09-25 entry) plus `trade_data(listenKey)` / `risk_data(listenKey)` streams (`websocket_streams.py`). The session-based resubscribe code is for Spot's `userDataStream.subscribe`, not margin.
- Threading fit: asyncio only (aiohttp), the SDK owns task lifecycle. We would still need our own thread + loop bridge (the freqtrade pattern, 3.5). Callbacks are invoked from the loop.
- Migration cost: kline only: moderate and low value (kline works). Margin user stream: not feasible without writing our own ws-api subscribe on top of the SDK transport, which is the in-house option with a heavier dependency.
- Assessment: best transport robustness, wrong coverage for our one hard problem, and the churn of major versions is a maintenance tax. Revisit if Binance adds listenToken to the margin SDK. [I]

### 3.3 Thin in-house client on `websockets` (or `aiohttp`)

- Licences: `websockets` BSD-3-Clause (<https://github.com/python-websockets/websockets>), `aiohttp` Apache-2.0 (<https://github.com/aio-libs/aiohttp>). Both are actively maintained (pushed 2026-10-04) and already installed transitively via python-binance, so no new dependency.
- Scope for the user stream only (about 300 lines [I]):
  - Dedicated connection to `wss://ws-api.binance.com:443/ws-api/v3`, not shared with anything else.
  - Lifecycle on every (re)connect: `POST /sapi/v1/userListenToken` (the sync `Client.margin_create_listen_token` already exists, `client.py:5738`), then `userDataStream.subscribe.listenToken`, then await the response with a timeout.
  - Renew at `expirationTime - 60 s` by minting a new token and subscribing again (same approach as ccxt, 3.4).
  - Proactive rotation at about 23 h; handle `serverShutdown` and `eventStreamTerminated` as "rebuild now".
  - Backoff with jitter, bounded by the 300 connections per 5 min per IP limit [V].
  - Ping/pong via the library's keepalive.
  - Callbacks into the existing `UserDataProcessor.enqueue`, preserving the generation-token and event-gated recovery invariants of #717.
- What it fixes by construction: re-subscribe on reconnect, token renewal, 24 h handling, deterministic teardown (no shared `ws_api`, no -2036 noise, no frame-pattern exception handler needed).
- Costs and risks: we own protocol drift (four Binance changes in twelve months, section 2); we must get reconnect storms, token-mint rate limits (`sapi` weights) and IP-ban handling right; staging cannot exercise it (paper mode never starts a margin user socket, per the #723 issue body), so the only real-venue test is prod shadow mode.
- Threading fit: own daemon thread with its own loop, `run_coroutine_threadsafe` for control calls. Same shape as freqtrade's `ExchangeWS` (3.5) and as our current TWM arrangement.
- Migration cost: moderate (about one week including fake-server tests and shadow rollout [I]). Kline and REST untouched.

### 3.4 ccxt / ccxt.pro

- [V] MIT, 713 open issues, pushed 2026-10-03, `ccxt 4.5.85` released 2026-10-01 (<https://pypi.org/project/ccxt/>, <https://github.com/ccxt/ccxt>). WebSocket support is bundled in the main package.
- [V] `ccxt/pro/binance.py` implements the margin path Binance documents: `sapiPostUserListenToken`, `userDataStream.subscribe.listenToken`, a 23 h default refresh and renewal one minute before `expirationTime`, single-flight guard against minting two tokens (`ensure_user_data_stream_ws_subscribe_listen_token`, `renew_listen_token`, <https://github.com/ccxt/ccxt/blob/master/python/ccxt/pro/binance.py>). Client side: aiohttp, manual pong, ping loop (`async_support/base/ws/client.py`).
- [I] Reconnect semantics in ccxt are caller-driven: a closed socket rejects pending `watch_*` futures with `NetworkError` and the caller re-awaits, which reconnects and re-subscribes. Read from `client.py` but not exercised.
- Threading fit: asyncio-only; needs a thread + loop bridge. Binance REST via ccxt would be a separate question we do not need to answer.
- Migration cost: high. We consume raw Binance `executionReport` / `outboundAccountPosition` / `balanceUpdate` dicts (`UserDataProcessor`, `OrderTracker`). ccxt hands back unified order/balance structures with the raw payload under `info`, so the processing layer needs a rewrite or an adapter. Large dependency for one stream.
- Assessment: the only maintained library that implements margin listenToken renewal correctly. Useful as a reference implementation for 3.3 and as a fallback adapter; not worth adopting for its own sake. [I]

### 3.5 What other bots do

- freqtrade (GPL-3.0): ccxt.pro `watch_ohlcv` only, on a dedicated thread with its own event loop, controlled through `run_coroutine_threadsafe` (`freqtrade/exchange/exchange_ws.py`, <https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/exchange/exchange_ws.py>). No user-data stream; orders are polled. Evidence for the thread+loop bridge pattern, not for margin user streams.
- hummingbot (Apache-2.0): own aiohttp `WSAssistant`. Its Binance spot user stream subscribes `userDataStream.subscribe.signature` on a dedicated WS API connection (<https://github.com/hummingbot/hummingbot/blob/master/hummingbot/connector/exchange/binance/binance_api_user_stream_data_source.py>). That is spot only; I found no Binance cross-margin connector. Dedicated connection per user stream is the same design as 3.3.
- jesse, octobot: not researched [U].

### 3.6 unicorn-binance-websocket-api (UBWA)

- [V] MIT, single maintainer (Oliver Zehentleitner), 737 stars, 6 open issues, 2.16.1 released 2026-09-21 (<https://pypi.org/project/unicorn-binance-websocket-api/>, <https://github.com/oliver-zehentleitner/unicorn-binance-websocket-api>). Cadence is bursty: 2.10.2 on 2025-06-23, then nothing until 2.11.0 on 2026-04-10, then a release roughly every 2-6 weeks. Ships Cython wheels (build/CI implication) and pulls its own REST package.
- [V] Per its changelog (<https://github.com/oliver-zehentleitner/unicorn-binance-websocket-api/blob/master/CHANGELOG.md>): since 2.11.0 `!userData` on `binance.com-margin` uses a WS API subscription ("`userDataStream.subscribe.signature`") because Binance removed the listenKey endpoints; on reconnect it re-queues the full subscription state for market streams; one thread plus event loop per stream; 2.16.0 added a local stand-in-server scenario suite and a 24 h soak against binance.com.
- [U] Binance's margin docs describe listenToken, not `subscribe.signature`, for margin. Whether UBWA's margin user stream actually delivers on a live cross-margin account is unverified, and I found no evidence either way. It also says user-data streams on WS API are "unaffected" by the reconnect re-queue, which does not tell us how a lost ws-api subscription is restored [U].
- Threading fit: good (thread per stream, callbacks or buffer). Migration cost: moderate, needs a live-account test we cannot run here.
- Assessment: credible, single-maintainer, margin path unproven. Not recommended as the primary path; worth a one-hour live check by the owner if they would rather adopt than build. [I]

## 4. Comparison

| | python-binance 1.0.37 | Official modular SDK | In-house (`websockets`) | ccxt.pro | UBWA 2.16.1 |
|---|---|---|---|---|---|
| Licence | MIT | MIT | BSD-3 (lib) | MIT | MIT |
| Maintenance | last release 2026-06-08, quiet 4 months | very active, breaking majors every few weeks | ours | very active | active, single maintainer, bursty |
| Margin user stream per current Binance docs (listenToken) | yes (via ws_api, no renewal) | **no** | yes (we write it) | yes, with renewal | different method [U] |
| Re-subscribe after reconnect | no | yes (Spot WS API session) | yes | caller-driven [I] | partial [U] |
| 24 h / `serverShutdown` | no | yes | yes | partial [I] | not checked |
| Event shape change | none | pydantic models | none | unified, raw in `info` | raw |
| Thread model fit | current | needs bridge | own thread + loop | needs bridge | thread per stream |
| Migration cost | 0 | high, no margin user stream | moderate | high | moderate, unproven |

## 5. Recommendation and migration sketch

**Scope**: margin user-data stream only. Kline stays on python-binance: it self-heals (#662) and the only gains elsewhere (23 h rotation, `serverShutdown`) are cosmetic for market data. REST stays on python-binance.

**Step 0 (owner, before 2026-10-14, about an hour).** With the live API key, confirm that (a) `POST /sapi/v1/userListenToken` still returns a token, (b) `userDataStream.subscribe.listenToken` on `ws-api.binance.com` still delivers an event for a small test action, and (c) whether the account is on the migrated margin user data stream per the 2026-09-25 entry (ask Binance support or check the account in the UI; probing the legacy `POST /sapi/v1/margin/listen-key` on the live key creates a listen key, so only do it if you also `DELETE` it, and `-3086` means migrated). Until done, the migration risk is [U]. If (a) or (b) fail, this becomes P1 regardless of library choice.

**Step 1: adapter boundary (no behaviour change, one PR).** Extract a `UserStreamClient` protocol from `BinanceProvider`:
`start(on_event) -> bool`, `stop()`, `hard_restart()`, `verify_alive() -> bool`, `state`, `age_seconds()`, `generation` handling. Provide `PythonBinanceUserStream` (the current code moved verbatim: `:2792-2889`, `:2985-3051`, `:3107-3196`, handler `:2597-2711`). `BinanceProvider`'s public API (`start_user_stream`, `stop_user_stream`, `reconnect_user`, `hard_reconnect_user`, `verify_user_stream_alive`, `user_ws_healthy`, `user_stream_age_seconds`) stays, so `ws_health.py` and the engine are untouched. The existing tests (`tests/unit/test_binance_provider_websocket.py`, `test_trading_engine_ws_health.py`) are the regression net.

**Step 2: `NativeMarginUserStream` behind `FEATURE_NATIVE_USER_STREAM` (default off).** Design:
- Dedicated thread + loop; `websockets` client; one connection, one subscription.
- Connect -> mint token (sync `margin_create_listen_token` run via `run_in_executor`) -> subscribe -> await ack -> state PRIMARY.
- Renew at `expirationTime - 60 s`; rotate the connection at about 23 h; treat `serverShutdown`, `eventStreamTerminated`, any close, subscribe timeout and ack error as "rebuild", with jittered backoff and a hard cap on attempts per 5 min to stay under 300 connections/5 min [V].
- Liveness: use `session.status` if Step 0 shows it reflects margin subscriptions; otherwise keep ping as transport check and rely on `eventStreamTerminated`/close for subscription loss.
- Never set `_user_event_received` from a probe (the #717 invariant); keep the generation token under `_user_stream_lock`.
- Dedupe on resubscribe overlap (renewal re-subscribes while the old subscription is live): key on `(orderId, executionType, transactionTime)` before `enqueue`.

**What to test with a fake server** (a local `websockets.serve` standing in for ws-api plus a stub token endpoint; time-compressible):

| Scenario | Expectation |
|---|---|
| Normal subscribe + events | Events reach `enqueue` in order; state PRIMARY only after the first real event |
| Server closes with 1000/1001/1006 | Rebuild: new token, new connection, resubscribed, backoff observed |
| `serverShutdown` event | Rebuild before the close lands |
| `eventStreamTerminated` | Rebuild; stale-generation events dropped |
| Token expiry (short `expirationTime`) | New token minted at expiry-60 s, subscription extended, no gap, no duplicate events |
| Server never pongs | Ping-timeout close, rebuild |
| Subscribe ack never arrives / error ack (-2036, 400) | Bounded retry, REST fallback stays on |
| Token endpoint returns 429/418 (with `Retry-After`) | Backoff honours `Retry-After`, no storm |
| Reconnect storm (server closes immediately, repeatedly) | Attempts per 5 min stay under the cap |
| 24 h rotation (compressed) | Old connection closed after new one acked; no event loss/duplication |
| Slow consumer | Bounded memory; loop not blocked (`enqueue` is non-blocking) |
| `stop()` mid-handshake / mid-renewal | No leaked thread/loop/FD (CODE.md resource rules), no late callbacks after stop |

**Staged rollout** (staging has no margin user stream; LESSONS §5):
1. Merge Step 1; deploy to staging and prod; soak, no behaviour change.
2. Merge Step 2 with the flag off. Unit + fake-server suite in CI.
3. **Shadow on prod**: flag value `shadow` runs the native client on its own connection alongside the incumbent, logs and counts events and state transitions, and compares against the incumbent stream by `(orderId, executionType, transactTime)`. It never feeds `UserDataProcessor`. Criteria to proceed: at least 7 days, at least one real fill observed by both, zero missed events vs incumbent, at least one forced-reconnect cycle (24 h rotation) recovered with no gap. One extra connection is within limits [V: 300/5 min, 1,000 subscriptions per session].
4. **Primary** with the python-binance stream retained as cold fallback (`hard_restart` swaps implementations), 7 more days, alerts unchanged.
5. Remove the python-binance user-stream code and the frame-pattern exception handler for user streams (#716/#907/#729 workarounds).

Pre-committed abort: any missed fill in shadow, or any duplicate-event effect on `OrderTracker` state, stops the rollout.

## 6. Decision for the owner

1. **Spend about one week building the in-house user stream (Steps 1-2), or accept stay-and-fix?**
   - Build: removes the structural defect and a good part of the ~1,550 lines of WS compensation code (provider WS section plus `ws_health.py`; some of it serves kline and stays), at the cost of owning Binance protocol drift for one stream.
   - Stay-and-fix: low cost; accepts that a transport drop can leave the stream REST-degraded until a hard reconnect or restart (precedent: 2026-08-08 to 08-13). Mitigation is alert delivery (webhook is set in prod) and deciding whether to turn `FEATURE_WS_USER_HARD_RECONNECT` on by default. Whether it was still on during the August episode: [U].
   - My lean: build, because the exposure is small today (about $83 prod, exchange-side SL) and shadow mode makes the build cheap to validate. That is judgement, not a finding.
2. **Step 0** needs the owner's API key and an hour; it is independent of the decision above and time-boxed by 2026-10-14.
3. If the owner prefers to adopt rather than build: the live margin-subscription check on UBWA (3.6) is the only open question blocking it; ccxt.pro is the fallback and the reference implementation.

## 7. Not verified

- Whether the 2026-09-25 / 2026-10-14 margin migration affects our account.
- Whether `session.status` / `session.subscriptions` report margin listenToken subscriptions.
- Whether UBWA's `userDataStream.subscribe.signature` works on a live cross-margin account.
- Whether `FEATURE_WS_USER_HARD_RECONNECT` was still on in prod during the 2026-08-08 degradation, and the post-2026-07-06 count of `REST_DEGRADED` transitions (only the one 5-day episode is in `log.md`).
- ccxt reconnect/resubscribe behaviour beyond reading `client.py`.
- jesse and octobot WS layers.
- Binance docs were read through a fetch tool that summarises rendered pages; the dated facts in section 2 should be re-read directly on developers.binance.com before anyone acts on the 2026-10-14 date.

## Sources

- Binance margin changelog: <https://developers.binance.com/docs/margin_trading/change-log>
- Listen Token User Data Stream: <https://developers.binance.com/docs/margin_trading/trade-data-stream/Listen-Token-Websocket-API>
- Deprecated margin trade data stream: <https://developers.binance.com/en/docs/products/margin-trading/trade-data-stream>
- Margin risk data stream: <https://developers.binance.com/en/docs/products/margin-trading/risk-data-stream>
- Margin best practice: <https://developers.binance.com/docs/margin_trading/best-practice>
- Spot WS streams docs: <https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md>
- Spot WS API docs: <https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-api.md>
- python-binance: <https://github.com/sammchardy/python-binance>, changelog <https://github.com/sammchardy/python-binance/blob/master/docs/changelog.rst>, PyPI <https://pypi.org/project/python-binance/>, issues <https://github.com/sammchardy/python-binance/issues/1668>, <https://github.com/sammchardy/python-binance/issues/1678>, PRs <https://github.com/sammchardy/python-binance/pull/1670>, <https://github.com/sammchardy/python-binance/pull/1681>
- Official SDK: <https://github.com/binance/binance-connector-python>, <https://pypi.org/project/binance-sdk-margin-trading/>, <https://pypi.org/project/binance-sdk-spot/>, <https://pypi.org/project/binance-common/>, legacy <https://pypi.org/project/binance-connector/>
- UBWA: <https://github.com/oliver-zehentleitner/unicorn-binance-websocket-api>, <https://pypi.org/project/unicorn-binance-websocket-api/>
- ccxt: <https://github.com/ccxt/ccxt>, <https://pypi.org/project/ccxt/>
- freqtrade: <https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/exchange/exchange_ws.py>
- hummingbot: <https://github.com/hummingbot/hummingbot/blob/master/hummingbot/connector/exchange/binance/binance_api_user_stream_data_source.py>
- websockets keepalive: <https://websockets.readthedocs.io/en/stable/topics/keepalive.html>
- Repo issues: #616, #646, #650, #652, #662, #663, #665, #716, #717, #723, #724, #729, #907, #908, #911, #915; `.claude/state/log.md` (lines 248, 276, 297, 969, 1107); `.claude/LESSONS.md` §5
