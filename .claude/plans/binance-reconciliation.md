# Binance Reconciliation Plan

## Core Principle
**Binance is the source of truth.** The DB is a log/audit trail. When they disagree, Binance wins.

---

## Failure Scenarios

### Scenario 1: Bot Crashes After Order Sent (Network Timeout)
- **State:** Bot thinks order status unknown, Binance has the answer
- **Risk:** Duplicate orders if we retry
- **Solution:** Query order by clientOrderId (idempotency key) before acting

### Scenario 2: Position Closed Externally
- **State:** DB says position open, Binance says closed
- **Risk:** Bot tries to close already-closed position -> errors
- **Solution:** During recovery, query Binance positions first, reconcile DB

### Scenario 3: Partial Fill During Crash
- **State:** DB says full entry, Binance shows partial
- **Risk:** Position size mismatch -> wrong P&L, wrong exits
- **Solution:** Always verify order status, use executedQty from Binance

### Scenario 4: SL/TP Triggered While Bot Offline
- **State:** DB shows position open, SL order on Binance already filled
- **Risk:** Bot wakes up, sees position in DB, tries to manage non-existent position
- **Solution:** Recovery must query open positions on Binance, not just DB

### Scenario 5: Order Rejected (Insufficient Margin)
- **State:** DB records order submitted, Binance rejected it
- **Risk:** Bot thinks position exists, doesn't actually exist
- **Solution:** Check order status after submission, handle rejection explicitly

### Scenario 6: Manual User Intervention
- **State:** User closes position on Binance app
- **Risk:** Bot continues managing ghost position
- **Solution:** Periodic reconciliation, order operations verify state first

### Scenario 7: Entry Price Slippage > Expected
- **State:** Bot recorded entry_price=X, actual fill was X+slippage
- **Risk:** Wrong P&L calculations, wrong stop distances
- **Solution:** Use actual fill price from order confirmation, not expected price

### Scenario 8: WebSocket Disconnect During Order
- **State:** User event stream gaps, order execution missed
- **Risk:** Event handler never fires, state desynchronized
- **Solution:** REST API fallback recovery, event replay protection

---

## Architecture

### Phase 1: Recovery Reconciliation (Startup)
1. Query Binance for ALL open positions (not per-symbol)
2. Query Binance for ALL open orders (SL/TP/stop orders)
3. Build "truth set" from Binance data
4. Load DB positions (candidate set)
5. Reconcile:
   - Position in Binance but not DB -> Log warning, track anew
   - Position in DB but not Binance -> Log discrepancy, close in DB
   - Position in both but size differs -> Use Binance size
   - Position in both but price differs -> Use Binance price
   - Orphan orders (no position) -> Cancel and log

### Phase 2: Entry Phase (Order Placement)
1. Generate unique clientOrderId (idempotency key)
2. Send order to Binance
3. IMMEDIATELY query order status by clientOrderId
4. Verify: FILL, PARTIAL_FILL, REJECTED, EXPIRED
5. Only write to DB AFTER Binance confirmation
6. If timeout/unknown: Query by clientOrderId before retry
7. Never assume order succeeded without confirmation

### Phase 3: Exit Phase (Position Close)
1. Before exit: Verify position exists on Binance
2. Send exit order
3. Query order status for confirmation
4. On confirmed fill:
   - Close position in tracker
   - Write trade to DB (with actual exit price)
5. If position already closed on Binance:
   - Log "external close detected"
   - Reconcile DB to match
6. If order rejected: Handle gracefully, don't leave state bad

### Phase 4: Ongoing Reconciliation (Runtime)
1. Every N minutes (configurable, default 5min):
   - Query Binance positions
   - Compare with in-memory state
   - Log any discrepancies
   - If mismatch detected: Full reconciliation
2. On every order operation:
   - Pre-flight check: Verify current state on Binance
   - Post-operation: Verify new state on Binance
   - If mismatch: Log and trigger recovery
3. On WebSocket disconnect:
   - Mark state as "potentially stale"
   - On reconnect: Full reconciliation via REST API
   - Don't trade until reconciliation complete

---

## Discrepancy Handling

### Critical Discrepancies (stop trading immediately)
- Position exists in tracker but not on Binance
- Position size differs by > threshold
- Unexpected positions found on Binance
- **Action:** Log CRITICAL, pause trading, notify user

### Minor Discrepancies (log and auto-correct)
- Entry price differs (update to Binance value)
- SL/TP price differs (update to match active orders)
- Unrealized P&L differs (recalculate from Binance data)
- **Action:** Log WARNING, auto-correct, continue trading

---

## Implementation Structure

```
src/engines/live/execution/
  reconciliation.py          # NEW: Core reconciliation logic
    Reconciler               # Main reconciliation engine
    Discrepancy              # Discrepancy types and severity
    RecoveryResult           # Result of reconciliation

  position_tracker.py        # MODIFY: Add reconciliation hooks
    recover_positions()      # Add Binance verification
    verify_position_state()  # NEW: Check against Binance

  order_manager.py           # MODIFY: Add idempotency and confirmation
    place_order_with_confirmation()  # NEW: Wait for confirmation
    verify_order_status()    # NEW: Check by clientOrderId
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Write to DB AFTER Binance confirmation | DB should never claim something Binance doesn't know about |
| Use clientOrderId for idempotency | Can safely retry orders without duplicates |
| Full reconciliation on recovery | Can't trust any cached state after restart |
| Periodic reconciliation | Catches external changes (manual intervention) |
| Pre-flight checks before exits | Prevents errors from already-closed positions |
| CRITICAL vs WARNING discrepancies | Not all mismatches require stopping trading |

---

## Safety Guarantees

1. **No orphaned positions** - Recovery queries Binance directly
2. **No duplicate orders** - Idempotency via clientOrderId
3. **No ghost positions** - Reconciliation removes positions not on Binance
4. **Accurate P&L** - Entry/exit prices from Binance confirmations
5. **Graceful degradation** - Discrepancies logged and categorized
6. **Manual intervention detection** - Periodic reconciliation catches it
7. **Network resilience** - Order confirmation via REST API fallback

---

## Open Questions

1. How often should periodic reconciliation run? (5min? 15min? 30min?)
2. On discrepancy detection, should we pause trading or auto-correct and continue? (Recommendation: pause on critical, auto-correct on minor)
3. Should we expose a manual "reconcile now" CLI command?
4. Do we want webhook/alert integration for critical discrepancies?

---

## Status

- [ ] Plan reviewed and finalized
- [ ] Plan approved by Codex review
- [ ] Implementation started
- [ ] Tests written
- [ ] PR created
- [ ] Code reviewed
