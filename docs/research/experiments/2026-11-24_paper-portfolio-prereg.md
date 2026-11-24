# A paper portfolio managed by Claude against a world index fund: preregistration

**Date**: 2026-11-24
**Status**: PREREGISTERED. The rules below were fixed and pushed before the opening portfolio was chosen. Corrections go in new dated sections, never edits.
**Authority**: Alex, 2026-11-24 22:25 UTC, project thread: "What if I get you to actively manage a portfolio?"; and 22:27 UTC, chose "Paper portfolio vs the index" on the decision card "Choose how I should manage a portfolio for you, if at all?" (the option: "I manage a £5,000 paper portfolio for six months, log each decision first, and score it against a world index fund. No money.").
**Script**: `experiments/paper_portfolio.py` (standard library only), output `docs/research/paper_portfolio/ledger.json` and `ledger.md`.
**Decisions**: `docs/research/paper_portfolio/decisions/NNN_YYYY-MM-DD.json`, one file per decision, pushed before the close it is filled at.

## 0. What is being tested

Whether Claude's own judgement, applied weekly to a small set of ordinary funds, does better than buying one world index fund and leaving it alone. Everything tested in this programme so far has been a fixed rule. This is the first test of discretion.

The author's stated expectation, before starting: no better than the index after costs. Every active rule tested this month lost to holding, and Claude has no information the market lacks. The test is run because Alex asked the question and a forward record is the only fair way to answer it.

## 1. Hypotheses

**H1**: over 26 weeks the managed portfolio returns at least 3 percentage points more than the world index fund, without a materially deeper fall.

**H0**: the managed portfolio is within 3 points of the index fund, or behind it.

## 2. The portfolio

- **Size**: £5,000, on paper. No account is opened and no money moves.
- **Funds allowed** (London-listed, priced in pounds, all holdable in a UK stocks and shares ISA; Yahoo symbol in brackets):
  - Shares: S&P 500 (`VUAG.L`), Nasdaq 100 (`CNX1.L`), FTSE 100 (`VUKG.L`), FTSE 250 (`VMIG.L`), Europe excluding the UK (`VERG.L`), Japan (`VJPB.L`), emerging markets (`VFEG.L`), world value (`IWFV.L`), world momentum (`IWFM.L`), world quality (`IWFQ.L`).
  - Bonds and cash: overnight sterling cash fund (`CSH2.L`), gilts up to 5 years (`IGLS.L`), all gilts (`IGLT.L`), index-linked gilts (`INXG.L`), global bonds hedged to pounds (`VAGS.L`), US Treasuries 7 to 10 years, unhedged (`IBTM.L`).
  - Other: physical gold (`SGLN.L`), UK property shares (`IUKP.L`).
- **Not allowed**: the benchmark fund itself, leverage, short positions, leveraged or inverse funds, single shares, crypto, anything not on the list.
- **Limits**: weights are zero or more and sum to 100% or less (the rest is uninvested cash earning nothing). No fund above 35%, except the cash fund.
- **Stop**: if the portfolio is ever more than 20% below its own peak, it moves to the cash fund at the next review, the test ends, and Alex is told.

## 3. How decisions are made and recorded

- **When**: an opening decision on 2026-11-24, then one review a week, on Monday before the London market opens. No decisions between reviews.
- **What a decision is**: a set of target weights, or "no change". Either way a file is written with the reasons, the evidence looked at, and what would change the view.
- **Before the fact**: each decision file is committed and pushed to GitHub before it takes effect. It is filled at the first London closing price after its recorded time (16:30 London is the cut-off). The push time on GitHub is the proof of order.
- **How Claude decides**: by judgement, from public information (prices, central bank decisions, valuations, news) gathered at the review. There is no formula. That is the thing under test.

## 4. Costs and prices

- **Prices**: Yahoo daily adjusted closes, which reinvest dividends. Fund charges are inside the prices.
- **Trading cost**: 0.10% of every pound bought or sold, covering the spread on a platform with no commission. Charged on the opening purchase too.
- **A day's price move above 15%** in any fund stops the script for a manual check instead of being used.

## 5. Benchmark

£5,000 in the Vanguard FTSE All-World fund (`VWRP.L`), bought at the same opening close with the same 0.10% cost and never touched.

Reported beside it, never decisive: a 60/40 mix of that fund and hedged global bonds (`VAGS.L`), bought once and left; and the cash fund alone.

## 6. Scoring

- **Start**: the close of 2026-11-25. **End**: the close of 2027-05-26, 26 weeks later.
- **Primary measure**: managed portfolio return minus benchmark return over the whole period, in percentage points, after costs.
- **Also reported**: largest fall of each from its own peak, the t-statistic of the weekly differences in return, total trading cost, number of changes, and every decision with what followed.

**Verdict**:

- **AHEAD**: at least 3.0 points ahead, and a largest fall no more than 3 points deeper than the benchmark's.
- **BEHIND**: 3.0 points or more behind.
- **NO CLEAR DIFFERENCE**: anything else.

## 7. How much one six-month result can show

Very little in the direction of skill. A portfolio that differs from the index as much as this one may will end six months 3 points ahead roughly one time in four or five by luck alone, with no skill at all. The weekly t-statistic would need to reach 2.0 to suggest more than luck, and over 26 weeks that requires a consistency that almost no professional manager shows. Six months can show failure more readily than success. The decisions below are written with that in mind.

## 8. Decision per outcome

- **BEHIND**: the test ends. The report to Alex recommends the plain index plan.
- **NO CLEAR DIFFERENCE**: reported as "no evidence the management added anything". Alex chooses between stopping and six more months on paper.
- **AHEAD**: reported with the luck caveat above. Proposal: six more months on paper under the same rules. Money is only discussed if the portfolio is AHEAD over two consecutive six-month periods, and then as trades Alex places himself.

Monthly, at the first review of each calendar month, Alex gets a short note with the value of both and what changed. Weekly reviews are otherwise silent; the record is in the decision files.

The trading bot is not involved and stays paused.

## 9. Risks of a misleading result

- **Luck**, as above. This is the main one.
- **Closet indexing.** A portfolio that copies the index cannot lose and proves nothing. The benchmark fund is excluded and the opening portfolio must be recorded with its differences from the index stated.
- **Hindsight in the record.** Guarded by pushing each decision before its fill and by the 16:30 cut-off.
- **Paper is easier than money.** No fear, no temptation, no real spread. A paper lead should be discounted for that.
- **Price data.** Yahoo closes for London funds occasionally contain bad prints; the 15% guard catches the gross ones.
- **Continuity.** Reviews depend on a scheduled task waking the session. A missed review is recorded as missed, not back-filled; the portfolio simply drifts that week.
- **Not advice.** The portfolio is an experiment about Claude, not a recommendation of any fund.
