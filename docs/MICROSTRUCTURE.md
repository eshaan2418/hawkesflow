# Market structure notes: why order flow is self-exciting

## The empirical facts this reproduces
On NASDAQ ITCH for 2019-01-30 (six liquid names, regular hours, aggressor-signed displayed
executions):
1. **Poisson is badly wrong, even with intraday seasonality.** The KS distance of the
   time-rescaled residuals is 0.33–0.57. Executions cluster far more than a time-varying
   Poisson process allows.
2. **The flow is substantially endogenous.** In the 2-exponential model the spectral radius of
   the branching matrix is 0.41 (QQQ) to 0.65 (TSLA), so roughly half of all aggressive orders
   are, statistically, *responses* to earlier ones. That is in the range the literature reports
   for liquid equities at sub-second resolution (Filimonov & Sornette's "reflexivity",
   Hardiman-Bercot-Bouchaud).
3. **Excitation lives on several timescales at once.** It runs from about 50 µs through 1–8 ms
   to tens of milliseconds and beyond. Each extra exponential lowers BIC for every symbol, the
   signature of a heavy-tailed, roughly power-law kernel.
4. **Self beats cross.** Buys mostly trigger buys (0.33–0.61) and sells trigger sells, while
   cross-excitation is small (0.04–0.09).

## Where the clustering comes from
- **Order splitting.** Execution algorithms slice parent orders into child orders sent in quick
  succession: VWAP/POV schedules, and "sweep then refill" patterns.
- **Reaction to the book.** Latency-sensitive participants respond to a trade (and to the quote
  update it causes) within tens to hundreds of microseconds. The ~0.2 ms timescale is plausibly
  this: the round trip of co-located algorithms reacting to the feed.
- **Information arrival and herding** act at seconds and beyond.

## The fragmentation trap
- **Fragments.** A single aggressive order that consumes several resting orders appears as
  several executions at the **same nanosecond**: 85,757 executions become 52,954 aggressive
  events for AAPL (38% are fragments).
- **The trap.** Feeding fragments to a Hawkes model is worse than imprecise: the likelihood has
  no maximum (β → ∞ around zero gaps). Even with a cap, the "endogeneity" rises to 0.65–0.72,
  because order-book mechanics get counted as behaviour.
- **The rule.** Define the event as the decision (the aggressive order), not the print.

## What the residual misfit says
- **The short end.** All Hawkes models still deviate from Exp(1). The clustering region is
  captured well by two to three exponentials.
- **The long tail.** Long gaps are longer than predicted, which suggests regime-like
  variation in activity beyond the half-hour baseline: news, auctions of attention.
- **Remedies.** A finer baseline, a slow kernel component, or a doubly stochastic baseline.

## Conduct
The data is NASDAQ's public ITCH sample data, and the derived event files are not
redistributed. This is research on market structure, not a trading signal, and not investment
advice.
