"""Data adapters for the EGX MCP server.

Raw data:        market, disclosures, news, technicals, portfolio, universe
Decision layer:  fundamentals, macro, scoring, peers, sizing, calendar, decision
Simulation:      simulation, egx_listing
PM layer:        risk_free, liquidity, regime, factors, risk, optimizer, backtest
Agentic layer:   sentiment, debate, risk_gate, reflection
ML layer:        transformer_sentiment, events, forecast (all optional-dep,
                 graceful fallback to lexicon/keyword/naive estimators)
IR layer:        ir_fetch (discover + archive company IR docs per folder),
                 ir_extract (provisional fundamentals from PDFs — needs review)
"""
