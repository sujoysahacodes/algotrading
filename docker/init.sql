-- Initialize TimescaleDB for algorithmic trading data

-- Create extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create tables for market data
CREATE TABLE market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    PRIMARY KEY (time, symbol)
);

-- Convert to hypertable
SELECT create_hypertable('market_data', 'time');

-- Create tables for trades
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    commission DOUBLE PRECISION,
    pnl DOUBLE PRECISION
);

-- Create index on trades
CREATE INDEX idx_trades_time ON trades (time DESC);
CREATE INDEX idx_trades_strategy ON trades (strategy);
CREATE INDEX idx_trades_symbol ON trades (symbol);

-- Create tables for portfolio values
CREATE TABLE portfolio_values (
    time TIMESTAMPTZ NOT NULL,
    strategy TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    cash DOUBLE PRECISION,
    positions_value DOUBLE PRECISION,
    PRIMARY KEY (time, strategy)
);

-- Convert to hypertable
SELECT create_hypertable('portfolio_values', 'time');

-- Create tables for signals
CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    strength DOUBLE PRECISION,
    price DOUBLE PRECISION,
    metadata JSONB
);

-- Create index on signals
CREATE INDEX idx_signals_time ON signals (time DESC);
CREATE INDEX idx_signals_strategy ON signals (strategy);
CREATE INDEX idx_signals_symbol ON signals (symbol);

-- Create tables for risk metrics
CREATE TABLE risk_metrics (
    time TIMESTAMPTZ NOT NULL,
    strategy TEXT NOT NULL,
    var_95 DOUBLE PRECISION,
    expected_shortfall DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    volatility DOUBLE PRECISION,
    PRIMARY KEY (time, strategy)
);

-- Convert to hypertable
SELECT create_hypertable('risk_metrics', 'time');
