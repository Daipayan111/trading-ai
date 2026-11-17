"""
Supabase Direct PostgreSQL Migration Script
Creates tables, RLS policies, and seed data in Supabase Cloud
"""

import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

DB_PASS = "Rdx@743147047Sona"
PROJECT_REF = "myqxkdptssxaatixdcng"
REGION = "ap-northeast-2"

# Connection strings to try (Pooler session mode, Pooler transaction mode, Direct host)
CONN_OPTIONS = [
    # Option 1: Supabase Connection Pooler (Session mode - Port 5432)
    {
        "host": f"aws-0-{REGION}.pooler.supabase.com",
        "port": 5432,
        "user": f"postgres.{PROJECT_REF}",
        "password": DB_PASS,
        "dbname": "postgres",
        "sslmode": "require"
    },
    # Option 2: Supabase Connection Pooler (Transaction mode - Port 6543)
    {
        "host": f"aws-0-{REGION}.pooler.supabase.com",
        "port": 6543,
        "user": f"postgres.{PROJECT_REF}",
        "password": DB_PASS,
        "dbname": "postgres",
        "sslmode": "require"
    },
    # Option 3: Direct DB connection (Port 5432)
    {
        "host": f"db.{PROJECT_REF}.supabase.co",
        "port": 5432,
        "user": "postgres",
        "password": DB_PASS,
        "dbname": "postgres",
        "sslmode": "require"
    }
]

SQL_STATEMENTS = """
-- 1. Create Users Table
CREATE TABLE IF NOT EXISTS public.users (
    id TEXT PRIMARY KEY DEFAULT ('usr_' || floor(extract(epoch from now()))::text),
    full_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    plan TEXT DEFAULT 'FREE' CHECK (plan IN ('FREE', 'PRO', 'VIP_INSTITUTIONAL')),
    status TEXT DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'BANNED', 'SUSPENDED')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_active TIMESTAMPTZ DEFAULT NOW(),
    notes TEXT
);

-- 2. Create System Config Table
CREATE TABLE IF NOT EXISTS public.system_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Create Signal Archive Table
CREATE TABLE IF NOT EXISTS public.signal_archive (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price NUMERIC(10, 2) NOT NULL,
    target NUMERIC(10, 2) NOT NULL,
    stop_loss NUMERIC(10, 2) NOT NULL,
    conviction NUMERIC(5, 2),
    confluence_score TEXT,
    outcome TEXT DEFAULT 'PENDING',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Enable Row Level Security (RLS)
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.system_config ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.signal_archive ENABLE ROW LEVEL SECURITY;

-- 5. Policies
DROP POLICY IF EXISTS "Public Config Read" ON public.system_config;
CREATE POLICY "Public Config Read" ON public.system_config FOR SELECT USING (true);

DROP POLICY IF EXISTS "Anon Full Access" ON public.users;
CREATE POLICY "Anon Full Access" ON public.users FOR ALL USING (true);

DROP POLICY IF EXISTS "Anon Signal Access" ON public.signal_archive;
CREATE POLICY "Anon Signal Access" ON public.signal_archive FOR ALL USING (true);

-- 6. Insert Default Configs
INSERT INTO public.system_config (key, value, description) VALUES
('min_confluence_score', '7', 'Minimum confluence checks passed out of 9 required'),
('tp1_multiplier', '0.65', 'Safe TP1 target ATR multiplier for momentum 50% lock'),
('tp2_multiplier', '1.40', 'Institutional primary supply zone target multiplier'),
('tp3_multiplier', '2.60', 'Trend runner target multiplier'),
('sl_buffer', '1.60', 'Stop loss ATR risk cushion buffer'),
('liquidity_wick_shield', 'true', 'Ignore intraday wick stop-outs unless candle body closes beyond SL'),
('traps_filter_active', 'true', 'Auto-disqualify trades when Bull/Bear or Consolidation Chop Traps detected'),
('kill_switch', 'false', 'Emergency Kill-Switch: Halt all trade signals across all user apps immediately'),
('darkpool_min_cr', '5.0', 'Minimum transaction value in Crores to trigger Dark Pool badge')
ON CONFLICT (key) DO NOTHING;

-- 7. Insert Initial Demo / Admin Traders
INSERT INTO public.users (id, full_name, email, phone, plan, status, notes) VALUES
('usr_admin_1', 'Tapas Adhikary (Admin)', 'tapas@tradingai.pro', '+91 98765 43210', 'VIP_INSTITUTIONAL', 'ACTIVE', 'Founder / System Administrator'),
('usr_demo_2', 'Vikram Malhotra', 'vikram.m@wealthcorp.in', '+91 98112 34567', 'VIP_INSTITUTIONAL', 'ACTIVE', 'Hedge Fund Desk Trader - TCS & BankNifty'),
('usr_demo_3', 'Ananya Sharma', 'ananya.s@traderdesk.com', '+91 97234 56789', 'PRO', 'ACTIVE', 'Swing Trader'),
('usr_demo_4', 'Rahul Verma', 'rahul.v@gmail.com', '+91 91234 56780', 'FREE', 'ACTIVE', 'Trial Trader (7-day evaluation)')
ON CONFLICT (id) DO NOTHING;
"""

def run_migration():
    connected = False
    conn = None
    for idx, opt in enumerate(CONN_OPTIONS, 1):
        print(f"Trying connection option {idx}: host={opt['host']}, port={opt['port']}, user={opt['user']}...")
        try:
            conn = psycopg2.connect(**opt, connect_timeout=10)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            print(f">>> Successfully connected via option {idx}!")
            connected = True
            break
        except Exception as e:
            print(f"Option {idx} failed: {e}")

    if not connected or not conn:
        print("ERROR: Could not establish direct PostgreSQL connection to Supabase.")
        sys.exit(1)

    print("\nExecuting Table Creation SQL Script...")
    with conn.cursor() as cur:
        cur.execute(SQL_STATEMENTS)
    print(">>> All tables, RLS policies, and seed data created successfully in Supabase!")

    # Verify tables
    with conn.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        tables = [row[0] for row in cur.fetchall()]
        print(f"\nPublic Tables in Supabase: {tables}")

        cur.execute("SELECT COUNT(*) FROM public.users;")
        u_count = cur.fetchone()[0]
        print(f"Users in public.users: {u_count}")

        cur.execute("SELECT COUNT(*) FROM public.system_config;")
        c_count = cur.fetchone()[0]
        print(f"Configs in public.system_config: {c_count}")

    conn.close()
    print("\n>>> SUPABASE MIGRATION 100% COMPLETE! <<<")

if __name__ == "__main__":
    run_migration()
