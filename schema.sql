create table public.idx_fear_and_greed (
  date date not null,
  momentum real not null,
  strength real not null,
  volatility real not null,
  recovery real not null,
  trend_strength real not null,
  fear_and_greed_index real not null,
  updated_on timestamp with time zone not null default (now() AT TIME ZONE 'utc'::text),
  constraint idx_fear_and_greed_pkey primary key (date)
) TABLESPACE pg_default;