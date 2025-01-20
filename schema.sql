create table
  public.idx_fear_and_greed (
    date date not null,
    momentum real not null,
    strength real not null,
    volatility real not null,
    volume_breadth real not null,
    safe_haven real not null,
    exchange_rate real not null,
    interest_rate real not null,
    buffett real not null,
    fear_and_greed_index real not null,
    updated_on timestamp with time zone not null,
    constraint idx_fear_and_greed_pkey primary key (date)
  ) tablespace pg_default;