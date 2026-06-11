-- FleetOps AI - Snowflake DDL
-- 50 aircraft, 6 bases, 2 years of history

CREATE OR REPLACE SCHEMA fleetops;
USE SCHEMA fleetops;

-- ============================================================
-- DIMENSION TABLES
-- ============================================================

CREATE TABLE bases (
    base_id         INTEGER PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    icao_code       VARCHAR(4) NOT NULL,
    lat             FLOAT NOT NULL,
    lon             FLOAT NOT NULL,
    region          VARCHAR(50) NOT NULL,
    timezone        VARCHAR(50) NOT NULL
);

CREATE TABLE aircraft (
    aircraft_id     INTEGER PRIMARY KEY,
    tail_number     VARCHAR(10) NOT NULL UNIQUE,
    aircraft_type   VARCHAR(50) NOT NULL,       -- e.g. King Air 350, Bell 407, EC135
    category        VARCHAR(20) NOT NULL,       -- fixed_wing | rotary
    home_base_id    INTEGER NOT NULL REFERENCES bases(base_id),
    status          VARCHAR(20) NOT NULL DEFAULT 'active',  -- active | grounded | depot
    commissioned_date DATE NOT NULL,
    total_flight_hours FLOAT NOT NULL DEFAULT 0,
    -- maintenance intervals vary by type, stored here for agent access
    engine_service_interval_hours   FLOAT NOT NULL,
    airframe_inspection_interval_hours FLOAT NOT NULL,
    avionics_check_interval_days    INTEGER NOT NULL
);

CREATE TABLE crew (
    crew_id         INTEGER PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    rank            VARCHAR(30) NOT NULL,
    role            VARCHAR(30) NOT NULL,       -- pilot | copilot | flight_nurse | flight_paramedic
    base_id         INTEGER NOT NULL REFERENCES bases(base_id),
    status          VARCHAR(20) NOT NULL DEFAULT 'available',  -- available | on_assignment | on_leave | medical
    qualifications  ARRAY,                      -- aircraft types qualified on, e.g. ['Bell 407','EC135']
    total_flight_hours FLOAT NOT NULL DEFAULT 0
);

-- ============================================================
-- FACT TABLES
-- ============================================================

CREATE TABLE missions (
    mission_id      INTEGER PRIMARY KEY,
    mission_type    VARCHAR(30) NOT NULL,       -- cargo | medevac | organ_transport | charter | repositioning
    priority        VARCHAR(10) NOT NULL,       -- routine | urgent | critical
    requested_date  TIMESTAMP_NTZ NOT NULL,
    start_date      TIMESTAMP_NTZ,
    end_date        TIMESTAMP_NTZ,
    origin_base_id  INTEGER NOT NULL REFERENCES bases(base_id),
    dest_base_id    INTEGER REFERENCES bases(base_id),  -- nullable for local ops (repositioning)
    aircraft_id     INTEGER REFERENCES aircraft(aircraft_id),  -- nullable until assigned
    status          VARCHAR(20) NOT NULL DEFAULT 'planned',  -- planned | active | complete | cancelled
    pax_count       INTEGER,
    cargo_tons      FLOAT
);

CREATE TABLE flight_logs (
    flight_id       INTEGER PRIMARY KEY,
    mission_id      INTEGER NOT NULL REFERENCES missions(mission_id),
    aircraft_id     INTEGER NOT NULL REFERENCES aircraft(aircraft_id),
    pic_crew_id     INTEGER NOT NULL REFERENCES crew(crew_id),  -- pilot in command
    origin_base_id  INTEGER NOT NULL REFERENCES bases(base_id),
    dest_base_id    INTEGER NOT NULL REFERENCES bases(base_id),
    scheduled_departure TIMESTAMP_NTZ NOT NULL,
    actual_departure    TIMESTAMP_NTZ,
    scheduled_arrival   TIMESTAMP_NTZ NOT NULL,
    actual_arrival      TIMESTAMP_NTZ,
    flight_hours    FLOAT,
    fuel_burn_lbs   FLOAT,
    status          VARCHAR(20) NOT NULL DEFAULT 'scheduled',  -- scheduled | airborne | completed | diverted | cancelled
    delay_minutes   INTEGER DEFAULT 0,
    delay_reason    VARCHAR(50)                 -- weather | maintenance | crew | airspace | null
);

CREATE TABLE maintenance_events (
    event_id        INTEGER PRIMARY KEY,
    aircraft_id     INTEGER NOT NULL REFERENCES aircraft(aircraft_id),
    base_id         INTEGER NOT NULL REFERENCES bases(base_id),
    event_type      VARCHAR(20) NOT NULL,       -- scheduled | unscheduled | inspection
    category        VARCHAR(30) NOT NULL,       -- engine | avionics | airframe | hydraulic | rotor | landing_gear
    started_at      TIMESTAMP_NTZ NOT NULL,
    completed_at    TIMESTAMP_NTZ,
    aircraft_hours_at_event FLOAT NOT NULL,     -- total aircraft hours when maintenance triggered
    next_due_hours  FLOAT,                      -- next interval threshold
    next_due_date   DATE,
    grounded_days   INTEGER DEFAULT 0,
    description     VARCHAR(500)
);

CREATE TABLE weather_observations (
    observation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    base_id         INTEGER NOT NULL REFERENCES bases(base_id),
    observed_at     TIMESTAMP_NTZ NOT NULL,
    temp_c          FLOAT,
    wind_speed_kts  INTEGER,
    wind_gust_kts   INTEGER,
    visibility_sm   FLOAT,
    ceiling_ft      INTEGER,
    precip_type     VARCHAR(20),                -- none | rain | snow | ice | fog
    precip_intensity VARCHAR(10),               -- null | light | moderate | heavy
    flight_category VARCHAR(4) NOT NULL         -- VFR | MVFR | IFR | LIFR
);

-- ============================================================
-- CLUSTERING KEYS (Snowflake-specific optimization)
-- ============================================================

ALTER TABLE flight_logs CLUSTER BY (aircraft_id, scheduled_departure);
ALTER TABLE weather_observations CLUSTER BY (base_id, observed_at);
ALTER TABLE maintenance_events CLUSTER BY (aircraft_id, started_at);
