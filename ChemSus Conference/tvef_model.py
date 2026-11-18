"""Texas TVEF battery hydrogen case study as a compact gurobipy model.

Data loading, component declarations, the uncertainty-handle table (`PARAMS`), `make_inputs` and the `H2System`
model class live here so that notebooks only contain experiments.

Typical use::

    from tvef_model import load_case_data, make_inputs, H2System, PARAMS
    data = load_case_data()
    sys_ = H2System(make_inputs(data), objective='cost', demand_sign='eq', ramp=True, smoothing=True)
    sys_.solve()
    out = sys_.results()            # energiapy-style result dictionary (same keys as the Pyomo notebook)

Scopes: 1 = direct process emissions (CO2 to stack / vented), 2 = purchased electricity (grid; PPA power under the
location-based convention), 3 = upstream natural gas (and merchant hydrogen in the v2 case).

Case configurations (`CASES`, `case_config`, `make_inputs(data, overrides, case=...)`, `H2SystemTree(..., case=...)`):
  'legacy' (default) reproduces the ChemSus case study: 200 MWh battery bound with a free 25 MW rating and a 1,250
            USD/MWh energy cost, daily hydrogen demand, no hydrogen storage, emission-free demand-penalty tiers D1-D4;
  'v2'      costed battery (annualized energy + power capex, sized rating, loose bounds), hourly demand, compressed
            hydrogen storage, merchant hydrogen with its own scope-3 intensity (`merchant_h2_ef`) instead of D1-D4, and
            annualized process capex and fixed O&M from installed costs (`PROCESS_COSTED`, IEA Future of Hydrogen 2019)
            in place of the legacy per-(kg/hr) figures.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import scipy.sparse as _sp
import gurobipy as gp
from gurobipy import GRB

__author__ = "Marco De Sousa"

# ----------------------------------------------------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------------------------------------------------
BASE_DEFAULT = 'C:/Users/marcopdsousa/Documents/energiapy'
N_DAYS, N_HOURS = 365, 24
N_T = N_DAYS * N_HOURS
LOCATION = 'BG'              # location key kept for compatibility with the energiapy / Pyomo result dictionaries
NETWORK_PERIOD = 0
bigM = 1e8
smallM = 1e-3
SCOPES = (1, 2, 3)
H2_DEMAND_BASE = 360000.0    # kg/day

# resource names (identical to the energiapy Resource(name=...) strings)
Solar, Wind, Power, GridPower = 'Solar', 'Wind', 'Power', 'GridPower'
CO2_Stack, CO2_Air, CO2_Vent, CO2cpt = 'CO2 to Stack', 'CO2 from air', 'CO2 Vented from the process', 'CO2 Captured'
NG, H2Pure = 'Natural Gas', 'Pure Hydrogen'
H2OFresh, H2O_Air, H2O_Vent, H2O_Stack = 'H2O Fresh', 'H2O from air', 'H2O vented from the process', 'H2O to Stack'
H2O_Vent_ASU, H2O_Blowdown = 'H2O Vent from ASU', 'H2O in Blowdown'
N2_Air, N2_Vent, N2Prod, N2_Stack = 'N2 from Air', 'N2 vented from the process', 'N2 in product stream', 'N2 to Stack'
N2_Desorped, N2_Vented_Stack, N2_Blowdown = 'N2 Desorped', 'N2 in Stack', 'N2 in Blowdown'
O2_Air, O2_Vent, O2_Stack, O2_ASU = 'O2 from air', 'O2 vented from the process', 'O2 to Stack', 'O2 vented out of ASU'
DP1, DP2, DP3, DP4 = 'DP1', 'DP2', 'DP3', 'DP4'
LiI_stored = 'LiI_Power_stored'
H2_stored, H2_merchant = 'H2 stored', 'H2 merchant'

def _inf(x):
    """Treat big-M declarations (>= bigM) as no bound, to keep the LP well conditioned."""
    return GRB.INFINITY if x >= bigM else float(x)


PENALTY_PROCS = ('D1', 'D2', 'D3', 'D4')     # demand-deviation tiers: hydrogen not produced on site (legacy case)
MERCHANT_PROCS = ('H2_import',)              # merchant hydrogen import (v2 case): hydrogen not produced on site


def crf(rate, life):
    """Capital recovery factor: annual payment per unit of installed cost at discount `rate` over `life` years."""
    return rate / (1 - (1 + rate) ** -life) if rate > 0 else 1.0 / life

GRID_EF_PROFILES = ('0.1', '0.25', 'median', 'mean', '0.75', '0.9')
REFORMER_CO2_YIELD_NOMINAL = 9.3713
CCS_NOMINAL = {   # nominal (captured, vented) kg CO2 per kg H2 -> design capture rate = cpt / (cpt + vent)
    'SMRCCS':       dict(cpt=8.903,       vent=0.46856,             consumes_stack=True),
    'ATR_CCS_R':    dict(cpt=7.733322317, vent=0.44956544299999995, consumes_stack=False),
    'ATR_CCS_R_OC': dict(cpt=8.152141983, vent=0.663744554,         consumes_stack=False),
}


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def normalized(x, n):
    """energiapy `scale_changer` equivalent: pad/truncate to n, divide by the maximum, NaN -> 0."""
    s = pd.Series(np.asarray(x, dtype=float)).reindex(range(n))
    return (s / s.max(skipna=True)).fillna(0.0).to_numpy()


class CaseData:
    """Raw time series of the case study (same files and preprocessing as the energiapy notebook)."""

    def __init__(self, base=BASE_DEFAULT):
        self.base = base
        # hourly ERCOT hub-average prices (all monthly sheets concatenated)
        sheets = pd.read_excel(f'{base}/ChemSus Conference/Data for Model/Ercot Prices 2025.xlsx', sheet_name=None)
        ep = pd.concat(sheets.values(), ignore_index=True)
        ep = ep[ep['Settlement Point'] == 'HB_HUBAVG']['Settlement Point Price'].reset_index(drop=True).iloc[:N_T]
        self.elec_price = ep.to_numpy(dtype=float)                                   # USD/MWh, 8760
        # Henry Hub daily prices stretched to hours (each value repeated 30x), outliers > 3 sigma replaced by the mean
        ng = pd.read_csv(f'{base}/AICHE/Historical natural gas prices/Natural gas historical data.csv')
        ng = ng[['Henry Hub Natural Gas Spot Price (Dollars per Million Btu)']]
        self.ng_price_max = float(ng.max().iloc[0])                                 # base price in energiapy = max / 20
        ngh = ng.reindex(ng.index.repeat(30))[:N_T]
        mean, std = ngh.mean(), ngh.std()
        ngh = ngh.mask(np.abs(ngh - mean) > 3 * std)
        ngh = ngh.fillna(ngh.mean())
        self.ng_price_hourly = ngh.iloc[:, 0].to_numpy(dtype=float)                 # USD/MMBtu, 8760
        # daily hydrogen demand profile
        dem = pd.read_csv(f'{base}/ESCAPE/Belgium Data/Belgium_H2_Demand_2024.csv')['Hydrogen_Demand_kg_per_day']
        self.h2_demand_profile = dem.iloc[:N_DAYS].to_numpy(dtype=float)
        # ERCOT carbon intensity profiles (rolling-window quantiles): columns 0.1, 0.25, 0.75, 0.9, mean, median
        ei = pd.read_csv(f'{base}/ChemSus Conference/Data for Model/Ercot EI profile.csv')
        ei.columns = [str(c) for c in ei.columns]
        self.grid_ef = {c: ei[c].iloc[:N_T].to_numpy(dtype=float) for c in GRID_EF_PROFILES}   # kg CO2e/MWh
        # renewable availability
        wind = pd.read_csv(f'{base}/ChemSus Conference/Data for Model/Final Wind Availability ERCOT 2025 Hourly.csv')
        self.wind = wind['Wind availability'].iloc[:N_T].to_numpy(dtype=float)                 # 8759 rows -> last hour 0
        solar = pd.read_csv(f'{base}/ChemSus Conference/Data for Model/Final Solar Availability ERCOT 2025 Hourly.csv')
        self.solar = solar['HSLSystemWide'].iloc[:N_T].to_numpy(dtype=float)

        # normalized factors (energiapy semantics) and effective nominal series
        self.price_factor_ng = normalized(self.ng_price_hourly, N_T)
        self.capacity_factor_pv = normalized(self.solar, N_T)
        self.capacity_factor_wf = normalized(self.wind, N_T)
        self.demand_factor = normalized(self.h2_demand_profile, N_DAYS)
        self.h2_demand = H2_DEMAND_BASE * self.demand_factor                        # kg/day


def load_case_data(base=BASE_DEFAULT) -> CaseData:
    return CaseData(base)


# ----------------------------------------------------------------------------------------------------------------------
# components
# ----------------------------------------------------------------------------------------------------------------------
def resource(cons_max=0.0, price=0.0, gwp=0.0, sell=False, demand=False, varying=None, scope=None, basis='kg/hr', label=''):
    return dict(cons_max=cons_max, price=price, gwp=gwp, sell=sell or demand, demand=demand, varying=varying,
                scope=scope, basis=basis, label=label, store_max=0.0, store_min=0.0)


def process(conversion, capex=0.0, vopex=0.0, fopex=0.0, prod_min=0.0, prod_max=0.0, capacity_factor=None, basis='kg/hr', label=''):
    return dict(conversion=dict(conversion), capex=capex, vopex=vopex, fopex=fopex, prod_min=prod_min, prod_max=prod_max,
                capacity_factor=capacity_factor, basis=basis, label=label)


RESOURCES = {
    Solar: resource(cons_max=bigM, price=52.69, scope=2, basis='MW', label='Solar Power'),      # PPA, USD/MWh
    Wind: resource(cons_max=bigM, price=60.11, scope=2, basis='MW', label='Wind Power'),        # PPA, USD/MWh
    Power: resource(basis='MW', label='Power generated'),
    GridPower: resource(cons_max=bigM, price=None, gwp=None, varying=['price', 'gwp'], scope=2, basis='MW', label='Power bought'),
    CO2_Stack: resource(sell=True, gwp=1, scope=1, label='Carbon dioxide to Stack'),
    CO2_Air: resource(cons_max=bigM, label='Carbon dioxide from air'),
    CO2_Vent: resource(sell=True, gwp=1, scope=1, label='Carbon dioxide vented from the process'),
    CO2cpt: resource(sell=True, label='Captured carbon dioxide'),
    NG: resource(cons_max=bigM, price=None, gwp=0.960979, varying=['price'], scope=3, label='Natural Gas'),
    H2Pure: resource(demand=True, varying=['demand'], label='Pure Hydrogen'),
    H2OFresh: resource(cons_max=bigM, price=0.002, label='Fresh Water'),
    H2O_Air: resource(cons_max=bigM, label='Water from air'),
    H2O_Vent: resource(sell=True), H2O_Stack: resource(sell=True), H2O_Vent_ASU: resource(sell=True), H2O_Blowdown: resource(sell=True),
    N2_Air: resource(cons_max=bigM, label='Nitrogen from Air'),
    N2_Vent: resource(sell=True), N2Prod: resource(sell=True), N2_Stack: resource(sell=True),
    N2_Desorped: resource(sell=True), N2_Vented_Stack: resource(sell=True), N2_Blowdown: resource(sell=True),
    O2_Air: resource(cons_max=bigM, label='Oxygen from air'),
    O2_Vent: resource(sell=True), O2_Stack: resource(sell=True), O2_ASU: resource(sell=True),
    # demand-deviation penalty tiers: purchasable "deficit" hydrogen at increasing prices, converted by D1-D4
    DP1: resource(cons_max=750, price=10), DP2: resource(cons_max=2250, price=25),
    DP3: resource(cons_max=4000, price=50), DP4: resource(cons_max=5250, price=115),
    LiI_stored: resource(basis='MW', label='Power LiI(stored)'),
}
RESOURCES[LiI_stored].update(store_max=200.0, store_min=0.0)

# legacy (ChemSus) battery: 200 MWh bound, fixed 25 MW rating with no capex, 1,250 USD/MWh energy cost (not annualized)
BATTERY = dict(name='LiI', storage=Power, vopex=0.5, capex=0.0, fopex=0.0, prod_min=0, prod_max=25,
               store_max=200, store_min=0, storage_loss=0.05, storage_capex=1250, storage_cost=0.0)

# costed battery (v2): installed costs annualized with a capital recovery factor, fixed O&M as a fraction of the installed
# cost, the charge/discharge rating is a sized decision (one rating shared by both directions), bounds are loose and only
# there for conditioning. Energy 250 USD/kWh, power 150 USD/kW (utility-scale Li-ion, NREL ATB 2024 order of magnitude).
BATTERY_COSTED = dict(
    energy_capex_installed=250_000.0,   # USD per MWh installed
    power_capex_installed=150_000.0,    # USD per MW installed (one rating, charge and discharge)
    fom_fraction=0.025,                 # fixed O&M per year as a fraction of the installed cost
    discount_rate=0.08, life_years=15,
    vopex=0.5,                          # USD per MWh charged (degradation / cycling proxy, as in the legacy case)
    storage_loss=0.05,                  # fraction lost on discharge (round trip 95 %, as in the legacy case)
    store_max=10_000.0, rating_max=2_000.0,   # loose bounds [MWh], [MW]
)
BATTERY_COSTED['energy_capex_annual'] = BATTERY_COSTED['energy_capex_installed'] * (
    crf(BATTERY_COSTED['discount_rate'], BATTERY_COSTED['life_years']) + BATTERY_COSTED['fom_fraction'])    # USD/MWh/yr
BATTERY_COSTED['power_capex_annual'] = BATTERY_COSTED['power_capex_installed'] * crf(
    BATTERY_COSTED['discount_rate'], BATTERY_COSTED['life_years'])                                            # USD/MW/yr
BATTERY_COSTED['power_fopex_annual'] = BATTERY_COSTED['power_capex_installed'] * BATTERY_COSTED['fom_fraction']

# hydrogen storage (v2): compressed-gas buffer; compression draws power, release is free of losses
H2_STORAGE = dict(
    storage_capex_installed=500.0,      # USD per kg of storage capacity (compressed gas, 350 bar, DOE / HDSAM range)
    storage_fom_fraction=0.01, storage_life_years=20,
    compressor_capex_installed=2_250.0, # USD per (kg/hr) of compression capacity (~1.5 kW per kg/hr at ~1,500 USD/kW)
    compressor_fom_fraction=0.04, compressor_life_years=15,
    compression_power=0.0015,           # MW per (kg/hr), i.e. 1.5 kWh per kg
    discount_rate=0.08,
    store_max=3_000_000.0, rate_max=200_000.0,   # loose bounds [kg], [kg/hr]
)
H2_STORAGE['storage_capex_annual'] = H2_STORAGE['storage_capex_installed'] * (
    crf(H2_STORAGE['discount_rate'], H2_STORAGE['storage_life_years']) + H2_STORAGE['storage_fom_fraction'])   # USD/kg/yr
H2_STORAGE['compressor_capex_annual'] = H2_STORAGE['compressor_capex_installed'] * crf(
    H2_STORAGE['discount_rate'], H2_STORAGE['compressor_life_years'])                                            # USD/(kg/hr)/yr
H2_STORAGE['compressor_fopex_annual'] = H2_STORAGE['compressor_capex_installed'] * H2_STORAGE['compressor_fom_fraction']

# merchant hydrogen (v2): purchased hydrogen with its own scope-3 intensity (handle `merchant_h2_ef`); replaces D1-D4
MERCHANT_H2 = dict(price=5.0, cons_max=bigM)     # USD per kg delivered; no artificial volume limit

PROCESSES = {
    'Grid': process({GridPower: -1, Power: 1}, prod_min=bigM, prod_max=bigM, basis='MW', label='Grid'),
    'PV': process({Solar: -1, Power: 0.22}, prod_max=600, prod_min=600, capacity_factor='PV', basis='MW', label='Solar PV'),
    'WF': process({Wind: -1, Power: 0.4}, prod_max=650, prod_min=650, capacity_factor='WF', basis='MW', label='Wind mill array'),
    'SMR': process({Power: -0.00065, NG: -3.535189, H2OFresh: -16.78877421, N2_Air: -19.2134, O2_Air: -5.88716,
                    H2O_Air: -0.1582, CO2_Air: -0.01171,
                    H2Pure: 1, CO2_Stack: 9.3713, H2O_Stack: 3.94959, N2_Stack: 19.29814, O2_Stack: 0.535434},
                   capex=0.2, vopex=0.38, fopex=0.06, prod_min=smallM, prod_max=bigM, label='SMR'),
    'SMRCCS': process({Power: -0.00139, N2_Air: -22.2315, O2_Air: -6.811924, H2O_Air: -0.183055, CO2_Air: -0.013551,
                       CO2_Stack: -9.3713,
                       H2O_Vent: 3.826767, CO2cpt: 8.903, CO2_Vent: 0.46856, H2O_Stack: 0.558114, N2_Stack: 22.32218,
                       O2_Stack: 0.6203032},
                      capex=0.45, vopex=1.01, fopex=0.1, prod_min=smallM, prod_max=bigM, label='SMRCCS'),
    'PEM': process({Power: -0.052, H2OFresh: -9.5, H2Pure: 1},
                   capex=1.5, vopex=3.43, fopex=0.7, prod_min=smallM, prod_max=bigM, label='PEM'),
    'ATR_CCS_R': process({Power: -0.0025838509316770185, NG: -3.2080287, H2OFresh: -4.8123228, H2O_Air: -0.1041653,
                          H2O_Vent_ASU: 0.0019387, H2O_Blowdown: 1.7351222, H2O_Vent: 0.5220523,
                          N2_Air: -12.653394, N2_Vent: 9.970844643, N2Prod: 0.002780318, N2_Vented_Stack: 2.182324573,
                          N2_Desorped: 0.023329689, O2_Air: -3.87607534,
                          O2_ASU: 0.061984387, O2_Vent: 0.0647028, CO2_Vent: 0.44956544299999995, CO2cpt: 7.733322317,
                          H2Pure: 1},
                         capex=0.35, vopex=0.74, fopex=0.08, prod_min=smallM, prod_max=bigM, label='ATR_CCS_R'),
    'ATR_CCS_R_OC': process({Power: -0.002881987577, NG: -3.208028688, CO2_Vent: 0.663744554, CO2cpt: 8.152141983,
                             N2_Air: -12.56295099, N2_Vent: 11.88406435, N2_Blowdown: 0.008434949, N2_Desorped: 0.102417105,
                             N2Prod: 0.002780318,
                             H2OFresh: -4.812322829, H2O_Air: -0.103420802, H2O_Vent_ASU: 0.002310721,
                             H2O_Blowdown: 0.004504513, O2_Air: -4.804359335,
                             O2_Vent: 0.073878039, H2Pure: 1},
                            capex=0.35, vopex=0.75, fopex=0.1, prod_min=smallM, prod_max=bigM, label='ATR_CCS_R_OC'),
    'AWE': process({Power: -0.06, H2OFresh: -9.4635, H2Pure: 1},
                   capex=0.810127, vopex=3.850633, fopex=0.202532, prod_min=smallM, prod_max=bigM, label='AWE'),
    'D1': process({DP1: -1, H2Pure: 1}, prod_max=bigM, label='D1'),
    'D2': process({DP2: -1, H2Pure: 1}, prod_max=bigM, label='D2'),
    'D3': process({DP3: -1, H2Pure: 1}, prod_max=bigM, label='D3'),
    'D4': process({DP4: -1, H2Pure: 1}, prod_max=bigM, label='D4'),
    # lithium-ion battery: charging process (vopex on charging) and discharging process (loss on output)
    'LiI': process({Power: -1, LiI_stored: 1}, capex=BATTERY['capex'], vopex=BATTERY['vopex'], fopex=BATTERY['fopex'],
                   prod_min=BATTERY['prod_min'], prod_max=BATTERY['prod_max'], basis='MW', label='Lithium-ion battery'),
    'LiI_discharge': process({LiI_stored: -1, Power: 1 - BATTERY['storage_loss']}, capex=BATTERY['capex'],
                             prod_min=BATTERY['prod_min'], prod_max=BATTERY['prod_max'], basis='MW', label='Lithium-ion battery_storage'),
}
STORAGE_CAPEX = {LiI_stored: BATTERY['storage_capex']}     # USD per MWh installed
STORAGE_COST = {LiI_stored: BATTERY['storage_cost']}       # USD per MWh held per hour

# annualized process economics (v2): installed cost per unit of capacity, annualized with a capital recovery factor plus
# a fixed O&M fraction, the same treatment as BATTERY_COSTED and H2_STORAGE. Installed costs from the IEA Future of
# Hydrogen (2019) assumptions annex ("Today" column): natural gas reforming 910 USD/kW_H2 (OPEX 4.7 %/yr), reforming with
# carbon capture 1,680 USD/kW_H2 (OPEX 3 %/yr), water electrolysis 900 USD/kW_e (OPEX 1.5 %/yr), 25-year lifetime.
# Capacity basis of the processes is kg H2/hr: 1 kg/hr = 33.33 kW_H2 (LHV 120 MJ/kg); electrolysers are priced per kW_e
# of their own power draw (PEM 52 kW_e, AWE 60 kW_e per kg/hr). SMRCCS is the capture add-on on an SMR stack, priced at
# the increment between reforming with and without capture; ATR with capture is priced as reforming with capture.
KW_PER_KG_H2 = 1000.0 / 30.0        # 33.33 kW_H2 per (kg/hr), LHV
PROCESS_COSTED = {
    'SMR':          dict(installed_per_kw=910.0,  kw_per_unit=KW_PER_KG_H2, fom_fraction=0.047, life_years=25),
    'SMRCCS':       dict(installed_per_kw=770.0,  kw_per_unit=KW_PER_KG_H2, fom_fraction=0.03,  life_years=25),
    'ATR_CCS_R':    dict(installed_per_kw=1680.0, kw_per_unit=KW_PER_KG_H2, fom_fraction=0.03,  life_years=25),
    'ATR_CCS_R_OC': dict(installed_per_kw=1680.0, kw_per_unit=KW_PER_KG_H2, fom_fraction=0.03,  life_years=25),
    'PEM':          dict(installed_per_kw=900.0,  kw_per_unit=-1000.0 * PROCESSES['PEM']['conversion'][Power], fom_fraction=0.015, life_years=25),
    'AWE':          dict(installed_per_kw=900.0,  kw_per_unit=-1000.0 * PROCESSES['AWE']['conversion'][Power], fom_fraction=0.015, life_years=25),
}
PROCESS_DISCOUNT_RATE = 0.08
for _p, _d in PROCESS_COSTED.items():
    _d['capex_installed'] = _d['installed_per_kw'] * _d['kw_per_unit']                          # USD per (kg/hr)
    _d['capex_annual'] = _d['capex_installed'] * crf(PROCESS_DISCOUNT_RATE, _d['life_years'])    # USD per (kg/hr) per yr
    _d['fopex_annual'] = _d['capex_installed'] * _d['fom_fraction']                              # USD per (kg/hr) per yr
    _d['legacy_capex'], _d['legacy_fopex'] = PROCESSES[_p]['capex'], PROCESSES[_p]['fopex']
del _p, _d


def process_cost_table() -> pd.DataFrame:
    """PROCESS_COSTED as a table (installed, annualized capex, fixed O&M, legacy figures) for notebooks."""
    return pd.DataFrame(PROCESS_COSTED).T[['installed_per_kw', 'kw_per_unit', 'capex_installed', 'fom_fraction', 'life_years',
                                           'capex_annual', 'fopex_annual', 'legacy_capex', 'legacy_fopex']].astype(float)

# operational constraint parameters (ramp rate, minimum load, second-difference smoothing) for the reforming routes
# ERCOT four coincident peaks (4CP): transmission cost allocated on the load's average demand in the four 15-minute
# intervals of the ERCOT system peak in June-September, charged over the following year. 2025 intervals (ERCOT report to
# the PUCT, Dec 2025; interval times from Enverus): Jun 19 17:00, Jul 30 17:00, Aug 18 17:00, Sep 4 17:30, mapped to the
# hour that contains the interval (hour ending 17 = 16:00-17:00; the 17:30 interval falls in hour ending 18).
# Rate: ERCOT wholesale transmission rate 66.76 USD per kW-year (NRG remarks to the PUCT, Feb 2025); distribution
# service providers pass it through at 40,000-76,000 USD per MW-year.
ERCOT_4CP_2025 = {6: (19, 17), 7: (30, 17), 8: (18, 17), 9: (4, 18)}      # month: (day, hour ending)
ERCOT_4CP_RATE = 66.76                                                    # USD per kW of average 4CP demand per year
ERCOT_4CP_WINDOW = dict(hours=(16, 17, 18, 19), months=(6, 7, 8, 9))     # candidate hour endings (15:00-19:00) of a peak
RAMP = {'SMR': 0.05, 'ATR_CCS_R': 0.1, 'ATR_CCS_R_OC': 0.1}


def _hour_index(month, day, hour_ending, year=2025):
    """0-based hour index in the model year of (month, day, hour ending)."""
    import datetime as _dt
    doy = (_dt.date(year, month, day) - _dt.date(year, 1, 1)).days
    return doy * N_HOURS + hour_ending - 1


def _month_hours(month, hours, year=2025):
    """Hour indices of the given hour endings on every day of a month."""
    import calendar as _cal
    return [_hour_index(month, d, h, year) for d in range(1, _cal.monthrange(year, month)[1] + 1) for h in hours]
MINLOAD = {'SMR': 0.3, 'ATR_CCS_R': 0.2, 'ATR_CCS_R_OC': 0.2}
C2 = {'SMR': 0.008, 'ATR_CCS_R': 0.018, 'ATR_CCS_R_OC': 0.018}


# ----------------------------------------------------------------------------------------------------------------------
# case configuration
# ----------------------------------------------------------------------------------------------------------------------
# 'legacy' reproduces the ChemSus case study exactly; 'v2' is the revised case:
#   battery     'legacy' (200 MWh bound, free 25 MW rating, 1,250 USD/MWh) | 'costed' (BATTERY_COSTED)
#   demand      'daily' (daily totals, free intraday buffer)                | 'hourly' (hourly targets, `demand_shape`)
#   h2_storage  False                                                       | True (H2_STORAGE)
#   slack       'tiers' (D1-D4, emission free)                              | 'merchant' (MERCHANT_H2 + `merchant_h2_ef`)
#   process_capex 'legacy' (per-(kg/hr) figures of the case study)          | 'annualized' (PROCESS_COSTED)
#   prorate_capex False (fixed charges counted in full whatever the horizon) | True (capex + fixed O&M x horizon / year)
CASES = {
    'legacy': dict(battery='legacy', demand='daily', h2_storage=False, slack='tiers', process_capex='legacy', prorate_capex=False,
                   demand_shape=None, demand_interp=False, battery_cost_scale=1.0, h2_storage_cost_scale=1.0, process_cost_scale=1.0),
    'v2': dict(battery='costed', demand='hourly', h2_storage=True, slack='merchant', process_capex='annualized', prorate_capex=True,
               demand_shape=None, demand_interp=True, battery_cost_scale=1.0, h2_storage_cost_scale=1.0, process_cost_scale=1.0),
}
DEFAULT_CASE = 'legacy'


def case_config(case=None, **overrides) -> dict:
    """Configuration dict of a case: a name in CASES, a dict, or None (DEFAULT_CASE); `overrides` replace fields.
    demand_shape : 24 values (any scale) distributing each day's demand over its hours for demand='hourly'; None = flat;
    demand_interp: True interpolates the daily series linearly between day centres (no steps at midnight; each day's
                   total is preserved), False repeats each day's value for its 24 hours;
    battery_cost_scale, h2_storage_cost_scale, process_cost_scale: multipliers on the annualized capex and fixed O&M of
                   the costed battery (energy and power), of the hydrogen storage (vessel and compressor) and of the
                   annualized production processes (PROCESS_COSTED), for break-even studies;
    prorate_capex: True scales every capex and fixed O&M charge (all of them annual figures in the costed cases) by the
                   horizon length / one year, so that a short-horizon model (n_days < 365, or a tree over part of the
                   year) does not pay a full year of fixed charges against a few weeks of operation."""
    if case is None:
        cfg = dict(CASES[DEFAULT_CASE], name=DEFAULT_CASE)
    elif isinstance(case, str):
        cfg = dict(CASES[case], name=case)
    else:
        cfg = dict(CASES[DEFAULT_CASE], name='custom')
        cfg.update(case)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f'unknown case field(s): {sorted(unknown)}; known: {sorted(cfg)}')
    cfg.update(overrides)
    if cfg['battery'] not in ('legacy', 'costed') or cfg['demand'] not in ('daily', 'hourly') \
            or cfg['slack'] not in ('tiers', 'merchant', 'none') or cfg.get('process_capex', 'legacy') not in ('legacy', 'annualized'):
        raise ValueError(f'invalid case configuration {cfg}')
    return cfg


def build_components(cfg: dict) -> dict:
    """Resources, processes and storage economics of a case configuration.

    Returns dict(resources, processes, storage_capex, storage_cost, rating_links, slack_procs, intensity_basis).
    rating_links: pairs of processes whose capacities are tied (Cap_P[a] == Cap_P[b]);
    slack_procs: processes delivering hydrogen that is not produced on site (tiers or merchant import);
    intensity_basis: default denominator of an intensity cap, 'own' when the slack is emission free, else 'delivered'.
    """
    res = copy.deepcopy(RESOURCES)
    pr = copy.deepcopy(PROCESSES)
    storage_capex, storage_cost = dict(STORAGE_CAPEX), dict(STORAGE_COST)
    rating_links, slack_procs = [], list(PENALTY_PROCS)
    # battery
    if cfg['battery'] == 'costed':
        b, k = BATTERY_COSTED, float(cfg.get('battery_cost_scale', 1.0))
        res[LiI_stored].update(store_max=b['store_max'], store_min=0.0)
        pr['LiI'] = process({Power: -1, LiI_stored: 1}, capex=k * b['power_capex_annual'], fopex=k * b['power_fopex_annual'],
                            vopex=b['vopex'], prod_min=0.0, prod_max=b['rating_max'], basis='MW', label='Lithium-ion battery')
        pr['LiI_discharge'] = process({LiI_stored: -1, Power: 1 - b['storage_loss']}, prod_min=0.0, prod_max=b['rating_max'],
                                      basis='MW', label='Lithium-ion battery_storage')
        storage_capex[LiI_stored] = k * b['energy_capex_annual']
        rating_links.append(('LiI_discharge', 'LiI'))
    # hydrogen storage
    if cfg['h2_storage']:
        h, k = H2_STORAGE, float(cfg.get('h2_storage_cost_scale', 1.0))
        res[H2_stored] = resource(basis='kg', label='Hydrogen stored (compressed)')
        res[H2_stored].update(store_max=h['store_max'], store_min=0.0)
        pr['H2_compress'] = process({H2Pure: -1, Power: -h['compression_power'], H2_stored: 1},
                                    capex=k * h['compressor_capex_annual'], fopex=k * h['compressor_fopex_annual'],
                                    prod_min=0.0, prod_max=h['rate_max'], label='Hydrogen compression to storage')
        pr['H2_release'] = process({H2_stored: -1, H2Pure: 1}, prod_min=0.0, prod_max=h['rate_max'], label='Hydrogen release from storage')
        storage_capex[H2_stored] = k * h['storage_capex_annual']
        rating_links.append(('H2_release', 'H2_compress'))
    # production processes: annualized installed cost + fixed O&M in place of the legacy figures
    if cfg.get('process_capex', 'legacy') == 'annualized':
        k = float(cfg.get('process_cost_scale', 1.0))
        for q, d in PROCESS_COSTED.items():
            pr[q]['capex'], pr[q]['fopex'] = k * d['capex_annual'], k * d['fopex_annual']
    # slack supply
    if cfg['slack'] != 'tiers':
        for q, r in zip(PENALTY_PROCS, (DP1, DP2, DP3, DP4)):
            pr.pop(q); res.pop(r)
        slack_procs = []
    if cfg['slack'] == 'merchant':
        res[H2_merchant] = resource(cons_max=MERCHANT_H2['cons_max'], price=MERCHANT_H2['price'],
                                    gwp=float(PARAMS.loc['merchant_h2_ef', 'nominal']), scope=3, label='Merchant hydrogen')
        pr['H2_import'] = process({H2_merchant: -1, H2Pure: 1}, prod_max=bigM, label='Merchant hydrogen import')
        slack_procs = list(MERCHANT_PROCS)
    intensity_basis = 'own' if cfg['slack'] == 'tiers' else 'delivered'
    return dict(resources=res, processes=pr, storage_capex=storage_capex, storage_cost=storage_cost,
                rating_links=rating_links, slack_procs=slack_procs, intensity_basis=intensity_basis)


# ----------------------------------------------------------------------------------------------------------------------
# uncertainty handles
# ----------------------------------------------------------------------------------------------------------------------
PARAMS = pd.DataFrame([
    # handle                nominal     low        high        type         stage           unit              description                                                                                                   source
    ('gas_price_level',     1.0,        0.5,       1.5,        'aleatory',  'wait-and-see', '-',              'multiplier on the hourly natural gas price series (Henry Hub level)',                                        'Henry Hub 2021-2024 spanned ~2-9 USD/MMBtu'),
    ('elec_price_level',    1.0,        0.7,       1.3,        'aleatory',  'wait-and-see', '-',              'multiplier on the hourly ERCOT electricity price series (PPA prices unchanged)',                              'ERCOT year-to-year hub price variability'),
    ('grid_ef_profile',     'mean',     '0.1',     '0.9',      'aleatory',  'wait-and-see', 'column',         'column of "Ercot EI profile.csv" used as hourly grid emission factor: 0.1, 0.25, median, mean, 0.75, 0.9',   'rolling +/-3 day window quantiles of ERCOT 2017-2025 carbon intensity'),
    ('grid_ef_level',       1.0,        0.85,      1.15,       'aleatory',  'wait-and-see', '-',              'multiplier on the hourly grid emission factor',                                                              'interannual change of ERCOT average carbon intensity'),
    ('renewable_level',     1.0,        0.9,       1.1,        'aleatory',  'wait-and-see', '-',              'multiplier on the PV and wind capacity factors (capped at 1)',                                               'interannual resource variability'),
    ('ng_upstream_ef',      0.960979,   0.6,       2.5,        'epistemic', 'here-and-now', 'kg CO2e/kg NG',  'upstream (scope 3) emission factor of natural gas',                                                          '1-3 % methane leakage, GWP100 (29.8) vs GWP20 (82.5), AR6'),
    ('reformer_co2_yield',  9.3713,     8.9,       10.3,       'epistemic', 'here-and-now', 'kg CO2/kg H2',   'CO2 generated per kg H2 by SMR; CO2 generated by the ATR routes is scaled by the same relative factor',       'gas composition / heating value and reformer efficiency, ~ +/-5 %'),
    ('capture_rate_factor', 1.0,        0.70,      1.0,        'epistemic', 'here-and-now', '-',              'multiplier on the design capture rate of SMRCCS (0.950), ATR_CCS_R (0.945) and ATR_CCS_R_OC (0.925)',        'realized capture rates of operating CCS plants vs design'),
    ('scope2_convention',   'market',   'market',  'location', 'choice',    'here-and-now', '-',              'market-based: PPA power carries zero scope 2; location-based: PPA power carries the grid emission factor',   'GHG Protocol Scope 2 guidance'),
    ('merchant_h2_ef',      10.0,       9.0,       12.0,       'epistemic', 'here-and-now', 'kg CO2e/kg H2',  'scope-3 intensity of merchant (grey) hydrogen bought to cover demand (v2 case; unused with the D1-D4 tiers)',  'SMR without capture incl. upstream gas, GWP100 vs GWP20 / leakage'),
], columns=['handle', 'nominal', 'low', 'high', 'type', 'stage', 'unit', 'description', 'source']).set_index('handle')


def handle_values(overrides=None) -> dict:
    """Nominal handle values updated with `overrides` (dict handle -> value). Unknown handles raise."""
    p = PARAMS['nominal'].to_dict()
    if overrides:
        unknown = set(overrides) - set(p)
        if unknown:
            raise KeyError(f'unknown uncertainty handle(s): {sorted(unknown)}; known: {list(p)}')
        p.update(overrides)
    return p


def make_inputs(data: CaseData, overrides=None, case=None, **case_overrides) -> dict:
    """Effective model inputs for a set of handle values and a case configuration (see `case_config`).

    Returns dict(p, case, resources, processes, price_profile, gwp_profile, capacity_factor, h2_demand [kg/day],
    h2_demand_hourly [kg/hr], scope, storage_capex, storage_cost, rating_links, slack_procs, intensity_basis, ...).
    """
    p = handle_values(overrides)
    cfg = case_config(case, **case_overrides)
    comp = build_components(cfg)

    col = str(p['grid_ef_profile'])
    if col not in GRID_EF_PROFILES:
        raise ValueError(f'grid_ef_profile must be one of {GRID_EF_PROFILES}, got {col!r}')
    grid_gwp = float(p['grid_ef_level']) * np.nan_to_num(data.grid_ef[col])                     # kg CO2e/MWh
    grid_price = float(p['elec_price_level']) * data.elec_price                                  # USD/MWh
    ng_price = float(p['gas_price_level']) * (data.ng_price_max / 20) * data.price_factor_ng     # USD/kg
    cf_pv = np.minimum(1.0, float(p['renewable_level']) * data.capacity_factor_pv)
    cf_wf = np.minimum(1.0, float(p['renewable_level']) * data.capacity_factor_wf)

    res = comp['resources']
    res[NG]['gwp'] = float(p['ng_upstream_ef'])
    if H2_merchant in res:
        res[H2_merchant]['gwp'] = float(p['merchant_h2_ef'])
    price_profile = {GridPower: grid_price, NG: ng_price}
    gwp_profile = {GridPower: grid_gwp}
    if p['scope2_convention'] == 'location':
        gwp_profile[Solar] = grid_gwp
        gwp_profile[Wind] = grid_gwp
    elif p['scope2_convention'] != 'market':
        raise ValueError("scope2_convention must be 'market' or 'location'")

    pr = comp['processes']
    y = float(p['reformer_co2_yield'])
    rel = y / REFORMER_CO2_YIELD_NOMINAL
    pr['SMR']['conversion'][CO2_Stack] = y
    for name, d in CCS_NOMINAL.items():
        total = (d['cpt'] + d['vent']) * rel
        rate = min(1.0, float(p['capture_rate_factor']) * d['cpt'] / (d['cpt'] + d['vent']))
        pr[name]['conversion'][CO2cpt] = rate * total
        pr[name]['conversion'][CO2_Vent] = (1 - rate) * total
        if d['consumes_stack']:
            pr[name]['conversion'][CO2_Stack] = -y
    capacity_factor = {'PV': cf_pv, 'WF': cf_wf}

    grid_ef_all = {c: float(p['grid_ef_level']) * np.nan_to_num(data.grid_ef[c]) for c in GRID_EF_PROFILES}
    # hourly demand: each day's total spread over its hours with `demand_shape` (flat when None)
    shape = np.ones(N_HOURS) if cfg['demand_shape'] is None else np.asarray(cfg['demand_shape'], dtype=float)
    if shape.shape != (N_HOURS,) or (shape < 0).any() or shape.sum() <= 0:
        raise ValueError('demand_shape must be 24 non-negative values')
    daily = np.asarray(data.h2_demand, dtype=float)
    if cfg['demand_interp']:   # linear interpolation between day centres, rescaled so that every day keeps its total
        hours = np.arange(N_DAYS * N_HOURS) + 0.5
        base = np.interp(hours, N_HOURS * (np.arange(N_DAYS) + 0.5), daily).reshape(N_DAYS, N_HOURS)
    else:
        base = np.repeat(daily[:, None], N_HOURS, axis=1)
    base = base * (shape / shape.sum())[None, :] * N_HOURS
    h2_demand_hourly = (base * (daily / base.sum(axis=1))[:, None]).ravel()
    return dict(p=p, case=cfg, resources=res, processes=pr, price_profile=price_profile, gwp_profile=gwp_profile,
                capacity_factor=capacity_factor, h2_demand=data.h2_demand, h2_demand_hourly=h2_demand_hourly,
                scope={r: res[r]['scope'] for r in res}, storage_capex=comp['storage_capex'], storage_cost=comp['storage_cost'],
                rating_links=comp['rating_links'], slack_procs=comp['slack_procs'], intensity_basis=comp['intensity_basis'],
                grid_price=grid_price, ng_price=ng_price, grid_gwp=grid_gwp, grid_ef_all=grid_ef_all, _data=data)


# ----------------------------------------------------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------------------------------------------------
_STATUS = {GRB.OPTIMAL: 'optimal', GRB.INFEASIBLE: 'infeasible', GRB.INF_OR_UNBD: 'infeasibleOrUnbounded',
           GRB.UNBOUNDED: 'unbounded', GRB.TIME_LIMIT: 'timeLimit', GRB.INTERRUPTED: 'interrupted',
           GRB.SUBOPTIMAL: 'suboptimal'}


class H2System:
    """Single-location design + hourly scheduling MILP of the hydrogen system (gurobipy matrix API).

    Variables (arrays):  P[p, t] production, C[r, t] consumption, S[r, t] discharge, Inv[r, t] inventory,
                         Cap_P[p], Cap_S[r], X_P[p], X_S[r] (binaries).
    Expressions:         cost pieces, gwp_resource[r], gwp_scope[s], gwp_total.
    """

    def __init__(self, inputs, objective='cost', demand_sign='eq', ramp=True, smoothing=True,
                 ramp_rates=RAMP, min_loads=MINLOAD, c2=C2, annualization_factor=1.0, env=None, name='Texas_TVEF_Battery',
                 n_days=N_DAYS):
        t0 = time.time()
        self.inputs = inputs
        self.n_days = int(n_days)
        self.nT = nT = self.n_days * N_HOURS          # scheduling horizon (default: the full year)
        res, pr = inputs['resources'], inputs['processes']
        self.PROCS = sorted(pr)
        self.RES = sorted({r for p in pr for r in pr[p]['conversion']})
        self.ip = {p: i for i, p in enumerate(self.PROCS)}
        self.ir = {r: i for i, r in enumerate(self.RES)}
        nP, nR, ip, ir = len(self.PROCS), len(self.RES), self.ip, self.ir
        self.R_PURCH = [r for r in self.RES if res[r]['cons_max'] > 0]
        self.R_SELL = [r for r in self.RES if res[r]['sell']]
        self.R_STORE = [r for r in self.RES if res[r]['store_max'] > 0]
        self.R_DEMAND = [r for r in self.RES if res[r]['demand']]
        self.scope = {r: res[r]['scope'] for r in self.RES}
        self.case = inputs.get('case', case_config('legacy'))
        self.demand_resolution = self.case['demand']
        self.intensity_basis = inputs.get('intensity_basis', 'own')

        # conversion matrix A[r, p]
        A = np.zeros((nR, nP))
        for p in self.PROCS:
            for r, c in pr[p]['conversion'].items():
                A[ir[r], ip[p]] = c
        self.A = A
        A = _sp.csr_matrix(A)                  # sparse: avoids zero-coefficient terms in the balance rows

        m = gp.Model(name, env=env)
        self.m = m

        # ---------------- variables ----------------
        C_ub = np.zeros((nR, nT))
        for r in self.R_PURCH:   # energiapy bounds consumption only for resources without an explicit `varying` list
            C_ub[ir[r], :] = _inf(res[r]['cons_max']) if res[r]['varying'] is None else GRB.INFINITY
        S_ub = np.zeros((nR, nT)); S_ub[[ir[r] for r in self.R_SELL], :] = GRB.INFINITY
        Inv_ub = np.zeros((nR, nT)); Inv_ub[[ir[r] for r in self.R_STORE], :] = GRB.INFINITY
        store_max = np.array([res[r]['store_max'] for r in self.RES])
        store_min = np.array([res[r]['store_min'] for r in self.RES])
        prod_max = np.array([pr[p]['prod_max'] for p in self.PROCS])
        prod_min = np.array([pr[p]['prod_min'] for p in self.PROCS])
        prod_max_ub = np.array([_inf(v) for v in prod_max])              # big-M capacities are simply unbounded
        kfin = np.flatnonzero(prod_max < bigM)                           # facility rows only where the bound is real

        self.P = m.addMVar((nP, nT), lb=0.0, name='P')
        self.C = m.addMVar((nR, nT), lb=0.0, ub=C_ub, name='C')
        self.S = m.addMVar((nR, nT), lb=0.0, ub=S_ub, name='S')
        self.Inv = m.addMVar((nR, nT), lb=0.0, ub=Inv_ub, name='Inv')
        self.Cap_P = m.addMVar(nP, lb=0.0, ub=prod_max_ub, name='Cap_P')
        self.Cap_S = m.addMVar(nR, lb=0.0, ub=store_max, name='Cap_S')
        self.X_P = m.addMVar(nP, vtype=GRB.BINARY, name='X_P')
        self.X_S = m.addMVar(nR, vtype=GRB.BINARY, ub=(store_max > 0).astype(float), name='X_S')
        P, C, S, Inv, Cap_P, Cap_S, X_P, X_S = self.P, self.C, self.S, self.Inv, self.Cap_P, self.Cap_S, self.X_P, self.X_S

        # ---------------- design (facility sizing with network binaries) ----------------
        # processes declared with a big-M capacity (Grid, D1-D4) are unbounded and always available (X_P = 1)
        m.addConstr(Cap_P[kfin] <= prod_max[kfin] * X_P[kfin], name='facility_max')
        kbig = np.flatnonzero(prod_max >= bigM)
        if kbig.size:
            X_P[kbig].lb = 1.0
        kst = np.flatnonzero(store_max > 0)                                       # storage rows only (Cap_S is indexed by all resources)
        m.addConstr(Cap_S[kst] <= store_max[kst] * X_S[kst], name='storage_facility_max')
        kp, ks = np.flatnonzero((prod_min > 0) & (prod_min < bigM)), np.flatnonzero(store_min > 0)   # real minimum-size rows only
        if kp.size:
            m.addConstr(Cap_P[kp] >= prod_min[kp] * X_P[kp], name='facility_min')
        if ks.size:
            m.addConstr(Cap_S[ks] >= store_min[ks] * X_S[ks], name='storage_facility_min')
        for a, b in inputs.get('rating_links', []):                                # one rating for both directions
            m.addConstr(Cap_P[ip[a]] == Cap_P[ip[b]], name=f'rating_link[{a},{b}]')

        # ---------------- nameplate production and inventory ----------------
        for p in self.PROCS:
            cf = pr[p]['capacity_factor']
            if cf is None:
                m.addConstr(P[ip[p], :] <= Cap_P[ip[p]], name=f'nameplate[{p}]')
            else:   # hours with zero capacity factor are fixed to zero (avoids zero-coefficient rows)
                f = np.asarray(inputs['capacity_factor'][cf][:nT], dtype=float)
                on = np.flatnonzero(f > 1e-12)
                P[ip[p], np.flatnonzero(f <= 1e-12)].ub = 0.0
                if on.size:
                    m.addConstr(P[ip[p], on] <= f[on] * Cap_P[ip[p]], name=f'nameplate[{p}]')
        for r in self.R_STORE:
            m.addConstr(Inv[ir[r], :] <= Cap_S[ir[r]], name=f'nameplate_inventory[{r}]')

        # ---------------- resource balance: C + A P - S == Inv_t - Inv_{t-1}  (Inv_{-1} = 0) ----------------
        m.addConstr(C[:, 0] + A @ P[:, 0] - S[:, 0] == Inv[:, 0], name='balance_0')
        m.addConstr(C[:, 1:] + A @ P[:, 1:] - S[:, 1:] == Inv[:, 1:] - Inv[:, :-1], name='balance')

        # ---------------- demand: daily totals (legacy) or hourly targets ----------------
        if self.demand_resolution == 'hourly':
            daily = sum((S[ir[r], :] for r in self.R_DEMAND[1:]), S[ir[self.R_DEMAND[0]], :])
            target = np.asarray(inputs['h2_demand_hourly'], dtype=float)[:nT]
        else:
            daily = sum(S[ir[r], :].reshape((self.n_days, N_HOURS)).sum(axis=1) for r in self.R_DEMAND)
            target = np.asarray(inputs['h2_demand'], dtype=float)[:self.n_days]
        self.n_demand = len(target)
        if demand_sign == 'eq':
            self.demand_constr = m.addConstr(daily == target, name='demand')
        elif demand_sign == 'geq':
            self.demand_constr = m.addConstr(daily >= target, name='demand')
        elif demand_sign == 'leq':
            self.demand_constr = m.addConstr(daily <= target, name='demand')
        else:
            raise ValueError("demand_sign must be 'eq', 'geq' or 'leq'")

        # ---------------- operational constraints of the reforming routes ----------------
        if ramp:
            for p, rate in ramp_rates.items():
                i = ip[p]
                m.addConstr(P[i, 1:] - P[i, :-1] <= rate * Cap_P[i], name=f'ramp_up[{p}]')
                m.addConstr(P[i, 1:] - P[i, :-1] >= -rate * Cap_P[i], name=f'ramp_down[{p}]')
            for p, frac in min_loads.items():
                m.addConstr(P[ip[p], :] >= frac * Cap_P[ip[p]], name=f'min_load[{p}]')
        if smoothing:
            for p, c in c2.items():
                i = ip[p]
                accel = P[i, 2:] - 2 * P[i, 1:-1] + P[i, :-2]
                m.addConstr(accel <= c * Cap_P[i], name=f'smooth_up[{p}]')
                m.addConstr(accel >= -c * Cap_P[i], name=f'smooth_down[{p}]')

        # ---------------- cost expressions ----------------
        self.horizon_factor = nT / N_T if self.case.get('prorate_capex', False) else 1.0   # fixed charges per horizon
        capex = np.array([pr[p]['capex'] for p in self.PROCS]) * annualization_factor * self.horizon_factor
        fopex = np.array([pr[p]['fopex'] for p in self.PROCS]) * self.horizon_factor
        vopex = np.array([pr[p]['vopex'] for p in self.PROCS])
        s_capex = np.array([inputs.get('storage_capex', STORAGE_CAPEX).get(r, 0.0) for r in self.RES]) * annualization_factor * self.horizon_factor
        s_cost = np.array([inputs.get('storage_cost', STORAGE_COST).get(r, 0.0) for r in self.RES])
        price = np.zeros((nR, nT))
        for r in self.R_PURCH:
            price[ir[r], :] = inputs['price_profile'][r][:nT] if r in inputs['price_profile'] else res[r]['price']
        self.price = price
        self.Capex = capex @ Cap_P + s_capex @ Cap_S
        self.Fopex = fopex @ Cap_P
        self.Vopex = (vopex[:, None] * P).sum()
        self.Purchase = (price * C).sum()
        self.Storage_cost = (s_cost[:, None] * Inv).sum() if s_cost.any() else 0.0
        self.Cost = self.Capex + self.Fopex + self.Vopex + self.Purchase + self.Storage_cost

        # ---------------- hydrogen accounting (for intensity caps) ----------------
        # own hydrogen = hydrogen delivered against the demand minus the hydrogen not produced on site
        # (legacy: penalty tiers D1-D4; v2: merchant import H2_import)
        self.slack_procs = [p for p in self.PROCS if p in inputs.get('slack_procs', PENALTY_PROCS)]
        pen = self.slack_procs
        self.H2_delivered = sum((S[ir[r], :].sum() for r in self.R_DEMAND[1:]), S[ir[self.R_DEMAND[0]], :].sum())
        self.H2_penalty = sum((P[ip[p], :].sum() for p in pen[1:]), P[ip[pen[0]], :].sum()) if pen else 0.0
        self.H2_own = self.H2_delivered - self.H2_penalty

        # ---------------- emission expressions ----------------
        gwpC = np.zeros((nR, nT))     # kg CO2e per unit consumed, hourly
        gwpS = np.zeros(nR)            # kg CO2e per unit discharged
        for r in self.R_PURCH:
            if r in inputs['gwp_profile']:
                gwpC[ir[r], :] = inputs['gwp_profile'][r][:nT]
            elif res[r]['gwp']:
                gwpC[ir[r], :] = res[r]['gwp']
        for r in self.R_SELL:
            if res[r]['gwp']:
                gwpS[ir[r]] = res[r]['gwp']
        self.gwpC, self.gwpS = gwpC, gwpS
        self.gwp_resource = {}
        for r in self.RES:
            i = ir[r]
            if gwpC[i].any() or gwpS[i]:
                if self.scope[r] not in SCOPES:
                    raise ValueError(f'emitting resource {r!r} has no scope')
                self.gwp_resource[r] = (gwpC[i] * C[i, :]).sum() + gwpS[i] * S[i, :].sum()
        def _sum(exprs):
            exprs = list(exprs)
            return sum(exprs[1:], exprs[0]) if exprs else gp.LinExpr()
        self.gwp_scope = {s: _sum(e for r, e in self.gwp_resource.items() if self.scope[r] == s) for s in SCOPES}
        self.gwp_total = _sum(self.gwp_scope[s] for s in SCOPES)
        self.gwp_caps = {}

        self.set_objective(objective)
        m.update()
        self.status = None
        self.build_time = time.time() - t0

    # ------------------------------------------------------------------------------------------------------------------
    def set_objective(self, objective='cost'):
        """'cost', 'gwp' (total), or 'scope1' / 'scope2' / 'scope3'."""
        if objective == 'cost':
            expr = self.Cost
        elif objective == 'gwp':
            expr = self.gwp_total
        elif objective in ('scope1', 'scope2', 'scope3'):
            expr = self.gwp_scope[int(objective[-1])]
        else:
            raise ValueError(f'unknown objective {objective!r}')
        self.objective_name = objective
        self.m.setObjective(expr, GRB.MINIMIZE)
        return self

    def _h2_basis(self, basis=None):
        """Hydrogen expression used as the denominator of an intensity cap: 'own' (delivered minus slack supply) or
        'delivered' (everything delivered, used when the slack supply carries its own emissions). None = case default."""
        basis = basis or self.intensity_basis
        if basis == 'own':
            return self.H2_own
        if basis == 'delivered':
            return self.H2_delivered
        raise ValueError("intensity_basis must be 'own' or 'delivered'")

    def add_4cp_charge(self, rate=ERCOT_4CP_RATE, mode='known', basis='net', intervals=None, window=None, weights=None,
                       name='4cp'):
        """ERCOT four-coincident-peak transmission charge: rate [USD per kW-year] x average metered demand [kW] over the
        four monthly coincident-peak intervals (June-September). Added to the cost (and to the cost objective).

        mode  'known'    the intervals are known (`intervals`, default ERCOT_4CP_2025: {month: (day, hour ending)}):
                         D_m = load[t_m], the ex-post settlement with perfect foresight of the peak timing;
              'window'   the peak timing is uncertain within a candidate window per month (`window`, default
                         ERCOT_4CP_WINDOW: hour endings 16-19 of every day): D_m >= load[t] for every window hour, the
                         robust (worst-case timing) charge, which is what a load pays if it cannot predict the peak and
                         holds its demand down throughout the window;
              'expected' D_m = sum_t w_t load[t] with probability weights over the window hours (`weights`: {t: w} per
                         month, or None for uniform weights over the window), the expected charge for a load that does
                         not respond to peak forecasts.
        basis 'net'      metered demand = grid import (PPA solar and wind behind the meter reduce it);
              'gross'    metered demand = grid import + PPA generation (virtual / off-site PPAs: the whole plant draw is
                         metered, only on-site storage or load reduction can lower it).
        Months whose intervals / window lie outside the scheduling horizon are ignored (average over the months kept).
        Returns the charge expression; stores D_m in self.cp_demand[name]."""
        m, ip, P = self.m, self.ip, self.P
        load = P[ip['Grid'], :]
        if basis == 'gross':                                   # PPA power output = conversion coefficient x process level
            pr = self.inputs['processes']
            load = load + sum(pr[q]['conversion'][Power] * P[ip[q], :] for q in ('PV', 'WF') if q in ip)
        elif basis != 'net':
            raise ValueError(f'basis must be net or gross, not {basis!r}')
        intervals = dict(ERCOT_4CP_2025 if intervals is None else intervals)
        window = dict(ERCOT_4CP_WINDOW, **(window or {}))
        months = sorted(intervals) if mode == 'known' else list(window['months'])
        D = {}
        if mode == 'known':
            for mo in months:
                t = _hour_index(mo, *intervals[mo])
                if t < self.nT:
                    D[mo] = 1.0 * load[t]                          # expression (0-d MLinExpr), so .getValue() works
        elif mode in ('window', 'expected'):
            for mo in months:
                hours = [t for t in _month_hours(mo, window['hours']) if t < self.nT]
                if not hours:
                    continue
                if mode == 'window':
                    d = m.addVar(lb=0.0, name=f'{name}_demand[{mo}]')
                    m.addConstrs((d >= load[t] for t in hours), name=f'{name}_peak[{mo}]')
                    D[mo] = gp.LinExpr(d)
                else:
                    w = weights[mo] if weights else {t: 1.0 / len(hours) for t in hours}
                    D[mo] = sum(w[t] * load[t] for t in w)
        else:
            raise ValueError(f'mode must be known, window or expected, not {mode!r}')
        if not D:
            raise ValueError('no 4CP month lies inside the scheduling horizon')
        self.cp_demand = getattr(self, 'cp_demand', {})
        self.cp_demand[name] = D
        charge = rate * 1000.0 * sum(D.values()) / len(D) * (self.horizon_factor if self.case.get('prorate_capex') else 1.0)
        self.demand_charges = getattr(self, 'demand_charges', {})
        self.demand_charges[name] = dict(expr=charge, mode=mode, basis=basis, rate=rate, months=sorted(D))
        self.Cost = self.Cost + charge
        if self.objective_name == 'cost':
            self.set_objective('cost')
        m.update()
        return charge

    def add_gwp_cap(self, cap=None, scope=None, name=None, intensity=None, intensity_basis=None):
        """Emission cap on the total GWP or on one scope.

        cap       : absolute cap [kg CO2e per year];
        intensity : instead, an intensity cap [kg CO2e per kg H2]: GWP <= intensity * hydrogen, where the hydrogen is
                    the own hydrogen (delivered minus the emission-free penalty tiers D1-D4; legacy case) or all the
                    delivered hydrogen (v2 case, where merchant hydrogen carries its own scope-3 emissions);
                    `intensity_basis` ('own' / 'delivered') overrides the case default.
        Returns the constraint (use `.Pi` on a fixed/continuous model for duals).
        """
        if (cap is None) == (intensity is None):
            raise ValueError('give exactly one of cap or intensity')
        expr = self.gwp_total if scope is None else self.gwp_scope[scope]
        rhs = cap if intensity is None else intensity * self._h2_basis(intensity_basis)
        name = name or (('gwp_cap' if scope is None else f'gwp_cap_scope{scope}') + ('' if intensity is None else '_intensity'))
        c = self.m.addConstr(expr <= rhs, name=name)
        self.gwp_caps[name] = c
        return c

    def add_gwp_reduction(self, base, pct, scope=None):
        """energiapy-style reduction target: GWP <= base * (1 - pct/100)."""
        return self.add_gwp_cap(base * (1 - pct / 100), scope=scope)

    # ---------------- static budgeted robust counterpart ----------------
    ROBUST_ROUTES = ('SMR', 'SMRCCS', 'ATR_CCS_R', 'ATR_CCS_R_OC')

    def robust_deviations(self, band='0.75'):
        """Upper deviations of the emission coefficients used by `add_robust_gwp_cap`, derived from `PARAMS`.

        scope 1: per reforming route, net CO2 emitted per kg H2 (stack + vent coefficients) at the worst corner of the
                 `reformer_co2_yield` (high) and `capture_rate_factor` (low) ranges, minus the nominal value;
        scope 2: hourly grid factor of the `band` quantile profile minus the nominal profile (same level multiplier);
        scope 3: high value of `ng_upstream_ef` minus the nominal value ('scope3'); 'scope3_resources' holds the
                 deviation of every purchased scope-3 resource (natural gas; merchant hydrogen via `merchant_h2_ef`).
        """
        inp, p, data = self.inputs, self.inputs['p'], self.inputs['_data']
        corner = dict(p)
        corner.update(reformer_co2_yield=PARAMS.loc['reformer_co2_yield', 'high'], capture_rate_factor=PARAMS.loc['capture_rate_factor', 'low'])
        worst = make_inputs(data, corner, case=self.case)

        def e1(i):
            return {q: sum(i['processes'][q]['conversion'].get(r, 0.0) for r in (CO2_Stack, CO2_Vent)) for q in self.ROBUST_ROUTES}
        e_nom, e_worst = e1(inp), e1(worst)
        dev1 = {q: max(0.0, e_worst[q] - e_nom[q]) for q in self.ROBUST_ROUTES}
        band_profile = inp['grid_ef_all'][str(band)][:self.nT]
        dev2 = np.maximum(0.0, band_profile - inp['grid_gwp'][:self.nT])
        dev3 = max(0.0, float(PARAMS.loc['ng_upstream_ef', 'high']) - float(p['ng_upstream_ef']))
        dev3_res = {NG: dev3}
        if H2_merchant in inp['resources']:
            dev3_res[H2_merchant] = max(0.0, float(PARAMS.loc['merchant_h2_ef', 'high']) - float(p['merchant_h2_ef']))
        return dict(scope1=dev1, scope2=dev2, scope3=dev3, scope3_resources=dev3_res, e1_nominal=e_nom, e1_worst=e_worst,
                    band=str(band))

    def add_robust_gwp_cap(self, cap=None, gamma=None, band='0.75', name='robust_gwp_cap', intensity=None,
                           intensity_basis=None):
        """Static budgeted robust counterpart of the total-GWP cap with one uncertainty budget per scope.

        The emission coefficients may deviate upwards from their nominal values by at most the deviations of
        `robust_deviations`; the budget limits how many may deviate at once (Bertsimas & Sim):
          gamma[1] in [0, 4]        number of reforming routes whose net CO2 coefficient is at its worst value,
          gamma[2] in [0, n_days]   number of days whose hourly grid factors are at the `band` quantile,
          gamma[3] in [0, n3]       budget over the n3 purchased scope-3 resources (natural gas; plus merchant hydrogen
                                    in the v2 case): 1 = one factor fully at its worst value.
        gamma = 0 for every scope is the nominal cap; the maxima reproduce the deterministic model built at the worst
        corner (box). The counterpart adds one z per scope and one q per route / day / parameter and stays linear:
            GWP_nominal + sum_s ( gamma_s z_s + sum_k q_{s,k} ) <= cap,   z_s + q_{s,k} >= dev_{s,k} * x_{s,k}.
        With `intensity` instead of `cap` the right-hand side is intensity * own hydrogen (see `add_gwp_cap`).
        """
        if (cap is None) == (intensity is None):
            raise ValueError('give exactly one of cap or intensity')
        gamma = gamma or {}
        m, ip, ir = self.m, self.ip, self.ir
        dev = self.robust_deviations(band)
        g = {1: 0.0, 2: 0.0, 3: 0.0}
        g.update({int(k): float(v) for k, v in gamma.items()})
        rob = dict(gamma=g, dev=dev, margin={})
        # scope 1: reforming routes
        routes = [q for q in self.ROBUST_ROUTES if dev['scope1'][q] > 0]
        if g[1] > 0 and routes:
            z1 = m.addVar(lb=0.0, name='rob_z1')
            q1 = m.addMVar(len(routes), lb=0.0, name='rob_q1')
            for k, q in enumerate(routes):
                m.addConstr(q1[k] + z1 >= dev['scope1'][q] * self.P[ip[q], :].sum(), name=f'rob1[{q}]')
            rob['margin'][1] = g[1] * z1 + q1.sum()
        # scope 2: daily blocks of the hourly grid factor, for every resource carrying the grid factor
        rows = [ir[r] for r in self.R_PURCH if r in self.inputs['gwp_profile']]
        if g[2] > 0 and dev['scope2'].any() and rows:
            nd = self.n_days
            D = dev['scope2'].reshape(nd, N_HOURS)
            z2 = m.addVar(lb=0.0, name='rob_z2')
            q2 = m.addMVar(nd, lb=0.0, name='rob_q2')
            for d in range(nd):
                sl = slice(d * N_HOURS, (d + 1) * N_HOURS)
                expr = (D[d] * self.C[rows[0], sl]).sum()
                for i in rows[1:]:
                    expr = expr + (D[d] * self.C[i, sl]).sum()
                m.addConstr(q2[d] + z2 >= expr, name=f'rob2[{d}]')
            rob['margin'][2] = g[2] * z2 + q2.sum()
        # scope 3: upstream factors of the purchased scope-3 resources (natural gas; merchant hydrogen in the v2 case),
        # one parameter each, gamma[3] in [0, number of such resources]
        res3 = [(r, d) for r, d in dev['scope3_resources'].items() if d > 0 and r in ir]
        if g[3] > 0 and res3:
            z3 = m.addVar(lb=0.0, name='rob_z3')
            q3 = m.addMVar(len(res3), lb=0.0, name='rob_q3')
            for k, (r, d) in enumerate(res3):
                m.addConstr(q3[k] + z3 >= d * self.C[ir[r], :].sum(), name=f'rob3[{r}]')
            rob['margin'][3] = g[3] * z3 + q3.sum()
        margin = 0.0
        for e in rob['margin'].values():
            margin = margin + e
        rob['cap'], rob['intensity'] = cap, intensity
        self.robust = rob
        rhs = cap if intensity is None else intensity * self._h2_basis(intensity_basis)
        c = m.addConstr(self.gwp_total + margin <= rhs, name=name)
        self.gwp_caps[name] = c
        return c

    def robust_values(self):
        """Nominal GWP, worst-case GWP under the budgets and the per-scope robust margins of the solved model."""
        v = lambda e: float(e.getValue()) if hasattr(e, 'getValue') else float(e)
        rob = self.robust
        margins = {s: v(e) for s, e in rob['margin'].items()}
        nominal = v(self.gwp_total)
        own = v(self._h2_basis())
        return dict(nominal_gwp=nominal, worst_case_gwp=nominal + sum(margins.values()), margin=margins,
                    gamma=rob['gamma'], cap=rob['cap'], intensity_cap=rob.get('intensity'), h2_own=own,
                    nominal_intensity=nominal / own if own > 0 else np.nan,
                    worst_case_intensity=(nominal + sum(margins.values())) / own if own > 0 else np.nan)

    def solve(self, tee=True, **params):
        self.m.Params.OutputFlag = 1 if tee else 0
        for k, v in params.items():
            setattr(self.m.Params, k, v)
        t0 = time.time()
        self.m.optimize()
        self.solve_time = time.time() - t0
        self.status = _STATUS.get(self.m.Status, str(self.m.Status))
        if tee:
            print(f'termination: {self.status}  (build {self.build_time:.1f} s, solve {self.solve_time:.1f} s)')
        return self.status

    def shadow_prices(self, tee=False):
        """Duals of the GWP caps and the daily demand constraints with the binaries fixed at their solution values.

        Returns dict(name -> dual) for each cap and 'demand' -> array over days (USD per kg CO2e / per kg H2 if the
        objective is cost).
        """
        if self.status != 'optimal':
            raise RuntimeError('solve the model first')
        fixed = self.m.fixed()
        fixed.Params.OutputFlag = 1 if tee else 0
        fixed.optimize()
        out = {}
        for name in self.gwp_caps:
            out[name] = fixed.getConstrByName(name).Pi
        out['demand'] = np.array([fixed.getConstrByName(f'demand[{k}]').Pi for k in range(self.n_demand)])   # per day (legacy) or per hour
        return out

    # ------------------------------------------------------------------------------------------------------------------
    def frames(self):
        """Solution as DataFrames (rows = component, columns = hour)."""
        if self.status != 'optimal':
            raise RuntimeError(f'no solution (status {self.status})')
        return dict(
            P=pd.DataFrame(self.P.X, index=self.PROCS),
            C=pd.DataFrame(self.C.X, index=self.RES).loc[self.R_PURCH],
            S=pd.DataFrame(self.S.X, index=self.RES).loc[self.R_SELL],
            Inv=pd.DataFrame(self.Inv.X, index=self.RES).loc[self.R_STORE],
            Cap_P=pd.Series(self.Cap_P.X, index=self.PROCS),
            Cap_S=pd.Series(self.Cap_S.X, index=self.RES).loc[self.R_STORE],
        )

    def results(self):
        """energiapy-style result dictionary (same keys / index tuples as the Pyomo notebook) for post-processing."""
        out = {'termination': self.status, 'params': dict(self.inputs['p'])}
        if self.status != 'optimal':
            out['objective'] = None
            return out
        L, n = LOCATION, NETWORK_PERIOD
        key = lambda name, t: (L, name, n, t // N_HOURS, t % N_HOURS)
        ir, ip = self.ir, self.ip
        out['objective'] = self.m.ObjVal
        out['Cap_P'] = {(L, p, n): float(self.Cap_P.X[ip[p]]) for p in self.PROCS}
        out['Cap_S'] = {(L, r, n): float(self.Cap_S.X[ir[r]]) for r in self.R_STORE}
        out['X_P'] = {(L, p, n): int(round(self.X_P.X[ip[p]])) for p in self.PROCS}
        out['X_S'] = {(L, r, n): int(round(self.X_S.X[ir[r]])) for r in self.R_STORE}
        PX, CX, SX, IX = self.P.X, self.C.X, self.S.X, self.Inv.X
        out['P'] = {key(p, t): float(PX[ip[p], t]) for p in self.PROCS for t in range(self.nT)}
        out['C'] = {key(r, t): float(CX[ir[r], t]) for r in self.R_PURCH for t in range(self.nT)}
        out['S'] = {key(r, t): float(SX[ir[r], t]) for r in self.R_SELL for t in range(self.nT)}
        out['Inv'] = {key(r, t): float(IX[ir[r], t]) for r in self.R_STORE for t in range(self.nT)}
        out['P_location'] = {(L, p, n): float(PX[ip[p]].sum()) for p in self.PROCS}
        val = lambda e: float(e.getValue()) if hasattr(e, 'getValue') else float(e)
        out['Capex_network'] = {n: val(self.Capex)}
        out['Fopex_network'] = {n: val(self.Fopex)}
        out['Vopex_network'] = {n: val(self.Vopex)}
        out['B_network'] = {n: val(self.Purchase)}
        out['Cost'] = val(self.Cost)
        out['global_warming_potential_resource'] = {(L, r, n): (val(self.gwp_resource[r]) if r in self.gwp_resource else 0.0) for r in self.RES}
        out['global_warming_potential_scope'] = {(L, s, n): val(self.gwp_scope[s]) for s in SCOPES}
        out['global_warming_potential_location'] = {(L, n): val(self.gwp_total)}
        out['global_warming_potential_network'] = {n: val(self.gwp_total)}
        out['H2_delivered'] = val(self.H2_delivered)
        out['H2_penalty'] = val(self.H2_penalty)
        out['H2_own'] = val(self.H2_own)
        basis = val(self._h2_basis())
        out['gwp_intensity'] = val(self.gwp_total) / basis if basis > 0 else float('nan')
        for nm, dc in getattr(self, 'demand_charges', {}).items():
            out[f'demand_charge_{nm}'] = val(dc['expr'])
            out[f'cp_demand_{nm}'] = {mo: val(d) for mo, d in self.cp_demand[nm].items()}
        out['intensity_basis'] = self.intensity_basis
        out['case'] = dict(self.case)
        return out

    def summary(self):
        p = self.inputs['p']
        non_nominal = {k: v for k, v in p.items() if v != PARAMS.loc[k, 'nominal']}
        lines = [f'termination : {self.status}']
        if non_nominal:
            lines.append(f'handles     : {non_nominal}')
        if self.status == 'optimal':
            v = lambda e: float(e.getValue()) if hasattr(e, 'getValue') else float(e)
            lines.append(f'objective   : {self.m.ObjVal:,.2f}  ({self.objective_name})')
            lines.append(f'total cost  : {v(self.Cost):,.2f} USD  (capex {v(self.Capex):,.0f}, fopex {v(self.Fopex):,.0f}, '
                         f'vopex {v(self.Vopex):,.0f}, purchase {v(self.Purchase):,.0f})')
            lines.append(f'total GWP   : {v(self.gwp_total):,.2f} kg CO2e  (scope 1 {v(self.gwp_scope[1]):,.0f}, '
                         f'scope 2 {v(self.gwp_scope[2]):,.0f}, scope 3 {v(self.gwp_scope[3]):,.0f})')
            own, pen = v(self.H2_own), v(self.H2_penalty)
            basis = v(self._h2_basis())
            lines.append(f'hydrogen    : own {own:,.0f} kg, not produced on site ({", ".join(self.slack_procs) or "-"}) {pen:,.0f} kg'
                         f'  ->  intensity {(v(self.gwp_total) / basis if basis > 0 else float("nan")):.3f} kg CO2e/kg H2 '
                         f'({self.intensity_basis} basis, nominal coefficients)')
            for nm, dc in getattr(self, 'demand_charges', {}).items():
                cps = ', '.join(f'{mo}: {v(d):.2f}' for mo, d in self.cp_demand[nm].items())
                lines.append(f'4CP charge  : {v(dc["expr"]):,.0f} USD ({dc["mode"]} timing, {dc["basis"]} basis, '
                             f'{dc["rate"]} USD/kW-yr; coincident demand MW {{{cps}}})')
            caps = pd.Series(self.Cap_P.X, index=self.PROCS)
            lines.append('capacities  : ' + ', '.join(f'{p} {c:,.1f}' for p, c in caps.items() if c > 0))
            lines.append('storage     : ' + ', '.join(f'{r} {self.Cap_S.X[self.ir[r]]:,.2f}' for r in self.R_STORE))
        print('\n'.join(lines))


# ----------------------------------------------------------------------------------------------------------------------
# post-processing of an epsilon-constraint sweep (same sheets as the energiapy / Pyomo notebooks)
# ----------------------------------------------------------------------------------------------------------------------
def sweep_tables(loaded: dict, location=LOCATION, battery_keyword='LiI',
                 tracked_processes=('PV', 'WF', 'Grid', 'D1', 'D2', 'D3', 'D4', 'H2_import', 'H2_compress', 'H2_release')) -> dict:
    """Build the summary tables of an epsilon-constraint sweep.

    loaded: dict case_index -> result dictionary (as returned by `H2System.results`), case i = (i + 1) * 10 % reduction.
    Returns a dict of DataFrames (summary, battery capacity, storage profile, cycles, capacities, P profiles ...).
    """
    def optimal(o):
        return str(o.get('termination', '')).lower() == 'optimal'

    def pct(i):
        return (i + 1) * 10

    rows = []
    for i, o in loaded.items():
        gs = o.get('global_warming_potential_scope', {})
        rows.append({'gwp_reduction_pct': pct(i), 'termination': o.get('termination'),
                     'total_cost_USD': o.get('objective') if optimal(o) else None,
                     'total_emissions_kg_CO2eq': o.get('global_warming_potential_location', {}).get((location, 0)) if optimal(o) else None,
                     'scope1_kg_CO2eq': gs.get((location, 1, 0)), 'scope2_kg_CO2eq': gs.get((location, 2, 0)),
                     'scope3_kg_CO2eq': gs.get((location, 3, 0))})
    summary = pd.DataFrame(rows).set_index('gwp_reduction_pct')

    cap_rows, pcap_rows, inv_rows, p_rows = [], [], [], []
    for i, o in loaded.items():
        if not optimal(o):
            continue
        for (loc, r, n), v in o.get('Cap_S', {}).items():
            if loc == location and battery_keyword in str(r):
                cap_rows.append(dict(gwp_reduction_pct=pct(i), location=loc, stored_resource=r, network_period=n,
                                     battery_energy_capacity_MWh=v))
        for (loc, p, n), v in o.get('Cap_P', {}).items():
            if loc == location:
                pcap_rows.append(dict(gwp_reduction_pct=pct(i), location=loc, technology=p, network_period=n, production_capacity=v))
        for (loc, r, n, d, h), v in o.get('Inv', {}).items():
            if loc == location and battery_keyword in str(r):
                inv_rows.append(dict(gwp_reduction_pct=pct(i), location=loc, resource=r, scenario_idx=n, day=d, hour=h,
                                     timestep=d * N_HOURS + h, storage_level_MWh=v))
        for (loc, p, n, d, h), v in o.get('P', {}).items():
            if loc == location and str(p) in tracked_processes:
                p_rows.append(dict(gwp_reduction_pct=pct(i), location=loc, process=str(p), network_period=n, day=d, hour=h,
                                   timestep=d * N_HOURS + h, P_value=v))

    cap_df = pd.DataFrame(cap_rows)
    production_cap_df = pd.DataFrame(pcap_rows)
    storage_df = pd.DataFrame(inv_rows)
    production_df = pd.DataFrame(p_rows)

    tables = {'Cost_and_Emissions': summary, 'Battery_Capacity': cap_df, 'Production_Capacity_Raw': production_cap_df}
    if not production_cap_df.empty:
        piv = production_cap_df.pivot_table(index=['location', 'technology', 'network_period'], columns='gwp_reduction_pct',
                                            values='production_capacity', aggfunc='sum')
        piv.columns = [f'{int(c)}pct_reduction' for c in piv.columns]
        tables['Production_Capacity_Pivot'] = piv
        tables['Production_Capacity_Change'] = piv.diff(axis=1)
    if not storage_df.empty:
        storage_df = storage_df.sort_values(['gwp_reduction_pct', 'scenario_idx', 'timestep'])
        tables['Battery_Storage_Raw'] = storage_df
        piv = storage_df.pivot_table(index='timestep', columns='gwp_reduction_pct', values='storage_level_MWh', aggfunc='sum')
        piv.columns = [f'{int(c)}pct_reduction' for c in piv.columns]
        tables['Battery_Storage_Pivot'] = piv
        changes, cycles = [], []
        for g, grp in storage_df.groupby('gwp_reduction_pct'):
            grp = grp.sort_values('timestep').copy()
            grp['delta_storage_MWh'] = grp['storage_level_MWh'].diff().fillna(0.0)
            grp['charge_MWh'] = grp['delta_storage_MWh'].clip(lower=0)
            grp['discharge_MWh'] = -grp['delta_storage_MWh'].clip(upper=0)
            cap = cap_df.loc[cap_df['gwp_reduction_pct'] == g, 'battery_energy_capacity_MWh'].sum() if not cap_df.empty else 0.0
            ch, dis = grp['charge_MWh'].sum(), grp['discharge_MWh'].sum()
            cycles.append(dict(gwp_reduction_pct=g, battery_energy_capacity_MWh=cap, total_charged_MWh=ch, total_discharged_MWh=dis,
                               equivalent_cycles_charge_basis=ch / cap if cap > 0 else None,
                               equivalent_cycles_discharge_basis=dis / cap if cap > 0 else None,
                               equivalent_cycles_throughput_basis=(ch + dis) / (2 * cap) if cap > 0 else None))
            changes.append(grp)
        tables['Battery_Storage_Changes'] = pd.concat(changes, ignore_index=True)
        tables['Battery_Cycles'] = pd.DataFrame(cycles)
    if not production_df.empty:
        production_df = production_df.sort_values(['process', 'gwp_reduction_pct', 'network_period', 'timestep'])
        tables['P_Raw_PV_WF_Grid_D'] = production_df
        annual = (production_df.groupby(['gwp_reduction_pct', 'location', 'process', 'network_period'], as_index=False)['P_value']
                  .sum().rename(columns={'P_value': 'total_annual_P'}))
        tables['P_Annual_Summary'] = annual
        piv = annual.pivot_table(index=['location', 'process', 'network_period'], columns='gwp_reduction_pct', values='total_annual_P', aggfunc='sum')
        piv.columns = [f'{int(c)}pct_reduction' for c in piv.columns]
        tables['P_Annual_Pivot'] = piv
        piv = production_df.pivot_table(index=['process', 'timestep'], columns='gwp_reduction_pct', values='P_value', aggfunc='sum')
        piv.columns = [f'{int(c)}pct_reduction' for c in piv.columns]
        tables['P_Hourly_Pivot'] = piv
    return tables


def export_sweep(loaded: dict, out_path, **kwargs) -> pd.DataFrame:
    """Write the sweep tables to an Excel workbook (one sheet per table) and return the summary table."""
    tables = sweep_tables(loaded, **kwargs)
    index_sheets = {'Cost_and_Emissions', 'Production_Capacity_Pivot', 'Production_Capacity_Change',
                    'Battery_Storage_Pivot', 'P_Annual_Pivot', 'P_Hourly_Pivot'}
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for name, df in tables.items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=name, index=name in index_sheets)
    print(f'Saved results to: {out_path}')
    return tables['Cost_and_Emissions']


# ----------------------------------------------------------------------------------------------------------------------
# multistage scenario tree with non-anticipativity
# ----------------------------------------------------------------------------------------------------------------------
import scipy.sparse as _sp


class ScenarioNode:
    """A node of the scenario tree: a time slice [t0, t1) of hours with its own realization of the uncertain handles."""

    def __init__(self, id, stage, parent, label, overrides, cond_prob, t0, t1):
        self.id, self.stage, self.parent, self.label = id, stage, parent, label
        self.overrides = dict(overrides or {})      # handle overrides revealed at this node
        self.cond_prob = float(cond_prob)           # probability of this branch given the parent
        self.t0, self.t1 = int(t0), int(t1)
        self.children = []
        self.prob = None                            # unconditional probability of reaching the node (set by the tree)
        self.path_overrides = None                  # overrides merged along the path root -> node (set by the tree)

    @property
    def length(self):
        return self.t1 - self.t0

    @property
    def is_root(self):
        return self.parent is None

    def __repr__(self):
        return f'Node({self.id}, stage={self.stage}, label={self.label!r}, hours=[{self.t0},{self.t1}), p={self.prob:.3f})'


class ScenarioTree:
    """Scenario tree for the operational uncertainties.

    Stage 0 is the design root (no time slice). Every later stage covers a block of days; the realization of the
    uncertain handles for that block becomes known at the start of the block, so decisions inside the block may depend
    on it and on everything before it, but not on later blocks (non-anticipativity).

    Build with `ScenarioTree.product(stage_days, branches)` where `stage_days` is the list of block lengths in days and
    `branches` is a dict label -> (probability, overrides) applied at every stage (or a list with one dict per stage).
    Overrides use the handles of `PARAMS`; `make_inputs` is evaluated per node with the overrides merged along the path.
    """

    def __init__(self, root):
        self.root = root
        self.nodes = []
        queue = [root]
        while queue:                                  # breadth-first order: parents before children
            n = queue.pop(0)
            self.nodes.append(n)
            queue.extend(n.children)
        for n in self.nodes:
            if n.is_root:
                n.prob, n.path_overrides = 1.0, dict(n.overrides)
            else:
                n.prob = n.parent.prob * n.cond_prob
                n.path_overrides = {**n.parent.path_overrides, **n.overrides}
        self.leaves = [n for n in self.nodes if not n.children]
        self.n_days = max(n.t1 for n in self.nodes) // N_HOURS
        tot = sum(l.prob for l in self.leaves)
        if abs(tot - 1) > 1e-9:
            raise ValueError(f'leaf probabilities sum to {tot}, not 1')
        for n in self.nodes:
            if not n.is_root and (n.length % N_HOURS or n.length == 0):
                raise ValueError('node slices must be a positive whole number of days')

    @classmethod
    def product(cls, stage_days, branches, root_overrides=None):
        """Full product tree: every stage branches into the given labelled branches (independent across stages)."""
        if isinstance(branches, dict):
            branches = [branches] * len(stage_days)
        if len(branches) != len(stage_days):
            raise ValueError('one branch dict per stage is required')
        root = ScenarioNode(0, 0, None, 'root', root_overrides, 1.0, 0, 0)
        counter = [1]

        def grow(parent, k, t0):
            if k == len(stage_days):
                return
            t1 = t0 + stage_days[k] * N_HOURS
            for label, (prob, ov) in branches[k].items():
                node = ScenarioNode(counter[0], k + 1, parent, label, ov, prob, t0, t1)
                counter[0] += 1
                parent.children.append(node)
                grow(node, k + 1, t1)
        grow(root, 0, 0)
        return cls(root)

    @classmethod
    def chain(cls, stage_days, overrides_per_stage=None, root_overrides=None):
        """Single-scenario tree (no branching), e.g. for perfect-information or expected-value problems."""
        ov = overrides_per_stage or [{}] * len(stage_days)
        return cls.product(stage_days, [{f's{k + 1}': (1.0, ov[k])} for k in range(len(stage_days))], root_overrides)

    def path(self, node):
        out = []
        while node is not None:
            out.append(node)
            node = node.parent
        return out[::-1]

    def scenario_path(self, leaf):
        return self.path(leaf)[1:]                    # without the design root

    def unbundled(self):
        """Equivalent 'explicit' tree in which every scenario is its own chain of node copies.

        Returns (tree, copy_to_original): copy_to_original maps the id of a copied node to the original node id.
        Used to verify the node formulation against the explicit non-anticipativity-constraint formulation.
        """
        root = ScenarioNode(0, 0, None, 'root', self.root.overrides, 1.0, 0, 0)
        mapping = {0: 0}
        counter = [1]
        for leaf in self.leaves:
            parent = root
            for k, orig in enumerate(self.scenario_path(leaf)):
                cond = leaf.prob if k == 0 else 1.0
                node = ScenarioNode(counter[0], orig.stage, parent, orig.label, orig.overrides, cond, orig.t0, orig.t1)
                mapping[node.id] = orig.id
                counter[0] += 1
                parent.children.append(node)
                parent = node
        return ScenarioTree(root), mapping

    def table(self):
        rows = [dict(node=n.id, stage=n.stage, parent=(n.parent.id if n.parent else None), label=n.label,
                     day_start=n.t0 // N_HOURS, day_end=n.t1 // N_HOURS, cond_prob=n.cond_prob, prob=n.prob,
                     overrides=n.path_overrides) for n in self.nodes]
        return pd.DataFrame(rows).set_index('node')

    def __repr__(self):
        return f'ScenarioTree({len(self.nodes)} nodes, {len(self.leaves)} scenarios, {self.n_days} days)'


class H2SystemTree:
    """Multistage stochastic version of `H2System` on a `ScenarioTree`.

    Design variables (Cap_P, Cap_S, X_P, X_S) are shared by all nodes (first stage). Each tree node owns the
    operational variables of its time slice (P, C, S, Inv) built from the inputs of its own path of realizations, and is
    linked to its parent through the inventory, ramp and smoothing constraints at the boundary hour. Because a node's
    variables are common to every scenario passing through it, non-anticipativity holds by construction.

    `explicit_nac=True` builds the equivalent scenario-copy formulation with explicit non-anticipativity equalities
    (one chain per scenario, equal variables for nodes that are copies of the same tree node). Useful to verify the
    formulation; the node formulation is much smaller.

    objective: 'cost' (expected total cost), 'gwp' (expected total GWP) or 'scope1' / 'scope2' / 'scope3'.
    """

    def __init__(self, data, tree, objective='cost', demand_sign='eq', ramp=True, smoothing=True,
                 ramp_rates=RAMP, min_loads=MINLOAD, c2=C2, annualization_factor=1.0, explicit_nac=False, env=None,
                 name='Texas_TVEF_Battery_tree', fix_binaries=True, case=None, **case_overrides):
        """fix_binaries: True fixes every network binary at 1 (all facilities allowed; capacities stay continuous), a dict
        {process_or_resource: 0/1} fixes the given ones, False keeps them binary. With the binaries fixed the model is a
        pure LP and barrier without crossover solves the full-year tree in minutes; as a MIP, Gurobi needs a crossover /
        simplex root basis that takes an order of magnitude longer on this degenerate LP. Only Grid, PV and WF have a
        binding minimum size (prod_min = prod_max) and every solution of this case study builds all three."""
        t0_build = time.time()
        self.data, self.tree_original, self.explicit_nac = data, tree, explicit_nac
        self.fix_binaries = fix_binaries
        if explicit_nac:
            self.tree, self.copy_map = tree.unbundled()
        else:
            self.tree, self.copy_map = tree, {n.id: n.id for n in tree.nodes}
        tree = self.tree

        self.case = case_config(case, **case_overrides)
        base = make_inputs(data, tree.root.path_overrides if tree.root.path_overrides else None, case=self.case)
        res0, pr0 = base['resources'], base['processes']
        self.demand_resolution = self.case['demand']
        self.intensity_basis = base['intensity_basis']
        self.slack_procs = [p for p in sorted(pr0) if p in base['slack_procs']]
        self.PROCS = sorted(pr0)
        self.RES = sorted({r for p in pr0 for r in pr0[p]['conversion']})
        ip = self.ip = {p: i for i, p in enumerate(self.PROCS)}
        ir = self.ir = {r: i for i, r in enumerate(self.RES)}
        nP, nR = len(self.PROCS), len(self.RES)
        self.R_PURCH = [r for r in self.RES if res0[r]['cons_max'] > 0]
        self.R_SELL = [r for r in self.RES if res0[r]['sell']]
        self.R_STORE = [r for r in self.RES if res0[r]['store_max'] > 0]
        self.R_DEMAND = [r for r in self.RES if res0[r]['demand']]
        self.scope = {r: res0[r]['scope'] for r in self.RES}
        jc = {r: j for j, r in enumerate(self.R_PURCH)}
        js = {r: j for j, r in enumerate(self.R_SELL)}
        ji = {r: j for j, r in enumerate(self.R_STORE)}

        def selector(subset):      # sparse matrix mapping a compact array (subset x L) onto the resource rows (nR x L)
            E = _sp.lil_matrix((nR, len(subset)))
            for j, r in enumerate(subset):
                E[ir[r], j] = 1.0
            return E.tocsr()
        Ec, Es, Ei = selector(self.R_PURCH), selector(self.R_SELL), selector(self.R_STORE)

        m = gp.Model(name, env=env)
        self.m = m
        store_max = np.array([res0[r]['store_max'] for r in self.R_STORE])
        store_min = np.array([res0[r]['store_min'] for r in self.R_STORE])
        prod_max = np.array([pr0[p]['prod_max'] for p in self.PROCS])
        prod_min = np.array([pr0[p]['prod_min'] for p in self.PROCS])
        prod_max_ub = np.array([_inf(v) for v in prod_max])              # big-M capacities are simply unbounded
        kfin = np.flatnonzero(prod_max < bigM)

        # ---------------- first stage: design ----------------
        self.Cap_P = m.addMVar(nP, lb=0.0, ub=prod_max_ub, name='Cap_P')
        self.Cap_S = m.addMVar(len(self.R_STORE), lb=0.0, ub=store_max, name='Cap_S')
        if fix_binaries is False:
            self.X_P = m.addMVar(nP, vtype=GRB.BINARY, name='X_P')
            self.X_S = m.addMVar(len(self.R_STORE), vtype=GRB.BINARY, name='X_S')
        else:
            fixed = {} if fix_binaries is True else dict(fix_binaries)
            xP = np.array([float(fixed.get(p, 1)) for p in self.PROCS])
            xS = np.array([float(fixed.get(r, 1)) for r in self.R_STORE])
            self.X_P = m.addMVar(nP, lb=xP, ub=xP, name='X_P')
            self.X_S = m.addMVar(len(self.R_STORE), lb=xS, ub=xS, name='X_S')
        m.addConstr(self.Cap_P[kfin] <= prod_max[kfin] * self.X_P[kfin], name='facility_max')
        kbig = np.flatnonzero(prod_max >= bigM)
        if kbig.size:
            self.X_P[kbig].lb = 1.0                                               # big-M processes always available
        m.addConstr(self.Cap_S <= store_max * self.X_S, name='storage_facility_max')
        kp, ks = np.flatnonzero((prod_min > 0) & (prod_min < bigM)), np.flatnonzero(store_min > 0)   # real minimum-size rows only
        if kp.size:
            m.addConstr(self.Cap_P[kp] >= prod_min[kp] * self.X_P[kp], name='facility_min')
        if ks.size:
            m.addConstr(self.Cap_S[ks] >= store_min[ks] * self.X_S[ks], name='storage_facility_min')
        for a, b in base['rating_links']:                                          # one rating for both directions
            m.addConstr(self.Cap_P[ip[a]] == self.Cap_P[ip[b]], name=f'rating_link[{a},{b}]')
        self.horizon_factor = tree.n_days * N_HOURS / N_T if self.case.get('prorate_capex', False) else 1.0
        capex = np.array([pr0[p]['capex'] for p in self.PROCS]) * annualization_factor * self.horizon_factor
        fopex = np.array([pr0[p]['fopex'] for p in self.PROCS]) * self.horizon_factor
        s_capex = np.array([base['storage_capex'].get(r, 0.0) for r in self.R_STORE]) * annualization_factor * self.horizon_factor
        self.Capex = capex @ self.Cap_P + s_capex @ self.Cap_S
        self.Fopex = fopex @ self.Cap_P

        # ---------------- nodes: operations ----------------
        self.node = {}
        vopex = np.array([pr0[p]['vopex'] for p in self.PROCS])
        for n in tree.nodes:
            if n.is_root:
                continue
            inp = make_inputs(data, n.path_overrides, case=self.case)
            res, pr = inp['resources'], inp['processes']
            t0, t1, L = n.t0, n.t1, n.length
            A = np.zeros((nR, nP))
            for p in self.PROCS:
                for r, c in pr[p]['conversion'].items():
                    A[ir[r], ip[p]] = c
            A = _sp.csr_matrix(A)
            C_ub = np.array([[_inf(res[r]['cons_max']) if res[r]['varying'] is None else GRB.INFINITY] * L for r in self.R_PURCH])
            P = m.addMVar((nP, L), lb=0.0, name=f'P[{n.id}]')
            C = m.addMVar((len(self.R_PURCH), L), lb=0.0, ub=C_ub, name=f'C[{n.id}]')
            S = m.addMVar((len(self.R_SELL), L), lb=0.0, name=f'S[{n.id}]')
            Inv = m.addMVar((len(self.R_STORE), L), lb=0.0, name=f'Inv[{n.id}]')
            par = self.node.get(n.parent.id) if not n.parent.is_root else None

            # nameplate production / inventory
            for p in self.PROCS:
                cf = pr[p]['capacity_factor']
                if cf is None:
                    m.addConstr(P[ip[p], :] <= self.Cap_P[ip[p]], name=f'nameplate[{n.id},{p}]')
                else:   # hours with zero capacity factor are fixed to zero (avoids zero-coefficient rows)
                    f = np.asarray(inp['capacity_factor'][cf][t0:t1], dtype=float)
                    on = np.flatnonzero(f > 1e-12)
                    P[ip[p], np.flatnonzero(f <= 1e-12)].ub = 0.0
                    if on.size:
                        m.addConstr(P[ip[p], on] <= f[on] * self.Cap_P[ip[p]], name=f'nameplate[{n.id},{p}]')
            for r in self.R_STORE:
                m.addConstr(Inv[ji[r], :] <= self.Cap_S[ji[r]], name=f'nameplate_inventory[{n.id},{r}]')

            # resource balance; the inventory links to the parent's last hour (zero for first-stage nodes)
            if par is not None:
                m.addConstr(Ec @ C[:, 0] + A @ P[:, 0] - Es @ S[:, 0] == Ei @ Inv[:, 0] - Ei @ par['Inv'][:, -1], name=f'balance0[{n.id}]')
            else:
                m.addConstr(Ec @ C[:, 0] + A @ P[:, 0] - Es @ S[:, 0] == Ei @ Inv[:, 0], name=f'balance0[{n.id}]')
            if L > 1:
                m.addConstr(Ec @ C[:, 1:] + A @ P[:, 1:] - Es @ S[:, 1:] == Ei @ Inv[:, 1:] - Ei @ Inv[:, :-1], name=f'balance[{n.id}]')

            # demand inside the slice: daily totals (legacy) or hourly targets
            d0, d1 = t0 // N_HOURS, t1 // N_HOURS
            if self.demand_resolution == 'hourly':
                daily = sum((S[js[r], :] for r in self.R_DEMAND[1:]), S[js[self.R_DEMAND[0]], :])
                target = np.asarray(inp['h2_demand_hourly'], dtype=float)[t0:t1]
            else:
                daily = sum(S[js[r], :].reshape((d1 - d0, N_HOURS)).sum(axis=1) for r in self.R_DEMAND)
                target = np.asarray(inp['h2_demand'], dtype=float)[d0:d1]
            if demand_sign == 'eq':
                m.addConstr(daily == target, name=f'demand[{n.id}]')
            elif demand_sign == 'geq':
                m.addConstr(daily >= target, name=f'demand[{n.id}]')
            else:
                m.addConstr(daily <= target, name=f'demand[{n.id}]')

            # ramp, minimum load, smoothing (linked to the parent's boundary hours)
            if ramp:
                for p, rate in ramp_rates.items():
                    i = ip[p]
                    if L > 1:
                        m.addConstr(P[i, 1:] - P[i, :-1] <= rate * self.Cap_P[i], name=f'ramp_up[{n.id},{p}]')
                        m.addConstr(P[i, 1:] - P[i, :-1] >= -rate * self.Cap_P[i], name=f'ramp_down[{n.id},{p}]')
                    if par is not None:
                        m.addConstr(P[i, 0] - par['P'][i, -1] <= rate * self.Cap_P[i], name=f'ramp_up_link[{n.id},{p}]')
                        m.addConstr(P[i, 0] - par['P'][i, -1] >= -rate * self.Cap_P[i], name=f'ramp_down_link[{n.id},{p}]')
                for p, frac in min_loads.items():
                    m.addConstr(P[ip[p], :] >= frac * self.Cap_P[ip[p]], name=f'min_load[{n.id},{p}]')
            if smoothing:
                for p, c in c2.items():
                    i = ip[p]
                    if L > 2:
                        acc = P[i, 2:] - 2 * P[i, 1:-1] + P[i, :-2]
                        m.addConstr(acc <= c * self.Cap_P[i], name=f'smooth_up[{n.id},{p}]')
                        m.addConstr(acc >= -c * self.Cap_P[i], name=f'smooth_down[{n.id},{p}]')
                    if par is not None:
                        pP = par['P']
                        links = [P[i, 1] - 2 * P[i, 0] + pP[i, -1]]                     # centred on the node's first hour
                        if par['L'] > 1:
                            links.append(P[i, 0] - 2 * pP[i, -1] + pP[i, -2])            # centred on the parent's last hour
                        for k, acc in enumerate(links):
                            m.addConstr(acc <= c * self.Cap_P[i], name=f'smooth_up_link{k}[{n.id},{p}]')
                            m.addConstr(acc >= -c * self.Cap_P[i], name=f'smooth_down_link{k}[{n.id},{p}]')

            # node cost and emissions
            price = np.array([inp['price_profile'][r][t0:t1] if r in inp['price_profile'] else np.full(L, float(res[r]['price'] or 0.0))
                              for r in self.R_PURCH])
            opex = (vopex[:, None] * P).sum() + (price * C).sum()
            gwpC = np.array([inp['gwp_profile'][r][t0:t1] if r in inp['gwp_profile'] else np.full(L, float(res[r]['gwp'] or 0.0))
                             for r in self.R_PURCH])
            gwpS = np.array([float(res[r]['gwp'] or 0.0) for r in self.R_SELL])
            gwp_scope = {}
            for s in SCOPES:
                terms = [(gwpC[jc[r]] * C[jc[r], :]).sum() for r in self.R_PURCH if self.scope[r] == s and gwpC[jc[r]].any()]
                terms += [gwpS[js[r]] * S[js[r], :].sum() for r in self.R_SELL if self.scope[r] == s and gwpS[js[r]]]
                gwp_scope[s] = sum(terms[1:], terms[0]) if terms else gp.LinExpr()
            pen = self.slack_procs
            h2_delivered = sum((S[js[r], :].sum() for r in self.R_DEMAND[1:]), S[js[self.R_DEMAND[0]], :].sum())
            h2_penalty = sum((P[ip[q], :].sum() for q in pen[1:]), P[ip[pen[0]], :].sum()) if pen else 0.0
            self.node[n.id] = dict(P=P, C=C, S=S, Inv=Inv, L=L, opex=opex, gwp_scope=gwp_scope, inputs=inp,
                                   gwp=sum((gwp_scope[s] for s in SCOPES), gp.LinExpr()),
                                   h2_delivered=h2_delivered, h2_penalty=h2_penalty, h2_own=h2_delivered - h2_penalty)

        # ---------------- explicit non-anticipativity constraints (verification mode) ----------------
        self.n_nac = 0
        if explicit_nac:
            groups = {}
            for n in tree.nodes:
                if not n.is_root:
                    groups.setdefault(self.copy_map[n.id], []).append(n.id)
            for orig, ids in groups.items():
                rep = self.node[ids[0]]
                for other in ids[1:]:
                    o = self.node[other]
                    for key in ('P', 'C', 'S', 'Inv'):
                        if rep[key].shape[0]:
                            m.addConstr(o[key] == rep[key], name=f'nac_{key}[{orig},{other}]')
                            self.n_nac += int(np.prod(rep[key].shape))

        # ---------------- expected cost / emissions ----------------
        ops = [n for n in tree.nodes if not n.is_root]
        self.Opex_expected = sum((n.prob * self.node[n.id]['opex'] for n in ops[1:]), ops[0].prob * self.node[ops[0].id]['opex'])
        self.Cost_expected = self.Capex + self.Fopex + self.Opex_expected
        self.gwp_scope_expected = {s: sum((n.prob * self.node[n.id]['gwp_scope'][s] for n in ops), gp.LinExpr()) for s in SCOPES}
        self.gwp_expected = sum((self.gwp_scope_expected[s] for s in SCOPES), gp.LinExpr())
        self.H2_own_expected = sum((n.prob * self.node[n.id]['h2_own'] for n in ops[1:]), ops[0].prob * self.node[ops[0].id]['h2_own'])
        self.H2_delivered_expected = sum((n.prob * self.node[n.id]['h2_delivered'] for n in ops[1:]), ops[0].prob * self.node[ops[0].id]['h2_delivered'])
        self.gwp_caps = {}
        self.set_objective(objective)
        m.update()
        self.status = None
        self.build_time = time.time() - t0_build

    # ------------------------------------------------------------------------------------------------------------------
    def set_objective(self, objective='cost'):
        """'cost' / 'gwp' / 'scopeK' minimize the expectation over the tree; 'minmax_cost' / 'minmax_gwp' minimize the
        worst scenario (robust over the finite uncertainty set formed by the tree's leaves, epigraph formulation)."""
        if objective == 'cost':
            self.m.setObjective(self.Cost_expected, GRB.MINIMIZE)
        elif objective == 'gwp':
            self.m.setObjective(self.gwp_expected, GRB.MINIMIZE)
        elif objective in ('scope1', 'scope2', 'scope3'):
            self.m.setObjective(self.gwp_scope_expected[int(objective[-1])], GRB.MINIMIZE)
        elif objective in ('minmax_cost', 'minmax_gwp'):
            if not hasattr(self, '_epigraph'):
                self._epigraph = {}
            if objective not in self._epigraph:
                z = self.m.addVar(lb=-GRB.INFINITY, name=f'z_{objective}')
                for leaf in self.tree.leaves:
                    expr = self.scenario_cost(leaf) if objective == 'minmax_cost' else self.scenario_gwp(leaf)
                    self.m.addConstr(z >= expr, name=f'{objective}[{leaf.id}]')
                self._epigraph[objective] = z
            self.m.setObjective(self._epigraph[objective], GRB.MINIMIZE)
        else:
            raise ValueError(objective)
        self.objective_name = objective
        return self

    def solve_lexicographic(self, primary='minmax_cost', secondary='cost', method='weighted', weight=1e-3, tol=1e-6,
                            tee=False, **params):
        """Robust-then-efficient solve: minimize `primary` (e.g. the worst-scenario cost) and, among its (near-)optimal
        designs, the `secondary` objective (e.g. the expected cost). Removes the indifference of a pure min-max
        objective among the non-worst scenarios.

        method='weighted' (default): one solve of  primary + weight * secondary  (weight small, e.g. 1e-3, so the
            worst case is traded off against the expectation at 1000:1; the realized primary value is reported).
        method='strict': two solves; the second minimizes `secondary` subject to primary <= optimum * (1 + tol).
        """
        self.set_objective(primary)                       # creates the epigraph variable of `primary`
        z = self._epigraph[primary]
        sec = {'cost': self.Cost_expected, 'gwp': self.gwp_expected}.get(secondary)
        if sec is None:
            sec = self.gwp_scope_expected[int(secondary[-1])]
        if method == 'weighted':
            self.m.setObjective(z + weight * sec, GRB.MINIMIZE)
            st = self.solve(tee=tee, **params)
            if st == 'optimal':
                self.primary_value = z.X
            self.objective_name = f'{primary} + {weight:g} * {secondary}'
            return st
        if method == 'strict':
            st = self.solve(tee=tee, **params)
            if st != 'optimal':
                return st
            self.primary_value = self.m.ObjVal
            self.m.addConstr(z <= self.primary_value * (1 + tol), name='lexicographic_guard')
            self.set_objective(secondary)
            st = self.solve(tee=tee, **params)
            self.objective_name = f'{secondary} | {primary} <= {self.primary_value:,.0f}'
            return st
        raise ValueError("method must be 'weighted' or 'strict'")

    def scenario_gwp(self, leaf, scope=None):
        """GWP expression accumulated along the path of a leaf (scope=None: total)."""
        nodes = self.tree.scenario_path(leaf)
        terms = [self.node[n.id]['gwp'] if scope is None else self.node[n.id]['gwp_scope'][scope] for n in nodes]
        return sum(terms[1:], terms[0])

    def scenario_cost(self, leaf):
        nodes = self.tree.scenario_path(leaf)
        return self.Capex + self.Fopex + sum((self.node[n.id]['opex'] for n in nodes[1:]), self.node[nodes[0].id]['opex'])

    def scenario_h2(self, leaf, key='h2_own'):
        """Hydrogen accumulated along a scenario: key in 'h2_delivered', 'h2_penalty', 'h2_own'."""
        nodes = self.tree.scenario_path(leaf)
        return sum((self.node[n.id][key] for n in nodes[1:]), self.node[nodes[0].id][key])

    def add_gwp_cap(self, cap=None, scope=None, mode='each', intensity=None, intensity_basis=None):
        """Cap on total (or one scope's) GWP: mode='each' per scenario, mode='expected' on the expectation.
        cap is absolute [kg CO2e]; intensity instead caps kg CO2e per kg of hydrogen, on the own hydrogen (delivered
        minus the emission-free tiers; legacy default) or on all delivered hydrogen (v2 default, merchant hydrogen
        carries its own emissions); `intensity_basis` ('own' / 'delivered') overrides the case default."""
        if (cap is None) == (intensity is None):
            raise ValueError('give exactly one of cap or intensity')
        basis = intensity_basis or self.intensity_basis
        if basis not in ('own', 'delivered'):
            raise ValueError("intensity_basis must be 'own' or 'delivered'")
        key = 'h2_own' if basis == 'own' else 'h2_delivered'
        tag = ('gwp_cap' if scope is None else f'gwp_cap_scope{scope}') + ('' if intensity is None else '_intensity')
        if mode == 'expected':
            expr = self.gwp_expected if scope is None else self.gwp_scope_expected[scope]
            rhs = cap if intensity is None else intensity * (self.H2_own_expected if basis == 'own' else self.H2_delivered_expected)
            self.gwp_caps[tag] = self.m.addConstr(expr <= rhs, name=tag)
        elif mode == 'each':
            for leaf in self.tree.leaves:
                rhs = cap if intensity is None else intensity * self.scenario_h2(leaf, key)
                self.gwp_caps[f'{tag}[{leaf.id}]'] = self.m.addConstr(self.scenario_gwp(leaf, scope) <= rhs, name=f'{tag}[{leaf.id}]')
        else:
            raise ValueError("mode must be 'each' or 'expected'")
        return self

    def add_gwp_reduction(self, base, pct, scope=None, mode='each'):
        return self.add_gwp_cap(base * (1 - pct / 100), scope=scope, mode=mode)

    def fix_design(self, design):
        """Fix the first-stage decisions (dict with Cap_P / Cap_S Series, optionally X_P / X_S) to evaluate them."""
        capP = design['Cap_P'].reindex(self.PROCS).to_numpy(dtype=float)
        capS = design['Cap_S'].reindex(self.R_STORE).to_numpy(dtype=float)
        xP = design['X_P'].reindex(self.PROCS).to_numpy(dtype=float) if 'X_P' in design else (capP > 0).astype(float)
        xS = design['X_S'].reindex(self.R_STORE).to_numpy(dtype=float) if 'X_S' in design else (capS > 0).astype(float)
        self.Cap_P.lb, self.Cap_P.ub = capP, capP
        self.Cap_S.lb, self.Cap_S.ub = capS, capS
        self.X_P.lb, self.X_P.ub = xP, xP
        self.X_S.lb, self.X_S.ub = xS, xS
        self.m.update()
        return self

    def solve(self, tee=True, **params):
        """Solve. Defaults to barrier without crossover (Method=2, Crossover=0): the tree LP is large and highly
        degenerate, and crossover to a vertex can take far longer than the barrier itself. Pass Crossover=1 (or any
        Gurobi parameter) to override."""
        settings = dict(Method=2, Crossover=0)
        settings.update(params)
        self.m.Params.OutputFlag = 1 if tee else 0
        for k, v in settings.items():
            setattr(self.m.Params, k, v)
        t0 = time.time()
        self.m.optimize()
        self.solve_time = time.time() - t0
        self.status = _STATUS.get(self.m.Status, str(self.m.Status))
        if tee:
            print(f'termination: {self.status}  (build {self.build_time:.1f} s, solve {self.solve_time:.1f} s)')
        return self.status

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _val(e):
        return float(e.getValue()) if hasattr(e, 'getValue') else float(e)

    def design(self):
        return dict(Cap_P=pd.Series(self.Cap_P.X, index=self.PROCS), Cap_S=pd.Series(self.Cap_S.X, index=self.R_STORE),
                    X_P=pd.Series(np.round(self.X_P.X).astype(int), index=self.PROCS),
                    X_S=pd.Series(np.round(self.X_S.X).astype(int), index=self.R_STORE))

    def scenario_table(self):
        rows = []
        for leaf in self.tree.leaves:
            path = self.tree.scenario_path(leaf)
            own, pen, tot = (self._val(self.scenario_h2(leaf, k)) for k in ('h2_own', 'h2_penalty', 'h2_delivered'))
            gwp = self._val(self.scenario_gwp(leaf))
            basis = own if self.intensity_basis == 'own' else tot
            rows.append(dict(scenario=leaf.id, path=' > '.join(n.label for n in path), prob=leaf.prob,
                             cost=self._val(self.scenario_cost(leaf)),
                             **{f'gwp_scope{s}': self._val(self.scenario_gwp(leaf, s)) for s in SCOPES},
                             gwp_total=gwp, h2_own=own, h2_penalty=pen, h2_delivered=tot,
                             intensity=gwp / basis if basis > 0 else np.nan))
        return pd.DataFrame(rows).set_index('scenario')

    def node_table(self):
        rows = []
        for n in self.tree.nodes:
            if n.is_root:
                continue
            d = self.node[n.id]
            rows.append(dict(node=n.id, stage=n.stage, label=n.label, parent=n.parent.id, prob=n.prob,
                             day_start=n.t0 // N_HOURS, day_end=n.t1 // N_HOURS, opex=self._val(d['opex']),
                             **{f'gwp_scope{s}': self._val(d['gwp_scope'][s]) for s in SCOPES},
                             inv_end=(float(d['Inv'].X[:, -1].sum()) if d['Inv'].shape[0] else 0.0)))
        return pd.DataFrame(rows).set_index('node')

    def node_frames(self, node_id):
        """Operational solution of one node as DataFrames indexed by absolute hour."""
        n = next(x for x in self.tree.nodes if x.id == node_id)
        d = self.node[node_id]
        cols = range(n.t0, n.t1)
        return dict(P=pd.DataFrame(d['P'].X, index=self.PROCS, columns=cols),
                    C=pd.DataFrame(d['C'].X, index=self.R_PURCH, columns=cols),
                    S=pd.DataFrame(d['S'].X, index=self.R_SELL, columns=cols),
                    Inv=pd.DataFrame(d['Inv'].X, index=self.R_STORE, columns=cols))

    def scenario_frames(self, leaf):
        """Full-horizon operational solution along one scenario (concatenated node slices)."""
        parts = [self.node_frames(n.id) for n in self.tree.scenario_path(leaf)]
        return {k: pd.concat([p[k] for p in parts], axis=1) for k in parts[0]}

    def results(self):
        out = {'termination': self.status, 'objective': None, 'tree': self.tree_original.table()}
        if self.status != 'optimal':
            return out
        out.update(objective=self.m.ObjVal, expected_cost=self._val(self.Cost_expected), capex=self._val(self.Capex),
                   fopex=self._val(self.Fopex), expected_opex=self._val(self.Opex_expected),
                   expected_gwp=self._val(self.gwp_expected),
                   expected_gwp_scope={s: self._val(self.gwp_scope_expected[s]) for s in SCOPES},
                   design=self.design(), scenarios=self.scenario_table(), nodes=self.node_table())
        return out

    def summary(self):
        kind = 'explicit NAC' if self.explicit_nac else 'node formulation'
        lines = [f'termination : {self.status}  ({self.tree_original}; {kind})']
        if self.status == 'optimal':
            v = self._val
            lines.append(f'objective   : {self.m.ObjVal:,.2f}  ({self.objective_name})')
            lines.append(f'exp. cost   : {v(self.Cost_expected):,.2f} USD  (capex {v(self.Capex):,.0f}, fopex {v(self.Fopex):,.0f}, '
                         f'exp. opex {v(self.Opex_expected):,.0f})')
            lines.append(f'exp. GWP    : {v(self.gwp_expected):,.2f} kg CO2e  (scope 1 {v(self.gwp_scope_expected[1]):,.0f}, '
                         f'scope 2 {v(self.gwp_scope_expected[2]):,.0f}, scope 3 {v(self.gwp_scope_expected[3]):,.0f})')
            d = self.design()
            lines.append('capacities  : ' + ', '.join(f'{p} {c:,.1f}' for p, c in d['Cap_P'].items() if c > 0))
            lines.append('storage     : ' + ', '.join(f'{r} {c:,.2f}' for r, c in d['Cap_S'].items()))
        print('\n'.join(lines))
