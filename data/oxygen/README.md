# Oxygen 2021 case-study data

| File | What | Source | Status |
|---|---|---|---|
| `raw/states.csv` | daily cumulative Confirmed/Recovered/Deceased/Other by state | data.covid19india.org/csv/latest/states.csv (downloaded 2026-11-05) | downloaded |
| `active_cases_2021.csv` | active cases (= Confirmed - Recovered - Deceased - Other) per state on the oxygen-order dates and later peaks | derived from `raw/states.csv` | built |
| `oxygen_allocation.csv` | per-state demand and allocation (MT) for the 15/18/21 April 2021 orders, 19 states (blank = state not in that order) | Rajya Sabha Unstarred Q.1056 (Saket Gokhale), answered 12 Dec 2023, MoHFW, Annexure, entered by hand | **built; all 6 column sums equal the printed totals (4880/4880, 6293/6293, 6721/6722)** |

Manual download (sites block automated requests):
- Rajya Sabha Unstarred Q.1056, 12 Dec 2023 (annexure: demand + allocation by state for 15/18/21 April 2021): search sansad.in -> Rajya Sabha -> Questions.
- Rajya Sabha Q.349, 5 Apr 2022 (oxygen supply), archived copy:
  http://web.archive.org/web/20251206005547/https://rsdebate.nic.in/bitstream/123456789/727077/1/PQ_256_05042022_S349_p23_p33.pdf


## Documents checked (do not contain the allocation table)
| Document | Content | Useful facts |
|---|---|---|
| Rajya Sabha Starred Q.349, 5 Apr 2022 (Health), `PQ_256_05042022_S349_p23_p33.pdf` | deaths due to oxygen shortage | national only: 5,21,358 COVID deaths (4 Apr 2022); 20 States/UTs replied, none reported confirmed oxygen-shortage deaths; 4,055 PSA plants; 4,02,517 cylinders and 1,13,858 concentrators allocated |
| Rajya Sabha Unstarred Q.1056, 4 Dec 2024 (session 266, **Cooperation**), `_annex_266_AU1056_*.pdf` | PACS registrations; **wrong question** (same number, different session) | none |
| Rajya Sabha Unstarred Q.1056, 28 Jul 2026 (Health), `AU1056_NJBWBf.pdf` | private hospital scams; **wrong question** (same number, different session) | none |

**Found:** Rajya Sabha Unstarred Q.1056, answered **12 Dec 2023** (Saket Gokhale, MoHFW), "Medical oxygen supply during COVID-19 pandemic" (`AU1056.pdf`). Its Annexure is the source of `oxygen_allocation.csv`. The same reply says GoI did not itself procure oxygen between 1 Mar and 21 Apr 2021; facilities buy directly from manufacturers, and the allocation order (first issued 15 Apr 2021) assigns supply to states.

Note: question numbers repeat every session; identify this one by date + ministry.
