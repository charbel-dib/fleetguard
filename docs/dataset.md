# Dataset provenance

Dataset: **APS Failure at Scania Trucks**, UCI repository ID 421.

- Publisher page: <https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks>
- DOI: <https://doi.org/10.24432/C51S51>
- Archive: <https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip>
- Creator: Scania CV AB.
- Donors listed by UCI: Tony Lindgren and Jonas Biteus.
- Data description date: September 2016.

The archive contains `aps_failure_training_set.csv`, `aps_failure_test_set.csv` and
`aps_failure_description.txt`. FleetGuard preserves these files unchanged locally. There are
171 columns in each CSV: **one class column and 170 input features**. The original attribute
count of 171 includes the class column.

The challenge uses a cost of 10 for an unnecessary APS inspection and 500 for a missed APS failure.
These weights are evaluation assumptions, not current observed business costs or a euro estimate.

## Terms included with the source

As checked on 2026-10-07, the UCI page displays a CC BY 4.0 dataset license. The source CSV
preambles and the accompanying description include a Scania copyright notice and GNU GPL
version 3-or-later terms. FleetGuard records this discrepancy instead of rewriting the source
notice or asserting that its MIT code license also covers the dataset.

The repository and starter archive do not redistribute the Scania CSVs or original ZIP. The
download command retrieves them from the publisher and retains their notices. Reusing or
redistributing data or derived datasets requires considering the dataset's own terms.

## Integrity

Each local import records SHA-256 hashes for the archive and all three extracted source files.
These are snapshot identifiers, not publisher-provided authenticity signatures. `validate` and
`train` reject modified source members. Source identities are copied into each experiment run.

CSV schema checks reject missing labels, unexpected row counts, malformed sensor names and
non-numeric values. They do not detect all possible semantic changes that preserve the schema.

## Data interpretation

Features are anonymized operational counters and histogram bins. A negative example has a
failure unrelated to APS; it is not evidence of a healthy truck. Without a released vehicle ID,
timestamp or deployment sampling protocol, temporal validation, fleet isolation and real-world
prevalence correction cannot be established from the table alone.
