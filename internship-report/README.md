# Internship Report

Cumulative LaTeX report documenting all internship work. This report is incrementally updated as new work is completed and serves as the primary deliverable.

## Contents

| File | Description |
|------|-------------|
| `report.tex` | Main LaTeX source |
| `stage_hw.pdf` | Compiled report (latest version) |
| `qr1.png` | QR code linking to this repository |
| `sc1.png`, `sc2.png` | MLflow UI screenshots |

## Topics Covered

1. OpenStack discovery and environment setup
2. Static vs. dynamic load-balanced architecture
3. OpenStack component research (Nova, Keystone, Ceilometer, Gnocchi, Aodh, Heat)
4. Self-scaling infrastructure with Heat templates
5. Bug diagnosis and resolution (signal_url vs alarm_url)
6. DVC data versioning
7. MLflow experiment tracking
8. Data drift theory and analysis

## Building

```bash
# Compile the report (requires a LaTeX distribution)
cd internship-report
pdflatex report.tex
```
