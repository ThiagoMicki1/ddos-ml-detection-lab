# Dataset Notes

## Official Dataset

This project uses the CSE-CIC-IDS2018 on AWS dataset, also listed by AWS as "A Realistic Cyber Defense Dataset (CSE-CIC-IDS2018)."

Official references:

- UNB/CIC dataset page: https://www.unb.ca/cic/datasets/ids-2018.html
- AWS Open Data Registry: https://registry.opendata.aws/cse-cic-ids2018/

The AWS registry describes the dataset as a collaboration between the Communications Security Establishment and the Canadian Institute for Cybersecurity. The UNB page describes seven attack scenarios: brute force, Heartbleed, Botnet, DoS, DDoS, web attacks, and infiltration.

## Local Files Used

The local `data/` directory contains daily CSV exports:

- `02-14-2018.csv`
- `02-15-2018.csv`
- `02-16-2018.csv`
- `02-20-2018.csv`
- `02-21-2018.csv`
- `02-22-2018.csv`
- `02-23-2018.csv`
- `02-28-2018.csv`
- `03-01-2018.csv`
- `03-02-2018.csv`

The repaired experiment uses only the DDoS days:

- `02-20-2018.csv`
- `02-21-2018.csv`

That choice is intentional. These are the files containing the DDoS labels available in this local copy:

- `DDoS attacks-LOIC-HTTP`
- `DDOS attack-HOIC`
- `DDOS attack-LOIC-UDP`

Other attack families are excluded because this project is scoped to binary Benign vs DDoS detection, not general intrusion detection.

## Schema Finding

Most daily CSV files contain 80 columns. The file `02-20-2018.csv` contains 84 columns because it includes identifier fields:

- `Flow ID`
- `Src IP`
- `Src Port`
- `Dst IP`

The repaired pipeline normalizes column names and avoids using these identifier fields.

## Dataset Handling

The dataset and `archive.zip` are intentionally not committed. They are large research artifacts and may have redistribution restrictions. Reproduce the project by downloading the dataset from the official source and placing the CSV files under `data/`.

The AWS registry provides no-account S3 access:

```bash
aws s3 sync --no-sign-request --region ca-central-1 s3://cse-cic-ids2018/ data/raw-cse-cic-ids2018/
```

The exact local CSV layout may differ depending on how the dataset is downloaded or extracted.
