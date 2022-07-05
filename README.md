# biomarkers

Estimate A2CPS biomarker candidates from imaging data.

## Usage

```bash
$ pip install biomarkers
$ mark ${bids}
```

By default, outputs will be left in a newly created folder called `out`

### Singularity

```bash
$ singularity run --cleanenv docker://psadil/biomarkers --help
$ singularity run --cleanenv docker://psadil/biomarkers ${bids}
```
