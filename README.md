# IBF-rainfall

Heavy rainfall forecast pipeline for Kasese District to support Uganda RCS.
Adapted from [rainfall-monitor-metnoAPI](https://github.com/rodekruis/rainfall-monitor-metnoAPI)

## Introduction

This is a series of scripts running run daily to:
1. Extract latest rainfall forecast data from ECWMF (via Norwegian Meteorological Institute, using [MET Norway Location Forecast](https://github.com/Rory-Sullivan/metno-locationforecast)) and other input static data. 
2. Check if there will be a heavy rainfall event in a several lead times
3. Create rainfall extents and calculated affected population,
4. Load the output to locations where they can be served to the IBF Portal.

### Prerequisites

1. Install Docker

### Installation

1. Clone this directory to `<your_local_directory>`/IBF-pipeline/

2. Change `/pipeline/lib/pipeline/secrets.py.template` to `secrets.py` and fill in the necessary secrets.

3. Build Docker image (from the IBF-pipeline root folder) and run container with volume.

- Build image: `docker build -t ibf-rainfall-kasese .`
- Create + start container: `docker run -it --entrypoint /bin/bash ibf-rainfall-kasese`
- Test it (from within Docker container) through: `run-pipeline`

For operations, see status at [Azure Logic App](https://portal.azure.com/#@rodekruis.nl/resource/subscriptions/b2d243bd-7fab-4a8a-8261-a725ee0e3b47/resourceGroups/510Global-IBF-System/providers/Microsoft.Logic/workflows/510-ibf-rainfall-pipeline/logicApp).