######################
## COUNTRY SETTINGS ##
######################

COUNTRY_CODE = 'UGA'
SETTINGS = {
    "UGA": {
        'lead_times': {
            "1-day": 1,
            "2-day": 2,
            "3-day": 3
        },
        'thresholds': {
            "1-day": 60#,
            # "3-day": 150
        }, 
        'admin_level': 3,
        'EXPOSURE_DATA_SOURCES': {
            "population": {
                "source": "population/hrsl_uga_pop_resized_100",
                "rasterValue": 1
            }
        }
    }
}



####################
## OTHER SETTINGS ##
####################

# Nr. of max open files, when pipeline is ran from cronjob.
# Should be larger then the nr of admin-areas on the relevant admin-level handled (e.g. 1040 woreda's in ETH)
SOFT_LIMIT = 10000


###################
## PATH SETTINGS ##
###################
RASTER_DATA = '/pipeline/data/raster/'
RASTER_INPUT = RASTER_DATA + 'input/'
RASTER_OUTPUT = RASTER_DATA + 'output/'
PIPELINE_DATA = '/pipeline/data/other/'
PIPELINE_INPUT = PIPELINE_DATA + 'input/'
PIPELINE_OUTPUT = PIPELINE_DATA + 'output/'


#########################
## INPUT DATA SETTINGS ##
#########################

# METNO rainfall input
METNO_API = {'user_agent': '510Global'}
SHAPEFILE = {
    'observed_points': 'shape/uga_points',
    'boundary': 'shape/kasese_adm3'
    }