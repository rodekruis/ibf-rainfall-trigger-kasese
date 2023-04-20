from pipeline.rainfalldata import RainfallData
from pipeline.exposure import Exposure
from pipeline.dynamicDataDb import DatabaseManager
import pandas as pd
import json
from shapely import wkb, wkt
import geopandas
from pipeline.settings import *


class Forecast:
    def __init__(self, leadTimeLabel, leadTimeValue, countryCodeISO3, admin_level):
        self.leadTimeLabel = leadTimeLabel
        self.leadTimeValue = leadTimeValue
        self.admin_level = admin_level
        self.db = DatabaseManager(leadTimeLabel, countryCodeISO3)

        admin_area_json = PIPELINE_INPUT + SHAPEFILE['boundary'] + '.geojson' #self.db.apiGetRequest('admin-areas/raw',countryCodeISO3=countryCodeISO3)
        self.admin_area_gdf = geopandas.read_file(admin_area_json)
        # for index in range(len(admin_area_json)):
        #     admin_area_json[index]['geometry'] = admin_area_json[index]['geom']
        #     admin_area_json[index]['properties'] = {
        #         'placeCode': admin_area_json[index]['placeCode'],
        #         'name': admin_area_json[index]['name']
        #     }
        
        self.population_total = self.db.apiGetRequest('admin-area-data/{}/{}/{}'.format(countryCodeISO3, self.admin_level, 'populationTotal'), countryCodeISO3='')

        self.rainfallData = RainfallData(leadTimeLabel, leadTimeValue, countryCodeISO3, self.admin_area_gdf)#, self.rainfall_triggers)
        self.exposure = Exposure(leadTimeLabel, countryCodeISO3, self.admin_area_gdf, self.population_total, self.admin_level)
