import rasterio
import rasterio.mask
import rasterio.features
import rasterio.warp
from rasterio.features import shapes
import fiona
import numpy as np
import pandas as pd
from pandas import DataFrame
import json

from pipeline.settings import *
import os


class Exposure:

    """Class used to calculate the exposure per exposure type"""

    def __init__(self, leadTimeLabel, countryCodeISO3, admin_area_gdf, population_total, admin_level, district_mapping=None):
        self.leadTimeLabel = leadTimeLabel
        self.countryCodeISO3 = countryCodeISO3
        self.disasterExtentRaster = RASTER_OUTPUT + \
            '0/rainfall_extents/rainfall_extent_' + leadTimeLabel + '_' + countryCodeISO3 + '.tif'
        self.disasterExtentEmptyRaster = RASTER_OUTPUT + \
            '0/rainfall_extents/rainfall_extent_' + leadTimeLabel + '_' + countryCodeISO3 + '_notrigger.tif'
        self.selectionValue = 0.9
        self.outputPath = PIPELINE_OUTPUT + "out.tif"
        self.district_mapping = district_mapping
        self.ADMIN_AREA_GDF = admin_area_gdf
        self.ADMIN_AREA_GDF_TMP_PATH = PIPELINE_OUTPUT+"admin-areas_TMP.shp"
        self.EXPOSURE_DATA_SOURCES = SETTINGS[countryCodeISO3]['EXPOSURE_DATA_SOURCES']
        self.admin_level = admin_level
        if "population" in self.EXPOSURE_DATA_SOURCES:
            self.population_total = population_total
        # if "alert_threshold" in self.EXPOSURE_DATA_SOURCES:
        #     self.alert_threshold = alert_threshold

    def callAllExposure(self):
        for indicator, values in self.EXPOSURE_DATA_SOURCES.items():
            print('indicator: ', indicator)
            self.inputRaster = RASTER_INPUT + values['source'] + ".tif"
            self.outputRaster = RASTER_OUTPUT + "0/" + \
                values['source'] + self.leadTimeLabel

            if os.getenv('TRIGGER') == 'False': # filter only events has high impact
            #     for d in stats:
            #         d.update((k, 0) for k, v in d.items() if k=="amount")
                os.replace(self.disasterExtentEmptyRaster, self.disasterExtentRaster)
            stats = self.calcAffected(self.disasterExtentRaster, indicator, values['rasterValue'])

            result = {
                'countryCodeISO3': self.countryCodeISO3,
                'exposurePlaceCodes': stats,
                'leadTime': self.leadTimeLabel,
                'dynamicIndicator': indicator + '_affected',
                'adminLevel': self.admin_level
            }
            print('result: ', result)

            self.statsPath = PIPELINE_OUTPUT + 'calculated_affected/affected_' + \
                self.leadTimeLabel + '_' + self.countryCodeISO3 + '_' + indicator + '.json'
            print('self.statsPath: ', self.statsPath)

            with open(self.statsPath, 'w') as fp:
                json.dump(result, fp)

            # define alert_threshold layer
            alert_threshold = list(map(self.get_alert_threshold, stats))

            alert_threshold_file_path = PIPELINE_OUTPUT + 'calculated_affected/affected_' + \
                self.leadTimeLabel + '_' + self.countryCodeISO3 + '_' + 'alert_threshold' + '.json'

            alert_threshold_records = {
                'countryCodeISO3': self.countryCodeISO3,
                'exposurePlaceCodes': alert_threshold,
                'leadTime': self.leadTimeLabel,
                'dynamicIndicator': 'alert_threshold',
                'adminLevel': self.admin_level
            }

            with open(alert_threshold_file_path, 'w') as fp:
                json.dump(alert_threshold_records, fp)

    def get_population_affected_percentage(self, population_affected):
        population_total = next((x for x in self.population_total if x['placeCode'] == population_affected['placeCode']), None)
        population_affected_percentage = 0.0
        if population_total and population_total['value'] > 0:
            population_affected_percentage = population_affected['amount'] / population_total['value']
        return {
            'amount': population_affected_percentage,
            'placeCode': population_total['placeCode']
        }
    
    def get_alert_threshold(self, population_affected):
        alert_threshold = 0
        if (population_affected['amount'] > 0):
            alert_threshold = 1
        else:
            alert_threshold = 0
        return {
            'amount': alert_threshold,
            'placeCode': population_affected['placeCode']
        }

    def calcAffected(self, disasterExtentRaster, indicator, rasterValue):
        disasterExtentShapes = self.loadTiffAsShapes(disasterExtentRaster)
        if disasterExtentShapes != []:
            try:
                affectedImage, affectedMeta = self.clipTiffWithShapes(
                    self.inputRaster, disasterExtentShapes)
                with rasterio.open(self.outputRaster, "w", **affectedMeta) as dest:
                    dest.write(affectedImage)
            except ValueError:
                print('Rasters do not overlap')
        self.ADMIN_AREA_GDF.to_file(self.ADMIN_AREA_GDF_TMP_PATH)
        stats = self.calcStatsPerAdmin(
            indicator, disasterExtentShapes, rasterValue)

        return stats

    def calcStatsPerAdmin(self, indicator, disasterExtentShapes, rasterValue):
        stats = []
        with fiona.open(self.ADMIN_AREA_GDF_TMP_PATH, "r") as shapefile:

            # Clip affected raster per area
            for area in shapefile:
                if disasterExtentShapes != []:
                    try:
                        outImage, outMeta = self.clipTiffWithShapes(
                            self.outputRaster, [area["geometry"]])

                        # Write clipped raster to tempfile to calculate raster stats
                        with rasterio.open(self.outputPath, "w", **outMeta) as dest:
                            dest.write(outImage)

                        statsDistrict = self.calculateRasterStats(indicator,  str(
                            area['properties']['placeCode']), self.outputPath, rasterValue)

                        # # Overwrite non-triggered areas with positive exposure (due to rounding errors) to 0
                        # if self.countryCodeISO3 == 'UGA':
                        #     if 'EG' not in str(area['properties']['placeCode']):
                        #         statsDistrict = {'amount': 0, 'placeCode': str(
                        #             area['properties']['placeCode'])}
                    except (ValueError, rasterio.errors.RasterioIOError):
                        # If there is no disaster in the district set  the stats to 0
                        statsDistrict = {'amount': 0, 'placeCode': str(
                            area['properties']['placeCode'])}
                else:
                    statsDistrict = {'amount': 0, 'placeCode': str(
                        area['properties']['placeCode'])}
                stats.append(statsDistrict)
        os.remove(self.ADMIN_AREA_GDF_TMP_PATH)
        return stats

    def checkIfTriggeredArea(self, df_triggers, df_district_mapping, pcode):
        df_station_code = df_district_mapping[df_district_mapping['placeCode'] == pcode]
        if df_station_code.empty:
            return 0
        station_code = df_station_code['glofasStation'][0]
        if station_code == 'no_station':
            return 0
        df_trigger = df_triggers[df_triggers['stationCode'] == station_code]
        if df_trigger.empty:
            return 0
        trigger = df_trigger['fc_trigger'][0]
        return trigger

    def calculateRasterStats(self, indicator, district, outFileAffected, rasterValue):
        raster = rasterio.open(outFileAffected)
        stats = []

        array = raster.read(masked=True)
        band = array[0]
        theSum = band.sum() * rasterValue
        stats.append({
            'amount': float(str(theSum)),
            'placeCode': district
        })
        return stats[0]

    def loadTiffAsShapes(self, tiffLocaction):
        allgeom = []
        with rasterio.open(tiffLocaction) as dataset:
            # Read the dataset's valid data mask as a ndarray.
            image = dataset.read(1).astype(np.float32)
            mask = dataset.dataset_mask()
            theShapes = shapes(image, mask=mask, transform=dataset.transform)

            # Extract feature shapes and values from the array.
            for geom, val in theShapes:
                if val >= self.selectionValue:
                    # Transform shapes from the dataset's own coordinate
                    # reference system to CRS84 (EPSG:4326).
                    geom = rasterio.warp.transform_geom(
                        dataset.crs, 'EPSG:4326', geom, precision=6)
                    # Append everything to one geojson

                    allgeom.append(geom)
        return allgeom

    def clipTiffWithShapes(self, tiffLocaction, shapes):
        with rasterio.open(tiffLocaction) as src:
            outImage, out_transform = rasterio.mask.mask(
                src, shapes, crop=True)
            outMeta = src.meta.copy()

        outMeta.update({"driver": "GTiff",
                        "height": outImage.shape[1],
                        "width": outImage.shape[2],
                        "transform": out_transform})

        return outImage, outMeta
