import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from skyfield.api import load, Topos, Star
from skyfield.data import mpc
from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN
from swisseph import *

class AstroDataCollector:
    def __init__(self):
        # Initialize astronomy libraries
        self.planets = load('de421.bsp')
        self.earth = self.planets['earth']
        self.sun = self.planets['sun']
        self.moon = self.planets['moon']
        self.mercury = self.planets['mercury barycenter']
        self.venus = self.planets['venus barycenter']
        self.mars = self.planets['mars barycenter']
        self.jupiter = self.planets['jupiter barycenter']
        self.saturn = self.planets['saturn barycenter']
        self.uranus = self.planets['uranus barycenter']
        self.neptune = self.planets['neptune barycenter']
        self.pluto = self.planets['pluto barycenter']
        
        # Initialize ephemeris
        set_ephe_path('/path/to/ephemeris/files')  # Path to Swiss Ephemeris files
        
        # Connect to MetaTrader5
        if not mt5.initialize():
            print("MT5 initialization failed")
            mt5.shutdown()
    
    def get_planet_longitude(self, planet, time):
        """Calculate ecliptic longitude for a planet at a given time"""
        # Convert time to Julian date
        jd = time.tt
        
        # Use Swiss Ephemeris for precise calculations
        if planet == 'sun':
            lon, lat, distance = calc_ut(jd, SE_SUN, SEFLG_SPEED)
            return lon
        elif planet == 'moon':
            lon, lat, distance = calc_ut(jd, SE_MOON, SEFLG_SPEED)
            return lon
        # ... similar for other planets
    
    def is_retrograde(self, planet, time):
        """Check if a planet is in retrograde motion"""
        # Use Swiss Ephemeris to check speed
        jd = time.tt
        flags = SEFLG_SPEED
        
        planet_id = None
        if planet == 'mercury': planet_id = SE_MERCURY
        elif planet == 'venus': planet_id = SE_VENUS
        # ... map other planets
        
        if planet_id:
            lon, lat, distance = calc_ut(jd, planet_id, flags)
            # If longitude speed is negative, planet is retrograde
            return lon[3] < 0
        return False
    
    def calculate_aspects(self, time):
        """Calculate all major aspects between planets at given time"""
        aspects = {}
        planets = ['sun', 'moon', 'mercury', 'venus', 'mars', 
                   'jupiter', 'saturn', 'uranus', 'neptune', 'pluto']
        
        # Get all planet longitudes
        longitudes = {p: self.get_planet_longitude(p, time) for p in planets}
        
        # Calculate aspects between each planet pair
        for i, p1 in enumerate(planets):
            for p2 in planets[i+1:]:
                diff = abs(longitudes[p1] - longitudes[p2]) % 360
                
                # Define aspects with orbs
                if abs(diff - 0) < 8 or abs(diff - 360) < 8:
                    aspects[f"{p1}_conjunction_{p2}"] = 1
                elif abs(diff - 60) < 6:
                    aspects[f"{p1}_sextile_{p2}"] = 1
                elif abs(diff - 90) < 8:
                    aspects[f"{p1}_square_{p2}"] = 1
                elif abs(diff - 120) < 8:
                    aspects[f"{p1}_trine_{p2}"] = 1
                elif abs(diff - 180) < 10:
                    aspects[f"{p1}_opposition_{p2}"] = 1
        
        return aspects
    
    def get_lunar_phase(self, time):
        """Calculate lunar phase at given time"""
        sun_lon = self.get_planet_longitude('sun', time)
        moon_lon = self.get_planet_longitude('moon', time)
        
        phase_angle = (moon_lon - sun_lon) % 360
        
        # Determine lunar phase
        if 0 <= phase_angle < 45:
            return "new_moon"
        elif 45 <= phase_angle < 90:
            return "waxing_crescent"
        elif 90 <= phase_angle < 135:
            return "first_quarter"
        elif 135 <= phase_angle < 180:
            return "waxing_gibbous"
        elif 180 <= phase_angle < 225:
            return "full_moon"
        elif 225 <= phase_angle < 270:
            return "waning_gibbous"
        elif 270 <= phase_angle < 315:
            return "last_quarter"
        else:
            return "waning_crescent"
    
    def get_node_positions(self, time):
        """Get positions of lunar nodes (Rahu/Ketu)"""
        jd = time.tt
        
        # Calculate north node (Rahu)
        lon_rahu, lat, dist = calc_ut(jd, SE_TRUE_NODE, 0)
        
        # South node (Ketu) is always 180° opposite to north node
        lon_ketu = (lon_rahu + 180) % 360
        
        return lon_rahu, lon_ketu
    
    def get_xauusd_data(self, from_date, to_date, timeframe=mt5.TIMEFRAME_M15):
        """Get XAUUSD price data from MT5"""
        # Convert datetime to MT5 format
        from_date = mt5.datetime_to_timestamp(from_date)
        to_date = mt5.datetime_to_timestamp(to_date)
        
        # Get OHLC data
        rates = mt5.copy_rates_range("XAUUSD", timeframe, from_date, to_date)
        
        # Convert to DataFrame
        rates_df = pd.DataFrame(rates)
        rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')
        
        return rates_df
    
    def create_features_dataset(self, from_date, to_date):
        """Create complete feature dataset with price and astrological data"""
        # Get XAUUSD price data
        price_data = self.get_xauusd_data(from_date, to_date)
        
        # Create feature dataframe
        feature_data = []
        
        # Process each candle
        for idx, row in price_data.iterrows():
            candle_time = row['time']
            
            # Convert to skyfield time
            ts = load.timescale()
            t = ts.utc(candle_time.year, candle_time.month, candle_time.day, 
                       candle_time.hour, candle_time.minute, candle_time.second)
            
            # Get features
            features = {
                'time': candle_time,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['tick_volume'],
                # Planetary longitudes
                'sun_lon': self.get_planet_longitude('sun', t),
                'moon_lon': self.get_planet_longitude('moon', t),
                'mercury_lon': self.get_planet_longitude('mercury', t),
                'venus_lon': self.get_planet_longitude('venus', t),
                'mars_lon': self.get_planet_longitude('mars', t),
                'jupiter_lon': self.get_planet_longitude('jupiter', t),
                'saturn_lon': self.get_planet_longitude('saturn', t),
                'uranus_lon': self.get_planet_longitude('uranus', t),
                'neptune_lon': self.get_planet_longitude('neptune', t),
                'pluto_lon': self.get_planet_longitude('pluto', t),
                # Retrograde status
                'mercury_retrograde': 1 if self.is_retrograde('mercury', t) else 0,
                'venus_retrograde': 1 if self.is_retrograde('venus', t) else 0,
                'mars_retrograde': 1 if self.is_retrograde('mars', t) else 0,
                'jupiter_retrograde': 1 if self.is_retrograde('jupiter', t) else 0,
                'saturn_retrograde': 1 if self.is_retrograde('saturn', t) else 0,
                # Lunar phase
                'lunar_phase': self.get_lunar_phase(t),
                # Node positions
                'rahu_lon': self.get_node_positions(t)[0],
                'ketu_lon': self.get_node_positions(t)[1],
            }
            
            # Add aspects
            aspects = self.calculate_aspects(t)
            features.update(aspects)
            
            feature_data.append(features)
        
        return pd.DataFrame(feature_data)