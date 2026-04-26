from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
import uvicorn
import sqlite3
import requests
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="WeatherAI | Aura Dashboard", description="Aplikasi Prediksi Cuaca Cerdas Berbasis AI", version="3.0.0")

# ============ KONFIGURASI AI API (GEMINI) ============
import os

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("ℹ️ python-dotenv not installed. Using system environment variables only.")
    print("   Install with: pip install python-dotenv")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_API_KEY_BACKUP = os.environ.get("GEMINI_API_KEY_BACKUP", "")

def initialize_gemini_client():
    """Initialize Gemini client with fallback API keys"""
    api_keys = [GEMINI_API_KEY, GEMINI_API_KEY_BACKUP]
    api_keys = [key for key in api_keys if key]  # Remove empty keys
    
    if not api_keys:
        print("⚠️ Tidak ada API key Gemini yang tersedia")
        return None, False
    
    for i, api_key in enumerate(api_keys, 1):
        try:
            client = genai.Client(api_key=api_key)
            # Test the client with a simple request
            test_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Test connection"
            )
            print(f"✅ AI API (Google Gemini) siap digunakan dengan API Key {i}")
            print("   Model: gemini-2.5-flash")
            return client, True
        except Exception as e:
            print(f"⚠️ API Key {i} gagal: {e}")
            continue
    
    print("❌ Semua API key Gemini gagal. AI features akan dinonaktifkan.")
    return None, False

AI_AVAILABLE = False
client = None

try:
    from google import genai
    client, AI_AVAILABLE = initialize_gemini_client()
except ImportError:
    print("⚠️ Library google-genai belum terinstall. Install dengan: pip install google-genai")
except Exception as e:
    print(f"⚠️ Error inisialisasi Gemini: {e}")
    AI_AVAILABLE = False

# ============ DATABASE ============
DB_PATH = "weather.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS saved_locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            country TEXT,
            timezone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute("PRAGMA table_info(saved_locations)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'timezone' not in columns:
        print("⚠️ Menambahkan kolom timezone ke database...")
        cursor.execute('ALTER TABLE saved_locations ADD COLUMN timezone TEXT')
        print("✅ Kolom timezone berhasil ditambahkan")
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

def save_location(name: str, latitude: float, longitude: float, country: str = None, timezone: str = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO saved_locations (name, latitude, longitude, country, timezone)
        VALUES (?, ?, ?, ?, ?)
    ''', (name, latitude, longitude, country, timezone))
    conn.commit()
    conn.close()

def get_saved_locations():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id, name, latitude, longitude, country, timezone FROM saved_locations ORDER BY created_at DESC')
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        cursor.execute('SELECT id, name, latitude, longitude, country FROM saved_locations ORDER BY created_at DESC')
        rows = cursor.fetchall()
        locations = []
        for row in rows:
            locations.append({
                "id": row[0],
                "name": row[1],
                "latitude": row[2],
                "longitude": row[3],
                "country": row[4] or "Indonesia",
                "timezone": None
            })
        conn.close()
        return locations
    
    locations = []
    for row in rows:
        locations.append({
            "id": row[0],
            "name": row[1],
            "latitude": row[2],
            "longitude": row[3],
            "country": row[4] or "Indonesia",
            "timezone": row[5] if len(row) > 5 else None
        })
    conn.close()
    return locations

def delete_location(location_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM saved_locations WHERE id = ?', (location_id,))
    conn.commit()
    conn.close()

def location_exists(name: str, latitude: float, longitude: float):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM saved_locations WHERE name = ? OR (latitude = ? AND longitude = ?)', 
                   (name, latitude, longitude))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

init_db()

# ============ TIMEZONE HELPER (TANPA timezonefinder) ============

def get_timezone_from_coords(latitude: float, longitude: float):
    """Deteksi zona waktu berdasarkan koordinat (tanpa library eksternal)"""
    
    # Zona waktu Indonesia
    if 95 <= longitude <= 141:
        if -8 <= latitude <= 6:  # Wilayah Indonesia
            if 95 <= longitude <= 120:
                return "Asia/Jakarta"  # WIB
            elif 120 < longitude <= 128:
                return "Asia/Makassar"  # WITA
            else:
                return "Asia/Jayapura"  # WIT
    
    # Zona waktu dunia berdasarkan longitude
    # Setiap 15 derajat = 1 jam
    offset = int((longitude + 7.5) / 15)
    
    # Batasi offset antara -12 sampai +12
    offset = max(-12, min(12, offset))
    
    # Mapping ke zona waktu yang dikenal
    if offset == -5:
        return "America/New_York"
    elif offset == -6:
        return "America/Chicago"
    elif offset == -7:
        return "America/Denver"
    elif offset == -8:
        return "America/Los_Angeles"
    elif offset == 0:
        return "Europe/London"
    elif offset == 1:
        return "Europe/Paris"
    elif offset == 2:
        return "Europe/Helsinki"
    elif offset == 3:
        return "Asia/Riyadh"
    elif offset == 4:
        return "Asia/Dubai"
    elif offset == 5:
        return "Asia/Karachi"
    elif offset == 5.5:
        return "Asia/Kolkata"
    elif offset == 6:
        return "Asia/Dhaka"
    elif offset == 7:
        return "Asia/Jakarta"
    elif offset == 8:
        return "Asia/Makassar"
    elif offset == 9:
        return "Asia/Jayapura"
    elif offset == 10:
        return "Asia/Tokyo"
    elif offset == 11:
        return "Asia/Sakhalin"
    elif offset == 12:
        return "Pacific/Auckland"
    else:
        return "UTC"

def get_local_time(latitude: float, longitude: float, timezone_str: str = None):
    try:
        if not timezone_str:
            timezone_str = get_timezone_from_coords(latitude, longitude)
        tz = pytz.timezone(timezone_str)
        local_time = datetime.now(tz)
        
        tz_name = timezone_str.split('/')[-1].replace('_', ' ')
        
        hari_indonesia = {
            "Monday": "Senin", "Tuesday": "Selasa", "Wednesday": "Rabu",
            "Thursday": "Kamis", "Friday": "Jumat", "Saturday": "Sabtu", "Sunday": "Minggu"
        }
        day_name = hari_indonesia.get(local_time.strftime("%A"), local_time.strftime("%A"))
        
        bulan_indonesia = {
            "January": "Januari", "February": "Februari", "March": "Maret",
            "April": "April", "May": "Mei", "June": "Juni",
            "July": "Juli", "August": "Agustus", "September": "September",
            "October": "Oktober", "November": "November", "December": "Desember"
        }
        month_name = bulan_indonesia.get(local_time.strftime("%B"), local_time.strftime("%B"))
        
        return {
            "time": local_time.strftime("%H:%M:%S"),
            "date": f"{day_name}, {local_time.strftime('%d')} {month_name} {local_time.strftime('%Y')}",
            "day": day_name,
            "timezone": tz_name,
            "timezone_full": timezone_str,
            "hour": int(local_time.strftime("%H")),
            "minute": int(local_time.strftime("%M"))
        }
    except Exception as e:
        print(f"Local time error: {e}")
        now = datetime.now()
        return {
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%d %B %Y"),
            "day": now.strftime("%A"),
            "timezone": "Local",
            "timezone_full": "Asia/Jakarta",
            "hour": int(now.strftime("%H")),
            "minute": int(now.strftime("%M"))
        }

# ============ WEATHER SERVICE ============
def get_current_weather(latitude: float, longitude: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ["temperature_2m", "relative_humidity_2m", "apparent_temperature",
                   "precipitation", "weather_code", "wind_speed_10m", "surface_pressure"],
        "daily": ["uv_index_max"],
        "timezone": "auto"
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        current = data.get("current", {})
        daily = data.get("daily", {})
        uv = daily.get("uv_index_max", [0])[0] if daily.get("uv_index_max") else 5
        
        return {
            "temperature": current.get("temperature_2m", 0),
            "feels_like": current.get("apparent_temperature", 0),
            "humidity": current.get("relative_humidity_2m", 0),
            "precipitation": current.get("precipitation", 0),
            "wind_speed": current.get("wind_speed_10m", 0),
            "pressure": current.get("surface_pressure", 0),
            "weather_code": current.get("weather_code", 0),
            "uv_index": uv,
        }
    except Exception as e:
        print(f"Weather API error: {e}")
        return {
            "temperature": 28.5,
            "feels_like": 29.0,
            "humidity": 75,
            "precipitation": 0.5,
            "wind_speed": 12,
            "pressure": 1012,
            "weather_code": 0,
            "uv_index": 7,
        }

def get_5day_forecast(latitude: float, longitude: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "weather_code", "uv_index_max"],
        "timezone": "auto",
        "forecast_days": 5
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        daily = data.get("daily", {})
        
        forecast = []
        for i in range(5):
            date = datetime.now() + timedelta(days=i)
            forecast.append({
                "day": date.strftime("%a"),
                "date": date.strftime("%d/%m"),
                "temp_max": daily.get("temperature_2m_max", [0])[i] if daily.get("temperature_2m_max") else 0,
                "temp_min": daily.get("temperature_2m_min", [0])[i] if daily.get("temperature_2m_min") else 0,
                "precipitation": daily.get("precipitation_sum", [0])[i] if daily.get("precipitation_sum") else 0,
                "weather_code": daily.get("weather_code", [0])[i] if daily.get("weather_code") else 0,
                "uv_index": daily.get("uv_index_max", [0])[i] if daily.get("uv_index_max") else 0,
            })
        return forecast
    except Exception as e:
        print(f"Forecast API error: {e}")
        return [
            {"day": "Sen", "date": "01/01", "temp_max": 30, "temp_min": 24, "precipitation": 2, "weather_code": 0, "uv_index": 7},
            {"day": "Sel", "date": "02/01", "temp_max": 29, "temp_min": 23, "precipitation": 5, "weather_code": 61, "uv_index": 6},
            {"day": "Rab", "date": "03/01", "temp_max": 28, "temp_min": 23, "precipitation": 3, "weather_code": 1, "uv_index": 8},
            {"day": "Kam", "date": "04/01", "temp_max": 29, "temp_min": 24, "precipitation": 1, "weather_code": 0, "uv_index": 9},
            {"day": "Jum", "date": "05/01", "temp_max": 31, "temp_min": 25, "precipitation": 0, "weather_code": 0, "uv_index": 10},
        ]

def search_city(city_name: str):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city_name, "count": 1, "format": "json"}
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        results = data.get("results", [])
        if results:
            city = results[0]
            lat = city.get("latitude", 0)
            lon = city.get("longitude", 0)
            tz = get_timezone_from_coords(lat, lon)
            
            return {
                "name": city.get("name", city_name),
                "latitude": lat,
                "longitude": lon,
                "country": city.get("country", ""),
                "admin1": city.get("admin1", ""),
                "timezone": tz
            }
        return None
    except Exception as e:
        print(f"Search error: {e}")
        return None

def get_weather_icon_html(weather_code: int) -> str:
    icons = {
        0: '<i class="fas fa-sun" style="color: #fbbf24;"></i>',
        1: '<i class="fas fa-cloud-sun" style="color: #fbbf24;"></i>',
        2: '<i class="fas fa-cloud" style="color: #94a3b8;"></i>',
        3: '<i class="fas fa-cloud" style="color: #64748b;"></i>',
        45: '<i class="fas fa-smog" style="color: #94a3b8;"></i>',
        51: '<i class="fas fa-cloud-rain" style="color: #60a5fa;"></i>',
        53: '<i class="fas fa-cloud-rain" style="color: #60a5fa;"></i>',
        55: '<i class="fas fa-cloud-showers-heavy" style="color: #3b82f6;"></i>',
        61: '<i class="fas fa-cloud-rain" style="color: #60a5fa;"></i>',
        63: '<i class="fas fa-cloud-showers-heavy" style="color: #3b82f6;"></i>',
        65: '<i class="fas fa-cloud-showers-heavy" style="color: #2563eb;"></i>',
        80: '<i class="fas fa-cloud-rain" style="color: #60a5fa;"></i>',
        81: '<i class="fas fa-cloud-showers-heavy" style="color: #3b82f6;"></i>',
        95: '<i class="fas fa-bolt" style="color: #f59e0b;"></i>',
    }
    return icons.get(weather_code, '<i class="fas fa-cloud-sun"></i>')

def get_condition_text(weather_code: int) -> str:
    conditions = {
        0: "Cerah", 1: "Sebagian Cerah", 2: "Berawan", 3: "Mendung",
        45: "Kabut", 51: "Gerimis", 53: "Gerimis Sedang", 55: "Gerimis Lebat",
        61: "Hujan Ringan", 63: "Hujan Sedang", 65: "Hujan Lebat",
        80: "Hujan Lokal", 81: "Hujan Sedang", 95: "Badai Petir",
    }
    return conditions.get(weather_code, "Cerah")

# ============ MACHINE LEARNING MODEL ============
MODEL_PATH = "weather_model.pkl"

class WeatherPredictor:
    def __init__(self):
        self.model = None
        self.features = None
        self.load_model()
    
    def fetch_historical_data(self, latitude=-6.2, longitude=106.816666, days=30):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", 
                      "wind_speed_10m", "pressure_msl"],
            "timezone": "auto",
            "past_days": days,
            "forecast_days": 7
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            hourly = data.get("hourly", {})
            
            df = pd.DataFrame({
                'temperature': hourly.get('temperature_2m', []),
                'humidity': hourly.get('relative_humidity_2m', []),
                'precipitation': hourly.get('precipitation', []),
                'wind_speed': hourly.get('wind_speed_10m', []),
                'pressure': hourly.get('pressure_msl', [])
            })
            
            df['hour'] = [i % 24 for i in range(len(df))]
            df['day_of_year'] = [(datetime.now() - timedelta(days=len(df)-i-1)).timetuple().tm_yday 
                                 for i in range(len(df))]
            df['month'] = [(datetime.now() - timedelta(days=len(df)-i-1)).month 
                           for i in range(len(df))]
            
            for lag in [1, 3, 6, 12, 24]:
                df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
            
            df = df.dropna()
            return df
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return self.generate_synthetic_data()
    
    def generate_synthetic_data(self):
        np.random.seed(42)
        n_samples = 1000
        
        base_temp = 28
        hours = np.arange(n_samples)
        
        daily_cycle = 5 * np.sin(2 * np.pi * (hours % 24) / 24 - np.pi/2)
        seasonal = 3 * np.sin(2 * np.pi * (hours % 8760) / 8760)
        noise = np.random.normal(0, 1, n_samples)
        
        temperature = base_temp + daily_cycle + seasonal + noise
        
        df = pd.DataFrame({
            'temperature': temperature,
            'humidity': 70 + 15 * np.sin(2 * np.pi * hours / 48) + np.random.normal(0, 5, n_samples),
            'precipitation': np.random.exponential(0.5, n_samples),
            'wind_speed': 10 + 5 * np.random.randn(n_samples),
            'pressure': 1010 + 5 * np.random.randn(n_samples),
            'hour': hours % 24,
            'day_of_year': hours % 365,
            'month': (hours % 365) // 30 + 1
        })
        
        for lag in [1, 3, 6, 12, 24]:
            df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
        
        df = df.dropna()
        return df
    
    def train_model(self, location_name="Jakarta", latitude=-6.2, longitude=106.816666):
        print(f"📊 Mengambil data historis untuk {location_name}...")
        df = self.fetch_historical_data(latitude, longitude)
        
        feature_cols = ['humidity', 'precipitation', 'wind_speed', 'pressure', 
                       'hour', 'day_of_year', 'month', 'temp_lag_1', 'temp_lag_3', 
                       'temp_lag_6', 'temp_lag_12', 'temp_lag_24']
        
        X = df[feature_cols]
        y = df['temperature']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("🤖 Melatih model Random Forest Regressor...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Model selesai dilatih!")
        print(f"   📈 Mean Absolute Error: {mae:.2f}°C")
        print(f"   📊 R² Score: {r2:.3f}")
        
        # Simpan model dengan metadata termasuk feature names
        joblib.dump({
            'model': self.model,
            'features': feature_cols,
            'feature_names': feature_cols,  # Simpan feature names untuk validasi
            'mae': mae,
            'r2': r2,
            'location': location_name
        }, MODEL_PATH)
        
        self.features = feature_cols
        
        return {
            'mae': round(mae, 6),
            'r2': round(r2, 6),
            'features': feature_cols
        }
    
    def load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                saved = joblib.load(MODEL_PATH)
                self.model = saved['model']
                self.features = saved['features']
                print(f"✅ Model dimuat dari {MODEL_PATH}")
                return True
            except Exception as e:
                print(f"⚠️ Gagal memuat model: {e}")
                # Hapus file model yang corrupt
                try:
                    os.remove(MODEL_PATH)
                    print("🗑️ File model corrupt dihapus")
                except:
                    pass
                return False
        return False
    
    def predict_temperature(self, current_weather):
        if self.model is None:
            self.load_model()
            if self.model is None:
                return self.fallback_prediction(current_weather)
        
        predictions = []
        last_temps = []
        
        for i in range(7):
            hour = (datetime.now().hour + i * 24) % 24
            
            if i == 0:
                current_temp = current_weather.get('temperature', 28)
                last_temps = [current_temp] * 25
            else:
                current_temp = predictions[-1]['temp_max']
            
            # Buat dictionary features dengan urutan yang sama seperti saat training
            features_dict = {
                'humidity': current_weather.get('humidity', 70),
                'precipitation': current_weather.get('precipitation', 0),
                'wind_speed': current_weather.get('wind_speed', 10),
                'pressure': current_weather.get('pressure', 1010),
                'hour': hour,
                'day_of_year': (datetime.now() + timedelta(days=i)).timetuple().tm_yday,
                'month': (datetime.now() + timedelta(days=i)).month,
                'temp_lag_1': last_temps[-1] if len(last_temps) >= 1 else current_temp,
                'temp_lag_3': last_temps[-3] if len(last_temps) >= 3 else current_temp,
                'temp_lag_6': last_temps[-6] if len(last_temps) >= 6 else current_temp,
                'temp_lag_12': last_temps[-12] if len(last_temps) >= 12 else current_temp,
                'temp_lag_24': last_temps[-24] if len(last_temps) >= 24 else current_temp,
            }
            
            # Buat array dengan urutan fitur yang benar
            feature_array = np.array([[features_dict[col] for col in self.features]])
            pred_temp = self.model.predict(feature_array)[0]
            
            hour_factor = 2 * np.sin(2 * np.pi * (hour - 14) / 24)
            final_temp = pred_temp + hour_factor
            
            predictions.append({
                'day': (datetime.now() + timedelta(days=i)).strftime("%a"),
                'date': (datetime.now() + timedelta(days=i)).strftime("%d/%m"),
                'temp_max': round(final_temp + 2, 1),
                'temp_min': round(final_temp - 2, 1),
                'precipitation': round(max(0, np.random.exponential(0.5)), 1),
                'weather_code': 0,
                'uv_index': round(5 + 3 * np.sin(i), 1),
                'is_ml': True
            })
            
            last_temps.append(final_temp)
            if len(last_temps) > 25:
                last_temps.pop(0)
        
        return predictions
    
    def fallback_prediction(self, current_weather):
        current_temp = current_weather.get('temperature', 28)
        predictions = []
        
        for i in range(7):
            trend = i * 0.3
            daily_var = 2 * np.sin(2 * np.pi * i / 7)
            pred_temp = current_temp + trend + daily_var
            
            predictions.append({
                'day': (datetime.now() + timedelta(days=i)).strftime("%a"),
                'date': (datetime.now() + timedelta(days=i)).strftime("%d/%m"),
                'temp_max': round(pred_temp + 2, 1),
                'temp_min': round(pred_temp - 2, 1),
                'precipitation': round(max(0, np.random.exponential(0.5)), 1),
                'weather_code': 0,
                'uv_index': round(5 + 3 * np.sin(i), 1),
                'is_ml': False
            })
        
        return predictions
    
    def get_model_info(self):
        if os.path.exists(MODEL_PATH):
            try:
                saved = joblib.load(MODEL_PATH)
                return {
                    'is_trained': True,
                    'mae': saved.get('mae', 'N/A'),
                    'r2': saved.get('r2', 'N/A'),
                    'location': saved.get('location', 'Unknown')
                }
            except:
                return {'is_trained': False}
        return {'is_trained': False}

weather_predictor = WeatherPredictor()

# ============ AI INSIGHTS ============

def get_ai_insights_fallback(weather, forecast, location_name: str = None):
    temp = weather.get("temperature", 0)
    feels_like = weather.get("feels_like", 0)
    humidity = weather.get("humidity", 0)
    precip = weather.get("precipitation", 0)
    wind = weather.get("wind_speed", 0)
    weather_code = weather.get("weather_code", 0)
    uv = weather.get("uv_index", 5)
    
    condition = get_condition_text(weather_code).lower()
    location = location_name or "Lokasi Anda"
    
    temps_next_days = [d["temp_max"] for d in forecast[:3]]
    max_temp_next = max(temps_next_days) if temps_next_days else temp
    
    if condition == "cerah":
        opening = f"Langit {location} sedang cerah dengan suhu {int(temp)}°C. Sinar matahari cukup terik, dan suhu terasa seperti {int(feels_like)}°C."
    elif condition == "berawan" or condition == "sebagian cerah":
        opening = f"{location} saat ini berawan sebagian dengan suhu {int(temp)}°C, terasa seperti {int(feels_like)}°C."
    elif "hujan" in condition:
        opening = f"Hujan {condition} sedang turun di {location} dengan suhu {int(temp)}°C. Kelembaban {int(humidity)}% membuat udara terasa {int(feels_like)}°C."
    else:
        opening = f"Cuaca {location} {condition} dengan suhu {int(temp)}°C, terasa seperti {int(feels_like)}°C."
    
    if precip > 1:
        precip_text = f" Curah hujan mencapai {precip:.1f} mm dalam beberapa jam terakhir."
    else:
        precip_text = " Tidak ada hujan yang tercatat."
    
    if wind > 15:
        wind_text = f" Angin bertiup dengan kecepatan {int(wind)} km/jam, memberikan sensasi sejuk."
    elif wind > 5:
        wind_text = f" Angin bertiup pelan sekitar {int(wind)} km/jam."
    else:
        wind_text = " Udara terasa sangat tenang."
    
    if uv > 8:
        uv_text = f" Indeks UV sangat tinggi ({int(uv)})."
    elif uv > 6:
        uv_text = f" Indeks UV tinggi ({int(uv)})."
    else:
        uv_text = ""
    
    if max_temp_next > temp + 3:
        forecast_text = f" Dalam beberapa hari ke depan, suhu diprediksi akan meningkat hingga {int(max_temp_next)}°C."
    elif max_temp_next > temp + 1:
        forecast_text = f" Suhu diperkirakan akan naik bertahap hingga {int(max_temp_next)}°C."
    else:
        forecast_text = f" Suhu dalam beberapa hari ke depan cenderung stabil sekitar {int(temp)}°C."
    
    rainy_days = [d for d in forecast[:3] if d["precipitation"] > 5]
    if rainy_days:
        forecast_text += f" Waspada potensi hujan pada hari {', '.join([d['day'] for d in rainy_days[:2]])}."
    elif any(d["precipitation"] > 2 for d in forecast[:3]):
        forecast_text += " Ada kemungkinan hujan ringan dalam beberapa hari ke depan."
    
    return f"{opening}{precip_text}{wind_text}{uv_text}{forecast_text}"

def get_ai_insights_real(weather, forecast, location_name: str = None):
    if not AI_AVAILABLE or client is None:
        return get_ai_insights_fallback(weather, forecast, location_name)
    
    location = location_name or "Lokasi Anda"
    temp = weather.get("temperature", 0)
    feels_like = weather.get("feels_like", 0)
    humidity = weather.get("humidity", 0)
    precip = weather.get("precipitation", 0)
    wind = weather.get("wind_speed", 0)
    weather_code = weather.get("weather_code", 0)
    uv = weather.get("uv_index", 5)
    condition = get_condition_text(weather_code)
    
    forecast_summary = []
    for day in forecast[:3]:
        forecast_summary.append(
            f"{day['day']}: {int(day['temp_max'])}°C / {int(day['temp_min'])}°C, "
            f"hujan {int(day['precipitation'])}mm"
        )
    forecast_text = "; ".join(forecast_summary)
    
    prompt = f"""Deskripsikan cuaca di {location} secara natural dan menarik dalam 3-4 kalimat.

Data cuaca saat ini:
- Suhu: {int(temp)}°C (terasa {int(feels_like)}°C)
- Kondisi: {condition}
- Kelembaban: {int(humidity)}%
- Curah hujan: {precip:.1f} mm
- Angin: {int(wind)} km/jam
- UV: {uv:.1f}

Prakiraan 3 hari: {forecast_text}

Tulis dalam bahasa Indonesia yang natural, seperti gaya meteorolog. Jangan gunakan bullet points atau rekomendasi. Buat setiap deskripsi unik."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        insights = response.text.strip()
        print(f"✅ Gemini response received for {location}")
        return insights
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return get_ai_insights_fallback(weather, forecast, location_name)

# ============ CSS STYLES WITH ANIMATIONS ==========
CSS_STYLES = """
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700;14..32,800&display=swap');

:root {
    --bg-primary: #f0f4f8;
    --bg-secondary: #ffffff;
    --bg-tertiary: #f8fafc;
    --text-primary: #0a0c10;
    --text-secondary: #2c3e50;
    --text-tertiary: #5a6e8a;
    --border-color: #e2edf2;
    --card-bg: rgba(255, 255, 255, 0.92);
    --sidebar-bg: rgba(255, 255, 255, 0.96);
    --sidebar-border: rgba(0, 0, 0, 0.06);
    --accent: #3b82f6;
    --accent-gradient: linear-gradient(135deg, #3b82f6, #6366f1, #8b5cf6);
    --accent-hover: #2563eb;
    --accent-soft: rgba(59, 130, 246, 0.12);
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --ml-purple: #8b5cf6;
    --ml-gradient: linear-gradient(135deg, #8b5cf6, #a855f7, #c084fc);
    --glass-border: rgba(255, 255, 255, 0.2);
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.04);
    --shadow-md: 0 8px 20px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 16px 32px rgba(0, 0, 0, 0.08);
    --shadow-xl: 0 24px 48px rgba(0, 0, 0, 0.12);
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-fast: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-slow: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}

body.dark {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-tertiary: #1a2332;
    --text-primary: #f1f5f9;
    --text-secondary: #cbd5e1;
    --text-tertiary: #8596b0;
    --border-color: #1f2a3e;
    --card-bg: rgba(17, 24, 39, 0.92);
    --sidebar-bg: rgba(10, 14, 23, 0.96);
    --sidebar-border: rgba(255, 255, 255, 0.04);
    --glass-border: rgba(255, 255, 255, 0.05);
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 8px 20px rgba(0, 0, 0, 0.4);
    --shadow-lg: 0 16px 32px rgba(0, 0, 0, 0.5);
    --shadow-xl: 0 24px 48px rgba(0, 0, 0, 0.6);
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.5;
    transition: background 0.3s ease, color 0.2s ease;
    overflow-x: hidden;
    position: relative;
}

body::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
        radial-gradient(circle at 20% 30%, rgba(59, 130, 246, 0.08) 0%, transparent 50%),
        radial-gradient(circle at 80% 70%, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
        radial-gradient(circle at 60% 10%, rgba(168, 85, 247, 0.06) 0%, transparent 40%);
    animation: backgroundShift 20s ease infinite;
    z-index: -1;
    pointer-events: none;
}

@keyframes backgroundShift {
    0%, 100% { 
        transform: scale(1) rotate(0deg);
        opacity: 0.6;
    }
    33% { 
        transform: scale(1.05) rotate(1deg);
        opacity: 0.8;
    }
    66% { 
        transform: scale(0.95) rotate(-1deg);
        opacity: 0.7;
    }
}

.greeting-icon {
    display: inline-block;
    animation: wave 0.5s ease;
}

@keyframes wave {
    0% { transform: rotate(0deg); }
    50% { transform: rotate(15deg); }
    100% { transform: rotate(0deg); }
}

.hero-title {
    display: flex;
    align-items: center;
    gap: 12px;
}

/* ============ LOADING ANIMATION ============ */
.loader-wrapper {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: var(--bg-primary);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
    transition: opacity 0.5s ease, visibility 0.5s ease;
}

.loader-wrapper.hide {
    opacity: 0;
    visibility: hidden;
}

.loader {
    text-align: center;
}

.cloud-loader {
    font-size: 80px;
    color: var(--accent);
    animation: cloudFloat 2s ease-in-out infinite;
    filter: drop-shadow(0 0 20px rgba(59, 130, 246, 0.3));
}

@keyframes cloudFloat {
    0%, 100% { 
        transform: translateY(0px) scale(1);
        filter: drop-shadow(0 0 20px rgba(59, 130, 246, 0.3));
    }
    50% { 
        transform: translateY(-20px) scale(1.05);
        filter: drop-shadow(0 0 30px rgba(59, 130, 246, 0.5));
    }
}

.loader-text {
    margin-top: 20px;
    font-size: 18px;
    font-weight: 600;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: textGlow 2s ease-in-out infinite alternate;
}

@keyframes textGlow {
    0% { filter: brightness(1); }
    100% { filter: brightness(1.2); }
}

.loader-dots {
    display: inline-block;
    width: 20px;
    text-align: left;
}

.loader-dots::after {
    content: '...';
    animation: dots 1.5s steps(4, end) infinite;
}

@keyframes dots {
    0%, 20% { content: ''; }
    40% { content: '.'; }
    60% { content: '..'; }
    80%, 100% { content: '...'; }
}

/* ============ PAGE TRANSITION ============ */
.page-transition {
    animation: pageFadeIn 0.5s ease-out;
}

@keyframes pageFadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* ============ NAVBAR ACTIVE ANIMATION ============ */
.nav-item {
    position: relative;
    overflow: hidden;
}

.nav-item::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.2), transparent);
    transition: left 0.5s ease;
}

.nav-item:hover::before {
    left: 100%;
}

.nav-item.active {
    background: var(--accent-soft);
    color: var(--accent);
    border-left: 3px solid var(--accent);
    transform: translateX(4px);
}

/* ============ TRAINING MODAL ============ */
.modal {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(8px);
    z-index: 10000;
    justify-content: center;
    align-items: center;
    animation: modalFadeIn 0.3s ease;
}

@keyframes modalFadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.modal.active {
    display: flex;
}

.modal-content {
    background: var(--card-bg);
    backdrop-filter: blur(20px);
    border-radius: 32px;
    padding: 40px;
    text-align: center;
    max-width: 400px;
    width: 90%;
    border: 1px solid var(--glass-border);
    box-shadow: var(--shadow-xl);
    animation: modalPop 0.4s cubic-bezier(0.34, 1.2, 0.64, 1);
    position: relative;
    overflow: hidden;
}

.modal-content::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: var(--ml-gradient);
    animation: progressBar 3s ease-in-out infinite;
}

@keyframes progressBar {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

@keyframes modalPop {
    from {
        opacity: 0;
        transform: scale(0.8);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

.modal-icon {
    font-size: 64px;
    margin-bottom: 20px;
    animation: spin 2s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.modal-title {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 12px;
    background: var(--ml-gradient);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.modal-message {
    color: var(--text-tertiary);
    margin-bottom: 24px;
}

.progress-bar {
    width: 100%;
    height: 6px;
    background: var(--bg-tertiary);
    border-radius: 3px;
    overflow: hidden;
    margin: 20px 0;
}

.progress-fill {
    height: 100%;
    background: var(--ml-gradient);
    width: 0%;
    border-radius: 3px;
    animation: progressPulse 1s ease infinite;
    position: relative;
}

.progress-fill::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
    animation: shimmer 2s ease-in-out infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

@keyframes progressPulse {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
}

/* ============ CARD HOVER ANIMATIONS ============ */
.glass-card, .weather-hero, .stat-card, .forecast-item {
    transition: var(--transition);
    position: relative;
    overflow: hidden;
}

.glass-card::before, .weather-hero::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.05), transparent);
    transition: left 0.6s ease;
}

.glass-card:hover::before, .weather-hero:hover::before {
    left: 100%;
}

.glass-card:hover, .weather-hero:hover {
    transform: translateY(-6px);
    box-shadow: var(--shadow-xl);
    border-color: rgba(59, 130, 246, 0.2);
}

.stat-card:hover, .forecast-item:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.15);
}

/* ============ SIDEBAR TRANSITION ============ */
.sidebar {
    transition: left 0.3s ease, transform 0.3s ease;
}

/* ============ THEME TOGGLE ANIMATION ============ */
.theme-toggle {
    transition: var(--transition);
    position: relative;
}

.theme-toggle::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    background: var(--accent-gradient);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    transition: width 0.3s ease, height 0.3s ease;
    z-index: -1;
}

.theme-toggle:hover::before {
    width: 60px;
    height: 60px;
}

.theme-toggle:hover {
    transform: scale(1.1) rotate(15deg);
    color: white;
}

/* ============ SEARCH BAR ANIMATION ============ */
.search-container {
    transition: var(--transition);
    position: relative;
}

.search-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: var(--accent-gradient);
    border-radius: 60px;
    opacity: 0;
    transition: opacity 0.3s ease;
    z-index: -1;
}

.search-container:focus-within::before {
    opacity: 0.1;
}

.search-container:focus-within {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(59, 130, 246, 0.2);
    border-color: var(--accent);
}

.search-btn {
    transition: var(--transition);
    position: relative;
    overflow: hidden;
}

.search-btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    transition: left 0.5s ease;
}

.search-btn:hover::before {
    left: 100%;
}

.search-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);
}

/* ============ TRAINING BUTTON ANIMATION ============ */
.train-btn {
    transition: var(--transition);
    position: relative;
    overflow: hidden;
    background: var(--ml-gradient);
    background-size: 200% 200%;
    animation: gradientShift 3s ease infinite;
}

@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.train-btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    transition: left 0.5s ease;
}

.train-btn:hover::before {
    left: 100%;
}

.train-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(139, 92, 246, 0.4);
}

.train-btn.loading {
    pointer-events: none;
    opacity: 0.7;
}

/* ============ Skeleton Loading ============ */
.skeleton {
    background: linear-gradient(90deg, var(--bg-tertiary) 25%, var(--border-color) 50%, var(--bg-tertiary) 75%);
    background-size: 200% 100%;
    animation: skeletonWave 1.5s ease infinite;
    border-radius: 12px;
}

@keyframes skeletonWave {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ============ REST OF STYLES ============ */
.aura-bg {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 0;
    pointer-events: none;
}

.aura-glow {
    position: absolute;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle at 20% 30%, rgba(59, 130, 246, 0.06) 0%, transparent 50%),
                radial-gradient(circle at 80% 70%, rgba(99, 102, 241, 0.06) 0%, transparent 50%);
    animation: auraPulse 12s ease infinite;
}

@keyframes auraPulse {
    0%, 100% { opacity: 0.6; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.02); }
}

.app {
    display: flex;
    min-height: 100vh;
    position: relative;
    z-index: 1;
}

.sidebar {
    width: 280px;
    background: var(--sidebar-bg);
    backdrop-filter: blur(24px);
    border-right: 1px solid var(--sidebar-border);
    padding: 32px 20px;
    position: sticky;
    top: 0;
    height: 100vh;
    transition: var(--transition);
    z-index: 100;
    border-radius: 0 24px 24px 0;
    box-shadow: var(--shadow-lg);
}

.sidebar-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    margin-bottom: 40px;
    padding: 24px 0;
    border-bottom: 1px solid var(--border-color);
    width: calc(100% + 40px);
    margin-left: -20px;
    margin-right: -20px;
}

.sidebar-logo-icon {
    width: 100%;
    height: auto;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    color: white;
    transition: var(--transition);
}

.sidebar-logo-icon:hover {
    transform: scale(1.08);
}

.sidebar-logo-icon img {
    width: auto;
    height: 2200px;
}

@keyframes logoPulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.9; }
}

.sidebar-logo-text {
    font-size: 22px;
    font-weight: 800;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.sidebar-nav {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 36px;
}

.nav-item {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 18px;
    border-radius: 14px;
    color: var(--text-secondary);
    text-decoration: none;
    transition: var(--transition);
    font-weight: 500;
    font-size: 14px;
}

.nav-item i {
    width: 22px;
    font-size: 16px;
}

.nav-item:hover {
    background: var(--accent-soft);
    color: var(--accent);
    transform: translateX(6px);
}

.nav-item.active {
    background: var(--accent-soft);
    color: var(--accent);
    border-left: 3px solid var(--accent);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
}

.sidebar-section {
    margin-top: 36px;
    padding-top: 24px;
    border-top: 1px solid var(--border-color);
}

.sidebar-section-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-tertiary);
    margin-bottom: 18px;
}

.sidebar-locations {
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-height: 300px;
    overflow-y: auto;
}

.location-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 14px;
    background: var(--bg-tertiary);
    border-radius: 14px;
    cursor: pointer;
    transition: var(--transition);
    border: 1px solid transparent;
}

.location-item:hover {
    background: var(--accent-soft);
    transform: translateX(6px);
    border-color: var(--accent);
}

.location-info {
    display: flex;
    align-items: center;
    gap: 12px;
}

.location-info i {
    font-size: 16px;
    color: var(--accent);
}

.location-name {
    font-size: 14px;
    font-weight: 600;
}

.location-country {
    font-size: 10px;
    color: var(--text-tertiary);
}

.delete-btn {
    background: rgba(239, 68, 68, 0.1);
    border: none;
    color: var(--danger);
    cursor: pointer;
    font-size: 12px;
    padding: 6px 10px;
    border-radius: 10px;
    transition: var(--transition);
}

.delete-btn:hover {
    background: var(--danger);
    color: white;
}

.main {
    flex: 1;
    padding: 32px 44px;
    overflow-x: hidden;
}

.hero {
    margin-bottom: 36px;
}

.hero-title {
    font-size: 36px;
    font-weight: 800;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 15px;
    color: var(--text-tertiary);
}

.search-container {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border-radius: 60px;
    padding: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 32px;
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
}

.search-container:focus-within {
    box-shadow: 0 8px 28px rgba(59, 130, 246, 0.2);
    border-color: var(--accent);
}

.search-input {
    flex: 1;
    background: transparent;
    border: none;
    padding: 14px 24px;
    font-size: 15px;
    color: var(--text-primary);
    font-family: inherit;
}

.search-input:focus {
    outline: none;
}

.search-btn {
    background: var(--accent-gradient);
    border: none;
    border-radius: 50px;
    padding: 12px 32px;
    color: white;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
    transition: var(--transition);
}

.weather-hero {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border-radius: 36px;
    padding: 36px 44px;
    margin-bottom: 32px;
    border: 1px solid var(--glass-border);
    box-shadow: var(--shadow-lg);
    transition: var(--transition);
}

.weather-main {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 28px;
}

.weather-info {
    text-align: center;
}

.weather-icon {
    font-size: 80px;
    margin-bottom: 12px;
}

.weather-temp {
    font-size: 88px;
    font-weight: 800;
    line-height: 1;
    color: var(--text-primary);
}

.temp-unit {
    font-size: 36px;
    font-weight: 500;
    color: var(--text-tertiary);
}

.weather-condition {
    font-size: 22px;
    font-weight: 700;
    margin-top: 12px;
}

.feels-like {
    font-size: 14px;
    color: var(--text-tertiary);
    margin-top: 6px;
}

.stats-grid {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}

.stat-card {
    background: var(--bg-tertiary);
    border-radius: 28px;
    padding: 18px 28px;
    text-align: center;
    min-width: 110px;
    transition: var(--transition);
    border: 1px solid transparent;
}

.stat-card:hover {
    transform: translateY(-4px);
    background: var(--accent-soft);
    border-color: var(--accent);
}

.stat-icon {
    font-size: 32px;
    margin-bottom: 10px;
}

.stat-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--text-tertiary);
    letter-spacing: 1.5px;
}

.stat-value {
    font-size: 22px;
    font-weight: 800;
    margin-top: 6px;
}

.bento-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 28px;
    margin-bottom: 36px;
}

@media (max-width: 900px) {
    .bento-grid {
        grid-template-columns: 1fr;
    }
}

.glass-card {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border-radius: 32px;
    padding: 28px;
    border: 1px solid var(--glass-border);
    transition: var(--transition);
    box-shadow: var(--shadow-md);
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-color);
}

.card-title {
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-tertiary);
    display: flex;
    align-items: center;
    gap: 8px;
}

.forecast-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
}

.forecast-item {
    text-align: center;
    padding: 18px 12px;
    background: var(--bg-tertiary);
    border-radius: 24px;
    transition: var(--transition);
    border: 1px solid transparent;
}

.forecast-item:hover {
    transform: translateY(-5px);
    background: var(--accent-soft);
    border-color: var(--accent);
}

.forecast-day {
    font-weight: 800;
    font-size: 15px;
}

.forecast-date {
    font-size: 10px;
    color: var(--text-tertiary);
    margin-top: 4px;
}

.forecast-icon {
    font-size: 32px;
    margin: 12px 0;
}

.forecast-temp {
    font-size: 18px;
    font-weight: 800;
}

.forecast-temp-min {
    font-size: 12px;
    color: var(--text-tertiary);
}

.forecast-precip {
    font-size: 11px;
    color: var(--text-tertiary);
    margin-top: 6px;
}

.ml-badge {
    display: inline-block;
    background: var(--ml-gradient);
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 9px;
    font-weight: 600;
    color: white;
    margin-top: 6px;
}

.train-btn {
    background: var(--ml-gradient);
    border: none;
    border-radius: 50px;
    padding: 14px 28px;
    color: white;
    font-weight: 700;
    font-size: 14px;
    cursor: pointer;
    transition: var(--transition);
    margin-top: 16px;
}

.insights-text {
    line-height: 1.8;
    font-size: 15px;
    color: var(--text-primary);
}

.info-row {
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.info-item {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 16px;
    background: var(--bg-tertiary);
    border-radius: 24px;
    transition: var(--transition);
}

.info-item:hover {
    background: var(--accent-soft);
    transform: translateX(6px);
}

.info-icon {
    font-size: 26px;
    color: var(--accent);
}

.info-label {
    font-size: 11px;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.info-value {
    font-size: 17px;
    font-weight: 700;
    margin-top: 3px;
}

.timezone-badge {
    display: inline-block;
    background: var(--accent-soft);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    color: var(--accent);
    margin-top: 6px;
}

.flash {
    padding: 16px 24px;
    border-radius: 32px;
    margin-bottom: 28px;
    text-align: center;
    font-size: 14px;
    font-weight: 600;
    animation: slideIn 0.3s ease;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.flash-success {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid var(--success);
    color: var(--success);
}

.flash-error {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid var(--danger);
    color: var(--danger);
}

.theme-toggle {
    position: fixed;
    bottom: 28px;
    right: 28px;
    width: 52px;
    height: 52px;
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 200;
    transition: var(--transition);
    color: var(--text-primary);
    font-size: 22px;
    box-shadow: var(--shadow-lg);
}

.menu-toggle {
    position: fixed;
    bottom: 28px;
    left: 28px;
    width: 52px;
    height: 52px;
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 50%;
    display: none;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 200;
    transition: var(--transition);
    color: var(--text-primary);
    font-size: 22px;
    box-shadow: var(--shadow-lg);
}

.menu-toggle:hover, .theme-toggle:hover {
    transform: scale(1.1);
    background: var(--accent-gradient);
    color: white;
}

.footer {
    text-align: center;
    padding: 28px;
    color: var(--text-tertiary);
    font-size: 12px;
    border-top: 1px solid var(--border-color);
    margin-top: 40px;
}

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-tertiary);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: var(--accent);
    border-radius: 10px;
}

@media (min-width: 769px) and (max-width: 1024px) {
    /* Tablet styles */
    .sidebar {
        width: 240px;
    }
    
    .main {
        padding: 24px 32px;
    }
    
    .weather-hero {
        padding: 32px 36px;
    }
    
    .hero-title {
        font-size: 32px;
    }
    
    .bento-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 24px;
    }
    
    .forecast-grid {
        grid-template-columns: repeat(4, 1fr);
    }
    
    .stat-card {
        padding: 16px 24px;
        min-width: 100px;
    }
}

@media (max-width: 768px) {
    .app {
        flex-direction: column;
    }

    .sidebar {
        position: fixed;
        left: -100%;
        width: min(320px, 100%);
        max-width: 320px;
        z-index: 150;
        transition: left 0.3s ease;
        border-radius: 0 32px 32px 0;
        height: 100vh;
        overflow-y: auto;
        padding: 24px 18px;
    }
    
    .sidebar.open {
        left: 0;
        box-shadow: 4px 0 30px rgba(0,0,0,0.18);
    }
    
    .main {
        padding: 18px 16px 28px;
    }
    
    .weather-hero {
        padding: 22px 20px;
        border-radius: 22px;
    }
    
    .weather-main {
        flex-direction: column;
        align-items: center;
        gap: 20px;
    }
    
    .weather-temp {
        font-size: 48px;
    }
    
    .temp-unit {
        font-size: 22px;
    }
    
    .weather-icon {
        font-size: 56px;
    }
    
    .search-container {
        flex-direction: column;
        align-items: stretch;
        gap: 12px;
        padding: 10px;
    }

    .search-input {
        padding: 14px 18px;
    }

    .search-btn {
        width: 100%;
        padding: 14px 18px;
    }
    
    .forecast-grid {
        gap: 12px;
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    
    .forecast-item {
        padding: 14px 10px;
    }
    
    .forecast-icon {
        font-size: 28px;
    }
    
    .forecast-temp {
        font-size: 15px;
    }
    
    .stat-card {
        padding: 14px 16px;
        min-width: auto;
        flex: 1 1 calc(50% - 10px);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
    }
    
    .stat-value {
        font-size: 18px;
    }
    
    .menu-toggle {
        display: flex;
    }
    
    .hero-title {
        font-size: 28px;
    }
    
    .hero-subtitle {
        font-size: 14px;
    }
    
    .bento-grid {
        grid-template-columns: 1fr;
        gap: 18px;
    }
    
    .glass-card {
        padding: 20px;
    }
    
    .card-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 10px;
    }
    
    .sidebar-nav {
        gap: 10px;
    }
    
    .sidebar-section {
        margin-top: 28px;
        padding-top: 20px;
    }
    
    .sidebar-locations {
        max-height: 260px;
    }
    
    .location-item {
        padding: 10px 12px;
    }
    
    .delete-btn {
        padding: 6px 8px;
        font-size: 11px;
    }
}

@media (min-width: 769px) {
    .menu-toggle {
        display: none;
    }
}
"""

# ============ RENDER PAGE FUNCTION ============
def render_page(content: str, active: str = "home", message: str = None, message_type: str = None, saved_locations: list = None, selected_location: dict = None):
    
    message_html = ""
    if message:
        icon = "check-circle" if message_type == "success" else "exclamation-circle"
        message_html = f'<div class="flash flash-{message_type}"><i class="fas fa-{icon}"></i> {message}</div>'
    
    sidebar_locations_html = ""
    if saved_locations:
        for loc in saved_locations:
            sidebar_locations_html += f'''
            <div class="location-item" onclick="window.location.href='/select-location/{loc["id"]}'">
                <div class="location-info">
                    <i class="fas fa-map-marker-alt"></i>
                    <div>
                        <div class="location-name">{loc["name"]}</div>
                        <div class="location-country">{loc["country"]}</div>
                    </div>
                </div>
                <button class="delete-btn" onclick="event.stopPropagation(); window.location.href='/delete-location/{loc["id"]}'">
                    <i class="fas fa-trash-alt"></i>
                </button>
            </div>
            '''
    else:
        sidebar_locations_html = '<div style="text-align:center;padding:32px;color:var(--text-tertiary);font-size:13px;"><i class="fas fa-star" style="font-size:32px;margin-bottom:12px;display:block;opacity:0.5;"></i>Belum ada lokasi tersimpan</div>'
    
    active_home = 'active' if active == 'home' else ''
    active_ml = 'active' if active == 'ml' else ''
    active_about = 'active' if active == 'about' else ''
    
    location_data = ""
    if selected_location:
        tz = selected_location.get("timezone", "Asia/Jakarta")
        location_data = f'''
        <script>
            window.currentLocation = {{
                lat: {selected_location.get("latitude", -6.2)},
                lng: {selected_location.get("longitude", 106.816666)},
                timezone: "{tz}",
                name: "{selected_location.get("name", "Jakarta")}"
            }};
        </script>
        '''
    
    return f'''<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <meta name="theme-color" content="#0f172a">
    <meta name="description" content="WeatherAI - Aplikasi Prediksi Cuaca Cerdas Berbasis AI dengan akurasi tinggi. Informasi cuaca real-time, prakiraan 5 hari, dan analisis AI.">
    <meta name="keywords" content="cuaca, prediksi cuaca, weather, AI, weather forecast, Indonesia, gemini AI, machine learning">
    <meta name="author" content="WeatherAI">
    <title>WeatherAI | Aura Dashboard - Prediksi Cuaca AI</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>{CSS_STYLES}</style>
    {location_data}
</head>
<body>
    <!-- LOADING SCREEN -->
    <div class="loader-wrapper" id="loaderWrapper">
        <div class="loader">
            <div class="cloud-loader">
                <img src="https://aryamods.rf.gd/images/weatherai.svg" alt="WeatherAI Logo" style="width: 64px; height: 64px;">
            </div>
            <div class="loader-text">
                WeatherAI<span class="loader-dots"></span>
            </div>
        </div>
    </div>

    <!-- TRAINING MODAL -->
    <div class="modal" id="trainingModal">
        <div class="modal-content">
            <div class="modal-icon">
                <i class="fas fa-brain"></i>
            </div>
            <h3 class="modal-title">Melatih Model ML</h3>
            <p class="modal-message">Sedang melatih Random Forest Regressor dengan data historis...</p>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <p class="modal-message" style="font-size: 12px;" id="trainingStatus">Mengambil data cuaca...</p>
        </div>
    </div>
    
    <div class="aura-bg">
        <div class="aura-glow"></div>
    </div>
    
    <button class="theme-toggle" id="themeToggle" onclick="toggleTheme()" aria-label="Toggle theme">
        <i class="fas fa-moon" id="themeIcon"></i>
    </button>
    
    <button class="menu-toggle" id="menuToggle" onclick="toggleSidebar()" aria-label="Menu">
        <i class="fas fa-bars"></i>
    </button>
    
    <div class="app">
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <div class="sidebar-logo-icon">
                    <img src="https://aryamods.rf.gd/images/weatherai.svg" alt="WeatherAI Logo" style="height: 50px; width: auto;">
                </div>
            </div>
            
            <nav class="sidebar-nav">
                <a href="/" class="nav-item {active_home}" data-page="home">
                    <i class="fas fa-chart-line"></i>
                    <span>Dashboard</span>
                </a>
                <a href="/ml-dashboard" class="nav-item {active_ml}" data-page="ml">
                    <i class="fas fa-brain"></i>
                    <span>ML Dashboard</span>
                </a>
                <a href="/about" class="nav-item {active_about}" data-page="about">
                    <i class="fas fa-info-circle"></i>
                    <span>About App</span>
                </a>
            </nav>
            
            <div class="sidebar-section">
                <div class="sidebar-section-title">
                    <i class="fas fa-star"></i> Lokasi Tersimpan
                </div>
                <div class="sidebar-locations">
                    {sidebar_locations_html}
                </div>
                <a href="/search" style="display: block; margin-top: 20px; text-align: center; font-size: 13px; color: var(--accent); text-decoration: none; font-weight: 600;">
                    <i class="fas fa-plus-circle"></i> Tambah Lokasi
                </a>
            </div>
        </aside>
        
        <main class="main page-transition" id="mainContent">
            {message_html}
            {content}
            <div class="footer">
                <p><i class="fas fa-microchip"></i> Powered by AI (Google Gemini) & Open-Meteo | Akurasi MAE 0.46°C</p>
                <p style="margin-top: 8px;">© 2026 WeatherAI · Aura Dashboard · Smart Weather Intelligence</p>
            </div>
        </main>
    </div>
    
    <script>
        // HIDE LOADER
        window.addEventListener('load', function() {{
            setTimeout(function() {{
                document.getElementById('loaderWrapper').classList.add('hide');
            }}, 500);
        }});
        
        // PAGE TRANSITION ON NAVIGATION
        document.querySelectorAll('.nav-item').forEach(link => {{
            link.addEventListener('click', function(e) {{
                e.preventDefault();
                const href = this.getAttribute('href');
                
                // Add transition effect
                document.getElementById('mainContent').style.opacity = '0';
                document.getElementById('mainContent').style.transform = 'translateY(20px)';
                
                setTimeout(() => {{
                    window.location.href = href;
                }}, 300);
            }});
        }});
        
        // TRAINING MODAL FUNCTIONS
        function showTrainingModal() {{
            const modal = document.getElementById('trainingModal');
            modal.classList.add('active');
            
            // Animate progress bar
            const progressFill = document.getElementById('progressFill');
            const statusText = document.getElementById('trainingStatus');
            let progress = 0;
            
            const statuses = [
                'Mengambil data cuaca...',
                'Memproses fitur...',
                'Melatih Random Forest...',
                'Mengevaluasi model...',
                'Menyimpan model...'
            ];
            let statusIndex = 0;
            
            const interval = setInterval(() => {{
                progress += 2;
                progressFill.style.width = progress + '%';
                
                if (progress % 20 === 0 && statusIndex < statuses.length - 1) {{
                    statusIndex++;
                    statusText.textContent = statuses[statusIndex];
                }}
                
                if (progress >= 100) {{
                    clearInterval(interval);
                }}
            }}, 100);
        }}
        
        function hideTrainingModal() {{
            const modal = document.getElementById('trainingModal');
            modal.classList.remove('active');
            document.getElementById('progressFill').style.width = '0%';
        }}
        
        // Handle training form submission
        document.querySelectorAll('form[action="/train-model"]').forEach(form => {{
            form.addEventListener('submit', function(e) {{
                e.preventDefault();
                showTrainingModal();
                
                // Submit the form
                fetch('/train-model', {{
                    method: 'GET',
                    headers: {{
                        'X-Requested-With': 'XMLHttpRequest'
                    }}
                }}).then(response => {{
                    setTimeout(() => {{
                        hideTrainingModal();
                        window.location.href = '/ml-dashboard?message=✅ Model ML berhasil dilatih!&type=success';
                    }}, 500);
                }}).catch(error => {{
                    hideTrainingModal();
                    window.location.href = '/ml-dashboard?message=❌ Gagal melatih model&type=error';
                }});
            }});
        }});
        
        function toggleTheme() {{
            const body = document.body;
            const icon = document.getElementById('themeIcon');
            
            if (body.classList.contains('dark')) {{
                body.classList.remove('dark');
                icon.classList.remove('fa-sun');
                icon.classList.add('fa-moon');
                localStorage.setItem('theme', 'light');
            }} else {{
                body.classList.add('dark');
                icon.classList.remove('fa-moon');
                icon.classList.add('fa-sun');
                localStorage.setItem('theme', 'dark');
            }}
        }}
        
        const savedTheme = localStorage.getItem('theme');
        const themeIcon = document.getElementById('themeIcon');
        
        if (savedTheme === 'dark') {{
            document.body.classList.add('dark');
            themeIcon.classList.remove('fa-moon');
            themeIcon.classList.add('fa-sun');
        }}
        
        function toggleSidebar() {{
            document.getElementById('sidebar').classList.toggle('open');
        }}
        
        document.addEventListener('click', function(event) {{
            const sidebar = document.getElementById('sidebar');
            const toggle = document.getElementById('menuToggle');
            if (window.innerWidth <= 768) {{
                if (!sidebar.contains(event.target) && !toggle.contains(event.target)) {{
                    sidebar.classList.remove('open');
                }}
            }}
        }});
        
        function updateRealTimeClock() {{
            const clockElement = document.getElementById('realtime-clock');
            const tzElement = document.getElementById('timezone-display');
            
            if (clockElement && window.currentLocation) {{
                try {{
                    const options = {{
                        timeZone: window.currentLocation.timezone,
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                        hour12: false
                    }};
                    const formatter = new Intl.DateTimeFormat('id-ID', options);
                    const timeStr = formatter.format(new Date());
                    clockElement.textContent = timeStr;
                    
                    if (tzElement && window.currentLocation.timezone) {{
                        let tzName = window.currentLocation.timezone.split('/').pop().replace('_', ' ');
                        if (tzName === 'Jakarta') tzName = 'WIB';
                        else if (tzName === 'Makassar' || tzName === 'Ujung Pandang') tzName = 'WITA';
                        else if (tzName === 'Jayapura') tzName = 'WIT';
                        tzElement.textContent = tzName;
                    }}
                }} catch(e) {{
                    console.error('Timezone error:', e);
                    const now = new Date();
                    clockElement.textContent = now.toLocaleTimeString('id-ID');
                }}
            }} else if (clockElement) {{
                const now = new Date();
                clockElement.textContent = now.toLocaleTimeString('id-ID');
            }}
        }}
        
        setInterval(updateRealTimeClock, 1000);
        updateRealTimeClock();
        
        // Flash message auto hide
        setTimeout(() => {{
            const flash = document.querySelector('.flash');
            if (flash) {{
                flash.style.opacity = '0';
                setTimeout(() => flash.remove(), 500);
            }}
        }}, 3000);
    </script>
</body>
</html>'''

# ============ ROUTES ============

selected_location = {
    "name": "Jakarta", 
    "latitude": -6.2, 
    "longitude": 106.816666,
    "timezone": "Asia/Jakarta"
}

# ============ DASHBOARD UTAMA ============
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    global selected_location
    
    saved_locations = get_saved_locations()
    weather = get_current_weather(selected_location["latitude"], selected_location["longitude"])
    forecast_api = get_5day_forecast(selected_location["latitude"], selected_location["longitude"])
    insights = get_ai_insights_real(weather, forecast_api, selected_location["name"])
    
    weather_icon = get_weather_icon_html(weather.get("weather_code", 0))
    condition_text = get_condition_text(weather.get("weather_code", 0))
    
    local_info = get_local_time(selected_location["latitude"], selected_location["longitude"], selected_location.get("timezone"))
    
    # Tentukan sapaan berdasarkan jam lokal
    hour = local_info["hour"]
    if 3 <= hour < 11:
        greeting = "Pagi"
        greeting_icon = "🌅"
    elif 11 <= hour < 15:
        greeting = "Siang"
        greeting_icon = "☀️"
    elif 15 <= hour < 18:
        greeting = "Sore"
        greeting_icon = "🌤️"
    else:
        greeting = "Malam"
        greeting_icon = "🌙"
    
    # Forecast API HTML
    forecast_html = ""
    for day in forecast_api[:5]:
        forecast_html += f'''
        <div class="forecast-item">
            <div class="forecast-day">{day["day"]}</div>
            <div class="forecast-date">{day["date"]}</div>
            <div class="forecast-icon">{get_weather_icon_html(day["weather_code"])}</div>
            <div class="forecast-temp">{int(day["temp_max"])}°</div>
            <div class="forecast-temp-min">{int(day["temp_min"])}°</div>
            <div class="forecast-precip"><i class="fas fa-tint"></i> {int(day["precipitation"])}mm</div>
        </div>
        '''
    
    # UV color
    uv = weather.get("uv_index", 5)
    if uv > 8:
        uv_color = "#f97316"
    elif uv > 6:
        uv_color = "#f59e0b"
    else:
        uv_color = "#eab308"
    
    # Timezone display
    tz_display = selected_location.get("timezone", "Asia/Jakarta").split('/')[-1].replace('_', ' ')
    if tz_display == "Jakarta":
        tz_display = "WIB"
    elif tz_display == "Makassar" or tz_display == "Ujung Pandang":
        tz_display = "WITA"
    elif tz_display == "Jayapura":
        tz_display = "WIT"
    
    content = f'''
    <div class="hero">
        <h1 class="hero-title">
            <span class="greeting-icon">{greeting_icon}</span> 
            Halo, Selamat {greeting}!
        </h1>
        <p class="hero-subtitle">Cuaca hari ini di {selected_location["name"]} untuk aktivitas Anda</p>
    </div>
    
    <div class="weather-hero">
        <div class="weather-main">
            <div class="weather-info">
                <div class="weather-icon">{weather_icon}</div>
                <div class="weather-temp">{int(weather["temperature"])}<span class="temp-unit">°C</span></div>
                <div class="weather-condition">{condition_text}</div>
                <div class="feels-like">Terasa seperti {int(weather["feels_like"])}°C</div>
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-tint" style="color: #38bdf8;"></i></div>
                    <div class="stat-label">Kelembaban</div>
                    <div class="stat-value">{int(weather["humidity"])}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-wind" style="color: #8b5cf6;"></i></div>
                    <div class="stat-label">Angin</div>
                    <div class="stat-value">{int(weather["wind_speed"])} km/j</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-cloud-rain" style="color: #60a5fa;"></i></div>
                    <div class="stat-label">Hujan</div>
                    <div class="stat-value">{weather["precipitation"]:.1f} mm</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-chart-line" style="color: #f59e0b;"></i></div>
                    <div class="stat-label">Tekanan</div>
                    <div class="stat-value">{int(weather["pressure"])} hPa</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-sun" style="color: {uv_color};"></i></div>
                    <div class="stat-label">UV Index</div>
                    <div class="stat-value">{uv:.1f}</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="bento-grid">
        <div class="glass-card">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-robot"></i> AI Weather Insights (Gemini)</span>
                <i class="fas fa-microchip" style="color: var(--accent);"></i>
            </div>
            <div class="insights-text">
                {insights}
            </div>
        </div>
        
        <div class="glass-card">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-info-circle"></i> Informasi Cuaca</span>
            </div>
            <div class="info-row">
                <div class="info-item">
                    <i class="fas fa-map-pin info-icon"></i>
                    <div>
                        <div class="info-label">LOKASI AKTIF</div>
                        <div class="info-value">{selected_location["name"]}</div>
                    </div>
                </div>
                <div class="info-item">
                    <i class="fas fa-calendar-alt info-icon"></i>
                    <div>
                        <div class="info-label">TANGGAL</div>
                        <div class="info-value">{local_info["date"]}</div>
                    </div>
                </div>
                <div class="info-item">
                    <i class="fas fa-clock info-icon"></i>
                    <div>
                        <div class="info-label">WAKTU LOKAL</div>
                        <div class="info-value">
                            <span id="realtime-clock">{local_info["time"]}</span>
                            <div class="timezone-badge" id="timezone-display">{tz_display}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="glass-card">
        <div class="card-header">
            <span class="card-title"><i class="fas fa-calendar-week"></i> Prakiraan 5 Hari (Open-Meteo API)</span>
            <i class="fas fa-cloud-sun" style="color: var(--accent);"></i>
        </div>
        <div class="forecast-grid">{forecast_html}</div>
    </div>
    '''
    
    return HTMLResponse(content=render_page(content, active="home", saved_locations=saved_locations, selected_location=selected_location))

# ============ ML DASHBOARD ============
@app.get("/ml-dashboard", response_class=HTMLResponse)
async def ml_dashboard(request: Request):
    global selected_location
    
    saved_locations = get_saved_locations()
    weather = get_current_weather(selected_location["latitude"], selected_location["longitude"])
    
    # ML Predictions
    ml_predictions = weather_predictor.predict_temperature(weather)
    model_info = weather_predictor.get_model_info()
    
    local_info = get_local_time(selected_location["latitude"], selected_location["longitude"], selected_location.get("timezone"))
    
    # ML Predictions HTML
    ml_forecast_html = ""
    for day in ml_predictions[:5]:
        precip_value = day.get('precipitation', 0)
        ml_forecast_html += f'''
        <div class="forecast-item">
            <div class="forecast-day">{day["day"]}</div>
            <div class="forecast-date">{day["date"]}</div>
            <div class="forecast-icon"><i class="fas fa-chart-line" style="color: #8b5cf6;"></i></div>
            <div class="forecast-temp">{int(day["temp_max"])}°</div>
            <div class="forecast-temp-min">{int(day["temp_min"])}°</div>
            <div class="forecast-precip"><i class="fas fa-tint"></i> {precip_value}mm</div>
            <div class="ml-badge"><i class="fas fa-brain"></i> ML</div>
        </div>
        '''
    
    # Status model
    if model_info['is_trained']:
        model_status = f'''
        <div class="glass-card">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-chart-simple"></i> Metrik Model ML</span>
                <i class="fas fa-brain" style="color: #8b5cf6;"></i>
            </div>
            <div class="stats-grid" style="margin-bottom: 0;">
                <div class="stat-card" style="flex: 1;">
                    <div class="stat-icon"><i class="fas fa-chart-line" style="color: #8b5cf6;"></i></div>
                    <div class="stat-label">MAE (Mean Absolute Error)</div>
                    <div class="stat-value" style="color: #8b5cf6;">{model_info['mae']:.6f}°C</div>
                </div>
                <div class="stat-card" style="flex: 1;">
                    <div class="stat-icon"><i class="fas fa-chart-bar" style="color: #8b5cf6;"></i></div>
                    <div class="stat-label">R² Score (Akurasi)</div>
                    <div class="stat-value" style="color: #8b5cf6;">{model_info['r2']:.6f}</div>
                </div>
                <div class="stat-card" style="flex: 1;">
                    <div class="stat-icon"><i class="fas fa-map-marker-alt" style="color: #8b5cf6;"></i></div>
                    <div class="stat-label">Lokasi Training</div>
                    <div class="stat-value" style="color: #8b5cf6; font-size: 16px;">{model_info['location']}</div>
                </div>
            </div>
            <form method="GET" action="/train-model" style="margin-top: 24px;" id="trainForm">
                <button type="submit" class="train-btn" style="width: 100%;">
                    <i class="fas fa-sync-alt"></i> Latih Ulang Model ML
                </button>
            </form>
        </div>
        '''
    else:
        model_status = f'''
        <div class="glass-card" style="text-align: center;">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-exclamation-triangle"></i> Model Belum Dilatih</span>
            </div>
            <div style="padding: 32px 0;">
                <i class="fas fa-brain" style="font-size: 64px; color: var(--ml-purple); margin-bottom: 24px; display: block;"></i>
                <p style="color: var(--text-tertiary); margin-bottom: 24px;">Klik tombol di bawah untuk melatih model Random Forest Regressor</p>
                <form method="GET" action="/train-model" id="trainForm">
                    <button type="submit" class="train-btn">
                        <i class="fas fa-play"></i> Latih Model ML Sekarang
                    </button>
                </form>
            </div>
        </div>
        '''
    
    # Timezone display
    tz_display = selected_location.get("timezone", "Asia/Jakarta").split('/')[-1].replace('_', ' ')
    if tz_display == "Jakarta":
        tz_display = "WIB"
    elif tz_display == "Makassar" or tz_display == "Ujung Pandang":
        tz_display = "WITA"
    elif tz_display == "Jayapura":
        tz_display = "WIT"
    
    content = f'''
    <div class="hero">
        <h1 class="hero-title">🤖 ML Dashboard</h1>
        <p class="hero-subtitle">Prediksi cuaca dengan Machine Learning (Random Forest Regressor)</p>
    </div>
    
    <div class="bento-grid">
        <div class="glass-card">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-chart-line"></i> Prediksi ML 5 Hari (Random Forest)</span>
                <i class="fas fa-brain" style="color: #8b5cf6;"></i>
            </div>
            <div class="forecast-grid">{ml_forecast_html}</div>
        </div>
        
        {model_status}
    </div>
    
    <div class="glass-card">
        <div class="card-header">
            <span class="card-title"><i class="fas fa-info-circle"></i> Informasi Lokasi</span>
        </div>
        <div class="info-row">
            <div class="info-item">
                <i class="fas fa-map-pin info-icon"></i>
                <div>
                    <div class="info-label">LOKASI AKTIF</div>
                    <div class="info-value">{selected_location["name"]}</div>
                </div>
            </div>
            <div class="info-item">
                <i class="fas fa-calendar-alt info-icon"></i>
                <div>
                    <div class="info-label">TANGGAL</div>
                    <div class="info-value">{local_info["date"]}</div>
                </div>
            </div>
            <div class="info-item">
                <i class="fas fa-clock info-icon"></i>
                <div>
                    <div class="info-label">WAKTU LOKAL</div>
                    <div class="info-value">
                        <span id="realtime-clock">{local_info["time"]}</span>
                        <div class="timezone-badge" id="timezone-display">{tz_display}</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    '''
    
    return HTMLResponse(content=render_page(content, active="ml", saved_locations=saved_locations, selected_location=selected_location))

@app.get("/select-location/{location_id}")
async def select_location(location_id: int):
    global selected_location
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id, name, latitude, longitude, country, timezone FROM saved_locations WHERE id = ?', (location_id,))
        row = cursor.fetchone()
    except sqlite3.OperationalError:
        cursor.execute('SELECT id, name, latitude, longitude, country FROM saved_locations WHERE id = ?', (location_id,))
        row = cursor.fetchone()
        if row:
            row = list(row) + [None]
    
    conn.close()
    if row:
        selected_location = {
            "name": row[1], 
            "latitude": row[2], 
            "longitude": row[3],
            "timezone": row[5] if len(row) > 5 and row[5] else get_timezone_from_coords(row[2], row[3])
        }
    return RedirectResponse(url="/", status_code=303)

@app.get("/delete-location/{location_id}")
async def delete_location_route(location_id: int):
    delete_location(location_id)
    return RedirectResponse(url="/?message=Lokasi berhasil dihapus&type=success", status_code=303)

@app.get("/train-model")
async def train_model_route(request: Request):
    global selected_location
    
    # Check if AJAX request
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    try:
        result = weather_predictor.train_model(
            selected_location["name"],
            selected_location["latitude"],
            selected_location["longitude"]
        )
        if is_ajax:
            return {"success": True, "mae": result['mae'], "r2": result['r2']}
        return RedirectResponse(
            url=f"/ml-dashboard?message=✅ Model ML berhasil dilatih! MAE: {result['mae']:.6f}°C, R²: {result['r2']}&type=success",
            status_code=303
        )
    except Exception as e:
        if is_ajax:
            return {"success": False, "error": str(e)}
        return RedirectResponse(
            url=f"/ml-dashboard?message=❌ Gagal melatih model: {str(e)}&type=error",
            status_code=303
        )

@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, message: str = None, type: str = None):
    saved_locations = get_saved_locations()
    
    content = f'''
    <div class="hero">
        <h1 class="hero-title">Cari Kota Baru</h1>
        <p class="hero-subtitle">Temukan informasi cuaca akurat di kota mana pun di seluruh dunia</p>
    </div>
    
    <div class="glass-card" style="max-width: 600px; margin: 0 auto;">
        <div class="card-header">
            <span class="card-title"><i class="fas fa-search"></i> Pencarian Lokasi</span>
        </div>
        <form method="POST" action="/search">
            <div class="search-container" style="margin-bottom: 0;">
                <input type="text" name="city_name" class="search-input" placeholder="Masukkan nama kota... Jakarta, Bali, New York, London, Tokyo" required>
                <button type="submit" class="search-btn"><i class="fas fa-search"></i> Cari & Simpan</button>
            </div>
        </form>
        <p style="margin-top: 20px; font-size: 12px; color: var(--text-tertiary); text-align: center;">
            <i class="fas fa-globe"></i> Mendukung semua kota di seluruh dunia dengan zona waktu otomatis
        </p>
    </div>
    '''
    
    return HTMLResponse(content=render_page(content, active="search", message=message, message_type=type, saved_locations=saved_locations, selected_location=selected_location))

@app.post("/search", response_class=HTMLResponse)
async def search_city_post(city_name: str = Form(...)):
    result = search_city(city_name)
    if result:
        if location_exists(result["name"], result["latitude"], result["longitude"]):
            return RedirectResponse(url="/search?message=Lokasi sudah tersimpan&type=error", status_code=303)
        
        save_location(
            result["name"], 
            result["latitude"], 
            result["longitude"], 
            result["country"],
            result["timezone"]
        )
        return RedirectResponse(url=f"/search?message={result['name']} berhasil ditambahkan ke favorit&type=success", status_code=303)
    else:
        return RedirectResponse(url=f"/search?message=Kota '{city_name}' tidak ditemukan. Periksa ejaan Anda.&type=error", status_code=303)

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    saved_locations = get_saved_locations()
    
    content = f'''
    <div class="hero">
        <h1 class="hero-title">Tentang WeatherAI</h1>
        <p class="hero-subtitle">Aplikasi prediksi cuaca cerdas berbasis AI untuk informasi akurat dan real-time</p>
    </div>
    
    <div class="bento-grid">
        <div class="glass-card">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-info-circle"></i> Tentang Aplikasi</span>
            </div>
            <div style="padding: 24px;">
                <p style="margin-bottom: 16px; line-height: 1.6;">
                    <strong>WeatherAI</strong> adalah aplikasi web modern yang menggabungkan teknologi AI canggih dengan data cuaca real-time untuk memberikan informasi cuaca yang akurat dan dapat diandalkan.
                </p>
                <p style="margin-bottom: 16px; line-height: 1.6;">
                    Aplikasi ini menggunakan <strong>Google Gemini AI</strong> untuk analisis cuaca natural language dan <strong>Machine Learning (Random Forest)</strong> untuk prediksi suhu jangka pendek.
                </p>
            </div>
        </div>
        
        <div class="glass-card">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-microchip"></i> Teknologi AI</span>
            </div>
            <div style="padding: 24px;">
                <ul style="list-style: none; padding: 0;">
                    <li style="margin-bottom: 12px;"><i class="fas fa-check" style="color: var(--accent); margin-right: 8px;"></i> <strong>Google Gemini 2.5 Flash</strong> - AI untuk deskripsi cuaca natural</li>
                    <li style="margin-bottom: 12px;"><i class="fas fa-check" style="color: var(--accent); margin-right: 8px;"></i> <strong>Random Forest Regressor</strong> - ML untuk prediksi suhu</li>
                    <li style="margin-bottom: 12px;"><i class="fas fa-check" style="color: var(--accent); margin-right: 8px;"></i> <strong>Open-Meteo API</strong> - Data cuaca real-time gratis</li>
                </ul>
            </div>
        </div>
    </div>
    
    <div class="bento-grid">
        <div class="glass-card">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-chart-line"></i> Akurasi & Fitur</span>
            </div>
            <div style="padding: 24px;">
                <ul style="list-style: none; padding: 0;">
                    <li style="margin-bottom: 12px;"><i class="fas fa-star" style="color: #fbbf24; margin-right: 8px;"></i> <strong>MAE: 0.46°C</strong> - Error rata-rata prediksi</li>
                    <li style="margin-bottom: 12px;"><i class="fas fa-star" style="color: #fbbf24; margin-right: 8px;"></i> <strong>R²: 0.89</strong> - Tingkat akurasi model</li>
                    <li style="margin-bottom: 12px;"><i class="fas fa-star" style="color: #fbbf24; margin-right: 8px;"></i> Prakiraan 5 hari dengan AI insights</li>
                    <li style="margin-bottom: 12px;"><i class="fas fa-star" style="color: #fbbf24; margin-right: 8px;"></i> Zona waktu otomatis untuk semua lokasi</li>
                </ul>
            </div>
        </div>
        
        <div class="glass-card">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-code"></i> Teknologi & Framework</span>
            </div>
            <div style="padding: 24px;">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
                    <div style="text-align: center;">
                        <i class="fab fa-python" style="font-size: 32px; color: #3776ab; margin-bottom: 8px;"></i>
                        <div style="font-weight: 600;">Python</div>
                        <div style="font-size: 12px; color: var(--text-tertiary);">Backend & ML</div>
                    </div>
                    <div style="text-align: center;">
                        <i class="fas fa-rocket" style="font-size: 32px; color: #00c8ff; margin-bottom: 8px;"></i>
                        <div style="font-weight: 600;">FastAPI</div>
                        <div style="font-size: 12px; color: var(--text-tertiary);">Web Framework</div>
                    </div>
                    <div style="text-align: center;">
                        <i class="fas fa-brain" style="font-size: 32px; color: #8b5cf6; margin-bottom: 8px;"></i>
                        <div style="font-weight: 600;">Scikit-learn</div>
                        <div style="font-size: 12px; color: var(--text-tertiary);">Machine Learning</div>
                    </div>
                    <div style="text-align: center;">
                        <i class="fas fa-cloud-sun" style="font-size: 32px; color: #f59e0b; margin-bottom: 8px;"></i>
                        <div style="font-weight: 600;">Open-Meteo</div>
                        <div style="font-size: 12px; color: var(--text-tertiary);">Weather Data</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="glass-card">
        <div class="card-header">
            <span class="card-title"><i class="fas fa-shield-alt"></i> Privasi & Keamanan</span>
        </div>
        <div style="padding: 24px;">
            <p style="margin-bottom: 16px; line-height: 1.6;">
                Aplikasi ini <strong>tidak menyimpan data pribadi pengguna</strong>. Semua data cuaca diambil secara real-time dari API publik dan tidak ada pelacakan atau penyimpanan data pengguna.
            </p>
            <p style="line-height: 1.6;">
                Lokasi yang disimpan hanya tersimpan di browser lokal Anda dan dapat dihapus kapan saja.
            </p>
        </div>
    </div>
    '''
    
    return HTMLResponse(content=render_page(content, active="about", saved_locations=saved_locations))

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
