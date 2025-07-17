import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import logging
import threading
import queue
import sqlite3
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional
import psutil
import redis
from kafka import KafkaProducer
import asyncio
#import websockets
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from ml.traffic_predictor import TrafficFlowPredictor

roi_polygon_points = []
drawing_complete = False

# Red line mouse callback
def roi_mouse_callback(event, x, y, flags, param):
    global roi_polygon_points, drawing_complete
    if event == cv2.EVENT_LBUTTONDOWN and not drawing_complete:
        roi_polygon_points.append((x, y))
        print(f"ROI point {len(roi_polygon_points)}: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN and len(roi_polygon_points) >= 3:
        drawing_complete = True
        print("✅ ROI drawing completed.")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffic_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class VehicleData:
    """Data class for vehicle information"""
    id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center_point: Tuple[int, int]
    timestamp: datetime
    speed: Optional[float] = None
    lane: Optional[int] = None

@dataclass
class TrafficMetrics:
    """Data class for traffic metrics"""
    timestamp: datetime
    vehicle_counts: Dict[str, int]
    weighted_total: float
    traffic_status: str
    average_speed: float
    lane_occupancy: Dict[int, float]
    congestion_level: float
    
@dataclass
class KPIMetrics:
    """Data class for Key Performance Indicators"""
    timestamp: datetime
    traffic_flow_efficiency: float  # Average speed km/h
    average_waiting_time: float  # seconds
    congestion_level: float  # percentage
    incident_detection_time: Optional[float] = None  # seconds
    fuel_consumption_index: float = 0.0  # relative index
    co2_emissions_index: float = 0.0  # relative index
    system_uptime: float = 100.0  # percentage
    user_satisfaction_score: float = 0.0  # 0-100 scale

@dataclass
class IncidentData:
    """Data class for incident information"""
    id: int
    timestamp: datetime
    detection_time: float  # seconds to detect
    incident_type: str
    severity: str
    location: Tuple[int, int]
    resolved: bool = False
    resolution_time: Optional[float] = None

class ConfigManager:
    """Configuration management with validation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.validate_config()
    
    def load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found. Using defaults.")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """Default configuration"""
        return {
            'video': {
                'input_source': 'test_videos/test2.mov',
                'frame_skip': 2,
                'resize_width': 1280,
                'resize_height': 720
            },
            'detection': {
                'model_path': 'yolov8s.pt',
                'confidence_threshold': 0.5,
                'vehicle_classes': [1, 2, 3, 5, 6, 7],
                'tracking_threshold': 30
            },
            'counting': {
                'counting_lines': [
                    {'y': 360, 'x1': 345, 'x2': 635, 'direction': 'down'}
                ],
                'roi_polygon': [[0, 0], [1280, 0], [1280, 720], [0, 720]]
            },
            'traffic_weights': {
                'car': 1.0,
                'motorcycle': 0.5,
                'bus': 3.0,
                'truck': 3.0,
                'bicycle': 0.3
            },
            'thresholds': {
                'low_traffic': 10,
                'normal_traffic': 20,
                'high_traffic': 30
            },
            'logging': {
                'interval_seconds': 8,
                'database_path': 'traffic_data.db',
                'csv_export': True
            },
            'performance': {
                'max_fps': 30,
                'gpu_enabled': True,
                'multi_threading': True
            },
            'alerts': {
                'congestion_threshold': 80,
                'speed_limit': 50,
                'webhook_url': None,
                'email_notifications': False
            }
        }
    
    def validate_config(self):
        """Validate configuration parameters"""
        required_keys = ['video', 'detection', 'counting', 'traffic_weights']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config section: {key}")

class DatabaseManager:
    """Handle database operations with connection pooling"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Existing tables...
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                vehicle_counts TEXT NOT NULL,
                weighted_total REAL NOT NULL,
                traffic_status TEXT NOT NULL,
                average_speed REAL,
                lane_occupancy TEXT,
                congestion_level REAL,
                weather_condition TEXT,
                camera_id TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                speed REAL,
                lane INTEGER,
                camera_id TEXT
            )
        ''')
        
        # New KPI table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS kpi_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                traffic_flow_efficiency REAL NOT NULL,
                average_waiting_time REAL NOT NULL,
                congestion_level REAL NOT NULL,
                incident_detection_time REAL,
                fuel_consumption_index REAL NOT NULL,
                co2_emissions_index REAL NOT NULL,
                system_uptime REAL NOT NULL,
                user_satisfaction_score REAL NOT NULL,
                camera_id TEXT
            )
        ''')
        
        # Incident tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                detection_time REAL NOT NULL,
                incident_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                location_x INTEGER NOT NULL,
                location_y INTEGER NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_time REAL,
                camera_id TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_kpi_data(self, kpi_metrics: KPIMetrics, camera_id: str):
        """Insert KPI data into database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO kpi_metrics 
            (timestamp, traffic_flow_efficiency, average_waiting_time, congestion_level,
             incident_detection_time, fuel_consumption_index, co2_emissions_index,
             system_uptime, user_satisfaction_score, camera_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            kpi_metrics.timestamp.isoformat(),
            kpi_metrics.traffic_flow_efficiency,
            kpi_metrics.average_waiting_time,
            kpi_metrics.congestion_level,
            kpi_metrics.incident_detection_time,
            kpi_metrics.fuel_consumption_index,
            kpi_metrics.co2_emissions_index,
            kpi_metrics.system_uptime,
            kpi_metrics.user_satisfaction_score,
            camera_id
        ))
        
        conn.commit()
        conn.close()
        
    def insert_traffic_data(self, metrics: TrafficMetrics, camera_id: str):
        """Insert traffic data into database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO traffic_data 
            (timestamp, vehicle_counts, weighted_total, traffic_status, average_speed, 
            lane_occupancy, congestion_level, camera_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp.isoformat(),
            json.dumps(metrics.vehicle_counts),
            metrics.weighted_total,
            metrics.traffic_status,
            metrics.average_speed,
            json.dumps(metrics.lane_occupancy),
            metrics.congestion_level,
            camera_id
        ))
        
        conn.commit()
        conn.close()

class PerformanceMonitor:
    """Monitor system performance and resource usage"""
    
    def __init__(self):
        self.fps_history = deque(maxlen=30)
        self.cpu_history = deque(maxlen=30)
        self.memory_history = deque(maxlen=30)
        self.last_frame_time = time.time()
    
    def update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time)
        self.fps_history.append(fps)
        self.last_frame_time = current_time
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        self.cpu_history.append(psutil.cpu_percent())
        self.memory_history.append(psutil.virtual_memory().percent)
    
    def get_performance_stats(self) -> dict:
        """Get current performance statistics"""
        return {
            'avg_fps': sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0,
            'avg_cpu': sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0,
            'avg_memory': sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0,
            'current_fps': self.fps_history[-1] if self.fps_history else 0
        }
    
    

class AlertManager:
    """Handle alerts and notifications"""
    
    def __init__(self, config: dict):
        self.config = config
        self.kafka_producer = None
        self.redis_client = None
        self.setup_messaging()
    
    def setup_messaging(self):
        """Setup messaging systems"""
        try:
            if 'kafka' in self.config:
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=self.config['kafka']['servers'],
                    value_serializer=lambda x: json.dumps(x).encode('utf-8')
                )
            
            if 'redis' in self.config:
                self.redis_client = redis.Redis(
                    host=self.config['redis']['host'],
                    port=self.config['redis']['port'],
                    decode_responses=True
                )
        except Exception as e:
            logger.warning(f"Failed to setup messaging: {e}")
    
    def send_alert(self, alert_type: str, data: dict):
        """Send alert through configured channels"""
        alert = {
            'type': alert_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        # Send to Kafka
        if self.kafka_producer:
            try:
                self.kafka_producer.send('traffic_alerts', alert)
            except Exception as e:
                logger.error(f"Failed to send Kafka alert: {e}")
        
        # Send to Redis
        if self.redis_client:
            try:
                self.redis_client.lpush('traffic_alerts', json.dumps(alert))
            except Exception as e:
                logger.error(f"Failed to send Redis alert: {e}")


class KPICalculator:
    """Calculate and track Key Performance Indicators"""
    
    def __init__(self, config: dict):
        self.config = config
        self.incident_history = []
        self.speed_history = deque(maxlen=100)
        self.waiting_time_history = deque(maxlen=100)
        self.system_start_time = time.time()
        self.downtime_duration = 0.0
        self.fuel_baseline = 100.0  # baseline fuel consumption index
        self.co2_baseline = 100.0   # baseline CO2 emissions index
        
    def calculate_traffic_flow_efficiency(self, vehicles: List[VehicleData]) -> float:
        """Calculate average speed of vehicles (KPI 1)"""
        speeds = [v.speed for v in vehicles if v.speed is not None and v.speed > 0]
        if not speeds:
            return 0.0
        
        avg_speed = sum(speeds) / len(speeds)
        self.speed_history.append(avg_speed)
        return avg_speed
    
    def calculate_waiting_time(self, vehicles: List[VehicleData], 
                             intersection_areas: List[dict]) -> float:
        """Calculate average waiting time at intersections (KPI 2)"""
        total_waiting_time = 0.0
        waiting_vehicles = 0
        
        for vehicle in vehicles:
            # Check if vehicle is in intersection area and moving slowly
            for area in intersection_areas:
                if self._is_in_area(vehicle.center_point, area):
                    if vehicle.speed is not None and vehicle.speed < 5.0:  # Slow or stopped
                        # Estimate waiting time based on speed
                        waiting_time = max(0, (5.0 - vehicle.speed) * 2)  # Simplified estimation
                        total_waiting_time += waiting_time
                        waiting_vehicles += 1
        
        avg_waiting_time = total_waiting_time / waiting_vehicles if waiting_vehicles > 0 else 0.0
        self.waiting_time_history.append(avg_waiting_time)
        return avg_waiting_time
    
    def calculate_congestion_level(self, vehicles: List[VehicleData], 
                                 road_capacity: int) -> float:
        """Calculate traffic congestion level (KPI 3)"""
        vehicle_count = len(vehicles)
        congestion_level = (vehicle_count / road_capacity) * 100
        return min(congestion_level, 100.0)
    
    def detect_incidents(self, vehicles: List[VehicleData], 
                        previous_vehicles: List[VehicleData]) -> List[IncidentData]:
        """Detect incidents and calculate detection time (KPI 4)"""
        incidents = []
        detection_start = time.time()
        
        # Simple incident detection logic
        current_positions = {v.id: v.center_point for v in vehicles}
        prev_positions = {v.id: v.center_point for v in previous_vehicles}
        
        for vehicle in vehicles:
            # Check for stopped vehicles
            if (vehicle.speed is not None and vehicle.speed < 1.0 and 
                vehicle.id in prev_positions):
                
                # Check if vehicle has been stopped for a while
                if self._vehicle_stopped_duration(vehicle.id) > 30:  # 30 seconds
                    detection_time = time.time() - detection_start
                    
                    incident = IncidentData(
                        id=len(self.incident_history) + 1,
                        timestamp=datetime.now(),
                        detection_time=detection_time,
                        incident_type="stopped_vehicle",
                        severity="medium",
                        location=vehicle.center_point
                    )
                    incidents.append(incident)
                    self.incident_history.append(incident)
        
        return incidents
    
    def calculate_fuel_consumption_index(self, avg_speed: float, 
                                       congestion_level: float) -> float:
        """Calculate fuel consumption index (KPI 5)"""
        # Simplified model: fuel consumption increases with congestion and decreases with speed
        if avg_speed == 0:
            return 150.0  # High fuel consumption when stopped
        
        # Optimal speed for fuel efficiency is around 60 km/h
        speed_factor = abs(avg_speed - 60) / 60
        congestion_factor = congestion_level / 100
        
        fuel_index = self.fuel_baseline * (1 + speed_factor * 0.3 + congestion_factor * 0.5)
        return fuel_index
    
    def calculate_co2_emissions_index(self, fuel_index: float) -> float:
        """Calculate CO2 emissions index (KPI 5)"""
        # CO2 emissions are proportional to fuel consumption
        return fuel_index * 0.95  # Slight adjustment factor
    
    def calculate_system_uptime(self) -> float:
        """Calculate system uptime (KPI 6)"""
        total_time = time.time() - self.system_start_time
        uptime_percentage = ((total_time - self.downtime_duration) / total_time) * 100
        return min(uptime_percentage, 100.0)
    
    def record_downtime(self, duration: float):
        """Record system downtime"""
        self.downtime_duration += duration
    
    def calculate_user_satisfaction(self, avg_speed, waiting_time, congestion):
        speed_score = min(avg_speed / 60.0, 1.0) * 40
        wait_score = max(0, (60 - waiting_time) / 60.0) * 30
        congestion_score = max(0, (100 - congestion) / 100.0) * 30
        return round(speed_score + wait_score + congestion_score, 2)

    
    def _is_in_area(self, point: Tuple[int, int], area: dict) -> bool:
        """Check if point is within specified area"""
        x, y = point
        return (area['x1'] <= x <= area['x2'] and 
                area['y1'] <= y <= area['y2'])
    
    def _vehicle_stopped_duration(self, vehicle_id: int) -> float:
        """Get duration for which vehicle has been stopped"""
        # This would require tracking vehicle states over time
        # Simplified implementation
        return 35.0  # Placeholder
    
    def generate_kpi_report(self, vehicles: List[VehicleData]) -> KPIMetrics:
        """Generate comprehensive KPI report"""
        # Define intersection areas (these would be configured per location)
        intersection_areas = [
            {'x1': 300, 'y1': 300, 'x2': 700, 'y2': 400},  # Example intersection
        ]
        
        # Calculate all KPIs
        traffic_flow_efficiency = self.calculate_traffic_flow_efficiency(vehicles)
        waiting_time = self.calculate_waiting_time(vehicles, intersection_areas)
        congestion_level = self.calculate_congestion_level(vehicles, 50)  # Assume capacity of 50
        fuel_index = self.calculate_fuel_consumption_index(traffic_flow_efficiency, congestion_level)
        co2_index = self.calculate_co2_emissions_index(fuel_index)
        system_uptime = self.calculate_system_uptime()
        user_satisfaction = self.calculate_user_satisfaction(
            traffic_flow_efficiency, waiting_time, congestion_level
        )
        
        return KPIMetrics(
            timestamp=datetime.now(),
            traffic_flow_efficiency=traffic_flow_efficiency,
            average_waiting_time=waiting_time,
            congestion_level=congestion_level,
            fuel_consumption_index=fuel_index,
            co2_emissions_index=co2_index,
            system_uptime=system_uptime,
            user_satisfaction_score=user_satisfaction
        )

class LaneDetector:
    """Detect and track vehicle lanes"""
    
    def __init__(self, frame_width: int, num_lanes: int = 4):
        self.frame_width = frame_width
        self.num_lanes = num_lanes
        self.lane_width = frame_width // num_lanes
    
    def get_lane(self, center_x: int) -> int:
        """Determine lane number from center x coordinate"""
        lane = min(center_x // self.lane_width, self.num_lanes - 1)
        return max(0, lane)
    
    def calculate_lane_occupancy(self, vehicles: List[VehicleData]) -> Dict[int, float]:
        """Calculate occupancy percentage for each lane"""
        lane_counts = defaultdict(int)
        for vehicle in vehicles:
            lane = self.get_lane(vehicle.center_point[0])
            lane_counts[lane] += 1
        
        # Convert to occupancy percentages
        total_vehicles = sum(lane_counts.values())
        if total_vehicles == 0:
            return {i: 0.0 for i in range(self.num_lanes)}
        
        return {
            lane: (count / total_vehicles) * 100 
            for lane, count in lane_counts.items()
        }

class SpeedEstimator:
    """Estimate vehicle speeds using optical flow"""
    
    def __init__(self, pixels_per_meter: float = 10.0):
        self.pixels_per_meter = pixels_per_meter
        self.vehicle_history = defaultdict(deque)
        self.max_history = 10
    
    def update_position(self, vehicle_id: int, position: Tuple[int, int], timestamp: datetime):
        """Update vehicle position history"""
        self.vehicle_history[vehicle_id].append((position, timestamp))
        if len(self.vehicle_history[vehicle_id]) > self.max_history:
            self.vehicle_history[vehicle_id].popleft()
    
    def estimate_speed(self, vehicle_id: int) -> Optional[float]:
        """Estimate speed in km/h"""
        if len(self.vehicle_history[vehicle_id]) < 2:
            return None
        
        positions = list(self.vehicle_history[vehicle_id])
        
        # Calculate distance and time difference
        pos1, time1 = positions[-2]
        pos2, time2 = positions[-1]
        
        distance_pixels = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)

        distance_meters = distance_pixels / self.pixels_per_meter
        
        time_diff = (time2 - time1).total_seconds()
        if time_diff == 0:
            return None
        
        speed_mps = distance_meters / time_diff
        speed_kmh = speed_mps * 3.6
        
        return speed_kmh

class IndustrialTrafficMonitor:
    """Main industrial-grade traffic monitoring system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        
        
        # Initialize components
        self.model = YOLO(self.config['detection']['model_path'])
        self.db_manager = DatabaseManager(self.config['logging']['database_path'])
        self.performance_monitor = PerformanceMonitor()
        self.alert_manager = AlertManager(self.config.get('alerts', {}))
        self.lane_detector = LaneDetector(self.config['video']['resize_width'])
        self.speed_estimator = SpeedEstimator()
        self.kpi_calculator = KPICalculator(self.config)  # NEW: KPI calculator
        
        # State variables
        self.class_counts = defaultdict(int)
        self.interval_counts = defaultdict(int)
        self.crossed_ids = set()
        self.track_history = {}
        self.current_vehicles = []
        self.previous_vehicles = []  # NEW: Track previous frame vehicles
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=30)
        self.running = False
        
        # Timing
        self.interval_start_time = time.time()
        self.last_alert_time = defaultdict(float)
        self.last_kpi_time = time.time()  # NEW: KPI calculation timing
        
        self.traffic_predictor = TrafficFlowPredictor()
        self.counted_vehicle_ids = set()

        
        logger.info("Industrial Traffic Monitor with KPIs initialized successfully")
        
    def draw_clean_overlay(self, frame, metrics, kpi_metrics, predicted_flow, fps):
        """Draw clean organized UI on the frame"""
        display = frame.copy()

        # 🟣 Draw ROI polygon if defined
        if roi_polygon_points:
            cv2.polylines(display, [np.array(roi_polygon_points, np.int32)], True, (255, 0, 255), 2)

        # ✅ Top-left: Summary of vehicle counts and prediction
        summary_lines = [
            f"Predicted Flow: {predicted_flow:.1f}",
            f"Vehicle Count: {sum(self.interval_counts.values())}",
            f"Cars: {self.interval_counts.get('car', 0)}",
            f"Bikes: {self.interval_counts.get('motorcycle', 0)}",
            f"Buses: {self.interval_counts.get('bus', 0)}",
            f"Trucks: {self.interval_counts.get('truck', 0)}"
        ]

        for i, line in enumerate(summary_lines):
            y = 30 + i * 22
            cv2.putText(display, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ✅ Bottom-left: FPS and status
        status_text = f"Status: {metrics.traffic_status}"
        cv2.putText(display, status_text, (20, display.shape[0] - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if "Severe" in status_text else (0, 255, 0), 2)
        cv2.putText(display, f"FPS: {fps:.1f}", (20, display.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, f"Avg Speed: {metrics.average_speed:.1f} km/h", (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return display

        
    def is_in_roi(self, point: Tuple[int, int]) -> bool:
        """Check if a point is inside the user-drawn ROI polygon"""
        if not roi_polygon_points or not drawing_complete:
            return False
        return cv2.pointPolygonTest(np.array(roi_polygon_points, np.int32), point, False) >= 0

    # Add new method to draw KPI visualizations
    def draw_kpi_overlay(self, frame: np.ndarray, kpi_metrics: KPIMetrics) -> np.ndarray:
        """Draw KPI metrics overlay on frame"""
        # KPI display area
        overlay = frame.copy()
        kpi_area = (frame.shape[1] - 400, 10, frame.shape[1] - 10, 300)
        cv2.rectangle(overlay, (kpi_area[0], kpi_area[1]), (kpi_area[2], kpi_area[3]), 
                     (0, 0, 0), -1)
        
        # Add transparency
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # KPI text
        kpi_texts = [
            f"Traffic Flow: {kpi_metrics.traffic_flow_efficiency:.1f} km/h",
            f"Avg Wait Time: {kpi_metrics.average_waiting_time:.1f}s",
            f"Congestion: {kpi_metrics.congestion_level:.1f}%",
            f"Fuel Index: {kpi_metrics.fuel_consumption_index:.1f}",
            f"CO2 Index: {kpi_metrics.co2_emissions_index:.1f}",
            f"System Uptime: {kpi_metrics.system_uptime:.2f}%",
            f"Satisfaction: {kpi_metrics.user_satisfaction_score:.1f}/100"
        ]
        
        y_offset = 30
        for text in kpi_texts:
            cv2.putText(frame, text, (kpi_area[0] + 10, kpi_area[1] + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 30
        
        return frame

    # Add missing methods to IndustrialTrafficMonitor class
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Advanced frame preprocessing"""
        # Resize frame
        frame = cv2.resize(frame, 
                          (self.config['video']['resize_width'], 
                           self.config['video']['resize_height']))
        
        # Apply histogram equalization for better detection
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return frame
    
    def detect_and_track(self, frame: np.ndarray) -> List[VehicleData]:
        """Perform detection and tracking with error handling"""
        try:
            results = self.model.track(
                frame, 
                persist=True, 
                classes=self.config['detection']['vehicle_classes'],
                conf=self.config['detection']['confidence_threshold']
            )
            
            vehicles = []
            if results and results[0].boxes and results[0].boxes.xyxy is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
                class_indices = results[0].boxes.cls.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                current_time = datetime.now()
                
                for i, (box, class_idx, conf) in enumerate(zip(boxes, class_indices, confidences)):
                    track_id = track_ids[i] if i < len(track_ids) else -1
                    if track_id == -1:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box)
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    class_name = self.model.names[class_idx]
                    
                    # Estimate speed
                    self.speed_estimator.update_position(track_id, (center_x, center_y), current_time)
                    speed = self.speed_estimator.estimate_speed(track_id)
                    
                    # Determine lane
                    lane = self.lane_detector.get_lane(center_x)
                    
                    vehicle = VehicleData(
                        id=track_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center_point=(center_x, center_y),
                        timestamp=current_time,
                        speed=speed,
                        lane=lane
                    )
                    vehicles.append(vehicle)
            
            return vehicles
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def check_vehicles_in_roi(self, vehicles: List[VehicleData]):
        """Track and count each vehicle only once per interval"""
        for vehicle in vehicles:
            if vehicle.id in self.counted_vehicle_ids:
                continue  # already counted this vehicle

            if self.is_in_roi(vehicle.center_point):
                self.class_counts[vehicle.class_name] += 1
                self.interval_counts[vehicle.class_name] += 1
                self.counted_vehicle_ids.add(vehicle.id)


    
    def calculate_traffic_metrics(self, vehicles: List[VehicleData]) -> TrafficMetrics:
        """Calculate comprehensive traffic metrics"""
        # Calculate weighted total
        weighted_total = sum(
            count * self.config['traffic_weights'].get(class_name, 1.0)
            for class_name, count in self.interval_counts.items()
        )
        
        # Determine traffic status
        thresholds = self.config['thresholds']
        if weighted_total <= thresholds['low_traffic']:
            traffic_status = "Low Traffic"
        elif weighted_total <= thresholds['normal_traffic']:
            traffic_status = "Normal Traffic"
        elif weighted_total <= thresholds['high_traffic']:
            traffic_status = "High Traffic"
        else:
            traffic_status = "Severe Congestion"
        
        # Calculate average speed
        speeds = [v.speed for v in vehicles if v.speed is not None]
        average_speed = sum(speeds) / len(speeds) if speeds else 0.0
        
        # Calculate lane occupancy
        lane_occupancy = self.lane_detector.calculate_lane_occupancy(vehicles)
        
        # Calculate congestion level (0-100)
        congestion_level = min(100, (weighted_total / thresholds['high_traffic']) * 100)
        
        return TrafficMetrics(
            timestamp=datetime.now(),
            vehicle_counts=dict(self.interval_counts),
            weighted_total=weighted_total,
            traffic_status=traffic_status,
            average_speed=average_speed,
            lane_occupancy=lane_occupancy,
            congestion_level=congestion_level
        )
    
    def check_alerts(self, metrics: TrafficMetrics, vehicles: List[VehicleData]):
        """Check for alert conditions"""
        current_time = time.time()
        
        # Congestion alert
        if (metrics.congestion_level > self.config['alerts']['congestion_threshold'] and
            current_time - self.last_alert_time['congestion'] > 300):  # 5 min cooldown
            
            self.alert_manager.send_alert('congestion', {
                'level': metrics.congestion_level,
                'status': metrics.traffic_status,
                'weighted_total': metrics.weighted_total
            })
            self.last_alert_time['congestion'] = current_time
        
        # Speed violation alerts
        speed_limit = self.config['alerts']['speed_limit']
        for vehicle in vehicles:
            if (vehicle.speed and vehicle.speed > speed_limit and
                current_time - self.last_alert_time[f'speed_{vehicle.id}'] > 60):
                
                self.alert_manager.send_alert('speed_violation', {
                    'vehicle_id': vehicle.id,
                    'speed': vehicle.speed,
                    'limit': speed_limit,
                    'class': vehicle.class_name
                })
                self.last_alert_time[f'speed_{vehicle.id}'] = current_time
    
    def draw_visualizations(self, frame: np.ndarray, vehicles: List[VehicleData], metrics: TrafficMetrics) -> np.ndarray:
        """Draw comprehensive visualizations"""
        # Draw counting lines
        if roi_polygon_points:
            cv2.polylines(frame, [np.array(roi_polygon_points, np.int32)], isClosed=True, color=(255, 0, 255), thickness=2)


        # Draw vehicles
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle.bbox
            
            # Color based on speed
            if vehicle.speed:
                if vehicle.speed > self.config['alerts']['speed_limit']:
                    color = (0, 0, 255)  # Red for speeding
                else:
                    color = (0, 255, 0)  # Green for normal speed
            else:
                color = (255, 0, 0)  # Blue for unknown speed
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw info
            info_text = f"ID:{vehicle.id} {vehicle.class_name}"
            if vehicle.speed:
                info_text += f" {vehicle.speed:.1f}km/h"
            if vehicle.lane is not None:
                info_text += f" L{vehicle.lane}"
            
            cv2.putText(frame, info_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw center point
            cv2.circle(frame, vehicle.center_point, 4, color, -1)
        
        # Draw metrics overlay
        y_offset = 30
        for class_name, count in self.class_counts.items():
            cv2.putText(frame, f"{class_name}: {count}", (50, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 30
        
        # Traffic status
        status_color = {
            "Low Traffic": (0, 255, 0),
            "Normal Traffic": (0, 255, 255),
            "High Traffic": (0, 165, 255),
            "Severe Congestion": (0, 0, 255)
        }.get(metrics.traffic_status, (255, 255, 255))
        
        cv2.putText(frame, f"Status: {metrics.traffic_status}", 
                   (50, frame.shape[0] - 80),
                   cv2.FONT_HERSHEY_TRIPLEX, 0.9, status_color, 2)
        
        # Performance metrics
        perf_stats = self.performance_monitor.get_performance_stats()
        cv2.putText(frame, f"FPS: {perf_stats['current_fps']:.1f}", 
                   (50, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Avg Speed: {metrics.average_speed:.1f} km/h", 
                   (50, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def save_interval_data(self, metrics: TrafficMetrics):
        """Save interval data to DB and CSV with consistent production-ready formatting"""
        try:
            print(f"DEBUG: save_interval_data called at {datetime.now()}")

            self.db_manager.insert_traffic_data(metrics, "camera_001")

            # Generate KPI metrics
            kpi_metrics = self.kpi_calculator.generate_kpi_report(self.current_vehicles)
            self.db_manager.insert_kpi_data(kpi_metrics, "camera_001")

            # Build CSV record
            def fmt(x): return round(x or 0, 2)

            csv_data = {
                'timestamp': metrics.timestamp.isoformat(),
                'weighted_total': fmt(metrics.weighted_total),
                'traffic_status': metrics.traffic_status,
                'average_speed': fmt(metrics.average_speed),
                'congestion_level': fmt(metrics.congestion_level),
                'traffic_flow_efficiency': fmt(kpi_metrics.traffic_flow_efficiency),
                'average_waiting_time': fmt(kpi_metrics.average_waiting_time),
                'incident_detection_time': fmt(kpi_metrics.incident_detection_time),
                'fuel_consumption_index': fmt(kpi_metrics.fuel_consumption_index),
                'co2_emissions_index': fmt(kpi_metrics.co2_emissions_index),
                'system_uptime': fmt(kpi_metrics.system_uptime),
                'user_satisfaction_score': fmt(kpi_metrics.user_satisfaction_score),
            }

            for vtype in ['car', 'motorcycle', 'bus', 'truck', 'bicycle']:
                csv_data[f"vehicle_{vtype}_count"] = metrics.vehicle_counts.get(vtype, 0)

            for lane in range(4):
                csv_data[f'lane_{lane}_occupancy'] = fmt(metrics.lane_occupancy.get(lane, 0.0))

            df = pd.DataFrame([csv_data])
            csv_filename = f"traffic_data_{datetime.now().strftime('%Y%m%d')}.csv"
            csv_path = Path.cwd() / csv_filename

            if csv_path.exists():
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_path, index=False)

            logger.info(f"Exported traffic + KPI data to CSV: {csv_filename}")

        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            import traceback
            print(f"DEBUG: CSV error traceback:\n{traceback.format_exc()}")



        
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if hasattr(self.alert_manager, 'kafka_producer') and self.alert_manager.kafka_producer:
            self.alert_manager.kafka_producer.close()
        logger.info("Cleanup completed")

# Update the main run method to include KPI calculations
    def run(self, input_source: Optional[str] = None):
        """Main execution loop with polygonal ROI-based detection"""
        source = input_source or self.config['video']['input_source']
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {source}")
            return

        ret, first_frame = cap.read()
        if not ret:
            logger.error("Could not read first frame for ROI selection.")
            return

        resized_first_frame = self.preprocess_frame(first_frame)

        # 🖱️ ROI drawing setup
        cv2.imshow("Draw ROI - Left Click to Add Points, Right Click to Finish", resized_first_frame)
        cv2.setMouseCallback("Draw ROI - Left Click to Add Points, Right Click to Finish", roi_mouse_callback)

        # ⏳ Draw until right-click
        while not drawing_complete:
            temp = resized_first_frame.copy()
            for pt in roi_polygon_points:
                cv2.circle(temp, pt, 5, (0, 255, 255), -1)
            if len(roi_polygon_points) > 1:
                cv2.polylines(temp, [np.array(roi_polygon_points, np.int32)], False, (255, 0, 255), 2)
            cv2.imshow("Draw ROI - Left Click to Add Points, Right Click to Finish", temp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        cv2.destroyWindow("Draw ROI - Left Click to Add Points, Right Click to Finish")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.running = True
        frame_count = 0
        kpi_interval = 30
        logger.info(f"Starting traffic monitoring with ROI on source: {source}")

        try:
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break

                frame_count += 1
                if frame_count % self.config['video']['frame_skip'] != 0:
                    continue

                processed_frame = self.preprocess_frame(frame)
                vehicles = self.detect_and_track(processed_frame)
                self.kpi_calculator.detect_incidents(vehicles, self.previous_vehicles)
                self.previous_vehicles = vehicles.copy()
                self.current_vehicles = vehicles

                # ✅ ROI-based detection
                self.check_vehicles_in_roi(vehicles)

                metrics = self.calculate_traffic_metrics(vehicles)

                # ✅ Predict future flow
                try:
                    predicted_flow = self.traffic_predictor.predict({
                        'weighted_total': metrics.weighted_total,
                        'average_speed': metrics.average_speed,
                        'congestion_level': metrics.congestion_level,
                        'traffic_flow_efficiency': self.kpi_calculator.calculate_traffic_flow_efficiency(vehicles),
                        'average_waiting_time': self.kpi_calculator.calculate_waiting_time(vehicles, [{'x1': 300, 'y1': 300, 'x2': 700, 'y2': 400}])
                    })
                
                    pred_df = pd.DataFrame([{
                        'timestamp': datetime.now().isoformat(),
                        'predicted_flow': round(predicted_flow, 2)
                    }])
                    pred_path = Path("predicted_traffic_flow.csv")
                    if pred_path.exists():
                        pred_df.to_csv(pred_path, mode='a', header=False, index=False)
                    else:
                        pred_df.to_csv(pred_path, index=False)
                except Exception as e:
                    logger.warning(f"❌ Failed to save predicted flow: {e}")

                
                    predicted_flow = metrics.weighted_total

                # KPI interval
                current_time = time.time()
                if current_time - self.interval_start_time >= self.config['logging']['interval_seconds']:
                    if self.interval_counts:
                        metrics = self.calculate_traffic_metrics(vehicles)
                        self.save_interval_data(metrics)
                    self.interval_counts = defaultdict(int)
                    self.counted_vehicle_ids = set()
                    self.interval_start_time = current_time

                if current_time - self.last_kpi_time >= kpi_interval:
                    kpi_metrics = self.kpi_calculator.generate_kpi_report(vehicles)
                    self.db_manager.insert_kpi_data(kpi_metrics, "camera_001")
                    logger.info(f"KPI Update - Flow: {kpi_metrics.traffic_flow_efficiency:.1f} km/h, "
                                f"Congestion: {kpi_metrics.congestion_level:.1f}%, "
                                f"Satisfaction: {kpi_metrics.user_satisfaction_score:.1f}")
                    self.last_kpi_time = current_time
                else:
                    kpi_metrics = self.kpi_calculator.generate_kpi_report(vehicles)

                self.check_alerts(metrics, vehicles)
                self.performance_monitor.update_fps()
                self.performance_monitor.update_system_metrics()

                try:
                    fps = self.performance_monitor.get_performance_stats()['current_fps']
                    display_frame = self.draw_clean_overlay(processed_frame, metrics, kpi_metrics, predicted_flow, fps)
                    display_frame = self.draw_kpi_overlay(display_frame, kpi_metrics)
                except Exception as viz_err:
                    logger.error(f"Visualization error: {viz_err}")
                    display_frame = processed_frame.copy()
                
                cv2.imwrite("latest_frame.jpg", display_frame)

                cv2.imshow("Industrial Traffic Monitor with ROI", display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
            self.kpi_calculator.record_downtime(5.0)
        finally:
            
            if self.interval_counts:
                metrics = self.calculate_traffic_metrics(self.current_vehicles)
                self.save_interval_data(metrics)
                
            self.cleanup()
            cap.release()
            cv2.destroyAllWindows()


            
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if hasattr(self.alert_manager, 'kafka_producer') and self.alert_manager.kafka_producer:
            self.alert_manager.kafka_producer.close()
        logger.info("Cleanup completed")
        


def main():
    """Main entry point"""
    monitor = IndustrialTrafficMonitor("config.yaml")
    monitor.run()

if __name__ == "__main__":
    main()