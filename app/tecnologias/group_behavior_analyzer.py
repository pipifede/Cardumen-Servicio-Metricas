# group_behavior_analyzer.py
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from collections import defaultdict, deque
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import math

@dataclass
class GroupEvent:
    """Representa un evento de comportamiento grupal"""
    event_type: str
    group_ids: List[int]
    start_frame: int
    end_frame: int
    confidence: float
    centroid: Tuple[float, float]
    duration: float
    metadata: Dict

class GroupBehaviorAnalyzer:
    def __init__(self, 
                 proximity_threshold: float = 200.0,
                 min_group_size: int = 2,
                 max_group_size: int = 10,
                 temporal_window: int = 60,
                 min_duration: int = 15):
        """
        Inicializa el analizador de comportamiento grupal
        
        Args:
            proximity_threshold: Distancia máxima para considerar objetos como grupo
            min_group_size: Tamaño mínimo de grupo
            max_group_size: Tamaño máximo de grupo
            temporal_window: Ventana temporal para análisis (frames)
            min_duration: Duración mínima para considerar un comportamiento válido
        """
        self.proximity_threshold = proximity_threshold
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.temporal_window = temporal_window
        self.min_duration = min_duration
        
        # Historiales para análisis temporal
        self.position_history = defaultdict(deque)
        self.group_history = defaultdict(deque)
        self.velocity_history = defaultdict(deque)
        
        # Eventos detectados
        self.detected_events = []
        self.active_groups = {}
        
        # Configuraciones de comportamientos
        self.behavior_configs = {
            'reunion': {
                'min_objects': 3,
                'max_distance': 80,
                'min_duration': 20,
                'velocity_threshold': 5.0
            },
            'siguiendo': {
                'min_objects': 2,
                'max_distance': 150,
                'min_duration': 30,
                'direction_similarity': 0.7
            },
            'dispersion': {
                'min_objects': 3,
                'max_distance': 200,
                'min_duration': 15,
                'velocity_threshold': 8.0
            },
            'convergen': {
                'min_objects': 2,
                'max_distance': 120,
                'min_duration': 25,
                'convergence_rate': 2.0
            }
        }
    
    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calcula distancia euclidiana entre dos posiciones"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def calculate_velocity(self, positions: deque) -> Tuple[float, float]:
        """Calcula velocidad basada en posiciones históricas"""
        if len(positions) < 2:
            return (0.0, 0.0)
        
        recent_pos = list(positions)[-5:]  # Últimas 5 posiciones
        if len(recent_pos) < 2:
            return (0.0, 0.0)
        
        dx = recent_pos[-1][0] - recent_pos[0][0]
        dy = recent_pos[-1][1] - recent_pos[0][1]
        dt = len(recent_pos) - 1
        
        return (dx/dt, dy/dt)
    
    def find_groups(self, frame_detections: List[Dict]) -> List[List[int]]:
        """Encuentra grupos usando clustering DBSCAN (alg usado para encontrar grupos de puntos fisicamente cercanos en un frame)"""
        if len(frame_detections) < self.min_group_size:
            return []
        
        # Extraer posiciones centrales
        positions = []
        object_ids = []
        
        for detection in frame_detections:
            if detection.get('id') is not None:
                x1, y1, x2, y2 = detection['xyxy']
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                positions.append([center_x, center_y])
                object_ids.append(detection['id'])
        
        if len(positions) < self.min_group_size:
            return []
        
        # Aplicar DBSCAN clustering
        clustering = DBSCAN(
            eps=self.proximity_threshold,
            min_samples=self.min_group_size
        ).fit(positions)
        
        # Organizar grupos
        groups = defaultdict(list)
        for i, label in enumerate(clustering.labels_):
            if label != -1:  # -1 significa ruido (no agrupado)
                groups[label].append(object_ids[i])
        
        # Filtrar grupos por tamaño
        valid_groups = []
        for group_ids in groups.values():
            if self.min_group_size <= len(group_ids) <= self.max_group_size:
                valid_groups.append(group_ids)
        
        return valid_groups
    
    def analyze_gathering_behavior(self, group_ids: List[int], frame_num: int) -> Optional[GroupEvent]:
        """Detecta comportamiento de congregación"""
        config = self.behavior_configs['reunion']
        
        if len(group_ids) < config['min_objects']:
            return None
        
        # Verificar velocidades bajas (objetos relativamente estáticos)
        low_velocity_count = 0
        positions = []
        
        for obj_id in group_ids:
            if obj_id in self.velocity_history:
                vx, vy = self.calculate_velocity(self.position_history[obj_id])
                speed = math.sqrt(vx**2 + vy**2)
                if speed < config['velocity_threshold']:
                    low_velocity_count += 1
                
                if self.position_history[obj_id]:
                    positions.append(self.position_history[obj_id][-1])
        
        # Mayoría de objetos deben tener velocidad baja
        if low_velocity_count < len(group_ids) * 0.7:
            return None
        
        # Calcular centro del grupo
        if positions:
            centroid = (
                sum(pos[0] for pos in positions) / len(positions),
                sum(pos[1] for pos in positions) / len(positions)
            )
        else:
            centroid = (0, 0)
        
        return GroupEvent(
            event_type='reunion',
            group_ids=group_ids,
            start_frame=frame_num,
            end_frame=frame_num,
            confidence=0.8,
            centroid=centroid,
            duration=1,
            metadata={'average_speed': sum(math.sqrt(vx**2 + vy**2) for vx, vy in [self.calculate_velocity(self.position_history[obj_id]) for obj_id in group_ids]) / len(group_ids)}
        )
    
    def analyze_siguiendo_behavior(self, group_ids: List[int], frame_num: int) -> Optional[GroupEvent]:
        """Detecta comportamiento de seguimiento"""
        config = self.behavior_configs['siguiendo']
        
        if len(group_ids) < config['min_objects']:
            return None
        
        # Calcular vectores de dirección
        directions = []
        positions = []
        
        for obj_id in group_ids:
            if obj_id in self.velocity_history and len(self.position_history[obj_id]) >= 2:
                vx, vy = self.calculate_velocity(self.position_history[obj_id])
                if vx != 0 or vy != 0:
                    # Normalizar vector 
                    magnitude = math.sqrt(vx**2 + vy**2)
                    directions.append((vx/magnitude, vy/magnitude))
                    positions.append(self.position_history[obj_id][-1])
        
        if len(directions) < config['min_objects']:
            return None
        
        # Calcular similitud de direcciones
        avg_direction = (
            sum(d[0] for d in directions) / len(directions),
            sum(d[1] for d in directions) / len(directions)
        )
        
        similarity_scores = []
        for dx, dy in directions:
            dot_product = dx * avg_direction[0] + dy * avg_direction[1]
            similarity_scores.append(abs(dot_product))
        
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        if avg_similarity < config['direction_similarity']:
            return None
        
        # Calcular centroide
        centroid = (
            sum(pos[0] for pos in positions) / len(positions),
            sum(pos[1] for pos in positions) / len(positions)
        ) if positions else (0, 0)
        
        return GroupEvent(
            event_type='siguiendo',
            group_ids=group_ids,
            start_frame=frame_num,
            end_frame=frame_num,
            confidence=avg_similarity,
            centroid=centroid,
            duration=1,
            metadata={'direction_similarity': avg_similarity, 'average_direction': avg_direction}
        )
    
    def analyze_dispersion_behavior(self, group_ids: List[int], frame_num: int) -> Optional[GroupEvent]:
        """Detecta comportamiento de dispersión"""
        config = self.behavior_configs['dispersion']
        
        if len(group_ids) < config['min_objects']:
            return None
        
        # Calcular centro del grupo y velocidades
        positions = []
        velocities = []
        
        for obj_id in group_ids:
            if obj_id in self.position_history and self.position_history[obj_id]:
                pos = self.position_history[obj_id][-1]
                positions.append(pos)
                
                vx, vy = self.calculate_velocity(self.position_history[obj_id])
                velocities.append((vx, vy))
        
        if len(positions) < config['min_objects']:
            return None
        
        # Calcular centroide del grupo
        centroid = (
            sum(pos[0] for pos in positions) / len(positions),
            sum(pos[1] for pos in positions) / len(positions)
        )
        
        # Verificar si los objetos se alejan del centro
        diverging_count = 0
        for i, (pos, vel) in enumerate(zip(positions, velocities)):
            # Vector del centro hacia el objeto
            to_object = (pos[0] - centroid[0], pos[1] - centroid[1])
            
            # Verificar si la velocidad apunta lejos del centro
            dot_product = to_object[0] * vel[0] + to_object[1] * vel[1]
            speed = math.sqrt(vel[0]**2 + vel[1]**2)
            
            if dot_product > 0 and speed > config['velocity_threshold']:
                diverging_count += 1
        
        # Mayoría debe estar divergiendo
        if diverging_count < len(group_ids) * 0.6:
            return None
        
        return GroupEvent(
            event_type='dispersion',
            group_ids=group_ids,
            start_frame=frame_num,
            end_frame=frame_num,
            confidence=diverging_count / len(group_ids),
            centroid=centroid,
            duration=1,
            metadata={'diverging_objects': diverging_count, 'total_objects': len(group_ids)}
        )
    
    def analyze_converging_behavior(self, group_ids: List[int], frame_num: int) -> Optional[GroupEvent]:
        """Detecta comportamiento de convergencia"""
        config = self.behavior_configs['convergen']
        
        if len(group_ids) < config['min_objects']:
            return None
        
        # Similar a dispersion pero en dirección opuesta
        positions = []
        velocities = []
        
        for obj_id in group_ids:
            if obj_id in self.position_history and self.position_history[obj_id]:
                pos = self.position_history[obj_id][-1]
                positions.append(pos)
                
                vx, vy = self.calculate_velocity(self.position_history[obj_id])
                velocities.append((vx, vy))
        
        if len(positions) < config['min_objects']:
            return None
        
        # Calcular centroide del grupo
        centroid = (
            sum(pos[0] for pos in positions) / len(positions),
            sum(pos[1] for pos in positions) / len(positions)
        )
        
        # Verificar si los objetos se acercan al centro
        converging_count = 0
        for i, (pos, vel) in enumerate(zip(positions, velocities)):
            # Vector del objeto hacia el centro
            to_center = (centroid[0] - pos[0], centroid[1] - pos[1])
            
            # Verificar si la velocidad apunta hacia el centro
            dot_product = to_center[0] * vel[0] + to_center[1] * vel[1]
            speed = math.sqrt(vel[0]**2 + vel[1]**2)
            
            if dot_product > 0 and speed > config['convergence_rate']:
                converging_count += 1
        
        # Mayoría debe estar convergiendo
        if converging_count < len(group_ids) * 0.6:
            return None
        
        return GroupEvent(
            event_type='convergen',
            group_ids=group_ids,
            start_frame=frame_num,
            end_frame=frame_num,
            confidence=converging_count / len(group_ids),
            centroid=centroid,
            duration=1,
            metadata={'converging_objects': converging_count, 'total_objects': len(group_ids)}
        )
    
    def update_histories(self, frame_detections: List[Dict], frame_num: int):
        """Actualiza los historiales de posición y velocidad"""
        for detection in frame_detections:
            obj_id = detection.get('id')
            if obj_id is not None:
                x1, y1, x2, y2 = detection['xyxy']
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Actualizar historial de posiciones
                self.position_history[obj_id].append((center_x, center_y))
                if len(self.position_history[obj_id]) > self.temporal_window:
                    self.position_history[obj_id].popleft()
                
                # Calcular y actualizar velocidad
                velocity = self.calculate_velocity(self.position_history[obj_id])
                self.velocity_history[obj_id].append(velocity)
                if len(self.velocity_history[obj_id]) > self.temporal_window:
                    self.velocity_history[obj_id].popleft()
    
    def analyze_frame(self, frame_detections: List[Dict], frame_num: int) -> List[GroupEvent]:
        """Analiza un frame y detecta comportamientos grupales"""
        # Actualizar historiales
        self.update_histories(frame_detections, frame_num)
        
        # Encontrar grupos
        groups = self.find_groups(frame_detections)
        
        # Analizar comportamientos para cada grupo
        frame_events = []
        
        for group_ids in groups:
            # Analizar diferentes tipos de comportamiento
            behaviors = [
                self.analyze_gathering_behavior(group_ids, frame_num),
                self.analyze_siguiendo_behavior(group_ids, frame_num),
                self.analyze_dispersion_behavior(group_ids, frame_num),
                self.analyze_converging_behavior(group_ids, frame_num)
            ]
            
            # Añadir eventos válidos
            for behavior in behaviors:
                if behavior is not None:
                    frame_events.append(behavior)
                    
                    # Actualizar historial de grupos
                    group_key = tuple(sorted(group_ids))
                    self.group_history[group_key].append(behavior)
                    if len(self.group_history[group_key]) > self.temporal_window:
                        self.group_history[group_key].popleft()
        
        return frame_events
    
    def consolidate_events(self) -> List[GroupEvent]:
        """Consolida eventos temporales en comportamientos duraderos"""
        consolidated_events = []
        
        for group_key, events in self.group_history.items():
            if len(events) < self.min_duration:
                continue
            
            # Agrupar eventos consecutivos del mismo tipo
            current_event = None
            event_start = None
            
            for i, event in enumerate(events):
                if current_event is None or current_event.event_type != event.event_type:
                    # Guardar evento anterior si existe
                    if current_event is not None and (i - event_start) >= self.min_duration:
                        consolidated_event = GroupEvent(
                            event_type=current_event.event_type,
                            group_ids=current_event.group_ids,
                            start_frame=event_start,
                            end_frame=i - 1,
                            confidence=current_event.confidence,
                            centroid=current_event.centroid,
                            duration=i - event_start,
                            metadata=current_event.metadata
                        )
                        consolidated_events.append(consolidated_event)
                    
                    # Iniciar nuevo evento
                    current_event = event
                    event_start = i
            
            # Guardar último evento si es válido
            if current_event is not None and (len(events) - event_start) >= self.min_duration:
                consolidated_event = GroupEvent(
                    event_type=current_event.event_type,
                    group_ids=current_event.group_ids,
                    start_frame=event_start,
                    end_frame=len(events) - 1,
                    confidence=current_event.confidence,
                    centroid=current_event.centroid,
                    duration=len(events) - event_start,
                    metadata=current_event.metadata
                )
                consolidated_events.append(consolidated_event)
        
        return consolidated_events
    
    def get_behavior_summary(self) -> Dict:
        """Genera un resumen de todos los comportamientos detectados"""
        consolidated_events = self.consolidate_events()
        
        behavior_counts = defaultdict(int)
        behavior_durations = defaultdict(list)
        
        for event in consolidated_events:
            behavior_counts[event.event_type] += 1
            behavior_durations[event.event_type].append(event.duration)
        
        summary = {
            'total_events': len(consolidated_events),
            'behavior_types': dict(behavior_counts),
            'events': [
                {
                    'type': event.event_type,
                    'group_ids': event.group_ids,
                    'start_frame': event.start_frame,
                    'end_frame': event.end_frame,
                    'duration': event.duration,
                    'confidence': event.confidence,
                    'centroid': event.centroid,
                    'metadata': event.metadata
                }
                for event in consolidated_events
            ]
        }
        
        # Agregar estadísticas
        for behavior_type, durations in behavior_durations.items():
            summary[f'{behavior_type}_avg_duration'] = sum(durations) / len(durations)
            summary[f'{behavior_type}_max_duration'] = max(durations)
        
        return summary

def visualize_group_behavior(frame: np.ndarray, 
                           groups: List[List[int]], 
                           events: List[GroupEvent],
                           detections: List[Dict]) -> np.ndarray:
    """Visualiza comportamientos grupales en el frame"""
    result_frame = frame.copy()
    
    # Colores para diferentes comportamientos
    behavior_colors = {
        'reunion': (0, 255, 0),      # Verde
        'siguiendo': (255, 0, 0),      # Azul
        'dispersion': (0, 0, 255),     # Rojo
        'convergen': (255, 255, 0)    # Cian
    }
    
    # Dibujar grupos
    for group_ids in groups:
        # Encontrar posiciones de objetos en el grupo
        group_positions = []
        for detection in detections:
            if detection.get('id') in group_ids:
                x1, y1, x2, y2 = detection['xyxy']
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                group_positions.append((center_x, center_y))
        
        # Dibujar conexiones entre objetos del grupo
        if len(group_positions) > 1:
            for i in range(len(group_positions)):
                for j in range(i + 1, len(group_positions)):
                    cv2.line(result_frame, group_positions[i], group_positions[j], (128, 128, 128), 1)
    
    # Dibujar eventos de comportamiento
    for event in events:
        color = behavior_colors.get(event.event_type, (255, 255, 255))
        centroid = (int(event.centroid[0]), int(event.centroid[1]))
        
        # Dibujar círculo en el centroide
        cv2.circle(result_frame, centroid, 30, color, 3)
        cv2.circle(result_frame, centroid, 30, color, 2)
        # Añadir etiqueta
        label = f"{event.event_type.upper()}: {event.confidence:.2f}"
        cv2.putText(result_frame, label, (centroid[0] - 60, centroid[1] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return result_frame