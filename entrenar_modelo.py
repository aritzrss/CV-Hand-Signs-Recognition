"""
Script para entrenar un modelo clasificador de gestos LSE
Usa varios algoritmos y guarda el mejor modelo
"""

import numpy as np
import pickle
import json
import glob
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class GestureModelTrainer:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64, 32), 
                                           max_iter=500, 
                                           random_state=42)
        }
        
        self.best_model = None
        self.best_model_name = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Carga todos los datos disponibles"""
        data_files = sorted(glob.glob('data/gestures_data_*.npy'))
        label_files = sorted(glob.glob('data/gestures_labels_*.npy'))
        
        if not data_files or not label_files:
            raise FileNotFoundError("No se encontraron archivos de datos. Ejecuta capture_gestures.py primero.")
        
        print(f"Encontrados {len(data_files)} archivos de datos")
        
        # Cargar el archivo más reciente
        data = np.load(data_files[-1])
        labels = np.load(label_files[-1])
        
        print(f"✓ Datos cargados: {data.shape[0]} muestras, {data.shape[1]} características")
        print(f"✓ Clases: {np.unique(labels)}")
        
        return data, labels
    
    def prepare_data(self, data, labels):
        """Prepara los datos para entrenamiento"""
        # Codificar las etiquetas
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Dividir en train y test
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels_encoded, 
            test_size=0.2, 
            random_state=42,
            stratify=labels_encoded
        )
        
        print(f"\n✓ Conjunto de entrenamiento: {X_train.shape[0]} muestras")
        print(f"✓ Conjunto de prueba: {X_test.shape[0]} muestras")
        
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Entrena y evalúa todos los modelos"""
        results = {}
        
        print("\n" + "="*60)
        print("ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\n🔄 Entrenando {name}...")
            
            # Entrenar
            model.fit(X_train, y_train)
            
            # Validación cruzada
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Predicciones
            y_pred = model.predict(X_test)
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Seleccionar el mejor modelo
        best_name = max(results, key=lambda x: results[x]['accuracy'])
        self.best_model = results[best_name]['model']
        self.best_model_name = best_name
        
        print("\n" + "="*60)
        print(f"✨ MEJOR MODELO: {best_name}")
        print(f"   Accuracy: {results[best_name]['accuracy']:.4f}")
        print("="*60)
        
        return results
    
    def plot_results(self, results, y_test):
        """Genera visualizaciones de los resultados"""
        print("\n📊 Generando visualizaciones...")
        
        # 1. Comparación de modelos
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_means = [results[name]['cv_mean'] for name in model_names]
        
        # Gráfico de barras de accuracy
        axes[0].bar(model_names, accuracies, color=['#2ecc71' if name == self.best_model_name else '#3498db' for name in model_names])
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Comparación de Modelos - Test Set')
        axes[0].set_ylim([0, 1])
        axes[0].tick_params(axis='x', rotation=45)
        
        for i, (name, acc) in enumerate(zip(model_names, accuracies)):
            axes[0].text(i, acc + 0.02, f'{acc:.3f}', ha='center')
        
        # Gráfico de barras de CV
        axes[1].bar(model_names, cv_means, color='#9b59b6')
        axes[1].set_ylabel('CV Score')
        axes[1].set_title('Validación Cruzada (5-Fold)')
        axes[1].set_ylim([0, 1])
        axes[1].tick_params(axis='x', rotation=45)
        
        for i, (name, score) in enumerate(zip(model_names, cv_means)):
            axes[1].text(i, score + 0.02, f'{score:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('data/model_comparison.png', dpi=300, bbox_inches='tight')
        print("   ✓ Guardado: data/model_comparison.png")
        
        # 2. Matriz de confusión del mejor modelo
        y_pred_best = results[self.best_model_name]['predictions']
        cm = confusion_matrix(y_test, y_pred_best)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title(f'Matriz de Confusión - {self.best_model_name}')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        plt.savefig('data/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("   ✓ Guardado: data/confusion_matrix.png")
        
        plt.close('all')
    
    def generate_report(self, results, y_test):
        """Genera reporte detallado de clasificación"""
        print("\n📋 Reporte de Clasificación del Mejor Modelo:")
        print("="*60)
        
        y_pred_best = results[self.best_model_name]['predictions']
        report = classification_report(
            y_test, 
            y_pred_best,
            target_names=self.label_encoder.classes_,
            digits=4
        )
        
        print(report)
        
        # Guardar reporte
        with open('data/classification_report.txt', 'w') as f:
            f.write(f"REPORTE DE CLASIFICACIÓN - {self.best_model_name}\n")
            f.write("="*60 + "\n\n")
            f.write(report)
        
        print("\n   ✓ Guardado: data/classification_report.txt")
    
    def save_model(self):
        """Guarda el mejor modelo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'label_encoder': self.label_encoder,
            'classes': self.label_encoder.classes_.tolist(),
            'timestamp': timestamp
        }
        
        filename = f'data/gesture_model_{timestamp}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        # También guardar una versión "latest"
        with open('data/gesture_model_latest.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Modelo guardado:")
        print(f"   ✓ {filename}")
        print(f"   ✓ data/gesture_model_latest.pkl")
    
    def run(self):
        """Ejecuta el proceso completo de entrenamiento"""
        print("\n" + "="*60)
        print("ENTRENADOR DE MODELOS DE GESTOS LSE")
        print("="*60)
        
        # Cargar datos
        data, labels = self.load_data()
        
        # Preparar datos
        X_train, X_test, y_train, y_test = self.prepare_data(data, labels)
        
        # Entrenar y evaluar
        results = self.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Generar reporte
        self.generate_report(results, y_test)
        
        # Visualizaciones
        self.plot_results(results, y_test)
        
        # Guardar modelo
        self.save_model()
        
        print("\n✅ ¡Entrenamiento completado exitosamente!")
        print("\nArchivos generados:")
        print("   • data/gesture_model_latest.pkl")
        print("   • data/model_comparison.png")
        print("   • data/confusion_matrix.png")
        print("   • data/classification_report.txt")

if __name__ == "__main__":
    trainer = GestureModelTrainer()
    trainer.run()