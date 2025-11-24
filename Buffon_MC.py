import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from numba import njit, prange

# Paramètres
NEEDLE_LENGTH = 1.0  # Longueur de l'aiguille (modifiable)
LINE_SPACING = 1.0   # Espacement entre les lignes (modifiable)
N_NEEDLES_PER_FRAME = 10  # Augmenté grâce à l'optimisation
MAX_NEEDLES = 2000   # Augmenté grâce à l'optimisation

# Configuration de la figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Problème des Aiguilles de Buffon (L={NEEDLE_LENGTH}, D={LINE_SPACING})", 
             fontsize=14, fontweight='bold')

# Configuration du subplot gauche (aiguilles)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_aspect('equal')
ax1.set_title('Simulation des aiguilles')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Dessiner les lignes horizontales
n_lines = int(10 / LINE_SPACING) + 2
for i in range(n_lines):
    y = i * LINE_SPACING
    ax1.axhline(y=y, color='black', linewidth=2, alpha=0.3)

# Configuration du subplot droit (convergence)
theoretical_p = 2*NEEDLE_LENGTH/(np.pi*LINE_SPACING)
ax2.set_xlim(0, MAX_NEEDLES)
ax2.set_ylim(0, min(1.0, theoretical_p * 2.5))
ax2.set_title('Convergence vers 2L/(πD)')
ax2.set_xlabel('Nombre d\'aiguilles')
ax2.set_ylabel('Proportion d\'intersections (p)')
ax2.axhline(y=theoretical_p, color='blue', linestyle='--', linewidth=2, 
            label=f'Asymptote théorique: p = 2L/(πD) = {theoretical_p:.4f}')
ax2.grid(True, alpha=0.3)

# Données globales
n_intersections = 0
n_total = 0
history_n = []
history_ratio = []

# Collections pour un rendu optimisé
green_needles = []  # Stocke les segments qui intersectent
red_needles = []    # Stocke les segments qui n'intersectent pas
green_collection = LineCollection([], colors='green', linewidths=1.5, alpha=0.6)
red_collection = LineCollection([], colors='red', linewidths=1.5, alpha=0.6)
ax1.add_collection(green_collection)
ax1.add_collection(red_collection)

# Ligne pour le graphique
line_ratio, = ax2.plot([], [], 'r-', linewidth=2, label='p mesuré')
ax2.legend()

@njit(parallel=True, fastmath=True)
def generate_needles_batch(n, length, x_min, x_max, y_min, y_max):
    """Génère un batch d'aiguilles de manière optimisée avec Numba"""
    positions = np.empty((n, 4), dtype=np.float64)
    
    for i in prange(n):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        angle = np.random.uniform(0, 2 * np.pi)
        
        x1 = x
        y1 = y
        x2 = x + length * np.cos(angle)
        y2 = y + length * np.sin(angle)
        
        positions[i, 0] = x1
        positions[i, 1] = y1
        positions[i, 2] = x2
        positions[i, 3] = y2
    
    return positions

@njit(parallel=True, fastmath=True)
def check_intersections_batch(positions, spacing):
    """Vérifie les intersections pour un batch d'aiguilles de manière optimisée"""
    n = positions.shape[0]
    intersects = np.empty(n, dtype=np.bool_)
    
    for i in prange(n):
        y1 = positions[i, 1]
        y2 = positions[i, 3]
        
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        
        # Trouver les lignes qui pourraient être intersectées
        line_below = int(y_min / spacing) * spacing
        line_above = (int(y_max / spacing) + 1) * spacing
        
        # Vérifier si une ligne est entre y_min et y_max
        intersects[i] = False
        line_y = line_below
        while line_y <= line_above:
            if y_min <= line_y <= y_max:
                intersects[i] = True
                break
            line_y += spacing
    
    return intersects

def init():
    """Initialisation de l'animation"""
    return []

def update(frame):
    """Mise à jour pour chaque frame de l'animation"""
    global n_intersections, n_total, green_needles, red_needles
    
    if n_total >= MAX_NEEDLES:
        return []
    
    # Générer un batch d'aiguilles de manière optimisée
    n_to_add = min(N_NEEDLES_PER_FRAME, MAX_NEEDLES - n_total)
    positions = generate_needles_batch(n_to_add, NEEDLE_LENGTH, 0, 10, 0, 10)
    intersects = check_intersections_batch(positions, LINE_SPACING)
    
    # Séparer les aiguilles qui intersectent de celles qui n'intersectent pas
    for i in range(n_to_add):
        segment = [(positions[i, 0], positions[i, 1]), 
                   (positions[i, 2], positions[i, 3])]
        
        if intersects[i]:
            green_needles.append(segment)
            n_intersections += 1
        else:
            red_needles.append(segment)
        
        n_total += 1
    
    # Mettre à jour les collections (plus efficace que de dessiner ligne par ligne)
    green_collection.set_segments(green_needles)
    red_collection.set_segments(red_needles)
    
    # Réduire l'opacité après 100 aiguilles
    if n_total > 100:
        green_collection.set_alpha(0.4)
        red_collection.set_alpha(0.4)
    
    # Mettre à jour l'historique
    if n_total > 0:
        ratio = n_intersections / n_total
        history_n.append(n_total)
        history_ratio.append(ratio)
        
        # Mettre à jour le graphique (seulement tous les 5 frames pour optimiser)
        if frame % 5 == 0 or n_total >= MAX_NEEDLES:
            line_ratio.set_data(history_n, history_ratio)
            line_ratio.set_label(f'p mesuré = {ratio:.4f}')
            ax2.legend()
        
        # Estimation de π avec protection contre la divergence
        if ratio > 0.01:
            pi_estimate = 2*NEEDLE_LENGTH/(ratio*LINE_SPACING)
            pi_text = f'{pi_estimate:.5f}'
        else:
            pi_text = 'N/A'
        
        # Mettre à jour les titres
        percentage = 100 * n_intersections / n_total
        ax1.set_title(f'Aiguilles: {n_total} | Intersections: {n_intersections}')
        ax2.set_title(f'Estimation de π = 2L/(p×D) : {pi_text} | Attendu: {np.pi:.5f}')
    
    return []

# Créer l'animation
anim = FuncAnimation(fig, update, init_func=init, 
                    frames=MAX_NEEDLES // N_NEEDLES_PER_FRAME + 1,
                    interval=1, repeat=False, blit=False)

plt.tight_layout()
plt.show()