import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import csv

# Dataset pour value network 9x9
class GoValueDataset9x9(Dataset):
    def __init__(self, npz_path=None, augment=False):
        data = np.load(npz_path)
        self.X = data['X'].astype('float32')  # (N, 2, 9, 9)
        self.y = data['value_targets'].astype('float32')  # (N,)
        self.augment = augment
        bs = self.X.shape[2]
        self.bs = bs
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        x = self.X[idx].copy()
        v = float(self.y[idx])
        
        if self.augment:
            # Déterminer la transformation aléatoire (8 possibilités - groupe diédral D4)
            transform_type = np.random.randint(0, 8)
            k_rot = transform_type % 4  # Nombre de rotations de 90°
            do_flip = transform_type >= 4  # Si True, on fait un flip horizontal
            
            # Appliquer les rotations
            if k_rot > 0:
                x = np.rot90(x, k=k_rot, axes=(1, 2))
                
            # Appliquer le flip horizontal
            if do_flip:
                x = np.flip(x, axis=2)
                
        # Conversion en tenseurs PyTorch
        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        # Conversion de la valeur v de [-1, 1] à [0, 1]
        v = (v + 1) / 2  # Remap v from [-1, 1] to [0, 1]
        v = torch.tensor(v, dtype=torch.float32)
        
        return x, v

# ValueNet avec architecture améliorée
class ValueNet9x9(nn.Module):
    def __init__(self, board_size=9, n_blocks=6, filters=96):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        # Blocs résiduels améliorés avec attention
        self.resblocks = nn.Sequential(*[EnhancedResidualBlock(filters) for _ in range(n_blocks)])
        
        # Branche d'attention globale
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(filters, filters, 1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Value head améliorée
        self.value_head = nn.Sequential(
            nn.Conv2d(filters, filters, 1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )
        
        # Value network final
        self.value_fc = nn.Sequential(
            nn.Linear(filters * board_size * board_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Change activation to Sigmoid for output in [0, 1]
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.resblocks(x)
        
        # Appliquer l'attention globale
        attention = self.global_attention(x)
        x = x * attention
        
        x = self.value_head(x)
        x = x.view(x.size(0), -1)
        value = self.value_fc(x).squeeze(-1)
        return value

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + res)

class EnhancedResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Squeeze-and-Excitation block pour attention par canal
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Appliquer SE attention
        se_weight = self.se(out)
        out = out * se_weight
        
        return F.relu(out + res)

class EarlyStopper:
    def __init__(self, patience=50, min_delta=1e-4, base_min_lr_factor=10):
        self.patience = patience
        self.base_min_delta = min_delta  # Valeur de base
        self.min_delta = min_delta       # Valeur actuelle (potentiellement ajustée)
        self.base_min_lr_factor = base_min_lr_factor
        self.min_lr_factor = base_min_lr_factor  # Sera dynamiquement ajusté
        self.best_loss = math.inf
        self.counter = 0
        
    def adjust_for_lr(self, current_lr):
        """Ajuste min_delta et min_lr_factor en fonction du learning rate actuel"""
        # Ajuster dynamiquement le min_lr_factor selon le LR
        if current_lr < 1e-9:
            # LR ultra-faible (< 1e-9): facteur très élevé
            self.min_lr_factor = 10000
        elif current_lr < 1e-8:
            # LR très faible (< 1e-8): facteur élevé
            self.min_lr_factor = 5000
        elif current_lr < 1e-7:
            # LR faible (< 1e-7): facteur plus élevé
            self.min_lr_factor = 1000
        elif current_lr < 1e-6:
            # LR assez faible (< 1e-6): facteur modérément élevé
            self.min_lr_factor = 500
        elif current_lr < 1e-5:
            # LR moyen-faible (< 1e-5): facteur légèrement élevé
            self.min_lr_factor = 100
        else:
            # LR normal: facteur standard
            self.min_lr_factor = self.base_min_lr_factor
        
        # Calculer le min_delta avec le nouveau facteur
        if current_lr < self.base_min_delta / self.min_lr_factor:
            self.min_delta = current_lr * self.min_lr_factor
        else:
            self.min_delta = self.base_min_delta
            
        print(f"Adjusted min_lr_factor: {self.min_lr_factor}, min_delta: {self.min_delta:.10e}")
        return self.min_delta
        
    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def refresh_model(net, optimizer, device):
    """
    Effectue un rafraîchissement du modèle pour améliorer les performances.
    
    Cette technique sauvegarde et recharge le modèle, ce qui a pour effet:
    - De réinitialiser partiellement les états internes de l'optimiseur
    - De stabiliser les poids du modèle pour améliorer la convergence
    - De fournir un nouvel élan d'apprentissage en sortant des plateaux
    
    Args:
        net: Le modèle à rafraîchir
        optimizer: L'optimiseur associé au modèle
        device: L'appareil sur lequel le modèle est chargé
    
    Returns:
        tuple: (modèle rafraîchi, optimiseur modifié)
    """
    print("🔄 Performing model refreshing...")
    
    # 1. Sauvegarder les poids du modèle dans un fichier temporaire
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'refresh_tmp_model.pth')
    torch.save(net.state_dict(), tmp_path)
    
    # 2. Réinitialiser partiellement l'optimiseur (réduire le momentum)
    for param_group in optimizer.param_groups:
        for p in param_group['params']:
            param_state = optimizer.state.get(p)
            if param_state is not None and 'momentum_buffer' in param_state:
                # Réduire le momentum de moitié pour permettre de nouveaux chemins d'optimisation
                param_state['momentum_buffer'] *= 0.5
    
    # 3. Recharger le modèle
    state_dict = torch.load(tmp_path)
    net.load_state_dict(state_dict)
    
    # 4. Nettoyer le fichier temporaire
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    
    print("✅ Model refreshed successfully")
    return net, optimizer

def load_metrics_from_csv(csv_path):
    """
    Charge les métriques existantes depuis un fichier CSV
    
    Returns:
        tuple: (headers, metrics, last_epoch, last_lr)
            - headers: Liste des en-têtes de colonnes
            - metrics: Liste de toutes les lignes de métriques
            - last_epoch: Dernière époque enregistrée
            - last_lr: Dernier learning rate utilisé
    """
    if not os.path.exists(csv_path):
        return None, [], 0, None
    
    metrics = []
    headers = None
    last_lr = None
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Lire les en-têtes
        
        # Trouver l'index de la colonne Learning Rate
        lr_index = None
        for i, header in enumerate(headers):
            if header == 'Learning Rate':
                lr_index = i
                break
        
        for row in reader:
            metrics.append(row)
            # Extraire le learning rate de la dernière ligne
            if lr_index is not None and len(row) > lr_index:
                try:
                    last_lr = float(row[lr_index])
                except ValueError:
                    pass
    
    if not metrics:
        return headers, [], 0, None
    
    # Obtenir la dernière époque
    last_epoch = int(metrics[-1][0])
    
    return headers, metrics, last_epoch, last_lr


def plot_training_metrics(csv_path, output_dir, csv_headers):
    """
    Fonction dédiée pour tracer les graphiques d'entraînement de manière robuste
    """
    all_data = {header: [] for header in csv_headers}
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for line in reader:
                for header in csv_headers:
                    if header in line and line[header] and line[header].strip():
                        try:
                            all_data[header].append(float(line[header]))
                        except (ValueError, TypeError):
                            pass
    except Exception as e:
        print(f"Erreur lors de la lecture du CSV: {e}")
        return
    
    # Vérifier qu'on a des données à tracer
    if not all_data['Epoch']:
        print("Pas de données à tracer")
        return
    
    # Configuration des graphiques à tracer
    plot_config = [
        (['Train Loss', 'Val Loss'], None, 'Loss'),
        (['MSE', 'MAE'], None, 'Error Metrics'),
        (['R²'], None, 'R-squared')
    ]
    
    plt.figure(figsize=(15, 15))
    
    for i, (metrics, ylim, title) in enumerate(plot_config):
        plt.subplot(len(plot_config), 1, i+1)
        
        for metric in metrics:
            if metric in all_data and all_data[metric]:
                # Tracer uniquement les données disponibles
                metric_data = all_data[metric]
                plot_epochs = list(range(1, len(metric_data) + 1))
                plt.plot(plot_epochs, metric_data, label=metric)
                    
        plt.title(title)
        plt.xlabel('Epochs')
        if ylim:
            plt.ylim(ylim)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics.png'))
    plt.close()


def train_value_9x9(npz_path, batch_size=128, lr=3e-4, epochs=1000, val_split=0.1, device=None,
                    resume_from=None, continue_output_dir=None, ultra_low_lr=False,
                    existing_metrics=None, start_epoch=1, last_lr=None):
    """
    Fonction principale d'entraînement du modèle de valeur pour Go 9x9
    
    Args:
        npz_path: Chemin vers le dataset NPZ
        batch_size: Taille des batchs
        lr: Taux d'apprentissage initial
        epochs: Nombre d'époques d'entraînement
        val_split: Proportion des données pour la validation
        device: Appareil à utiliser ('cuda' ou 'cpu')
        resume_from: Chemin vers le modèle à charger pour reprendre l'entraînement
        continue_output_dir: Répertoire pour continuer l'entraînement
        ultra_low_lr: Si True, utilise un LR très bas (5e-6) pour la reprise
        existing_metrics: Métriques existantes à utiliser
        start_epoch: Époque de démarrage
        last_lr: Dernier learning rate utilisé
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Définir le répertoire de sortie
    if continue_output_dir:
        output_dir = continue_output_dir
        print(f"Using output directory: {output_dir}")
    else:
        timestamp = __import__('datetime').datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Starting training run at: {timestamp}")
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'training_logs_value9x9_v5_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
    
    # Checkpointing périodique
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialisation des en-têtes CSV et du fichier
    csv_headers = ['Epoch', 'Train Loss', 'Val Loss', 'MSE', 'MAE', 'R²', 'Learning Rate']
    csv_path = os.path.join(output_dir, 'metrics_all.csv')
    
    # Utiliser les métriques existantes si elles sont fournies
    if existing_metrics is None:
        existing_metrics = []
    
    # Dataset avec augmentation améliorée
    ds = GoValueDataset9x9(npz_path, augment=True)
    
    # Analyse des statistiques de valeur
    values = ds.y
    print(f"\nStatistiques des valeurs cibles:")
    print(f"  Moyenne: {np.mean(values):.4f}")
    print(f"  Médiane: {np.median(values):.4f}")
    print(f"  Min: {np.min(values):.4f}")
    print(f"  Max: {np.max(values):.4f}")
    print(f"  Écart-type: {np.std(values):.4f}")
    
    n_val = int(len(ds)*val_split)
    n_train = len(ds)-n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    
    # DataLoaders optimisés pour entraînement long
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=os.cpu_count(),  # Utiliser tous les cœurs disponibles
        persistent_workers=True,
        drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=os.cpu_count())
    
    # Variables pour l'état de l'optimiseur et du scheduler
    checkpoint = None
    has_optimizer_state = False
    has_scheduler_state = False
    
    # Créer ou charger le modèle
    if resume_from:
        print(f"Loading model from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from)
        
        # Vérifier si c'est un dictionnaire d'état ou un checkpoint complet
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # C'est un checkpoint complet
            model_state_dict = checkpoint['model_state_dict']
            has_optimizer_state = 'optimizer_state_dict' in checkpoint
            has_scheduler_state = 'scheduler_state_dict' in checkpoint
        else:
            # C'est juste un état du modèle
            model_state_dict = checkpoint
            
        # Créer l'instance du modèle
        net = ValueNet9x9().to(device)

        # Convertir les clés si nécessaire (modèle DataParallel)
        keys = list(model_state_dict.keys())
        if len(keys) > 0:
            if keys[0].startswith('module.'):
                print("Converting DataParallel model to single-device model")
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in model_state_dict.items():
                    name = k[7:] if k.startswith('module.') else k  # remove 'module.'
                    new_state_dict[name] = v
                model_state_dict = new_state_dict

        # Tenter le chargement avec gestion d'erreur robuste
        try:
            net.load_state_dict(model_state_dict)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Warning: Strict loading failed: {e}")
            print("Attempting non-strict loading...")
            net.load_state_dict(model_state_dict, strict=False)
            print("Model loaded with non-strict loading")
    else:
        print("Initializing new model")
        net = ValueNet9x9().to(device)
    
    # Configuration GPU
    if device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        net = nn.DataParallel(net)
    else:
        print(f"Using single {device}")
        
    # Déterminer le learning rate initial
    initial_lr = lr
    
    # Si on reprend l'entraînement, utiliser le learning rate du checkpoint si disponible
    if resume_from and last_lr is not None:
        initial_lr = last_lr
        print(f"Using provided learning rate: {initial_lr:.14f}")
    # Ou si ultra_low_lr est demandé
    elif resume_from and ultra_low_lr:
        initial_lr = 5e-6  # Commencer très bas
        print(f"Using ultra-low starting learning rate: {initial_lr:.14f}")

    # Optimiseur RAdam avec weight decay adapté
    optimizer = optim.RAdam(net.parameters(), lr=initial_lr, weight_decay=1e-4)
    
    # Charger l'état de l'optimiseur si disponible
    if resume_from and has_optimizer_state:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
    
    # Scheduler ReduceLROnPlateau pour adaptation du LR, sans limite inférieure
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=7,
        verbose=True,
        min_lr=0,  # Pas de limite inférieure pour le LR
        cooldown=3
    )
    
    # Charger l'état du scheduler si disponible
    if resume_from and has_scheduler_state:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}")
    
    # Early stopper avec patience élevée
    stopper = EarlyStopper(patience=200)
    
    # Critère MSE
    criterion = nn.MSELoss()
    
    print(f"Training for {epochs} epochs starting from epoch {start_epoch}")
    
    # S'assurer que le fichier CSV existe avec les en-têtes appropriés
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
            
            # Écrire les métriques existantes si disponibles
            if existing_metrics:
                for old_row in existing_metrics:
                    if int(old_row[0]) < start_epoch:
                        writer.writerow(old_row)
                print(f"Initialized metrics file with metrics up to epoch {start_epoch - 1}")
            else:
                print(f"Initialized empty metrics file")

    for epoch in range(start_epoch, start_epoch + epochs):
        # Libérer mémoire au début de l'époque
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        net.train()
        total_loss = 0
        for xb, vb in tqdm(train_loader, desc=f"Epoch {epoch}/{start_epoch + epochs - 1} [Train]"):
            xb, vb = xb.to(device), vb.to(device)
            pred = net(xb)
            loss = criterion(pred, vb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / n_train
        
        # Appliquer le rafraîchissement du modèle toutes les 50 époques
        if epoch % 50 == 0:
            net, optimizer = refresh_model(net, optimizer, device)

        net.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for xb, vb in val_loader:
                xb, vb = xb.to(device), vb.to(device)
                pred = net(xb)
                loss = criterion(pred, vb)
                val_loss += loss.item() * xb.size(0)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(vb.cpu().numpy())
        
        val_loss /= n_val
        
        # Libérer mémoire après validation
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # Calculer métriques supplémentaires
        mse = np.mean((np.array(all_preds) - np.array(all_targets))**2)
        mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
        r2 = np.corrcoef(np.array(all_preds), np.array(all_targets))[0, 1]**2
        
        # Mise à jour du scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, lr={current_lr:.14f}")
        
        # Ajuster min_delta en fonction du LR actuel
        stopper.adjust_for_lr(current_lr)

        is_improved = not stopper.step(val_loss)

        # Log et plot des métriques
        row = [
            epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{mse:.6f}", f"{mae:.6f}", f"{r2:.6f}",
            f"{current_lr:.12f}"
        ]
        
        # Ajouter la nouvelle ligne au CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        plot_training_metrics(csv_path, output_dir, csv_headers)

        # Sauvegarder le meilleur modèle complet quand la validation s'améliore significativement
        if is_improved:
            best_model_path = os.path.join(output_dir, 'best_valuenet_9x9.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr
            }, best_model_path)
            
            # Également enregistrer l'information dans un fichier texte pour référence rapide
            best_info_path = os.path.join(output_dir, 'best_model_info.txt')
            with open(best_info_path, 'w') as f:
                f.write(f"Best model:\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Val Loss: {val_loss:.6f}\n")
                f.write(f"Train Loss: {train_loss:.6f}\n")
                f.write(f"Learning Rate: {current_lr:.14f}\n")
                f.write(f"MSE: {mse:.6f}\n")
                f.write(f"MAE: {mae:.6f}\n")
                f.write(f"R²: {r2:.6f}\n")
            
            print(f"New best model saved at epoch {epoch} (LR: {current_lr:.14f})")

        # Sauvegarder l'état complet à chaque époque (écrase la version précédente)
        last_model_path = os.path.join(output_dir, 'last_model_full.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,   
            'learning_rate': current_lr
        }, last_model_path)
            
    print("Training complete. Models saved.")


def resume_training(checkpoint_path, npz_path, epochs=1000, batch_size=512, 
                   ultra_low_lr=False):
    """Reprend l'entraînement en continuant les métriques existantes"""
    print(f"Resuming training from: {checkpoint_path}")
    
    # Charger le checkpoint
    try:
        checkpoint = torch.load(checkpoint_path)
        
        if not isinstance(checkpoint, dict):
            raise ValueError("Le checkpoint n'est pas un dictionnaire complet")
        
        # Vérifier les informations minimales requises
        required_keys = ['model_state_dict', 'epoch', 'learning_rate']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            raise ValueError(f"Informations manquantes dans le checkpoint: {missing_keys}")
            
        # Extraire les informations
        checkpoint_epoch = checkpoint['epoch']
        checkpoint_lr = checkpoint['learning_rate']
        
        print(f"Checkpoint chargé: époque {checkpoint_epoch}, LR {checkpoint_lr:.14}")
    except Exception as e:
        print(f"ERREUR lors du chargement: {e}")
        return
    
    # Identifier le répertoire source des métriques
    source_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    csv_path = os.path.join(source_dir, 'metrics_all.csv')
    
    # Charger les métriques existantes
    if os.path.exists(csv_path):
        csv_headers, existing_metrics, _, _ = load_metrics_from_csv(csv_path)
        print(f"Métriques chargées: {len(existing_metrics)} époques")
    else:
        existing_metrics = []
        print("Aucune métrique existante trouvée")
    
    # Créer un nouveau répertoire pour cette reprise basé sur le dossier original
    source_folder_name = os.path.basename(source_dir)
    
    # Créer le nom du nouveau dossier avec _resumed à la fin
    new_folder_name = f"{source_folder_name}_resumed"
    
    # Si le dossier existe déjà, ajouter un index incrémental
    base_path = os.path.dirname(os.path.abspath(__file__))
    counter = 1
    new_output_dir = os.path.join(base_path, new_folder_name)
    
    while os.path.exists(new_output_dir):
        new_folder_name = f"{source_folder_name}_resumed_{counter}"
        new_output_dir = os.path.join(base_path, new_folder_name)
        counter += 1
    
    os.makedirs(new_output_dir, exist_ok=True)
    print(f"Résultats dans: {new_output_dir}")
    
    # Démarrer l'entraînement avec l'historique complet
    train_value_9x9(
        npz_path=npz_path,
        batch_size=batch_size,
        epochs=epochs,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        resume_from=checkpoint_path,
        continue_output_dir=new_output_dir,
        existing_metrics=existing_metrics,  # Passer les métriques existantes
        start_epoch=checkpoint_epoch + 1,
        last_lr=checkpoint_lr
    )


# python3 go_value_9x9_v3.py --resume training_logs_value9x9_v5_2025-06-06_18-21-00/last_model_full.pth --epochs 10000 --bs 512

if __name__ == '__main__':
    import argparse
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rel_path = os.path.join('dataset', 'value_dataset', 'value_dataset_9x9.npz')
    default_data_path = os.path.join(base_dir, rel_path)
    
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=default_data_path, help='Path to training data')
    p.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train')
    p.add_argument('--bs', type=int, default=512, help='Batch size')
    p.add_argument('--lr', type=float, default=3e-4, help='Initial learning rate')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                  help='Device to use for training')
    p.add_argument('--resume', type=str, default=None,
                  help='Path to model checkpoint to resume training')
    p.add_argument('--ultra_low_lr', action='store_true',
                  help='Start with extremely low learning rate (5e-6)')
    p.add_argument('--fresh', action='store_true', 
                  help='Start a fresh training run (ignore previous metrics)')
    
    args = p.parse_args()
    
    if args.resume and not args.fresh:
        resume_training(
            checkpoint_path=args.resume,
            npz_path=args.data,
            epochs=args.epochs,
            batch_size=args.bs,
            ultra_low_lr=args.ultra_low_lr
        )
    else:
        train_value_9x9(
            args.data,
            batch_size=args.bs,
            lr=args.lr,
            epochs=args.epochs,
            device=args.device
        )
